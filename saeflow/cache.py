"""Activation caching for TransformerLens models.

Layout of a cache directory:

    manifest.json                                   config + list of finished shards
    <hook>/<shard:05d>.npy                          dense activations [rows, d]
    <hook>/<shard:05d>.{indptr,indices,values}.npy  CSR SAE features [rows, d_sae] (hooks with an encoder)

A row is one token position, in sequence-major order. A shard covers
`batches_per_shard` forward batches. Its files are fsynced and renamed into
place before the manifest records it, so after an interruption the cache holds
only whole shards and a rerun with the same arguments resumes at the first
missing one. Shards are plain .npy, so the reader memory-maps them.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import torch

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_CSR_PARTS = ("indptr", "indices", "values")


def _to_numpy(t):
    t = t.detach().contiguous().cpu()
    # numpy has no bfloat16: store the raw bits, the manifest records the dtype.
    return (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy()


def _from_numpy(a, dtype):
    t = torch.from_numpy(np.ascontiguousarray(a))
    return t.view(dtype) if dtype == torch.bfloat16 else t


def _atomic_write(path, write):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        write(fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _save(path, t):
    _atomic_write(path, lambda fh: np.save(fh, _to_numpy(t)))


def _cat_csr(parts):
    indptrs, offset = [torch.zeros(1, dtype=torch.int64)], 0
    for indptr, _, _ in parts:
        indptrs.append(indptr[1:].cpu() + offset)
        offset += int(indptr[-1])
    return (
        torch.cat(indptrs),
        torch.cat([p[1].cpu() for p in parts]),
        torch.cat([p[2].cpu() for p in parts]),
    )


def _stop_layer(hook_names):
    """Run only as many blocks as the deepest requested hook needs."""
    layers = [re.match(r"blocks\.(\d+)\.", h) for h in hook_names]
    return max(int(m.group(1)) for m in layers) + 1 if all(layers) else None


@torch.inference_mode()
def cache_activations(model, tokens, hook_names, out_dir, *, batch_size=8, batches_per_shard=32,
                      dtype=torch.float32, encoders=None):
    """Run `model` over `tokens` [n_seqs, seq_len] and cache activations at `hook_names`.

    model: a TransformerLens HookedTransformer (anything with a compatible run_with_cache).
    encoders: optional {hook_name: sae}; those hooks are SAE-encoded on the fly with
      saeflow's kernels and stored as CSR feature activations instead of dense ones.
    Calling again with the same arguments after an interruption resumes the run.
    Returns the cache directory as a Path.
    """
    out_dir, hook_names, encoders = Path(out_dir), list(hook_names), encoders or {}
    unknown = set(encoders) - set(hook_names)
    if unknown:
        raise ValueError(f"encoders given for hooks not being cached: {sorted(unknown)}")
    if encoders:
        from .ops import encode_sparse

    config = {
        "hook_names": hook_names,
        "n_seqs": len(tokens),
        "seq_len": int(tokens.shape[1]),
        "batch_size": batch_size,
        "batches_per_shard": batches_per_shard,
        "dtype": str(dtype).removeprefix("torch."),
        "encoded": sorted(encoders),
    }
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config"] != config:
            raise ValueError(f"{out_dir} holds a cache with a different config: {manifest['config']}")
    else:
        manifest = {"config": config, "hooks": {}, "shards": []}
    for h in hook_names:
        (out_dir / h).mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    stop = _stop_layer(hook_names)
    seqs_per_shard = batch_size * batches_per_shard
    for start in range(len(manifest["shards"]) * seqs_per_shard, len(tokens), seqs_per_shard):
        end = min(start + seqs_per_shard, len(tokens))
        parts = {h: [] for h in hook_names}
        for b in range(start, end, batch_size):
            batch = tokens[b:min(b + batch_size, end)].to(device)
            _, acts = model.run_with_cache(batch, names_filter=hook_names, stop_at_layer=stop, return_type=None)
            for h in hook_names:
                a = acts[h].flatten(0, 1).flatten(1)  # [batch * pos, d]; per-head hooks are concatenated
                if h in encoders:
                    sae = encoders[h]
                    indptr, indices, values = encode_sparse(sae, a.to(sae.W_enc.dtype))
                    parts[h].append((indptr.cpu(), indices.cpu(), values.to(dtype).cpu()))
                else:
                    parts[h].append(a.to(dtype).cpu())

        shard = len(manifest["shards"])
        for h in hook_names:
            if h in encoders:
                for name, t in zip(_CSR_PARTS, _cat_csr(parts[h])):
                    _save(out_dir / h / f"{shard:05d}.{name}.npy", t)
                manifest["hooks"][h] = {"format": "csr", "d": int(encoders[h].W_enc.shape[1])}
            else:
                dense = torch.cat(parts[h])
                _save(out_dir / h / f"{shard:05d}.npy", dense)
                manifest["hooks"][h] = {"format": "dense", "d": int(dense.shape[1])}
        manifest["shards"].append({"n_seqs": end - start, "n_rows": (end - start) * config["seq_len"]})
        _atomic_write(manifest_path, lambda fh: fh.write(json.dumps(manifest, indent=2).encode()))
    return out_dir


class ActivationReader:
    """Streams a cache written by cache_activations back as batches."""

    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.dtype = _DTYPES[self.manifest["config"]["dtype"]]

    @property
    def hook_names(self):
        return self.manifest["config"]["hook_names"]

    def n_rows(self):
        return sum(s["n_rows"] for s in self.manifest["shards"])

    def _is_csr(self, hook):
        return self.manifest["hooks"][hook]["format"] == "csr"

    def _open(self, hook, shard):
        d = self.path / hook
        if self._is_csr(hook):
            return [np.load(d / f"{shard:05d}.{name}.npy", mmap_mode="r") for name in _CSR_PARTS]
        return np.load(d / f"{shard:05d}.npy", mmap_mode="r")

    def _rows(self, hook, arrays, a, b):
        if not self._is_csr(hook):
            return _from_numpy(arrays[a:b], self.dtype)
        indptr, indices, values = arrays
        lo, hi = int(indptr[a]), int(indptr[b])
        return (
            torch.from_numpy(indptr[a:b + 1] - lo),
            torch.from_numpy(np.ascontiguousarray(indices[lo:hi])),
            _from_numpy(values[lo:hi], self.dtype),
        )

    def _join(self, hook, pieces):
        return _cat_csr(pieces) if self._is_csr(hook) else torch.cat(pieces)

    def iter_batches(self, hook, batch_size):
        """Yield `batch_size`-row batches in cache order (the last may be short), crossing
        shard boundaries. Dense hooks yield [rows, d] tensors; encoded hooks yield CSR
        (indptr, indices, values) tuples, which saeflow.decode_sparse takes directly.
        """
        pending, n_pending = [], 0
        for i, shard in enumerate(self.manifest["shards"]):
            arrays, row = self._open(hook, i), 0
            while row < shard["n_rows"]:
                take = min(batch_size - n_pending, shard["n_rows"] - row)
                pending.append(self._rows(hook, arrays, row, row + take))
                n_pending, row = n_pending + take, row + take
                if n_pending == batch_size:
                    yield self._join(hook, pending)
                    pending, n_pending = [], 0
        if pending:
            yield self._join(hook, pending)
