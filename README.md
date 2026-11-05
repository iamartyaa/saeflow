# saeflow

Triton kernels and an activation-caching pipeline for sparse-autoencoder (SAE) interpretability.

## The problem

SAEs decompose a model's activations into a large dictionary of features: a residual
stream vector of width `d_in` (768 for GPT-2 small, 2304 for Gemma-2-2B) is encoded into
`d_sae` features (16k to 1M), and decoded back. Interpretability work runs this pair
over a lot of tokens: caching activations from the model, training SAEs on them, and
then encoding whole datasets to find what each feature responds to. Encode and decode
are each a `tokens x d_in x d_sae` matrix multiply, so that is where the compute goes.

The features are very sparse. A trained SAE typically has an L0 (active features per
token) in the tens to low hundreds out of tens of thousands. Standard PyTorch SAE code
ignores this:

- **Encode** writes the full `[tokens, d_sae]` pre-activation to memory, then reads it
  back to apply the bias and activation, then writes the result again.
- **Decode** runs a dense `[tokens, d_sae] @ [d_sae, d_in]` GEMM on a matrix that is
  more than 99% zeros. At `d_sae = 65536` and `L0 = 64`, 1023 of every 1024
  multiply-adds are by zero.
- **Caching** stores dense activations, or dense feature activations that are mostly zeros.

saeflow attacks each of these.

## Design

### Fused encode (`saeflow/kernels/encode.py`)

`fused_encode` computes `act((x - b_dec) @ W_enc + b_enc)` in one tiled Triton GEMM:

- **Prologue:** each `[BLOCK_M, BLOCK_K]` tile of `x` has `b_dec` subtracted as it is
  loaded, in fp32, before being cast to the input dtype for the tensor-core dot.
- **Main loop:** fp32 accumulation over `d_in`, with grouped tile ordering so that
  `GROUP_M` row-tiles reuse each `W_enc` column-tile while it is in L2.
- **Epilogue:** `b_enc` and the activation (ReLU, or JumpReLU with a per-feature
  threshold) are applied to the fp32 accumulator in registers. The pre-activation
  matrix is never written to global memory. Only the final features are stored.
- **Optional nonzero counting:** the epilogue can also add each tile's per-row nonzero
  count into an int32 buffer, one atomic per row per tile. That is exactly the input the
  dense -> CSR compaction needs, so the counting pass costs nothing extra.

Inputs can be fp32, bf16 or fp16, with fp32 accumulation. fp32 uses IEEE products
unless `torch.backends.cuda.matmul.allow_tf32` (or `allow_tf32=True`) says otherwise,
so it matches the PyTorch reference. Block sizes are autotuned over `CONFIGS` keyed on
`(M, N, K)`. Pass `config={...}` to pin them, and use `best_config()` to see what the
autotuner picked.

### Sparse decode (`saeflow/kernels/decode.py`)

Features are stored as CSR: `indptr [tokens + 1]`, `indices [nnz]` (int32), and
`values [nnz]`. `sparse_decode` computes

```
x_hat[m] = b_dec + sum over j in row m of values[j] * W_dec[indices[j]]
```

with one program per (token, `BLOCK_D` slice of `d_in`). Each program walks its row's
active features `BLOCK_NNZ` at a time, gathers those rows of `W_dec`, and accumulates
in fp32. `W_dec` is `[d_sae, d_in]` row-major, so every gathered row is a contiguous,
coalesced load.

**Why low L0 matters.** For `T` tokens, the dense decode does `2 * T * d_sae * d_in`
FLOPs and must first materialise and read a `T x d_sae` feature matrix. The sparse
decode does `2 * T * L0 * d_in` FLOPs and reads `T * L0` rows of `W_dec` plus the CSR
arrays. FLOPs drop by a factor of `d_sae / L0` (for example 1024 at `d_sae = 65536`,
`L0 = 64`). The kernel becomes bandwidth-bound on the gather rather than compute-bound
on zeros. A dense feature matrix for 16k tokens at 65k features in bf16 is 2 GiB, while
the CSR form at `L0 = 64` is 6 MiB. Hot features (real SAEs have heavy-tailed feature
frequencies) also stay in L2 across tokens. Where the kernel actually lands is what the
roofline benchmark is for.

**Dense -> CSR fast path.** `dense_to_csr` turns encoder output into CSR. The row
offsets are a cumulative sum of the per-row counts, which `fused_encode` can produce for
free. Then a compaction kernel, one program per row, scans the row in chunks,
prefix-sums the nonzero mask with `tl.cumsum` to get each nonzero's output slot, and
scatters indices and values there. The output is sorted by feature index (canonical
CSR). The only host sync is the one that sizes the output. For TopK SAEs there is no
compaction at all: `torch.topk` already gives fixed-width `(index, value)` pairs.

### SAE-level API (`saeflow/ops.py`)

```python
import saeflow
from saeflow import SAE

sae = SAE(d_in=2304, d_sae=65536, activation="jumprelu").cuda().bfloat16()
f   = saeflow.encode(sae, x)          # dense features, fused kernel
csr = saeflow.encode_sparse(sae, x)   # (indptr, indices, values)
x_hat = saeflow.decode_sparse(sae, csr)
x_hat = saeflow.forward(sae, x)       # encode_sparse -> decode_sparse
```

`saeflow.reference.SAE` is a plain PyTorch ReLU / JumpReLU / TopK SAE with SAELens
parameter names and shapes. It is the ground truth for every test. The ops accept any
object with the same attributes.

### Activation cache (`saeflow/cache.py`)

```python
from transformer_lens import HookedTransformer
from saeflow.cache import ActivationReader, cache_activations

model = HookedTransformer.from_pretrained("gpt2")
cache_activations(model, tokens, ["blocks.8.hook_resid_pre"], "cache/gpt2-l8",
                  batch_size=16, batches_per_shard=64, dtype=torch.bfloat16,
                  encoders={"blocks.8.hook_resid_pre": sae})   # optional: store SAE features only

for batch in ActivationReader("cache/gpt2-l8").iter_batches("blocks.8.hook_resid_pre", 4096):
    ...   # dense [4096, d] tensor, or a CSR tuple for encoded hooks
```

- **Forward passes:** batched `run_with_cache` with a `names_filter`, `return_type=None`
  (no unembed), and `stop_at_layer` set to just past the deepest requested hook.
- **Storage:** shards are `.npy` files. The reader memory-maps them and streams
  fixed-size batches across shard boundaries. bf16 is stored as raw bits, with the dtype
  recorded in the manifest.
- **Resumable writes:** each shard's files are fsynced and atomically renamed before
  `manifest.json` records the shard. An interrupted run leaves only whole shards behind,
  and rerunning with the same arguments resumes at the first missing shard. Resuming
  with different arguments is refused.
- **On-the-fly encoding:** with `encoders`, a hook's activations go through
  `encode_sparse` on the GPU and only CSR features are written. At typical L0 that is
  a fraction of the dense activation size.

The cache targets TransformerLens 2.x / 3.x `HookedTransformer` (`pip install -e ".[cache]"`).
TransformerLens 4 moved to `TransformerBridge`. Support for that is on the roadmap.

## Running the tests

The tests compare every kernel with the reference across fp32 / bf16 / fp16 and shapes
that are not multiples of any block size, including empty rows and an all-zero matrix.
They also check the cache: write, interrupt-then-resume, exact read-back, bf16 storage,
and sparse encoding on the fly. With no CUDA device they run on CPU through the Triton
interpreter (`TRITON_INTERPRET=1` is set automatically). Triton ships Linux wheels
only, so on macOS run them in a Linux container or rely on CI.

```bash
pip install -e ".[test]"
pytest -q
```

On a GPU the same tests run the compiled, autotuned kernels.

## Running the benchmarks

```bash
pip install -e ".[bench]"
python bench/bench_encode.py    # fused encode, and full forward at a target L0
python bench/bench_decode.py    # sparse decode and dense -> CSR conversion
python bench/plot.py            # roofline + time-vs-tokens plots
```

- **What runs:** each script prints the GPU, CUDA, torch, Triton and SAELens versions,
  and checks saeflow's output against the reference before timing anything.
- **Sweep:** SAE widths `2304x16384`, `768x24576` and `2304x65536` (`--widths`), token
  counts via `--tokens`, and L0 via `--l0`. It runs in fp32 and in the GPU's native
  16-bit type (bf16 on sm80+, fp16 on T4).
- **Baselines:** saeflow is compared with the PyTorch reference, SAELens, and (for
  decode) torch's own CSR matmul.
- **Output:** rows go to `bench/results/<op>_<gpu>.csv` with ms, TFLOP/s, GB/s and
  arithmetic intensity. `plot.py` places them on a roofline using datasheet peaks for
  common GPUs (`--peak-tflops` / `--peak-gbps` to override).
- **Colab:** `notebooks/bench_colab.ipynb` runs the tests and the whole sweep on a
  Colab T4.

## Benchmarks

Results will be added after the first GPU run.

## Roadmap

- **TopK fused select:** keep a running per-row top-k across `d_sae` tiles inside the
  encoder, so TopK SAEs never write the dense `[tokens, d_sae]` output.
- **Encode straight to sparse:** have the epilogue append nonzeros to per-row slots via
  atomics, skipping the dense intermediate for ReLU / JumpReLU too.
- **Multi-GPU caching:** shard the token stream across ranks, with one manifest per
  rank merged at the end, and overlap GPU -> host copies with the next forward pass.
- **SAELens integration:** take `sae_lens.SAE` objects directly (architecture and
  activation-normalisation handling), and read saeflow caches as an SAELens activation
  store.
- **TransformerLens 4:** cache from `TransformerBridge` models as well as `HookedTransformer`.
- **Backward kernels** for training SAEs on sparse features.

## License

MIT
