"""Shared benchmark helpers: environment report, timing, GPU peaks, SAELens adapter, CSV output."""

import csv
import json
import platform
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).parent / "results"

# Dense (no structured sparsity) vendor-datasheet peaks: TFLOP/s per input dtype, DRAM GB/s.
# float32 is the CUDA-core rate: saeflow and torch both default to non-TF32 fp32 matmuls.
# Matched against torch.cuda.get_device_name(); first entry whose substrings all appear wins.
# Override with bench/plot.py --peak-tflops / --peak-gbps for anything missing or different.
PEAKS = [
    (("H100", "PCIe"), {"bfloat16": 756, "float16": 756, "float32": 51}, 2000),
    (("H100",), {"bfloat16": 989, "float16": 989, "float32": 67}, 3350),
    (("A100", "80GB"), {"bfloat16": 312, "float16": 312, "float32": 19.5}, 2039),
    (("A100",), {"bfloat16": 312, "float16": 312, "float32": 19.5}, 1555),
    (("L40S",), {"bfloat16": 362, "float16": 362, "float32": 91.6}, 864),
    (("L4",), {"bfloat16": 121, "float16": 121, "float32": 30.3}, 300),
    (("A10",), {"bfloat16": 125, "float16": 125, "float32": 31.2}, 600),
    (("RTX 4090",), {"bfloat16": 165, "float16": 165, "float32": 82.6}, 1008),
    (("T4",), {"float16": 65, "float32": 8.1}, 320),
]

DTYPE_NAMES = {torch.float32: "float32", torch.bfloat16: "bfloat16", torch.float16: "float16"}


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_name():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else f"cpu-{platform.machine()}"


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def peaks(gpu, dtype_name, peak_tflops=None, peak_gbps=None):
    """(TFLOP/s, GB/s) ceilings for GPU `gpu`; None where unknown and not overridden."""
    for subs, flops, gbps in PEAKS:
        if all(s in gpu for s in subs):
            return peak_tflops or flops.get(dtype_name), peak_gbps or gbps
    return peak_tflops, peak_gbps


def parse_dtypes(s):
    """'auto' -> fp32 plus the GPU's native 16-bit type (bf16 needs sm80+; T4 gets fp16).
    Otherwise a comma list such as 'float32,bfloat16'."""
    if s != "auto":
        return [getattr(torch, d) for d in s.split(",")]
    if not torch.cuda.is_available():
        return [torch.float32]
    half = torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16
    return [torch.float32, half]


def env_info():
    def version(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    info = {
        "gpu": gpu_name(),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": version("triton"),
        "sae_lens": version("sae-lens"),
        "python": platform.python_version(),
    }
    for k, v in info.items():
        print(f"{k:>20}: {v}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"env_{slug(info['gpu'])}.json").write_text(json.dumps(info, indent=2))
    return info


def time_ms(fn, warmup=10, rep=50):
    """Median wall time of fn() in ms. CUDA: triton's do_bench (L2 flushed between runs)."""
    if torch.cuda.is_available():
        from triton.testing import do_bench

        return do_bench(fn, warmup=25, rep=100, return_mode="median")
    for _ in range(min(warmup, 2)):
        fn()
    times = []
    for _ in range(min(rep, 3)):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1e3)
    return statistics.median(times)


def saelens_sae(ref):
    """A SAELens SAE holding the same weights as reference SAE `ref`, or None (reason printed)."""
    try:
        import sae_lens

        kinds = {
            "relu": ("StandardSAE", "StandardSAEConfig", {}),
            "jumprelu": ("JumpReLUSAE", "JumpReLUSAEConfig", {}),
            "topk": ("TopKSAE", "TopKSAEConfig", {"k": ref.k}),
        }
        cls, cfg_cls, extra = kinds[ref.activation]
        cfg = getattr(sae_lens, cfg_cls)(
            d_in=ref.d_in, d_sae=ref.d_sae, dtype=DTYPE_NAMES[ref.W_enc.dtype], device=str(ref.W_enc.device), **extra
        )
        sae = getattr(sae_lens, cls)(cfg)
        with torch.no_grad():
            for name in ("W_enc", "W_dec", "b_enc", "b_dec", "threshold"):
                if getattr(ref, name, None) is not None:
                    getattr(sae, name).copy_(getattr(ref, name))
        return sae.requires_grad_(False)
    except Exception as e:  # noqa: BLE001 - SAELens is optional and its API moves; report and skip.
        print(f"SAELens comparison skipped: {type(e).__name__}: {e}")
        return None


FIELDS = [
    "gpu", "op", "impl", "dtype", "activation", "d_in", "d_sae", "tokens", "l0",
    "ms", "tflops", "gbps", "intensity", "config",
]


class Results:
    """Writes rows to bench/results/<name>_<gpu>[_<tag>].csv as they come, so an
    interrupted sweep keeps everything measured so far."""

    def __init__(self, name, tag=None):
        RESULTS.mkdir(exist_ok=True)
        self.path = RESULTS / f"{name}_{slug(gpu_name())}{'_' + tag if tag else ''}.csv"
        self.rows = []

    def add(self, *, flops, bytes_moved, ms, **row):
        row.update(
            gpu=gpu_name(), ms=round(ms, 5),
            tflops=round(flops / ms / 1e9, 3), gbps=round(bytes_moved / ms / 1e6, 2),
            intensity=round(flops / bytes_moved, 3),
        )
        self.rows.append(row)
        print("  ".join(f"{k}={row.get(k)}" for k in ("op", "impl", "dtype", "d_sae", "tokens", "l0", "ms", "tflops", "gbps")))
        with open(self.path, "w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS)
            w.writeheader()
            w.writerows(self.rows)


def parse_widths(s):
    """'2304x16384,768x24576' -> [(2304, 16384), (768, 24576)]"""
    return [tuple(int(v) for v in w.split("x")) for w in s.split(",")]


def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def oom_guard(fn):
    """Run fn(); on CUDA OOM free memory and return None so the sweep continues."""
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        free_memory()
        print("  skipped: out of memory")
        return None
