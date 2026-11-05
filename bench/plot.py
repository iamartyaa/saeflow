"""Plot bench/results/*.csv: a roofline per GPU and dtype, and time vs tokens per op and width.

    python bench/plot.py [--peak-tflops X --peak-gbps Y]
"""

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import RESULTS, peaks, slug  # noqa: E402

COLORS = {"saeflow": "tab:red", "torch": "tab:blue", "torch_csr": "tab:green", "saelens": "tab:purple"}
MARKERS = {"encode": "o", "decode": "s"}


def roofline(rows, gpu, dtype, peak_tflops, peak_gbps):
    pts = [r for r in rows if r["op"] in MARKERS and float(r["tflops"]) > 0]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = [float(r["intensity"]) for r in pts]
    lo, hi = min(xs) / 3, max(xs) * 3
    tflops, gbps = peaks(gpu, dtype, peak_tflops, peak_gbps)
    if gbps:
        ridge = tflops * 1e3 / gbps if tflops else hi
        ax.plot([lo, min(ridge, hi)], [gbps * lo / 1e3, gbps * min(ridge, hi) / 1e3], "k-", lw=1.5)
        ax.text(lo, gbps * lo / 1e3, f" {gbps:g} GB/s", va="bottom", fontsize=8)
    if tflops:
        ridge = tflops * 1e3 / gbps if gbps else lo
        ax.plot([max(ridge, lo), hi], [tflops, tflops], "k-", lw=1.5)
        ax.text(hi, tflops, f"{tflops:g} TFLOP/s ", ha="right", va="bottom", fontsize=8)
    seen = set()
    for r in pts:
        label = f"{r['op']} / {r['impl']}"
        ax.scatter(float(r["intensity"]), float(r["tflops"]), c=COLORS.get(r["impl"], "gray"),
                   marker=MARKERS[r["op"]], s=28, alpha=0.8, label=None if label in seen else label)
        seen.add(label)
    ax.set(xscale="log", yscale="log", xlabel="arithmetic intensity (FLOP / byte)", ylabel="achieved TFLOP/s",
           title=f"Roofline: {gpu}, {dtype}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    path = RESULTS / f"roofline_{slug(gpu)}_{dtype}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def time_vs_tokens(rows, gpu, op):
    groups = defaultdict(list)
    for r in rows:
        if r["op"] == op:
            groups[(r["dtype"], r["d_in"], r["d_sae"], r["activation"], r["l0"])].append(r)
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.8), squeeze=False)
    for ax, (key, rs) in zip(axes[0], sorted(groups.items())):
        dtype, d_in, d_sae, act, l0 = key
        by_impl = defaultdict(list)
        for r in rs:
            by_impl[r["impl"]].append((int(r["tokens"]), float(r["ms"])))
        for impl, pts in by_impl.items():
            pts.sort()
            ax.plot(*zip(*pts), "o-", c=COLORS.get(impl, "gray"), label=impl, ms=4)
        extra = f" {act}" if act else ""
        extra += f" L0={l0}" if l0 else ""
        ax.set(xscale="log", yscale="log", xlabel="tokens", ylabel="ms", title=f"{d_in}x{d_sae} {dtype}{extra}")
        ax.title.set_fontsize(9)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(f"{op}: {gpu}")
    fig.tight_layout()
    path = RESULTS / f"{op}_time_{slug(gpu)}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--peak-tflops", type=float)
    p.add_argument("--peak-gbps", type=float)
    args = p.parse_args()

    rows = []
    for path in sorted(RESULTS.glob("*.csv")):
        with open(path) as fh:
            rows += list(csv.DictReader(fh))
    by_gpu = defaultdict(list)
    for r in rows:
        by_gpu[r["gpu"]].append(r)
    for gpu, rs in by_gpu.items():
        for dtype in sorted({r["dtype"] for r in rs}):
            roofline([r for r in rs if r["dtype"] == dtype], gpu, dtype, args.peak_tflops, args.peak_gbps)
        for op in ("encode", "decode", "forward", "to_csr"):
            time_vs_tokens(rs, gpu, op)


if __name__ == "__main__":
    main()
