"""Encoder benchmark: saeflow fused encode vs PyTorch reference vs SAELens.

Also times the full forward pass (encode -> decode) with a JumpReLU threshold
set for a target L0, where saeflow uses encode_sparse + sparse decode.

    python bench/bench_encode.py                      # default sweep
    python bench/bench_encode.py --widths 2304x65536 --tokens 4096,16384
"""

import argparse

import torch
from common import DTYPE_NAMES, Results, parse_dtypes, device, env_info, free_memory, oom_guard, parse_widths, saelens_sae, time_ms

import saeflow
from saeflow.kernels import fused_encode
from saeflow.kernels.encode import best_config
from saeflow.reference import SAE


def check(got, want, dtype):
    tol = (1e-3 if dtype == torch.float32 else 5e-2) * (1 + want.abs().max().item())
    bad = ((got.float() - want.float()).abs() > tol).float().mean().item()
    # A few elements sitting on the activation cut may flip under rounding; anything more is a bug.
    assert bad < 1e-3, f"saeflow output disagrees with reference on {bad:.2%} of elements"


def make_sae(d_in, d_sae, activation, dtype, l0):
    sae = SAE(d_in, d_sae, activation, device=device())
    sae.b_enc.normal_(0, 0.05)
    if activation == "jumprelu":
        # One global threshold that leaves ~l0 active features per token on Gaussian inputs.
        pre = sae.encode_pre(torch.randn(256, d_in, device=device()))
        sae.threshold.fill_(pre.topk(l0, dim=1).values[:, -1].mean().item())
    return sae.to(dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--widths", default="2304x16384,768x24576,2304x65536", help="d_in x d_sae pairs")
    p.add_argument("--tokens", default="1024,4096,16384")
    p.add_argument("--activations", default="relu,jumprelu")
    p.add_argument("--l0", type=int, default=64, help="target L0 for the forward-pass benchmark")
    p.add_argument("--dtypes", default="auto", help="'auto' or a list like float32,bfloat16")
    p.add_argument("--tag", help="suffix for the results CSV, to keep several runs apart")
    p.add_argument("--smoke", action="store_true", help="tiny shapes, for CI under TRITON_INTERPRET=1")
    args = p.parse_args()
    if args.smoke:
        args.widths, args.tokens = "64x256", "37"
        args.l0 = 8

    env_info()
    out = Results("encode", args.tag)
    for d_in, d_sae in parse_widths(args.widths):
        for dtype in parse_dtypes(args.dtypes):
            e = torch.finfo(dtype).bits // 8
            for activation in args.activations.split(","):
                sae = make_sae(d_in, d_sae, activation, dtype, args.l0)
                lens = saelens_sae(sae)
                for M in map(int, args.tokens.split(",")):
                    row = dict(dtype=DTYPE_NAMES[dtype], activation=activation, d_in=d_in, d_sae=d_sae, tokens=M)

                    def run():
                        x = torch.randn(M, d_in, device=device(), dtype=dtype)
                        check(saeflow.encode(sae, x), sae.encode(x), dtype)
                        impls = {
                            "saeflow": lambda: fused_encode(x, sae.W_enc, sae.b_enc, sae.b_dec, sae.threshold, activation),
                            "torch": lambda: sae.encode(x),
                        }
                        if lens is not None:
                            impls["saelens"] = lambda: lens.encode(x)
                        flops = 2 * M * d_in * d_sae
                        bytes_moved = (M * d_in + d_in * d_sae + M * d_sae) * e
                        for impl, fn in impls.items():
                            ms = oom_guard(lambda: time_ms(fn))
                            if ms is not None:
                                cfg = str(best_config()) if impl == "saeflow" else ""
                                out.add(op="encode", impl=impl, flops=flops, bytes_moved=bytes_moved, ms=ms,
                                        config=cfg, **row)

                        if activation != "jumprelu":
                            return
                        csr = saeflow.encode_sparse(sae, x)
                        l0 = csr[1].numel() / M
                        check(saeflow.forward(sae, x), sae(x), dtype)
                        fwd = {"saeflow": lambda: saeflow.forward(sae, x), "torch": lambda: sae(x)}
                        if lens is not None:
                            fwd["saelens"] = lambda: lens(x)
                        # Dense-equivalent FLOPs and minimal I/O for every impl, so tflops reads as "effective".
                        flops = 4 * M * d_in * d_sae
                        bytes_moved = (2 * M * d_in + 2 * d_in * d_sae) * e
                        for impl, fn in fwd.items():
                            ms = oom_guard(lambda: time_ms(fn))
                            if ms is not None:
                                out.add(op="forward", impl=impl, flops=flops, bytes_moved=bytes_moved, ms=ms,
                                        config="", **{**row, "l0": round(l0, 1)})

                    oom_guard(run)
                del sae, lens  # free before the next SAE so two never share the GPU
                free_memory()
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
