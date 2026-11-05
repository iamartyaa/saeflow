"""Decoder benchmark: saeflow sparse decode vs dense PyTorch, torch CSR (cuSPARSE) and SAELens,
plus the dense -> CSR conversion (saeflow compaction kernel vs torch.Tensor.to_sparse_csr).

Active features are drawn uniformly at random (L0 per token, unique per row). Real SAE
feature frequencies are heavy-tailed, which helps L2 reuse of hot W_dec rows, so this is
closer to a worst case for the gather.

    python bench/bench_decode.py
    python bench/bench_decode.py --widths 2304x65536 --l0 32,128 --tokens 4096,65536
"""

import argparse

import torch
from common import DTYPE_NAMES, Results, default_dtypes, device, env_info, oom_guard, parse_widths, saelens_sae, time_ms

from saeflow.kernels import dense_to_csr, sparse_decode
from saeflow.reference import SAE


def random_csr(M, d_sae, l0, dtype):
    # Unique sorted indices: one random pick inside each of l0 equal strides of [0, d_sae).
    step = d_sae // l0
    idx = torch.arange(l0, device=device()) * step + torch.randint(0, step, (M, l0), device=device())
    values = (torch.rand(M, l0, device=device()) + 0.1).to(dtype)
    indptr = torch.arange(0, M * l0 + 1, l0, device=device(), dtype=torch.int64)
    return idx, values, (indptr, idx.flatten().to(torch.int32), values.flatten())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--widths", default="2304x16384,768x24576,2304x65536", help="d_in x d_sae pairs")
    p.add_argument("--tokens", default="1024,4096,16384,65536")
    p.add_argument("--l0", default="32,64,128")
    p.add_argument("--smoke", action="store_true", help="tiny shapes, for CI under TRITON_INTERPRET=1")
    args = p.parse_args()
    if args.smoke:
        args.widths, args.tokens, args.l0 = "64x256", "37", "8"

    env_info()
    out = Results("decode")
    for d_in, d_sae in parse_widths(args.widths):
        for dtype in default_dtypes():
            e = torch.finfo(dtype).bits // 8
            sae = SAE(d_in, d_sae, "relu", device=device()).to(dtype)
            sae.b_dec.normal_(0, 0.1)
            lens = saelens_sae(sae)
            for l0 in map(int, args.l0.split(",")):
                for M in map(int, args.tokens.split(",")):
                    row = dict(dtype=DTYPE_NAMES[dtype], activation="", d_in=d_in, d_sae=d_sae, tokens=M, l0=l0, config="")

                    def run():
                        idx, vals, csr = random_csr(M, d_sae, l0, dtype)
                        flops = 2 * M * l0 * d_in
                        bytes_moved = M * l0 * (d_in * e + 4 + e) + M * d_in * e + (M + 1) * 8
                        ms = time_ms(lambda: sparse_decode(*csr, sae.W_dec, sae.b_dec))
                        out.add(op="decode", impl="saeflow", flops=flops, bytes_moved=bytes_moved, ms=ms, **row)

                        f_csr = torch.sparse_csr_tensor(csr[0], csr[1].long(), csr[2], (M, d_sae))
                        bias = sae.b_dec.expand(M, d_in)
                        try:
                            ms = oom_guard(lambda: time_ms(lambda: torch.addmm(bias, f_csr, sae.W_dec)))
                        except (RuntimeError, NotImplementedError) as err:  # cuSPARSE lacks some dtypes
                            print(f"  torch_csr skipped: {err}")
                            ms = None
                        if ms is not None:
                            out.add(op="decode", impl="torch_csr", flops=flops, bytes_moved=bytes_moved, ms=ms, **row)

                        # Dense baselines need the [M, d_sae] feature matrix; skipped on OOM.
                        f = torch.zeros(M, d_sae, device=device(), dtype=dtype).scatter_(1, idx, vals)
                        want = sae.decode(f)
                        got = sparse_decode(*csr, sae.W_dec, sae.b_dec)
                        tol = 1e-3 if dtype == torch.float32 else 5e-2
                        torch.testing.assert_close(got.float(), want.float(), atol=tol * l0 ** 0.5, rtol=tol)

                        flops_dense = 2 * M * d_sae * d_in
                        bytes_dense = (M * d_sae + d_sae * d_in + M * d_in) * e
                        dense = {"torch": lambda: sae.decode(f)}
                        if lens is not None:
                            dense["saelens"] = lambda: lens.decode(f)
                        for impl, fn in dense.items():
                            ms = oom_guard(lambda: time_ms(fn))
                            if ms is not None:
                                out.add(op="decode", impl=impl, flops=flops_dense, bytes_moved=bytes_dense, ms=ms, **row)

                        nnz = M * l0
                        bytes_conv = M * d_sae * e + nnz * (4 + e) + (M + 1) * 8
                        for impl, fn in {"saeflow": lambda: dense_to_csr(f), "torch": lambda: f.to_sparse_csr()}.items():
                            ms = oom_guard(lambda: time_ms(fn))
                            if ms is not None:
                                out.add(op="to_csr", impl=impl, flops=0, bytes_moved=bytes_conv, ms=ms, **row)

                    oom_guard(run)
    out.write()


if __name__ == "__main__":
    main()
