"""Sparse SAE decode and the dense -> CSR fast path.

SAE features are CSR: row m's active features are indices[indptr[m]:indptr[m+1]]
with matching values. Decode computes

    x_hat[m] = b_dec + sum_j values[j] * W_dec[indices[j]]

so each token reads only its L0 active rows of W_dec (each row contiguous,
so the gather is coalesced) instead of running a dense
[tokens, d_sae] @ [d_sae, d_in] GEMM that is ~99.9% multiplications by zero.
"""

import torch
import triton
import triton.language as tl

from . import INTERPRET

CONFIGS = [
    triton.Config({"BLOCK_NNZ": nnz, "BLOCK_D": d}, num_warps=w)
    for nnz, d, w in [(16, 128, 2), (16, 256, 4), (32, 256, 4), (16, 512, 4), (32, 512, 8), (64, 256, 8)]
]
INTERPRET_CONFIG = {"BLOCK_NNZ": 16, "BLOCK_D": 32}
COMPACT_BLOCK = 1024
INTERPRET_COMPACT_BLOCK = 64


@triton.jit
def _decode_kernel(
    indptr_ptr, idx_ptr, val_ptr, w_ptr, b_ptr, out_ptr,
    D, stride_wn, stride_wd, stride_om, stride_od,
    HAS_BIAS: tl.constexpr, BLOCK_NNZ: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # One program = one token row x one BLOCK_D slice of d_in.
    row = tl.program_id(0).to(tl.int64)
    offs_d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for j0 in range(start, end, BLOCK_NNZ):
        j = j0 + tl.arange(0, BLOCK_NNZ)
        mask_j = j < end
        idx = tl.load(idx_ptr + j, mask=mask_j, other=0).to(tl.int64)
        val = tl.load(val_ptr + j, mask=mask_j, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + idx[:, None] * stride_wn + offs_d[None, :] * stride_wd,
            mask=mask_j[:, None] & mask_d[None, :], other=0.0,
        )
        acc += tl.sum(val[:, None] * w.to(tl.float32), axis=0)

    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_om + offs_d * stride_od, acc.to(out_ptr.dtype.element_ty), mask=mask_d)


_decode_autotuned = triton.autotune(CONFIGS, key=["D"])(_decode_kernel)


@triton.jit
def _compact_kernel(f_ptr, indptr_ptr, idx_ptr, val_ptr, N, stride_fm, stride_fn, BLOCK_N: tl.constexpr):
    # One program per row: scan the row in BLOCK_N chunks, prefix-sum the nonzero
    # mask to get each nonzero's slot, scatter it there. Output is sorted by index.
    row = tl.program_id(0).to(tl.int64)
    base = tl.load(indptr_ptr + row)
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        f = tl.load(f_ptr + row * stride_fm + offs * stride_fn, mask=offs < N, other=0.0)
        nz = f != 0.0
        slot = base + tl.cumsum(nz.to(tl.int32), axis=0) - 1
        tl.store(idx_ptr + slot, offs, mask=nz)
        tl.store(val_ptr + slot, f, mask=nz)
        base += tl.sum(nz.to(tl.int32), axis=0)


def dense_to_csr(f, row_nnz=None):
    """Dense features [M, d_sae] -> (indptr int64 [M+1], indices int32 [nnz], values [nnz]).

    row_nnz: per-row nonzero counts if already known (fused_encode can produce
    them for free); otherwise counted here. One host sync sizes the outputs.
    """
    M, N = f.shape
    if row_nnz is None:
        row_nnz = torch.count_nonzero(f, dim=1)
    indptr = torch.zeros(M + 1, dtype=torch.int64, device=f.device)
    indptr[1:] = row_nnz.cumsum(0)
    nnz = int(indptr[-1])
    indices = torch.empty(nnz, dtype=torch.int32, device=f.device)
    values = torch.empty(nnz, dtype=f.dtype, device=f.device)
    if nnz:
        block = INTERPRET_COMPACT_BLOCK if INTERPRET else COMPACT_BLOCK
        _compact_kernel[(M,)](f, indptr, indices, values, N, f.stride(0), f.stride(1), BLOCK_N=block)
    return indptr, indices, values


def sparse_decode(indptr, indices, values, W_dec, b_dec=None, *, out_dtype=None, config=None):
    """CSR features @ W_dec (+ b_dec) -> [M, d_in], accumulated in fp32.

    W_dec: [d_sae, d_in], any strides (row-major is the fast case: gathered rows are contiguous).
    config: dict with BLOCK_NNZ / BLOCK_D (and optionally num_warps) to pin instead of autotuning.
    """
    M = indptr.numel() - 1
    D = W_dec.shape[1]
    out = torch.empty(M, D, device=W_dec.device, dtype=out_dtype or W_dec.dtype)
    if M == 0:
        return out
    args = (
        indptr, indices, values, W_dec,
        b_dec if b_dec is not None else out,  # unused unless HAS_BIAS
        out, D, W_dec.stride(0), W_dec.stride(1), out.stride(0), out.stride(1),
    )
    if config is None and INTERPRET:
        config = INTERPRET_CONFIG
    if config is not None:
        _decode_kernel[(M, triton.cdiv(D, config["BLOCK_D"]))](*args, HAS_BIAS=b_dec is not None, **config)
    else:
        grid = lambda m: (M, triton.cdiv(D, m["BLOCK_D"]))  # noqa: E731
        _decode_autotuned[grid](*args, HAS_BIAS=b_dec is not None)
    return out


def csr_to_dense(indptr, indices, values, d_sae):
    """Inverse of dense_to_csr, via torch's own CSR tensor. For tests and inspection."""
    M = indptr.numel() - 1
    return torch.sparse_csr_tensor(indptr, indices.long(), values, (M, d_sae)).to_dense()
