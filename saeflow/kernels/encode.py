"""Fused SAE encoder: act((x - b_dec) @ W_enc + b_enc) in a single kernel.

The b_dec subtraction runs in the GEMM prologue on each loaded x tile, and the
bias plus ReLU / JumpReLU threshold run in the epilogue on the fp32
accumulator, so the [tokens, d_sae] pre-activation only ever exists in
registers. The epilogue can also count each row's nonzeros (one atomic per row
per tile), which is exactly what the dense -> CSR compaction needs.
"""

import torch
import triton
import triton.language as tl

from . import INTERPRET

ACTIVATIONS = {"relu": 0, "jumprelu": 1}

# Autotune search space; configs that do not fit a GPU's shared memory are skipped by the autotuner.
CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8}, num_warps=w, num_stages=s)
    for bm, bn, bk, w, s in [
        (128, 256, 64, 8, 3),
        (256, 128, 64, 8, 3),
        (128, 128, 64, 4, 4),
        (128, 128, 32, 4, 4),
        (64, 256, 32, 4, 4),
        (128, 64, 32, 4, 4),
        (64, 128, 32, 4, 4),
        (64, 64, 32, 2, 5),
        (32, 64, 32, 2, 5),
    ]
]
INTERPRET_CONFIG = {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 4}


@triton.jit
def _encode_kernel(
    x_ptr, w_ptr, b_enc_ptr, b_dec_ptr, thr_ptr, out_ptr, nnz_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wk, stride_wn, stride_om, stride_on,
    ACT: tl.constexpr, COUNT: tl.constexpr, PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # Grouped tile order: GROUP_M row-tiles share each W_enc column-tile while it is hot in L2.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pids_per_group = GROUP_M * num_pid_n
    first_pid_m = (pid // pids_per_group) * GROUP_M
    group_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % pids_per_group) % group_m
    pid_n = (pid % pids_per_group) // group_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        b_dec = tl.load(b_dec_ptr + offs_k, mask=mask_k, other=0.0)
        w = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0,
        )
        x = (x.to(tl.float32) - b_dec[None, :].to(tl.float32)).to(w.dtype)
        acc += tl.dot(x, w, input_precision=PRECISION)

    pre = acc + tl.load(b_enc_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    out = tl.maximum(pre, 0.0)
    if ACT == 1:
        thr = tl.load(thr_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        out = tl.where(pre > thr[None, :], out, 0.0)

    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out.to(out_ptr.dtype.element_ty), mask=mask,
    )
    if COUNT:
        nnz = tl.sum(((out != 0.0) & mask).to(tl.int32), axis=1)
        tl.atomic_add(nnz_ptr + offs_m, nnz, mask=mask_m)


_encode_autotuned = triton.autotune(CONFIGS, key=["M", "N", "K"], reset_to_zero=["nnz_ptr"])(_encode_kernel)


def fused_encode(x, W_enc, b_enc, b_dec, threshold=None, activation="relu", *,
                 out_dtype=None, row_nnz=None, config=None, allow_tf32=None):
    """act((x - b_dec) @ W_enc + b_enc) -> [M, d_sae], accumulated in fp32.

    x: [M, d_in] fp32 / bf16 / fp16. W_enc: [d_in, d_sae], any strides.
    activation: "relu" or "jumprelu" (needs `threshold` [d_sae]).
    row_nnz: optional zeroed int32 [M] buffer; receives each row's nonzero count.
    config: dict of block sizes (keys as in CONFIGS) to pin instead of autotuning.
    allow_tf32: fp32 inputs only; defaults to torch.backends.cuda.matmul.allow_tf32.
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"fused_encode supports {list(ACTIVATIONS)}, got {activation!r}")
    if activation == "jumprelu" and threshold is None:
        raise ValueError("jumprelu needs a threshold")
    M, K = x.shape
    K_w, N = W_enc.shape
    if K != K_w:
        raise ValueError(f"x has d_in={K} but W_enc has d_in={K_w}")
    out = torch.empty(M, N, device=x.device, dtype=out_dtype or x.dtype)
    if M == 0:
        return out
    if allow_tf32 is None:
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    precision = "ieee" if x.dtype == torch.float32 and not allow_tf32 else "tf32"

    args = (
        x, W_enc, b_enc, b_dec,
        threshold if threshold is not None else b_enc,  # unused unless jumprelu
        out,
        row_nnz if row_nnz is not None else out,  # unused unless COUNT
        M, N, K,
        x.stride(0), x.stride(1), W_enc.stride(0), W_enc.stride(1), out.stride(0), out.stride(1),
    )
    meta = dict(ACT=ACTIVATIONS[activation], COUNT=row_nnz is not None, PRECISION=precision)
    if config is None and INTERPRET:
        config = INTERPRET_CONFIG
    if config is not None:
        grid = (triton.cdiv(M, config["BLOCK_M"]) * triton.cdiv(N, config["BLOCK_N"]),)
        _encode_kernel[grid](*args, **meta, **config)
    else:
        grid = lambda m: (triton.cdiv(M, m["BLOCK_M"]) * triton.cdiv(N, m["BLOCK_N"]),)  # noqa: E731
        _encode_autotuned[grid](*args, **meta)
    return out


def best_config():
    """Block sizes the autotuner picked on the most recent autotuned launch (None before any)."""
    return getattr(_encode_autotuned, "best_config", None)
