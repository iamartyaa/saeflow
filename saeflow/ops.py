"""SAE-level entry points. `sae` is a saeflow.reference.SAE, or anything with the
same attributes (W_enc, b_enc, W_dec, b_dec, threshold, activation, k).

Sparse features are CSR tuples (indptr, indices, values); see kernels/decode.py.
"""

import torch

from .kernels import dense_to_csr, fused_encode, sparse_decode


def _topk(sae, x):
    # ponytail: TopK writes the dense ReLU output, then torch.topk selects; a fused
    # in-epilogue select (roadmap) would skip the [M, d_sae] write entirely.
    f = fused_encode(x, sae.W_enc, sae.b_enc, sae.b_dec, activation="relu")
    return f.topk(sae.k, dim=-1)


def encode(sae, x):
    """Dense feature activations [M, d_sae]."""
    if sae.activation == "topk":
        top = _topk(sae, x)
        return torch.zeros(x.shape[0], sae.W_enc.shape[1], device=x.device, dtype=x.dtype).scatter_(
            -1, top.indices, top.values
        )
    return fused_encode(x, sae.W_enc, sae.b_enc, sae.b_dec, sae.threshold, sae.activation)


def encode_sparse(sae, x):
    """Feature activations as CSR (indptr, indices, values)."""
    M = x.shape[0]
    if sae.activation == "topk":
        # TopK is fixed-width already: no compaction pass needed.
        top = _topk(sae, x)
        indptr = torch.arange(0, M * sae.k + 1, sae.k, device=x.device, dtype=torch.int64)
        return indptr, top.indices.flatten().to(torch.int32), top.values.flatten()
    row_nnz = torch.zeros(M, device=x.device, dtype=torch.int32)
    f = fused_encode(x, sae.W_enc, sae.b_enc, sae.b_dec, sae.threshold, sae.activation, row_nnz=row_nnz)
    return dense_to_csr(f, row_nnz)


def decode_sparse(sae, csr):
    """Reconstruction [M, d_in] from CSR features."""
    return sparse_decode(*csr, sae.W_dec, sae.b_dec)


def forward(sae, x):
    """encode_sparse then decode_sparse: the dense feature matrix never reaches the decoder."""
    return decode_sparse(sae, encode_sparse(sae, x))
