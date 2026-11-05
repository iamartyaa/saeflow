import pytest
import torch

import saeflow
from saeflow.kernels import csr_to_dense, dense_to_csr, fused_encode, sparse_decode
from saeflow.reference import SAE

# (tokens, d_in, d_sae): includes sizes not divisible by any block size.
SHAPES = [(1, 16, 32), (37, 50, 129), (64, 64, 256), (100, 33, 70)]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
TOL = {torch.float32: 1e-4, torch.bfloat16: 3e-2, torch.float16: 5e-3}


def make_sae(d_in, d_sae, activation, device, dtype, k=None):
    sae = SAE(d_in, d_sae, activation, k=k, device=device)
    sae.b_enc.normal_(0, 0.1)
    sae.b_dec.normal_(0, 0.1)
    if activation == "jumprelu":
        sae.threshold.uniform_(0.0, 0.5)
    return sae.to(dtype)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", ["relu", "jumprelu"])
def test_fused_encode(shape, dtype, activation, device):
    M, d_in, d_sae = shape
    sae = make_sae(d_in, d_sae, activation, device, dtype)
    x = torch.randn(M, d_in, device=device).to(dtype)
    row_nnz = torch.zeros(M, device=device, dtype=torch.int32)
    got = fused_encode(x, sae.W_enc, sae.b_enc, sae.b_dec, sae.threshold, activation, row_nnz=row_nnz).float()

    # Ground truth in fp32 from the same (rounded) inputs.
    ref_sae = make_sae(d_in, d_sae, activation, device, torch.float32)
    ref_sae.load_state_dict({k: v.float() for k, v in sae.state_dict().items()})
    want = ref_sae.encode(x.float())
    pre = ref_sae.encode_pre(x.float())

    # Elements within rounding distance of the ReLU / JumpReLU cut may legitimately flip.
    cut = pre - (ref_sae.threshold if activation == "jumprelu" else 0)
    tol = TOL[dtype] * (1 + pre.abs().max())
    keep = cut.abs() > tol
    torch.testing.assert_close(got[keep], want[keep], atol=tol, rtol=TOL[dtype])
    assert torch.equal(row_nnz.cpu(), torch.count_nonzero(got, dim=1).int().cpu())


def test_fused_encode_strided_weights(device):
    sae = make_sae(48, 80, "relu", device, torch.float32)
    W_t = sae.W_enc.T.contiguous().T  # same values, column-major strides
    x = torch.randn(20, 48, device=device)
    got = fused_encode(x, W_t, sae.b_enc, sae.b_dec)
    torch.testing.assert_close(got, sae.encode(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", [(1, 32), (37, 129), (64, 300)])
def test_dense_to_csr_matches_torch(shape, device):
    f = torch.relu(torch.randn(*shape, device=device) - 1.0)
    f[0] = 0  # an empty row
    indptr, indices, values = dense_to_csr(f)
    want = f.to_sparse_csr()
    assert torch.equal(indptr.cpu(), want.crow_indices().cpu())
    assert torch.equal(indices.long().cpu(), want.col_indices().cpu())
    assert torch.equal(values.cpu(), want.values().cpu())


def test_dense_to_csr_all_zero(device):
    indptr, indices, values = dense_to_csr(torch.zeros(5, 40, device=device))
    assert indptr.tolist() == [0] * 6 and indices.numel() == 0 and values.numel() == 0


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_decode(shape, dtype, device):
    M, d_in, d_sae = shape
    sae = make_sae(d_in, d_sae, "relu", device, dtype)
    f = torch.relu(torch.randn(M, d_sae, device=device) - 1.5).to(dtype)  # ~7% density, variable L0
    got = sparse_decode(*dense_to_csr(f), sae.W_dec, sae.b_dec).float()
    want = f.float() @ sae.W_dec.float() + sae.b_dec.float()
    torch.testing.assert_close(got, want, atol=TOL[dtype] * 4, rtol=TOL[dtype])


@pytest.mark.parametrize("activation", ["relu", "jumprelu", "topk"])
def test_ops_match_reference(activation, device):
    sae = make_sae(40, 150, activation, device, torch.float32, k=8)
    x = torch.randn(33, 40, device=device)
    torch.testing.assert_close(saeflow.encode(sae, x), sae.encode(x), atol=1e-4, rtol=1e-4)
    csr = saeflow.encode_sparse(sae, x)
    torch.testing.assert_close(csr_to_dense(*csr, 150), sae.encode(x), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(saeflow.forward(sae, x), sae(x), atol=1e-4, rtol=1e-4)
