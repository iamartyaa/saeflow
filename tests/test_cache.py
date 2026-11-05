import pytest
import torch

from saeflow.cache import ActivationReader, cache_activations
from saeflow.kernels import csr_to_dense
from saeflow.reference import SAE

tl = pytest.importorskip("transformer_lens")

HOOKS = ["blocks.0.hook_resid_post", "blocks.1.attn.hook_z"]


@pytest.fixture
def model():
    cfg = tl.HookedTransformerConfig(
        n_layers=2, d_model=32, n_ctx=16, d_head=8, n_heads=4, d_mlp=64, d_vocab=100, act_fn="gelu", seed=0, device="cpu"
    )
    return tl.HookedTransformer(cfg).eval()


@pytest.fixture
def tokens():
    return torch.randint(0, 100, (22, 16))  # 22 seqs: last shard and last batch are partial


def expected(model, tokens, hook):
    with torch.inference_mode():
        _, acts = model.run_with_cache(tokens, names_filter=[hook])
    return acts[hook].flatten(0, 1).flatten(1)


class Interrupted(Exception):
    pass


class FlakyModel:
    """Delegates to the real model but dies after `n` forward passes."""

    def __init__(self, model, n):
        self.model, self.n = model, n

    def parameters(self):
        return self.model.parameters()

    def run_with_cache(self, *args, **kwargs):
        if self.n == 0:
            raise Interrupted
        self.n -= 1
        return self.model.run_with_cache(*args, **kwargs)


def read_all(reader, hook, batch_size):
    return torch.cat(list(reader.iter_batches(hook, batch_size)))


def test_write_and_read_back_exactly(model, tokens, tmp_path):
    cache_activations(model, tokens, HOOKS, tmp_path, batch_size=4, batches_per_shard=2)
    reader = ActivationReader(tmp_path)
    assert len(reader.manifest["shards"]) == 3
    for hook in HOOKS:
        assert torch.equal(read_all(reader, hook, 7), expected(model, tokens, hook))


def test_resume_after_interruption(model, tokens, tmp_path):
    with pytest.raises(Interrupted):
        cache_activations(FlakyModel(model, 3), tokens, HOOKS, tmp_path, batch_size=4, batches_per_shard=2)
    assert len(ActivationReader(tmp_path).manifest["shards"]) == 1  # half-written shard 1 discarded

    cache_activations(model, tokens, HOOKS, tmp_path, batch_size=4, batches_per_shard=2)
    reader = ActivationReader(tmp_path)
    assert len(reader.manifest["shards"]) == 3
    assert not list(tmp_path.rglob("*.tmp"))
    for hook in HOOKS:
        assert torch.equal(read_all(reader, hook, 5), expected(model, tokens, hook))


def test_resume_with_different_config_refuses(model, tokens, tmp_path):
    cache_activations(model, tokens, HOOKS[:1], tmp_path, batch_size=4)
    with pytest.raises(ValueError, match="different config"):
        cache_activations(model, tokens, HOOKS[:1], tmp_path, batch_size=8)


def test_bfloat16_storage(model, tokens, tmp_path):
    hook = HOOKS[0]
    cache_activations(model, tokens, [hook], tmp_path, batch_size=8, dtype=torch.bfloat16)
    got = read_all(ActivationReader(tmp_path), hook, 64)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, expected(model, tokens, hook).bfloat16())


def test_encode_on_the_fly_stores_sparse(model, tokens, tmp_path):
    hook = HOOKS[0]
    sae = SAE(32, 96, "jumprelu")
    sae.threshold.fill_(0.5)
    cache_activations(model, tokens, [hook], tmp_path, batch_size=4, batches_per_shard=2, encoders={hook: sae})
    reader = ActivationReader(tmp_path)
    assert reader.manifest["hooks"][hook] == {"format": "csr", "d": 96}
    dense = torch.cat([csr_to_dense(*csr, 96) for csr in reader.iter_batches(hook, 13)])
    torch.testing.assert_close(dense, sae.encode(expected(model, tokens, hook)), atol=1e-4, rtol=1e-4)
