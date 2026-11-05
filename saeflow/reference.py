"""Plain PyTorch SAE. Ground truth for every kernel test and benchmark."""

import torch
from torch import nn

ACTIVATIONS = ("relu", "jumprelu", "topk")


class SAE(nn.Module):
    """x -> f = act((x - b_dec) @ W_enc + b_enc) -> x_hat = f @ W_dec + b_dec.

    activation: "relu", "jumprelu" (per-feature `threshold`) or "topk" (keep the
    `k` largest pre-activations per token). Parameter names and shapes follow
    SAELens: W_enc [d_in, d_sae], W_dec [d_sae, d_in].
    """

    def __init__(self, d_in, d_sae, activation="relu", k=None, device=None, dtype=torch.float32):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation!r}")
        if activation == "topk" and not k:
            raise ValueError("topk activation needs k")
        self.d_in, self.d_sae, self.activation, self.k = d_in, d_sae, activation, k

        W_dec = torch.randn(d_sae, d_in, device=device)
        W_dec /= W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec.to(dtype))
        self.W_enc = nn.Parameter(W_dec.T.contiguous().to(dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))
        self.threshold = (
            nn.Parameter(torch.zeros(d_sae, device=device, dtype=dtype)) if activation == "jumprelu" else None
        )
        self.requires_grad_(False)

    def encode_pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x):
        pre = self.encode_pre(x)
        if self.activation == "relu":
            return torch.relu(pre)
        if self.activation == "jumprelu":
            return torch.relu(pre) * (pre > self.threshold)
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, torch.relu(top.values))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))
