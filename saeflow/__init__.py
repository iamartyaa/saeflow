"""saeflow: GPU kernels and caching for sparse-autoencoder interpretability."""

from .ops import decode_sparse, encode, encode_sparse, forward
from .reference import SAE

__version__ = "0.1.0"
__all__ = ["SAE", "decode_sparse", "encode", "encode_sparse", "forward"]
