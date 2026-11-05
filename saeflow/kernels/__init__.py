import os

# Under the Triton CPU interpreter there is no autotuning: kernels run with small fixed tiles.
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

from .decode import csr_to_dense, dense_to_csr, sparse_decode  # noqa: E402
from .encode import fused_encode  # noqa: E402

__all__ = ["INTERPRET", "csr_to_dense", "dense_to_csr", "fused_encode", "sparse_decode"]
