import os

import pytest
import torch

# No GPU: run every Triton kernel on CPU through the interpreter. Must be set before triton is imported.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def device():
    return DEVICE


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
