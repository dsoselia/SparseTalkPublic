import pytest


@pytest.fixture(autouse=True)
def cuda_only():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SparseTalkPublic tests require CUDA; CPU execution is disabled")
