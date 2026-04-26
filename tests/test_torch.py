import torch

def test_torch_installed():
    assert torch.__version__ is not None

def test_tensor_operations():
    x = torch.tensor([1, 2, 3])
    y = x * 2
    assert y.tolist() == [2, 4, 6]