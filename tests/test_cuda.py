# tests/test_cuda.py
import torch

def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA is not available on this system."

def test_cuda_device_name():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {name}")
        assert name is not None

def test_tensor_on_gpu():
    if torch.cuda.is_available():
        x = torch.tensor([1.0, 2.0]).cuda()
        y = x * 2
        assert y.device.type == "cuda"