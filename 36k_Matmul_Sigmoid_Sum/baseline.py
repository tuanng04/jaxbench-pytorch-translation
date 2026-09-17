"""Independent PyTorch baseline; see SOURCE.md for source-specific semantics."""
import math
import torch

CONFIG = {'name': '56_Matmul_Sigmoid_Sum',
 'batch_size': 4096,
 'input_size': 8192,
 'hidden_size': 8192}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=g)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=g) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=g) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    y = torch.matmul(x, weight) + bias
    return torch.sigmoid(y).sum(dim=1, keepdim=True)
