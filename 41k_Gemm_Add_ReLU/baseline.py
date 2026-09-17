"""Independent translation of the pinned JAX baseline; see SOURCE.md."""
import math
import torch

CONFIG = {'name': '76_Gemm_Add_ReLU',
 'batch_size': 4096,
 'in_features': 8192,
 'out_features': 8192}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    xgen = torch.Generator(device=device).manual_seed(0)
    wgen = torch.Generator(device=device).manual_seed(42)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=xgen)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=wgen) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=wgen) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    return torch.relu(torch.matmul(x, weight) + bias)
