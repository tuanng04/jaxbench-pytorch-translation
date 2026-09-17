"""Independent translation of the pinned JAX baseline; see SOURCE.md."""
import math
import torch

CONFIG = {'name': '86_Matmul_Divide_GELU',
 'batch_size': 4096,
 'input_size': 8192,
 'output_size': 8192,
 'divisor': 10.0}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    xgen = torch.Generator(device=device).manual_seed(0)
    wgen = torch.Generator(device=device).manual_seed(42)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=xgen)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=wgen) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=wgen) * 0.02
    return x, weight, bias


def approximate_gelu(x):
    coefficient = x.new_tensor(math.sqrt(2.0 / math.pi))
    cdf = 0.5 * (1.0 + torch.tanh(coefficient * (x + x.new_tensor(0.044715) * (x ** 3))))
    return x * cdf


def workload(x, weight, bias):
    y = (torch.matmul(x, weight) + bias) / 10.0
    return approximate_gelu(y)
