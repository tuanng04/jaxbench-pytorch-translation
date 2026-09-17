"""Independent translation of the pinned JAX baseline; see SOURCE.md."""
import math
import torch

CONFIG = {'name': '95_Matmul_Add_Swish_Tanh_GELU_Hardtanh',
 'batch_size': 4096,
 'in_features': 8192,
 'out_features': 8192}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    xgen = torch.Generator(device=device).manual_seed(0)
    wgen = torch.Generator(device=device).manual_seed(42)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=xgen)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=wgen) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=wgen) * 0.02
    add_value = torch.randn((8192,), dtype=dtype, device=device, generator=wgen) * 0.02
    return x, weight, bias, add_value


def approximate_gelu(x):
    coefficient = x.new_tensor(math.sqrt(2.0 / math.pi))
    cdf = 0.5 * (1.0 + torch.tanh(coefficient * (x + x.new_tensor(0.044715) * (x ** 3))))
    return x * cdf


def workload(x, weight, bias, add_value):
    y = torch.matmul(x, weight) + bias
    y = y + add_value
    y = y * torch.sigmoid(y)
    y = torch.tanh(y)
    return torch.clamp(approximate_gelu(y), -1.0, 1.0)
