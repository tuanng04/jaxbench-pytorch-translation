"""Independent translation of the pinned JAX baseline; see SOURCE.md."""
import math
import torch

CONFIG = {'name': '98_Matmul_AvgPool_GELU_Scale_Max',
 'batch_size': 4096,
 'in_features': 8192,
 'out_features': 8192,
 'pool_kernel_size': 16,
 'scale_factor': 2.0}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=g)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=g) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=g) * 0.02
    return x, weight, bias


def approximate_gelu(x):
    coefficient = x.new_tensor(math.sqrt(2.0 / math.pi))
    cdf = 0.5 * (1.0 + torch.tanh(coefficient * (x + x.new_tensor(0.044715) * x**3)))
    return x * cdf


def workload(x, weight, bias):
    y = torch.matmul(x, weight) + bias
    width = (y.shape[1] // 16) * 16
    y = y[:, :width].reshape(y.shape[0], -1, 16).sum(dim=-1) / 16
    y = approximate_gelu(y) * 2.0
    return torch.amax(y, dim=1)
