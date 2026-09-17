"""Independent PyTorch convolution; see SOURCE.md for reduction and residual semantics."""
import math
import torch
import torch.nn.functional as F

CONFIG = {'name': '67_Conv2d_GELU_GlobalAvgPool',
 'batch_size': 128,
 'in_channels': 8,
 'out_channels': 64,
 'kernel_size': 3}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((128, 8, 256, 256), dtype=dtype, device=device, generator=g)
    weight = torch.randn((64, 8, 3, 3), dtype=dtype, device=device, generator=g) * 0.02
    bias = torch.randn((64,), dtype=dtype, device=device, generator=g) * 0.02
    return x, weight, bias


def approximate_gelu(x):
    coefficient = x.new_tensor(math.sqrt(2.0 / math.pi))
    cdf = 0.5 * (1.0 + torch.tanh(coefficient * (x + x.new_tensor(0.044715) * x**3)))
    return x * cdf


def workload(x, weight, bias):
    y = F.conv2d(x, weight, bias=None, stride=1, padding=0)
    y = y + bias.reshape(1, -1, 1, 1)
    return approximate_gelu(y).mean(dim=(2, 3))
