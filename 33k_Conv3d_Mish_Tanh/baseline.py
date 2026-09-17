"""Independent source-faithful PyTorch convolution; see SOURCE.md."""
import math
import torch
import torch.nn.functional as F

CONFIG = {'name': '47_Conv3d_Mish_Tanh',
 'batch_size': 16,
 'in_channels': 32,
 'out_channels': 64,
 'kernel_size': 3}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((16, 32, 32, 64, 64), dtype=dtype, device=device, generator=g)
    weight = torch.randn((64, 32, 3, 3, 3), dtype=dtype, device=device, generator=g) * 0.02
    bias = torch.randn((64,), dtype=dtype, device=device, generator=g) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    y = F.conv3d(x, weight, bias=None, stride=1, padding=0)
    y = y + bias.reshape(1, -1, 1, 1, 1)
    y = y * torch.tanh(torch.log(1 + torch.exp(y)))
    return torch.tanh(y)
