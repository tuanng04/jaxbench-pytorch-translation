"""Independent PyTorch convolution; see SOURCE.md for reduction and residual semantics."""
import math
import torch
import torch.nn.functional as F

CONFIG = {'name': '92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp',
 'batch_size': 128,
 'in_channels': 8,
 'out_channels': 64,
 'kernel_size': 3,
 'groups': 16}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((128, 8, 128, 128), dtype=dtype, device=device, generator=g)
    weight = torch.randn((64, 8, 3, 3), dtype=dtype, device=device, generator=g) * 0.02
    bias = torch.randn((64,), dtype=dtype, device=device, generator=g) * 0.02
    scale = torch.ones((64,), dtype=dtype, device=device)
    offset = torch.zeros((64,), dtype=dtype, device=device)
    return x, weight, bias, scale, offset


def workload(x, conv_weight, conv_bias, gn_weight, gn_bias):
    conv = F.conv2d(x, conv_weight, bias=None, stride=1, padding=0)
    conv = conv + conv_bias.reshape(1, -1, 1, 1)
    n, c, h, w = conv.shape
    y = conv.reshape(n, 16, c // 16, h, w)
    mean = y.mean(dim=(2, 3, 4), keepdim=True)
    variance = y.var(dim=(2, 3, 4), correction=0, keepdim=True)
    y = (y - mean) / torch.sqrt(variance + 1e-5)
    y = y.reshape(n, c, h, w)
    y = y * gn_weight.reshape(1, -1, 1, 1) + gn_bias.reshape(1, -1, 1, 1)
    y = torch.tanh(y)
    y = y * torch.clamp(y + 3, min=0, max=6) / 6
    return torch.logsumexp(conv + y, dim=1, keepdim=True)
