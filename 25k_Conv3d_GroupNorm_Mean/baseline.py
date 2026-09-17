"""Independent source-faithful PyTorch convolution; see SOURCE.md."""
import math
import torch
import torch.nn.functional as F

CONFIG = {'name': '23_Conv3d_GroupNorm_Mean',
 'batch_size': 128,
 'in_channels': 3,
 'out_channels': 24,
 'kernel_size': 3,
 'num_groups': 8}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((128, 3, 24, 32, 32), dtype=dtype, device=device, generator=g)
    weight = torch.randn((24, 3, 3, 3, 3), dtype=dtype, device=device, generator=g) * 0.02
    bias = torch.randn((24,), dtype=dtype, device=device, generator=g) * 0.02
    scale = torch.ones((24,), dtype=dtype, device=device)
    offset = torch.randn((24,), dtype=dtype, device=device, generator=g) * 0.02
    return x, weight, bias, scale, offset


def workload(x, weight, conv_bias, gamma, beta):
    y = F.conv3d(x, weight, bias=None, stride=1, padding=0)
    y = y + conv_bias.reshape(1, -1, 1, 1, 1)
    n, c, d, h, w = y.shape
    y = y.reshape(n, 8, c // 8, d, h, w)
    mean = y.mean(dim=(2, 3, 4, 5), keepdim=True)
    variance = y.var(dim=(2, 3, 4, 5), correction=0, keepdim=True)
    y = (y - mean) / torch.sqrt(variance + 1e-5)
    y = y.reshape(n, c, d, h, w)
    y = y * gamma.reshape(1, -1, 1, 1, 1) + beta.reshape(1, -1, 1, 1, 1)
    return y.mean(dim=(1, 2, 3, 4))
