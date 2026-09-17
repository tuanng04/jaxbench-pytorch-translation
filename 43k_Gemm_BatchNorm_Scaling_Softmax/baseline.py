"""Independent PyTorch baseline; see SOURCE.md for source-specific semantics."""
import math
import torch

CONFIG = {'name': '84_Gemm_BatchNorm_Scaling_Softmax',
 'batch_size': 4096,
 'in_features': 8192,
 'out_features': 8192,
 'bn_eps': 1e-05,
 'bn_momentum': 0.1}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=g)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=g) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=g) * 0.02
    affine_scale = torch.ones((8192,), dtype=dtype, device=device)
    affine_bias = torch.randn((8192,), dtype=dtype, device=device, generator=g) * 0.02
    mean = torch.zeros((8192,), dtype=dtype, device=device)
    variance = torch.ones((8192,), dtype=dtype, device=device)
    scale = torch.ones((1,), dtype=dtype, device=device)
    return x, weight, bias, affine_scale, affine_bias, mean, variance, scale


def workload(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale):
    y = torch.matmul(x, weight) + bias
    normalized = (y - bn_mean) / torch.sqrt(bn_var + 1e-5)
    y = bn_scale * normalized + bn_bias
    y = scale * y
    exponential = torch.exp(y - torch.amax(y, dim=1, keepdim=True))
    return exponential / exponential.sum(dim=1, keepdim=True)
