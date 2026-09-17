"""Independent translation of the pinned JAX baseline; see SOURCE.md."""
import math
import torch

CONFIG = {'name': '97_Matmul_BatchNorm_BiasAdd_Divide_Swish',
 'batch_size': 4096,
 'in_features': 8192,
 'out_features': 8192,
 'bn_eps': 1e-05,
 'bn_momentum': 0.1,
 'divide_value': 1.0}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=g)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=g) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=g) * 0.02
    affine_scale = torch.ones((8192,), dtype=dtype, device=device)
    affine_bias = torch.zeros((8192,), dtype=dtype, device=device)
    mean = torch.zeros((8192,), dtype=dtype, device=device)
    variance = torch.ones((8192,), dtype=dtype, device=device)
    extra_bias = torch.randn((1,), dtype=dtype, device=device, generator=g) * 0.02
    return x, weight, bias, affine_scale, affine_bias, mean, variance, extra_bias


def workload(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias):
    tensors = (x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias)
    if (all(t.dtype == torch.bfloat16 and t.device.type == "cuda" for t in tensors)
            and tuple(tuple(t.shape) for t in tensors) ==
            ((4096, 8192), (8192, 8192), (8192,), (8192,), (8192,), (8192,), (8192,), (1,))):
        # The captured canonical executable materializes only its BF16
        # denominator before one fused dot/normalization/Swish output.
        epsilon = torch.tensor(1e-5, dtype=torch.bfloat16, device=x.device).float()
        denominator = torch.sqrt(bn_var.float() + epsilon).to(torch.bfloat16).float()
        accumulator = torch.mm(x, weight, out_dtype=torch.float32)
        value = bn_scale.float() * ((accumulator + linear_bias.float() - bn_mean.float()) / denominator)
        value = value + bn_bias.float() + bias.float()
        return (value * torch.sigmoid(value)).to(torch.bfloat16)
    y = torch.matmul(x, weight) + linear_bias
    normalized = (y - bn_mean) / torch.sqrt(bn_var + 1e-5)
    y = bn_scale * normalized + bn_bias
    y = y + bias
    y = y / 1.0
    return y * torch.sigmoid(y)
