"""Independent PyTorch translation of the pinned JAXBench forward baseline.

KernelBench supplies source guidance; shared JAX TPU fixtures supply acceptance.
See validation.py and SOURCE.md for case declarations and source reconciliation.
"""
import math
import torch

CONFIG = {'name': '40_Matmul_Scaling_ResidualAdd', 'batch_size': 16384, 'in_features': 4096, 'out_features': 4096, 'scaling_factor': 0.5}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    xgen = torch.Generator(device=device).manual_seed(0)
    wgen = torch.Generator(device=device).manual_seed(42)
    m, k, n = CONFIG['batch_size'], CONFIG['in_features'], CONFIG['out_features']
    x = torch.rand((m, k), dtype=dtype, device=device, generator=xgen)
    weight = torch.randn((k, n), dtype=dtype, device=device, generator=wgen) * 0.02
    bias = torch.randn((n,), dtype=dtype, device=device, generator=wgen) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    y = torch.matmul(x, weight) + bias
    return y * 0.5 + y
