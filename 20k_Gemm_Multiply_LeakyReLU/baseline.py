"""Independent PyTorch translation of the pinned JAXBench forward baseline.

KernelBench supplies source guidance; shared JAX TPU fixtures supply acceptance.
See validation.py and SOURCE.md for case declarations and source reconciliation.
"""
import math
import torch

CONFIG = {'name': '12_Gemm_Multiply_LeakyReLU', 'batch_size': 4096, 'in_features': 8192, 'out_features': 8192, 'multiplier': 2.0, 'negative_slope': 0.1}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    xgen = torch.Generator(device=device).manual_seed(0)
    wgen = torch.Generator(device=device).manual_seed(42)
    m, k, n = CONFIG['batch_size'], CONFIG['in_features'], CONFIG['out_features']
    x = torch.rand((m, k), dtype=dtype, device=device, generator=xgen)
    weight = torch.randn((k, n), dtype=dtype, device=device, generator=wgen) * 0.02
    bias = torch.randn((n,), dtype=dtype, device=device, generator=wgen) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    y = (torch.matmul(x, weight) + bias) * 2.0
    return torch.where(y >= 0, y, y * 0.1)
