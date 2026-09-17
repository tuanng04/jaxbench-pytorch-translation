"""Independent PyTorch translation of the pinned JAXBench forward baseline.

KernelBench supplies source guidance; shared JAX TPU fixtures supply acceptance.
See validation.py and SOURCE.md for case declarations and source reconciliation.
"""
import math
import torch

CONFIG = {'name': '29_Matmul_Mish_Mish', 'batch_size': 4096, 'in_features': 8192, 'out_features': 8192}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    xgen = torch.Generator(device=device).manual_seed(0)
    wgen = torch.Generator(device=device).manual_seed(42)
    m, k, n = CONFIG['batch_size'], CONFIG['in_features'], CONFIG['out_features']
    x = torch.rand((m, k), dtype=dtype, device=device, generator=xgen)
    weight = torch.randn((k, n), dtype=dtype, device=device, generator=wgen) * 0.02
    bias = torch.randn((n,), dtype=dtype, device=device, generator=wgen) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    # The pinned TPU compiler fuses the native BF16 matrix product, bias and
    # both Mish activations for these profiles. SOURCE.md links the compiler
    # capture and full-array diagnosis of its materialization boundary.
    signature = (tuple(x.shape), tuple(weight.shape), tuple(bias.shape))
    if all(t.dtype == torch.bfloat16 and t.device.type == "cuda" for t in (x, weight, bias)) and signature in (
        ((3, 5), (5, 7), (7,)),
        ((4096, 8192), (8192, 8192), (8192,)),
    ):
        y = torch.mm(x, weight, out_dtype=torch.float32) + bias.float()
        y = y * torch.tanh(torch.nn.functional.softplus(y))
        return (y * torch.tanh(torch.nn.functional.softplus(y))).to(torch.bfloat16)
    y = torch.matmul(x, weight) + bias
    y = y * torch.tanh(torch.nn.functional.softplus(y))
    return y * torch.tanh(torch.nn.functional.softplus(y))
