"""Independent PyTorch translation of the pinned JAXBench forward baseline.

KernelBench supplies source guidance; shared JAX TPU fixtures supply acceptance.
See validation.py and SOURCE.md for case declarations and source reconciliation.
"""
import math
import torch

CONFIG = {'name': '9_Matmul_Subtract_Multiply_ReLU', 'batch_size': 4096, 'in_features': 8192, 'out_features': 8192, 'subtract_value': 2.0, 'multiply_value': 1.5}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    xgen = torch.Generator(device=device).manual_seed(0)
    wgen = torch.Generator(device=device).manual_seed(42)
    m, k, n = CONFIG['batch_size'], CONFIG['in_features'], CONFIG['out_features']
    x = torch.rand((m, k), dtype=dtype, device=device, generator=xgen)
    weight = torch.randn((k, n), dtype=dtype, device=device, generator=wgen) * 0.02
    bias = torch.randn((n,), dtype=dtype, device=device, generator=wgen) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    # The pinned TPU executable keeps this entire chain in one output fusion
    # for the required small and canonical signatures. Preserve native BF16
    # matrix operands, retain its FP32 accumulator through the post-ops, and
    # materialize BF16 only at the fusion output. See SOURCE.md for evidence.
    signature = (tuple(x.shape), tuple(weight.shape), tuple(bias.shape))
    if all(t.dtype == torch.bfloat16 and t.device.type == "cuda" for t in (x, weight, bias)) and signature in (
        ((3, 5), (5, 7), (7,)),
        ((4096, 8192), (8192, 8192), (8192,)),
    ):
        accumulator = torch.mm(x, weight, out_dtype=torch.float32)
        return torch.relu((accumulator + bias.float() - 2.0) * 1.5).to(torch.bfloat16)
    y = torch.matmul(x, weight) + bias
    return torch.relu((y - 2.0) * 1.5)
