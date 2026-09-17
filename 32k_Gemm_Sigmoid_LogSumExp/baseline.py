"""Independent PyTorch baseline; see SOURCE.md for source-specific semantics."""
import math
import torch

CONFIG = {'name': '45_Gemm_Sigmoid_LogSumExp',
 'batch_size': 16384,
 'input_size': 2048,
 'hidden_size': 4096,
 'output_size': 1024}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((16384, 2048), dtype=dtype, device=device, generator=g)
    w1 = torch.randn((4096, 2048), dtype=dtype, device=device, generator=g) * 0.02
    b1 = torch.randn((4096,), dtype=dtype, device=device, generator=g) * 0.02
    w2 = torch.randn((1024, 4096), dtype=dtype, device=device, generator=g) * 0.02
    b2 = torch.randn((1024,), dtype=dtype, device=device, generator=g) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    y = torch.matmul(x, w1.T) + b1
    y = torch.sigmoid(y)
    y = torch.matmul(y, w2.T) + b2
    return torch.logsumexp(y, dim=1)
