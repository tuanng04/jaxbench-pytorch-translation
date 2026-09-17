"""Independent PyTorch translation; pinned JAX semantics are described in SOURCE.md."""
import torch

CONFIG = {'name': '18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp',
 'batch_size': 4096,
 'in_features': 8192,
 'out_features': 8192}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=generator)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=generator) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=generator) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    y = torch.matmul(x, weight.T) + bias
    y = y.sum(dim=1, keepdim=True)
    y = torch.amax(y, dim=1, keepdim=True)
    y = y.mean(dim=1, keepdim=True)
    y = torch.logsumexp(y, dim=1, keepdim=True)
    return torch.logsumexp(y, dim=1, keepdim=True)
