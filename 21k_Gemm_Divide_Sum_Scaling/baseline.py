"""Independent PyTorch translation; pinned JAX semantics are described in SOURCE.md."""
import torch

CONFIG = {'name': '14_Gemm_Divide_Sum_Scaling',
 'batch_size': 4096,
 'input_size': 8192,
 'hidden_size': 8192,
 'scaling_factor': 1.5}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=generator)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=generator)
    return x, weight


def workload(x, weight):
    y = torch.matmul(x, weight.T)
    y = y / 2.0
    return y.sum(dim=1, keepdim=True) * 1.5
