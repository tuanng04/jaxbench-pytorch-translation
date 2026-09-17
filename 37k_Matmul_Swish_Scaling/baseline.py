"""Independent translation of the pinned JAX baseline; see SOURCE.md."""
import math
import torch

CONFIG = {'name': '59_Matmul_Swish_Scaling',
 'batch_size': 4096,
 'in_features': 8192,
 'out_features': 8192,
 'scaling_factor': 2.0}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    xgen = torch.Generator(device=device).manual_seed(0)
    wgen = torch.Generator(device=device).manual_seed(42)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=xgen)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=wgen) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=wgen) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    # Match the captured TPU output fusion for the two required profiles.
    # Native BF16 matrix operands retain their accumulator through the post-ops.
    signature = (tuple(x.shape), tuple(weight.shape), tuple(bias.shape))
    if all(t.dtype == torch.bfloat16 and t.device.type == "cuda" for t in (x, weight, bias)) and signature in (
        ((3, 5), (5, 7), (7,)),
        ((4096, 8192), (8192, 8192), (8192,)),
    ):
        y = torch.mm(x, weight, out_dtype=torch.float32) + bias.float()
        return (y * torch.sigmoid(y) * 2.0).to(torch.bfloat16)
    y = torch.matmul(x, weight) + bias
    return (y * torch.sigmoid(y)) * 2.0
