"""Independent translation of the pinned JAX baseline; see SOURCE.md."""
import math
import torch

CONFIG = {'name': '80_Gemm_Max_Subtract_GELU',
 'batch_size': 4096,
 'in_features': 8192,
 'out_features': 8192,
 'max_dim': 1}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    xgen = torch.Generator(device=device).manual_seed(0)
    wgen = torch.Generator(device=device).manual_seed(42)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=xgen)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=wgen) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=wgen) * 0.02
    return x, weight, bias


def approximate_gelu(x):
    coefficient = x.new_tensor(math.sqrt(2.0 / math.pi))
    cdf = 0.5 * (1.0 + torch.tanh(coefficient * (x + x.new_tensor(0.044715) * (x ** 3))))
    return x * cdf


def workload(x, weight, bias):
    # Canonical TPU fusion 1 returns the BF16 dot and the BF16 row maximum
    # of accumulator+bias. Fusion 2 adds bias again to the stored dot, then
    # subtracts the maximum and evaluates GELU. Small HLO instead stores the
    # biased matrix, so this correction is restricted to the canonical shape.
    if all(t.dtype == torch.bfloat16 and t.device.type == "cuda" for t in (x, weight, bias)) and (
        tuple(x.shape), tuple(weight.shape), tuple(bias.shape)
    ) == ((4096, 8192), (8192, 8192), (8192,)):
        accumulator = torch.mm(x, weight, out_dtype=torch.float32)
        maximum = (accumulator + bias.float()).amax(dim=1, keepdim=True).to(torch.bfloat16)
        y = accumulator.to(torch.bfloat16).float() + bias.float() - maximum.float()
        # Preserve the BF16 literals shown in the pinned executable, while
        # retaining FP32 math inside this final fusion until its output.
        coefficient = torch.tensor(math.sqrt(2.0 / math.pi), dtype=torch.bfloat16, device=x.device).float()
        cubic = torch.tensor(0.044715, dtype=torch.bfloat16, device=x.device).float()
        cdf = (1.0 + torch.tanh(coefficient * (y + cubic * (y * y * y)))) * 0.5
        return (y * cdf).to(torch.bfloat16)
    y = torch.matmul(x, weight) + bias
    return approximate_gelu(y - torch.amax(y, dim=1, keepdim=True))
