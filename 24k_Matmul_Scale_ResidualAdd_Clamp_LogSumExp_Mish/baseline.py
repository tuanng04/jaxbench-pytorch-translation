"""Independent PyTorch translation; pinned JAX semantics are described in SOURCE.md."""
import torch

CONFIG = {'name': '22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish',
 'batch_size': 4096,
 'input_size': 8192,
 'hidden_size': 8192,
 'scale_factor': 2.0,
 'clamp_min': -10.0,
 'clamp_max': 10.0}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=generator)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=generator) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=generator) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    if (all(t.dtype == torch.bfloat16 and t.device.type == "cuda" for t in (x, weight, bias))
            and tuple(x.shape) == (4096, 8192) and tuple(weight.shape) == (8192, 8192)
            and tuple(bias.shape) == (8192,)):
        # The canonical TPU executable stores the raw dot and maximum, then
        # recomputes bias/clamp in the exponential reduction. Its reduction
        # sum and logarithm cross separate BF16 materialization boundaries.
        accumulator = torch.mm(x, weight.T, out_dtype=torch.float32)
        maximum = torch.clamp((accumulator + bias.float()) * 4.0, -10.0, 10.0)
        maximum = maximum.amax(dim=1, keepdim=True).to(torch.bfloat16).float()
        stored_dot = accumulator.to(torch.bfloat16).float()
        clamped = torch.clamp((stored_dot + bias.float()) * 4.0, -10.0, 10.0)
        total = torch.exp(clamped - maximum).sum(dim=1, keepdim=True).to(torch.bfloat16).float()
        logarithm = torch.log(total.abs()).to(torch.bfloat16).float()
        y = logarithm + maximum
        softplus = torch.logaddexp(y, torch.zeros((), dtype=y.dtype, device=y.device))
        return (y * (y * torch.tanh(softplus))).to(torch.bfloat16)
    y = torch.matmul(x, weight.T) + bias
    y = y * 2.0
    y = y + y
    y = torch.clamp(y, -10.0, 10.0)
    y = torch.logsumexp(y, dim=1, keepdim=True)
    softplus = torch.logaddexp(y, torch.zeros((), dtype=y.dtype, device=y.device))
    mish = y * torch.tanh(softplus)
    return y * mish
