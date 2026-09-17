"""Independent PyTorch translation; pinned JAX semantics are described in SOURCE.md."""
import torch

CONFIG = {'name': '28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply',
 'batch_size': 4096,
 'in_features': 8192,
 'out_features': 8192}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=generator)
    y = torch.rand((4096, 8192), dtype=dtype, device=device, generator=generator)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=generator) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=generator) * 0.02
    scale = torch.ones((8192,), dtype=dtype, device=device)
    offset = torch.randn((8192,), dtype=dtype, device=device, generator=generator) * 0.02
    return x, y, weight, bias, scale, offset


def workload(x, y, bmm_weight, bmm_bias, in_weight, in_bias):
    value = torch.matmul(x, bmm_weight.T) + bmm_bias
    value = value.unsqueeze(2).unsqueeze(3)
    mean = value.mean(dim=(2, 3), keepdim=True)
    variance = value.var(dim=(2, 3), correction=0, keepdim=True)
    value = (value - mean) / torch.sqrt(variance + 1e-5)
    value = value * in_weight.reshape(1, -1, 1, 1) + in_bias.reshape(1, -1, 1, 1)
    value = value.squeeze(3).squeeze(2)
    return (value + y) * y
