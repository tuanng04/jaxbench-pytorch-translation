"""PyTorch forward translation of the pinned JAXBench RMSNorm baseline.

Standalone random inputs support independent execution. Framework comparisons
must instead load the actual retained JAX-generated input values.
"""

import torch


CONFIG = {
    "name": "llama3_70b_rmsnorm",
    "model": "Llama-3.1-70B",
    "operator": "rms_norm",
    "batch": 8,
    "seq_len": 4096,
    "emb_dim": 8192,
    "epsilon": 1e-5,
}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    """Return canonical standalone (x, scale), including source scale offsets."""
    generator = torch.Generator(device=device).manual_seed(42)
    shape = (CONFIG["batch"], CONFIG["seq_len"], CONFIG["emb_dim"])
    x = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    scale_noise = torch.randn(
        (CONFIG["emb_dim"],), dtype=dtype, device=device, generator=generator
    )
    scale = scale_noise * scale_noise.new_tensor(0.1) + 1.0
    return x, scale


def workload(x, scale):
    """Accumulate in float32, cast normalization back, then apply scale."""
    x_f32 = x.to(torch.float32)
    mean2 = torch.mean(torch.square(x_f32), dim=-1, keepdim=True)
    normed = x_f32 * torch.rsqrt(mean2 + CONFIG["epsilon"])
    return normed.to(x.dtype) * scale
