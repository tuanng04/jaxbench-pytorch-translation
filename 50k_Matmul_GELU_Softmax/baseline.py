"""PyTorch translation of the pinned matmul/bias/approximate-GELU/softmax.

GELU follows JAX 0.7.2's explicit tanh formulation, including the coefficient's
input-dtype cast. Independent standalone inputs are not equivalence fixtures.
"""

import math

import torch


CONFIG = {
    "name": "99_Matmul_GELU_Softmax",
    "batch_size": 4096,
    "in_features": 8192,
    "out_features": 8192,
}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    """Return the source's canonical uniform x and scaled normal weight/bias."""
    input_generator = torch.Generator(device=device).manual_seed(0)
    weight_generator = torch.Generator(device=device).manual_seed(42)
    x = torch.rand(
        (4096, 8192), dtype=dtype, device=device, generator=input_generator
    )
    weight = torch.randn(
        (8192, 8192), dtype=dtype, device=device, generator=weight_generator
    )
    weight = weight * weight.new_tensor(0.02)
    bias = torch.randn(
        (8192,), dtype=dtype, device=device, generator=weight_generator
    )
    bias = bias * bias.new_tensor(0.02)
    return x, weight, bias


def approximate_gelu(x):
    """Preserve jax.nn.gelu(approximate=True)'s expression and operation order."""
    # https://raw.githubusercontent.com/jax-ml/jax/jax-v0.7.2/jax/_src/nn/functions.py
    sqrt_2_over_pi = x.new_tensor(math.sqrt(2.0 / math.pi))
    cubic_coefficient = x.new_tensor(0.044715)
    cdf = 0.5 * (1.0 + torch.tanh(sqrt_2_over_pi * (x + cubic_coefficient * (x ** 3))))
    return x * cdf


def workload(x, weight, bias):
    """Multiply, add bias, apply approximate GELU, and normalize each row."""
    logits = approximate_gelu(torch.matmul(x, weight) + bias)
    # Explicit operations retain the baseline's dtype at each source boundary.
    unnormalized = torch.exp(logits - torch.amax(logits, dim=1, keepdim=True))
    return unnormalized / torch.sum(unnormalized, dim=1, keepdim=True)
