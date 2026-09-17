"""PyTorch translation of JAXBench's independent grouped matrix products.

The actual source accepts two tensors; its input-factory docstring mentions
group_sizes, but no such argument is produced or used by the baseline.
"""

import torch


CONFIG = {
    "name": "mixtral_8x7b_ragged_dot",
    "model": "Mixtral-8x7B",
    "operator": "ragged_dot",
    "num_groups": 8,
    "M": 8192,
    "K": 4096,
    "N": 14336,
}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    """Return independent standalone x and weights at unchanged dimensions."""
    generator = torch.Generator(device=device).manual_seed(42)
    groups, rows, inner, cols = (
        CONFIG["num_groups"], CONFIG["M"], CONFIG["K"], CONFIG["N"]
    )
    x = torch.randn(
        (groups, rows // groups, inner),
        dtype=dtype, device=device, generator=generator,
    )
    weights = torch.randn(
        (groups, inner, cols), dtype=dtype, device=device, generator=generator
    )
    weights = weights * weights.new_tensor(0.02)
    return x, weights


def workload(x, weights):
    """Compute gmk,gkn->gmn without mixing groups or splitting reductions."""
    return torch.bmm(x, weights)
