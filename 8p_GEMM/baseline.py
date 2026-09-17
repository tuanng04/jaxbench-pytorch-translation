"""Ordinary PyTorch translation of JAXBench's dense Llama-70B GEMM.

The standalone input factory has its own PyTorch random stream. Equivalence
checks must load the actual shared JAX-generated arrays instead of using it.
"""

import torch


CONFIG = {
    "name": "gemm_llama70b",
    "model": "Llama-3.1-70B",
    "operator": "dense_matmul",
    "M": 8192,
    "K": 8192,
    "N": 28672,
}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    """Return standalone (A, B), retaining the source's 0.02 weight scale."""
    generator = torch.Generator(device=device).manual_seed(42)
    m, k, n = CONFIG["M"], CONFIG["K"], CONFIG["N"]
    a = torch.randn((m, k), dtype=dtype, device=device, generator=generator)
    b = torch.randn((k, n), dtype=dtype, device=device, generator=generator) * 0.02
    return a, b


def workload(A, B):
    """Dense matmul: C = A @ B, preserving input and output dtypes."""
    return torch.matmul(A, B)
