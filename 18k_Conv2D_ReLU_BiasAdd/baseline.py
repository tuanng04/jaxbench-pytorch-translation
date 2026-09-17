"""PyTorch translation of JAXBench's Conv2D, ReLU, and second bias add.

Standalone inputs use an independent random stream; equivalence checks load
retained JAX inputs. The external NCHW/OIHW layouts match the source interface.
"""

import torch
from torch.nn import functional as F


CONFIG = {
    "name": "1_Conv2D_ReLU_BiasAdd",
    "batch_size": 128,
    "in_channels": 64,
    "out_channels": 128,
    "kernel_size": 3,
}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    """Return the source's canonical x, weight, conv_bias, and post-ReLU bias."""
    input_generator = torch.Generator(device=device).manual_seed(0)
    weight_generator = torch.Generator(device=device).manual_seed(0xBADC0DE)
    x = torch.rand(
        (128, 64, 128, 128), dtype=dtype, device=device, generator=input_generator
    )
    weight = torch.randn(
        (128, 64, 3, 3), dtype=dtype, device=device, generator=weight_generator
    )
    weight = weight * weight.new_tensor(0.02)
    conv_bias = torch.randn(
        (128,), dtype=dtype, device=device, generator=weight_generator
    )
    conv_bias = conv_bias * conv_bias.new_tensor(0.02)
    bias = torch.randn(
        (128, 1, 1), dtype=dtype, device=device, generator=weight_generator
    )
    bias = bias * bias.new_tensor(0.02)
    return x, weight, conv_bias, bias


def workload(x, weight, conv_bias, bias):
    """VALID cross-correlation, first bias, ReLU, then a distinct second bias."""
    # Keep the convolution and first bias separate: the JAX convolution returns
    # an input-dtype tensor before the bias add (including for bfloat16).
    conv = F.conv2d(x, weight, bias=None, stride=1, padding=0)
    activated = torch.relu(conv + conv_bias.reshape(1, -1, 1, 1))
    return activated + bias
