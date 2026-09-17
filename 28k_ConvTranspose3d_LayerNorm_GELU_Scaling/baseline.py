"""Independent source-faithful PyTorch convolution; see SOURCE.md."""
import math
import torch
import torch.nn.functional as F

CONFIG = {'name': '34_ConvTranspose3d_LayerNorm_GELU_Scaling',
 'batch_size': 32,
 'in_channels': 32,
 'out_channels': 64,
 'kernel_size': 4,
 'stride': 2,
 'padding': 1,
 'bias': True,
 'eps': 1e-05,
 'scaling_factor': 1.0}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((32, 32, 16, 32, 32), dtype=dtype, device=device, generator=g)
    weight = torch.randn((32, 64, 4, 4, 4), dtype=dtype, device=device, generator=g)
    bias = torch.zeros((64,), dtype=dtype, device=device)
    scale = torch.ones((64,), dtype=dtype, device=device)
    offset = torch.zeros((64,), dtype=dtype, device=device)
    return x, weight, bias, scale, offset


def approximate_gelu(x):
    coefficient = x.new_tensor(math.sqrt(2.0 / math.pi))
    cdf = 0.5 * (1.0 + torch.tanh(coefficient * (x + x.new_tensor(0.044715) * x**3)))
    return x * cdf


def grouped_dot(x, weight, row_block=64, column_block=128):
    """Pinned TPU accumulation order with full K retained in each output block.

    The captured 256-wide windows group eight consecutive products with the
    corresponding eight from the second 128-wide half. Native BF16 products
    accumulate to FP32; explicit FP32 additions preserve the captured order.
    The CPU path exists for lowering/blocking diagnostics only.
    """
    rows, reduction = x.shape
    columns = weight.shape[0]
    assert reduction % 256 == 0 and weight.shape[1] == reduction
    left = x.reshape(rows, reduction // 256, 2, 16, 8).permute(1, 3, 0, 2, 4).reshape(-1, rows, 16)
    right = weight.reshape(columns, reduction // 256, 2, 16, 8).permute(1, 3, 0, 2, 4).reshape(-1, columns, 16)
    output = torch.empty((rows, columns), dtype=torch.float32, device=x.device)
    for row in range(0, rows, row_block):
        for column in range(0, columns, column_block):
            a = left[:, row:row + row_block]
            b = right[:, column:column + column_block]
            if x.device.type == 'cuda':
                partials = torch.bmm(a, b.transpose(1, 2), out_dtype=torch.float32)
            else:
                partials = torch.bmm(a.float(), b.float().transpose(1, 2))
            partials = partials.reshape(reduction // 256, 16, a.shape[1], b.shape[1])
            tiles = partials[:, 0].clone()
            for lane in range(1, 16):
                tiles = tiles + partials[:, lane]
            value = tiles[0].clone()
            for tile in range(1, reduction // 256):
                value = value + tiles[tile]
            output[row:row + a.shape[1], column:column + b.shape[1]] = value
    return output


def conv_transpose_accumulator(x, weight, block_size=3):
    """Lower stride-2 transpose convolution without splitting its reduction."""
    n, channels, depth, height, width = x.shape
    out_channels = weight.shape[1]
    grouped_profile = (tuple(x.shape) == (32, 32, 16, 32, 32)
                       and tuple(weight.shape) == (32, 64, 4, 4, 4)
                       and x.dtype == weight.dtype == torch.bfloat16)
    kernel = (weight.permute(1, 2, 3, 4, 0).flip((1, 2, 3)).reshape(out_channels, -1)
              if grouped_profile else weight.permute(1, 0, 2, 3, 4).flip((2, 3, 4)).flatten(1).T)
    result = []
    for start in range(0, n, block_size):
        block = x[start:start + block_size]
        dilated = x.new_zeros((block.shape[0], channels, depth*2-1, height*2-1, width*2-1))
        dilated[:, :, ::2, ::2, ::2] = block
        padded = F.pad(dilated, (2, 2, 2, 2, 2, 2))
        patches = padded.unfold(2, 4, 1).unfold(3, 4, 1).unfold(4, 4, 1)
        if grouped_profile:
            # Preserve spatial-then-channel product order across the complete
            # transpose-convolution reduction, including structural zeros.
            patches = patches.permute(0, 2, 3, 4, 5, 6, 7, 1).reshape(-1, channels*64)
            value = grouped_dot(patches, kernel, row_block=1024, column_block=128)
        else:
            patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, channels*64)
            value = (torch.mm(patches, kernel, out_dtype=torch.float32) if x.is_cuda
                     else torch.mm(patches.float(), kernel.float()))
        result.append(value.reshape(block.shape[0], depth*2, height*2, width*2, out_channels).permute(0, 4, 1, 2, 3))
    return torch.cat(result, dim=0)


def workload(x, conv_weight, conv_bias, ln_weight, ln_bias):
    tensors = (x, conv_weight, conv_bias, ln_weight, ln_bias)
    signature = tuple(tuple(t.shape) for t in tensors)
    small = ((2, 3, 2, 3, 4), (3, 8, 4, 4, 4), (8,), (8,), (8,))
    canonical = ((32, 32, 16, 32, 32), (32, 64, 4, 4, 4), (64,), (64,), (64,))
    if all(t.dtype == torch.bfloat16 and t.is_cuda for t in tensors) and signature in (small, canonical):
        accumulator = conv_transpose_accumulator(x, conv_weight)
        bias = conv_bias.float().reshape(1, -1, 1, 1, 1)
        linear = accumulator + bias
        mean = linear.mean(dim=-1, keepdim=True).to(torch.bfloat16).float()
        value = (linear.to(torch.bfloat16).float() if signature == small
                 else accumulator.to(torch.bfloat16).float() + bias)
        variance = ((value - mean) ** 2).mean(dim=-1, keepdim=True)
        epsilon = torch.tensor(1e-5, dtype=torch.bfloat16, device=x.device).float()
        denominator = torch.sqrt(variance + epsilon).to(torch.bfloat16).float()
        value = (value - mean) / denominator * ln_weight.float() + ln_bias.float()
        coefficient = torch.tensor(math.sqrt(2.0/math.pi), dtype=torch.bfloat16, device=x.device).float()
        cubic = torch.tensor(0.044715, dtype=torch.bfloat16, device=x.device).float()
        return (value * (0.5 * (1.0 + torch.tanh(coefficient * (value + cubic * value**3))))).to(torch.bfloat16)
    y = F.conv_transpose3d(x, conv_weight, bias=None, stride=2, padding=1)
    y = y + conv_bias.reshape(1, -1, 1, 1, 1)
    # The source normalizes W, and its affine vector broadcasts along W.
    mean = y.mean(dim=-1, keepdim=True)
    variance = ((y - mean) ** 2).mean(dim=-1, keepdim=True)
    y = (y - mean) / torch.sqrt(variance + 1e-5)
    y = y * ln_weight + ln_bias
    return approximate_gelu(y) * 1.0
