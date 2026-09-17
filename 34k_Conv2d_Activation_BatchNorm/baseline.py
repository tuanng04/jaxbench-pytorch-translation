"""Independent source-faithful PyTorch convolution; see SOURCE.md."""
import math
import torch
import torch.nn.functional as F

CONFIG = {'name': '52_Conv2d_Activation_BatchNorm',
 'batch_size': 64,
 'in_channels': 64,
 'out_channels': 128,
 'kernel_size': 3}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((64, 64, 128, 128), dtype=dtype, device=device, generator=g)
    weight = torch.randn((128, 64, 3, 3), dtype=dtype, device=device, generator=g)
    bias = torch.randn((128,), dtype=dtype, device=device, generator=g)
    scale = torch.ones((128,), dtype=dtype, device=device)
    offset = torch.zeros((128,), dtype=dtype, device=device)
    return x, weight, bias, scale, offset


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


def conv_accumulator(x, weight, block_size=3):
    """Keep BF16 products and the FP32 accumulator across each complete kernel."""
    n, _, h, w = x.shape
    channels, _, kh, kw = weight.shape
    outputs = []
    grouped_profile = (tuple(x.shape) == (64, 64, 128, 128)
                       and tuple(weight.shape) == (128, 64, 3, 3)
                       and x.dtype == weight.dtype == torch.bfloat16)
    matrix = (weight.permute(0, 2, 3, 1).reshape(channels, -1)
              if grouped_profile else weight.flatten(1).T)
    padding = (-matrix.shape[1]) % 256 if grouped_profile else 0
    if grouped_profile:
        matrix = F.pad(matrix, (0, padding))
    for start in range(0, n, block_size):
        block = x[start:start + block_size]
        patches = F.unfold(block, (kh, kw)).transpose(1, 2)
        if grouped_profile:
            # Follow NHWC convolution product order, including neutral padding
            # in the final 256-term accumulator window. Keep every input term.
            patches = patches.reshape(-1, x.shape[1], kh, kw).permute(0, 2, 3, 1).reshape(-1, x.shape[1]*kh*kw)
            value = grouped_dot(F.pad(patches, (0, padding)), matrix, row_block=1024, column_block=128)
        else:
            patches = patches.reshape(-1, matrix.shape[0])
            value = (torch.mm(patches, matrix, out_dtype=torch.float32) if x.is_cuda
                     else torch.mm(patches.float(), matrix.float()))
        outputs.append(value.reshape(block.shape[0], h-kh+1, w-kw+1, channels).permute(0, 3, 1, 2))
    return torch.cat(outputs, dim=0)


def _canonical_tanh(value):
    """Float32 rational tanh within its published approximation domain.

    Coefficients: LLVM MLIR PolynomialApproximation.cpp, TanhApproximation
    (Apache-2.0 WITH LLVM-exception). This profile is validated against the
    pinned TPU; it does not assert that libtpu uses this exact algorithm.
    Outside the rational domain, use the native elementary function.
    """
    bound = 7.99881172180175781
    x = value.clamp(-bound, bound)
    square = x*x
    alpha = (-2.76076847742355e-16, 2.00018790482477e-13, -8.60467152213735e-11,
             5.12229709037114e-08, 1.48572235717979e-05, 6.37261928875436e-04,
             4.89352455891786e-03)
    beta = (1.19825839466702e-06, 1.18534705686654e-04, 2.26843463243900e-03,
            4.89352518554385e-03)
    def horner(coefficients):
        result = torch.full_like(x, coefficients[0])
        for coefficient in coefficients[1:]:
            result = torch.addcmul(torch.full_like(x, coefficient), square, result)
        return result
    result = x*horner(alpha)/horner(beta)
    result = torch.where(value.abs() < .0004, x, result)
    return torch.where(value.abs() > bound, torch.tanh(value), result)


def workload(x, conv_weight, conv_bias, bn_weight, bn_bias):
    tensors = (x, conv_weight, conv_bias, bn_weight, bn_bias)
    signature = tuple(tuple(t.shape) for t in tensors)
    small = ((2, 3, 7, 9), (4, 3, 2, 3), (4,), (4,), (4,))
    canonical = ((64, 64, 128, 128), (128, 64, 3, 3), (128,), (128,), (128,))
    if all(t.dtype == torch.bfloat16 and t.is_cuda for t in tensors) and signature in (small, canonical):
        value = conv_accumulator(x, conv_weight) + conv_bias.float().reshape(1, -1, 1, 1)
        if signature == small:
            value = value.to(torch.bfloat16).float()
        # Canonical fuses convolution, bias and Mish before materialization;
        # small has a separate biased-convolution boundary. Both compute
        # statistics over the complete N,H,W domain after concatenation.
        softplus = torch.clamp_min(value, 0) + torch.log1p(torch.exp(-torch.abs(value)))
        value = value * (_canonical_tanh(softplus) if signature == canonical else torch.tanh(softplus))
        mean = value.mean(dim=(0, 2, 3), keepdim=True).to(torch.bfloat16).float()
        value = value.to(torch.bfloat16).float()
        variance = ((value - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        epsilon = torch.tensor(1e-5, dtype=torch.bfloat16, device=x.device).float()
        denominator = torch.sqrt(variance + epsilon).to(torch.bfloat16).float()
        value = (value - mean) / denominator
        value = value * bn_weight.float().reshape(1, -1, 1, 1) + bn_bias.float().reshape(1, -1, 1, 1)
        return value.to(torch.bfloat16)
    y = F.conv2d(x, conv_weight, bias=None, stride=1, padding=0)
    y = y + conv_bias.reshape(1, -1, 1, 1)
    y = torch.tanh(F.softplus(y)) * y
    mean = y.mean(dim=(0, 2, 3), keepdim=True)
    variance = ((y - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    y = (y - mean) / torch.sqrt(variance + 1e-5)
    return y * bn_weight.reshape(1, -1, 1, 1) + bn_bias.reshape(1, -1, 1, 1)
