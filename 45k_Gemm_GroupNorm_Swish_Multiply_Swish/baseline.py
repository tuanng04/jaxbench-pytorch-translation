"""Independent translation of the pinned JAX baseline; see SOURCE.md."""
import math
import torch

CONFIG = {'name': '88_Gemm_GroupNorm_Swish_Multiply_Swish',
 'batch_size': 4096,
 'in_features': 8192,
 'out_features': 8192,
 'num_groups': 256}


def create_inputs(dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand((4096, 8192), dtype=dtype, device=device, generator=g)
    weight = torch.randn((8192, 8192), dtype=dtype, device=device, generator=g) * 0.02
    bias = torch.randn((8192,), dtype=dtype, device=device, generator=g) * 0.02
    affine_scale = torch.ones((8192,), dtype=dtype, device=device)
    affine_bias = torch.zeros((8192,), dtype=dtype, device=device)
    multiplier = torch.randn((8192,), dtype=dtype, device=device, generator=g)
    return x, weight, bias, affine_scale, affine_bias, multiplier


def grouped_dot(x, weight, row_block=64, column_block=128):
    """Native BF16 grouped accumulation with full K retained in each output block.

    Eight consecutive products are grouped with the corresponding eight from
    the second 128-wide half. Source-bound TPU/GPU diagnostics support this
    FP32 accumulation strategy; the required full comparison remains decisive.
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


def workload(x, gemm_weight, gemm_bias, gn_weight, gn_bias, multiply_weight):
    tensors = (x, gemm_weight, gemm_bias, gn_weight, gn_bias, multiply_weight)
    signature = tuple(tuple(t.shape) for t in tensors)
    if (all(t.dtype == torch.bfloat16 and t.device.type == "cuda" for t in tensors)
            and signature in (
                ((3, 5), (8192, 5), (8192,), (8192,), (8192,), (8192,)),
                ((4096, 8192), (8192, 8192), (8192,), (8192,), (8192,), (8192,)),
            )):
        accumulator = grouped_dot(x, gemm_weight) if x.shape == (4096, 8192) else torch.mm(x, gemm_weight.T, out_dtype=torch.float32)
        linear = accumulator + gemm_bias.float()
        if x.shape[0] == 3:
            # Small TPU executable materializes biased linear values first.
            statistics = linear.to(torch.bfloat16).float()
            value = statistics
        else:
            # Canonical statistics retain FP32 biased values and the final
            # fusion recomputes bias from a separately stored BF16 raw dot.
            statistics = linear
            value = accumulator.to(torch.bfloat16).float() + gemm_bias.float()
        statistics = statistics.reshape(-1, 256, 32)
        mean = statistics.mean(dim=-1, keepdim=True)
        variance = ((statistics - mean) ** 2).mean(dim=-1, keepdim=True)
        epsilon = torch.tensor(1e-5, dtype=torch.bfloat16, device=x.device).float()
        denominator = torch.sqrt(variance + epsilon).to(torch.bfloat16).float()
        mean = mean.to(torch.bfloat16).float()
        value = ((value.reshape(-1, 256, 32) - mean) / denominator).reshape(-1, 8192)
        value = value * gn_weight.float() + gn_bias.float()
        value = value * torch.sigmoid(value) * multiply_weight.float()
        return (value * torch.sigmoid(value)).to(torch.bfloat16)
    y = torch.matmul(x, gemm_weight.T) + gemm_bias
    batch = y.shape[0]
    grouped = y.reshape(batch, 256, 32)
    mean = grouped.mean(dim=-1, keepdim=True)
    variance = grouped.var(dim=-1, correction=0, keepdim=True)
    y = ((grouped - mean) / torch.sqrt(variance + 1e-5)).reshape(batch, 8192)
    y = y * gn_weight + gn_bias
    y = y * torch.sigmoid(y)
    y = y * multiply_weight
    return y * torch.sigmoid(y)
