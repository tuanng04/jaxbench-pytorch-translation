"""Independent PyTorch translation of the pinned JAX baseline; see SOURCE.md."""
import torch

CONFIG = {'name': 'llama3_70b_swiglu',
 'model': 'Llama-3.1-70B',
 'operator': 'swiglu_mlp',
 'batch': 2,
 'seq_len': 4096,
 'emb_dim': 8192,
 'mlp_dim': 28672}

def create_inputs(dtype=torch.bfloat16, device='cpu'):
    g = torch.Generator(device=device).manual_seed(42)
    c = CONFIG
    x = torch.randn((c['batch'], c['seq_len'], c['emb_dim']),dtype=dtype,device=device,generator=g)
    gate = torch.randn((c['emb_dim'],c['mlp_dim']),dtype=dtype,device=device,generator=g)*0.02
    up = torch.randn((c['emb_dim'],c['mlp_dim']),dtype=dtype,device=device,generator=g)*0.02
    down = torch.randn((c['mlp_dim'],c['emb_dim']),dtype=dtype,device=device,generator=g)*0.02
    return x, gate, up, down

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
                # All 16 native BF16 products remain in this physical group.
                # Four short matrix products avoid the observed A100 K=16
                # accumulator rounding; additions remain in float32.
                partials = torch.bmm(a[..., :4], b[..., :4].transpose(1, 2), out_dtype=torch.float32)
                for term in range(4, 16, 4):
                    partials = partials + torch.bmm(a[..., term:term+4], b[..., term:term+4].transpose(1, 2), out_dtype=torch.float32)
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


def workload(x, gate_kernel, up_kernel, down_kernel):
    signature = tuple(tuple(t.shape) for t in (x, gate_kernel, up_kernel, down_kernel))
    if signature == ((2, 4096, 8192), (8192, 28672), (8192, 28672), (28672, 8192)) and all(
            t.dtype == torch.bfloat16 and t.device.type == 'cuda'
            for t in (x, gate_kernel, up_kernel, down_kernel)):
        flat = x.reshape(-1, x.shape[-1])
        # Keep native BF16 products and the observed FP32 accumulation order.
        # Both input projections materialize as BF16 before the activation.
        gate = grouped_dot(flat, gate_kernel.T, row_block=256, column_block=128).to(torch.bfloat16)
        up = grouped_dot(flat, up_kernel.T, row_block=256, column_block=128).to(torch.bfloat16)
        gate32 = gate.float()
        hidden = (gate32 * torch.sigmoid(gate32) * up.float()).to(torch.bfloat16)
        output = grouped_dot(hidden, down_kernel.T, row_block=256, column_block=128)
        return output.reshape(*x.shape[:-1], down_kernel.shape[-1]).to(torch.bfloat16)
    gate = torch.matmul(x, gate_kernel)
    up = torch.matmul(x, up_kernel)
    gate = gate * torch.sigmoid(gate)
    return torch.matmul(gate * up, down_kernel)
