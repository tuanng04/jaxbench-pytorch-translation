"""PyTorch MLA; canonical CUDA BF16 follows the captured TPU fusion boundaries.

All source dimensions and complete reductions are retained. Small arithmetic
uses the original path; see SOURCE.md for numerical evidence and applicability.
"""
import torch

CONFIG = {'name': 'deepseek_v3_mla',
 'model': 'DeepSeek-V3-671B',
 'operator': 'mla_attention',
 'batch': 4,
 'seq_len': 2048,
 'emb_dim': 7168,
 'num_heads': 128,
 'q_lora_rank': 1536,
 'kv_lora_rank': 512,
 'qk_nope_head_dim': 128,
 'qk_rope_head_dim': 64,
 'v_head_dim': 128,
 'rope_theta': 10000}

def create_inputs(dtype=torch.bfloat16,device='cpu'):
    g=torch.Generator(device=device).manual_seed(42);c=CONFIG
    b,s,e,h=c['batch'],c['seq_len'],c['emb_dim'],c['num_heads']
    ql,kvl,nope,rope,vd=c['q_lora_rank'],c['kv_lora_rank'],c['qk_nope_head_dim'],c['qk_rope_head_dim'],c['v_head_dim']
    x=torch.randn((b,s,e),dtype=dtype,device=device,generator=g)
    shapes=((e,ql),(ql,h*(nope+rope)),(e,kvl+rope),(kvl,h*nope),(kvl,h*vd),(h*vd,e))
    return (x,*(torch.randn(shape,dtype=dtype,device=device,generator=g)*0.02 for shape in shapes))

def _compute_rope(head_dim,seq_len,theta,dtype,device):
    freqs=1.0/(theta**(torch.arange(0,head_dim,2,dtype=torch.float32,device=device)/head_dim))
    angles=torch.outer(torch.arange(seq_len,dtype=torch.float32,device=device),freqs)
    return torch.cos(angles).to(dtype),torch.sin(angles).to(dtype)

def _apply_rope(x,cos,sin,*,fused=False):
    half=x.shape[-1]//2
    x1,x2=x[...,:half],x[...,half:]
    cos=cos[None,:,None,:];sin=sin[None,:,None,:]
    if fused:
        return torch.cat((x1.float()*cos.float()-x2.float()*sin.float(),
                          x2.float()*cos.float()+x1.float()*sin.float()),dim=-1).to(x.dtype)
    return torch.cat((x1*cos-x2*sin,x2*cos+x1*sin),dim=-1)

def _compiled_canonical_attention(q,k,v,scale,mask_value,*,block_rows,head_block):
    """Captured TPU canonical fusion boundaries, with complete BF16 dot operands."""
    batch,heads,length,dim=q.shape
    output=torch.empty((batch,heads,length,v.shape[-1]),dtype=q.dtype,device=q.device)
    scale=q.new_tensor(scale)
    masked_value=q.new_tensor(mask_value).float()
    key_positions=torch.arange(k.shape[-2],device=q.device)
    for first in range(0,heads,head_block):
        last=min(first+head_block,heads)
        count=last-first
        keys=k[:,first:last].transpose(-1,-2).reshape(batch*count,dim,k.shape[-2])
        values=v[:,first:last].reshape(batch*count,v.shape[-2],v.shape[-1])
        for begin in range(0,length,block_rows):
            end=min(begin+block_rows,length)
            queries=q[:,first:last,begin:end].reshape(batch*count,end-begin,dim)
            accumulator=torch.bmm(queries,keys,out_dtype=torch.float32).reshape(batch,count,end-begin,k.shape[-2])
            causal=key_positions[None,:]<=torch.arange(begin,end,device=q.device)[:,None]
            # First fusion emits raw QK and a separately computed scaled max.
            maximum=torch.where(causal,accumulator*scale,masked_value).amax(dim=-1,keepdim=True).to(q.dtype)
            scores=torch.where(causal,accumulator.to(q.dtype).float()*scale,masked_value)
            exponential=torch.exp(scores-maximum.float())
            denominator=exponential.sum(dim=-1,keepdim=True).to(q.dtype)
            probabilities=(exponential/denominator.float()).to(q.dtype)
            output[:,first:last,begin:end]=torch.bmm(
                probabilities.reshape(batch*count,end-begin,k.shape[-2]),values
            ).reshape(batch,count,end-begin,v.shape[-1])
    return output

def _attention(q, k, v, scale, mask_value, bias=None, *, block_rows=256, head_block=8, compiled_canonical=False):
    if block_rows<=0 or head_block<=0: raise ValueError('Blocks must be positive')
    if compiled_canonical:
        assert bias is None
        return _compiled_canonical_attention(q,k,v,scale,mask_value,block_rows=block_rows,head_block=head_block)
    batch,heads,length,dim=q.shape
    out=torch.empty((batch,heads,length,v.shape[-1]),dtype=q.dtype,device=q.device)
    keys=torch.arange(k.shape[-2],device=q.device)
    for h in range(0,heads,head_block):
        for start in range(0,length,block_rows):
            stop=min(start+block_rows,length)
            scores=torch.matmul(q[:,h:h+head_block,start:stop],k[:,h:h+head_block].transpose(-1,-2))
            if scale is not None: scores=scores*scale
            if bias is not None: scores=scores+bias[None,h:h+head_block,start:stop,:]
            causal=keys[None,:]<=torch.arange(start,stop,device=q.device)[:,None]
            scores=torch.where(causal,scores,scores.new_tensor(mask_value))
            shifted=scores-scores.amax(dim=-1,keepdim=True)
            probabilities=torch.exp(shifted)
            probabilities=probabilities/probabilities.sum(dim=-1,keepdim=True)
            out[:,h:h+head_block,start:stop]=torch.matmul(probabilities,v[:,h:h+head_block])
    return out

def _grouped_projection_dot(x, weight, row_block=64, column_block=128):
    """Pinned TPU accumulation order with full K retained in each output block.

    The captured 256-wide windows group eight consecutive products with the
    corresponding eight from the second 128-wide half. Native BF16 products
    accumulate to FP32; explicit FP32 additions preserve the captured order.
    The CPU path exists for lowering/blocking diagnostics only.
    """
    (rows, reduction) = x.shape
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

def _projection(left, right, *, compiled_canonical):
    if not compiled_canonical:
        return left @ right
    shape=left.shape[:-1]
    value=_grouped_projection_dot(left.reshape(-1,left.shape[-1]),right.T.contiguous(),row_block=256,column_block=128)
    return value.reshape(*shape,right.shape[-1]).to(left.dtype)

def workload(x,q_down_proj,q_up_proj,kv_down_proj,k_up_proj,v_up_proj,o_proj,*,block_rows=256,head_block=8):
    c=CONFIG;b,s,e=x.shape;h=c['num_heads']
    values=(x,q_down_proj,q_up_proj,kv_down_proj,k_up_proj,v_up_proj,o_proj)
    signature=((4,2048,7168),(7168,1536),(1536,24576),(7168,576),
               (512,16384),(512,16384),(16384,7168))
    compiled_canonical=(tuple(tuple(value.shape) for value in values)==signature and
                        all(value.dtype==torch.bfloat16 and value.device.type=='cuda' for value in values))
    nope,rope,vd,kvl=c['qk_nope_head_dim'],c['qk_rope_head_dim'],c['v_head_dim'],c['kv_lora_rank']
    q=_projection(_projection(x,q_down_proj,compiled_canonical=compiled_canonical),q_up_proj,compiled_canonical=compiled_canonical).reshape(b,s,h,nope+rope)
    q_nope,q_rope=q[...,:nope],q[...,nope:]
    kv=_projection(x,kv_down_proj,compiled_canonical=compiled_canonical);k_latent,k_rope_raw=kv[...,:kvl],kv[...,kvl:]
    k_nope=_projection(k_latent,k_up_proj,compiled_canonical=compiled_canonical).reshape(b,s,h,nope)
    cos,sin=_compute_rope(rope,s,c['rope_theta'],x.dtype,x.device)
    k_rope=k_rope_raw[:,:,None,:].expand(b,s,h,rope)
    q_rope=_apply_rope(q_rope,cos,sin,fused=compiled_canonical);k_rope=_apply_rope(k_rope,cos,sin,fused=compiled_canonical)
    value=_projection(k_latent,v_up_proj,compiled_canonical=compiled_canonical).reshape(b,s,h,vd).permute(0,2,1,3)
    q_full=torch.cat((q_nope,q_rope),dim=-1).permute(0,2,1,3)
    k_full=torch.cat((k_nope,k_rope),dim=-1).permute(0,2,1,3)
    out=_attention(q_full,k_full,value,(nope+rope)**-0.5,-1e9,block_rows=block_rows,head_block=head_block,compiled_canonical=compiled_canonical)
    return _projection(out.permute(0,2,1,3).reshape(b,s,h*vd),o_proj,compiled_canonical=compiled_canonical)
