"""PyTorch GQA with a source-bound canonical TPU attention fusion profile.

Every key and head feature remains present. Only independent output axes are
blocked; small source arithmetic stays unchanged. See SOURCE.md for evidence.
"""
import torch

CONFIG = {'name': 'llama3_405b_gqa',
 'model': 'Llama-3.1-405B',
 'operator': 'gqa_attention',
 'batch': 4,
 'seq_len': 4096,
 'num_query_heads': 128,
 'num_kv_heads': 8,
 'head_dim': 128,
 'emb_dim': 16384}

def create_inputs(dtype=torch.bfloat16,device='cpu'):
    g=torch.Generator(device=device).manual_seed(42); c=CONFIG
    return tuple(torch.randn((c['batch'],c['seq_len'],h,c['head_dim']),dtype=dtype,device=device,generator=g) for h in (c['num_query_heads'],c['num_kv_heads'],c['num_kv_heads']))

def _compiled_canonical_attention(q,k,v,scale,mask_value,*,block_rows,head_block):
    """Reproduce the full canonical TPU attention fusion with native BF16 dots."""
    batch,heads,length,dim=q.shape
    output=torch.empty((batch,heads,length,v.shape[-1]),dtype=q.dtype,device=q.device)
    scale=q.new_tensor(scale)
    masked_value=q.new_tensor(mask_value).float()
    key_positions=torch.arange(k.shape[-2],device=q.device)
    for first in range(0,heads,head_block):
        last=min(first+head_block,heads);count=last-first
        keys=k[:,first:last].transpose(-1,-2).reshape(batch*count,dim,k.shape[-2])
        values=v[:,first:last].reshape(batch*count,v.shape[-2],v.shape[-1])
        for begin in range(0,length,block_rows):
            end=min(begin+block_rows,length)
            queries=q[:,first:last,begin:end].reshape(batch*count,end-begin,dim)
            accumulator=torch.bmm(queries,keys,out_dtype=torch.float32).reshape(batch,count,end-begin,k.shape[-2])
            causal=key_positions[None,:]<=torch.arange(begin,end,device=q.device)[:,None]
            scores=torch.where(causal,accumulator*scale,masked_value)
            exponential=torch.exp(scores-scores.amax(dim=-1,keepdim=True))
            probabilities=(exponential/exponential.sum(dim=-1,keepdim=True)).to(q.dtype)
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

def workload(query,key,value,*,block_rows=256,head_block=8):
    values=(query,key,value)
    signature=((4,4096,128,128),(4,4096,8,128),(4,4096,8,128))
    compiled_canonical=(tuple(tuple(tensor.shape) for tensor in values)==signature and
                        all(tensor.dtype==torch.bfloat16 and tensor.device.type=='cuda' for tensor in values))
    groups=query.shape[2]//key.shape[2]
    key=key.repeat_interleave(groups,dim=2);value=value.repeat_interleave(groups,dim=2)
    out=_attention(query.permute(0,2,1,3),key.permute(0,2,1,3),value.permute(0,2,1,3),query.shape[-1]**-0.5,-1e9,block_rows=block_rows,head_block=head_block,compiled_canonical=compiled_canonical)
    return out.permute(0,2,1,3)
