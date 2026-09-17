"""Independent PyTorch translation of the pinned JAX baseline; see SOURCE.md."""
import torch

CONFIG = {'name': 'llama3_70b_flex_attention',
 'model': 'Llama-3.1-70B',
 'operator': 'flex_attention',
 'batch': 4,
 'seq_len': 4096,
 'num_heads': 64,
 'head_dim': 128}

def create_inputs(dtype=torch.bfloat16,device='cpu'):
    g=torch.Generator(device=device).manual_seed(42);c=CONFIG
    shape=(c['batch'],c['num_heads'],c['seq_len'],c['head_dim'])
    q=torch.randn(shape,dtype=dtype,device=device,generator=g)
    k=torch.randn(shape,dtype=dtype,device=device,generator=g)*0.02
    v=torch.randn(shape,dtype=dtype,device=device,generator=g)*0.02
    bias=torch.randn((c['num_heads'],c['seq_len'],c['seq_len']),dtype=dtype,device=device,generator=g)*0.01
    return q,k,v,bias

def _attention(q, k, v, scale, mask_value, bias=None, *, block_rows=256, head_block=8):
    if block_rows<=0 or head_block<=0: raise ValueError('Blocks must be positive')
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

def workload(q,k,v,rel_pos_bias,*,block_rows=256,head_block=8):
    if q.shape[-2]!=CONFIG['seq_len']: raise ValueError('Pinned causal mask requires the configured length')
    return _attention(q,k,v,CONFIG['head_dim']**-0.5,-1e30,rel_pos_bias,block_rows=block_rows,head_block=head_block)
