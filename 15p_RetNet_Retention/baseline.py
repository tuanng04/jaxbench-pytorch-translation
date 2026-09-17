"""Independent PyTorch translation of the pinned JAX baseline; see SOURCE.md."""
import torch

CONFIG = {'name': 'retnet_6_7b_retention',
 'model': 'RetNet-6.7B',
 'operator': 'multi_scale_retention',
 'batch': 4,
 'seq_len': 4096,
 'num_heads': 16,
 'head_dim': 256,
 'd_model': 4096}

def create_inputs(dtype=torch.bfloat16,device='cpu'):
    g=torch.Generator(device=device).manual_seed(42);c=CONFIG
    shape=(c['batch'],c['num_heads'],c['seq_len'],c['head_dim'])
    values=tuple(torch.randn(shape,dtype=dtype,device=device,generator=g) for _ in range(3))
    return values

def _compute(query,key,value,A_log=None,*,block_rows=256,head_block=8):
    if block_rows<=0 or head_block<=0:raise ValueError('Block size must be positive')
    b,h,s,d=query.shape;device=query.device
    positions=torch.arange(s,dtype=torch.float32,device=device)
    output=torch.empty_like(query)
    if A_log is None:
        gammas=1.0-torch.exp2(-5.0-torch.arange(h,dtype=torch.float32,device=device))
        log_gamma=torch.log(gammas)
    else:
        a=torch.sigmoid(A_log.to(torch.float32))
        cumulative=torch.cumsum(torch.log(a+1e-8),dim=-1)
    for first in range(0,h,head_block):
        last=min(first+head_block,h)
        for start in range(0,s,block_rows):
            stop=min(start+block_rows,s)
            distance=positions[start:stop,None]-positions[None,:]
            causal=distance>=0
            if A_log is None:
                decay=torch.exp(log_gamma[first:last,None,None]*distance.clamp_min(0)[None,:,:])*causal.to(torch.float32)[None,:,:]
                decay=decay[None,:,:,:]
            else:
                diff=cumulative[:,first:last,start:stop,None]-cumulative[:,first:last,None,:]
                decay=torch.exp(torch.where(causal,diff,diff.new_tensor(-1e30)))
            scores=query[:,first:last,start:stop].to(torch.float32)@key[:,first:last].to(torch.float32).transpose(-1,-2)
            scores=scores*decay
            if A_log is None:
                denominator=scores.abs().sum(dim=-1,keepdim=True).clamp_min(1.0)
            else:
                total=scores.sum(dim=-1,keepdim=True)
                total=torch.where(total.abs()<1e-6,1.0,total)
                denominator=total.abs().clamp_min(1.0)
            normalized=(scores/denominator).to(query.dtype)
            output[:,first:last,start:stop]=normalized@value[:,first:last]
    return output
def workload(query,key,value,*,block_rows=256,head_block=8):
    return _compute(query,key,value,block_rows=block_rows,head_block=head_block)
