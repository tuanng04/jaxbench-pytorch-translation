"""Independent PyTorch translation of the pinned JAX baseline; see SOURCE.md."""
import torch

CONFIG = {'name': 'megablox_gmm_qwen3_235b',
 'model': 'Qwen3-235B-A22B',
 'operator': 'grouped_matmul',
 'num_experts': 128,
 'num_experts_per_tok': 8,
 'emb_dim': 4096,
 'moe_mlp_dim': 1536,
 'seq_len': 4096}

_static_argnums=(3,)

def create_inputs(dtype=torch.bfloat16,device='cpu'):
    g=torch.Generator(device=device).manual_seed(42);c=CONFIG
    m=c['seq_len']*c['num_experts_per_tok'];groups=c['num_experts'];capacity=m//groups
    lhs=torch.randn((m,c['emb_dim']),dtype=dtype,device=device,generator=g).to(torch.bfloat16).to(dtype)
    rhs=(torch.randn((groups,c['emb_dim'],c['moe_mlp_dim']),dtype=dtype,device=device,generator=g)*0.02).to(torch.bfloat16).to(dtype)
    sizes=torch.full((groups,),capacity,dtype=torch.int32,device=device)
    return lhs,rhs,sizes,capacity

def workload(lhs,rhs,group_sizes,max_expert_size):
    m,k=lhs.shape;n=rhs.shape[2];capacity=max_expert_size
    ends=torch.cumsum(group_sizes,dim=0,dtype=torch.int32)
    starts=torch.cat((torch.zeros(1,dtype=torch.int32,device=lhs.device),ends[:-1]))
    output=torch.zeros((m+capacity,n),dtype=lhs.dtype,device=lhs.device)
    rows=torch.arange(capacity,device=lhs.device)
    for i in range(rhs.shape[0]):
        # dynamic_slice clamps its start so the full static-size slice fits.
        source_rows=starts[i].clamp(0,m-capacity).to(torch.int64)+rows
        source=lhs.index_select(0,source_rows)
        product=source.to(torch.float32)@rhs[i].to(torch.float32)
        product=torch.where(rows[:,None]<group_sizes[i],product,0.0).to(lhs.dtype)
        destination_rows=starts[i].clamp(0,m).to(torch.int64)+rows
        updated=output.index_select(0,destination_rows)+product
        output.index_copy_(0,destination_rows,updated)
    return output[:m]
