"""Independent PyTorch translation of the pinned JAX baseline; see SOURCE.md."""
import torch

CONFIG = {'name': 'llama3_70b_paged_attention',
 'model': 'Llama-3.1-70B',
 'operator': 'paged_attention',
 'num_seqs': 64,
 'max_seq_len': 4096,
 'num_query_heads': 64,
 'num_kv_heads': 8,
 'head_dim': 128,
 'page_size': 16,
 'pages_per_seq': 256}

def create_inputs(dtype=torch.bfloat16,device='cpu'):
    g=torch.Generator(device=device).manual_seed(42);c=CONFIG
    n,h,hkv,d,p,pps=c['num_seqs'],c['num_query_heads'],c['num_kv_heads'],c['head_dim'],c['page_size'],c['pages_per_seq']
    queries=torch.randn((n,h,d),dtype=dtype,device=device,generator=g)
    kv=[torch.randn((n*pps,p,hkv,d),dtype=dtype,device=device,generator=g)*0.02 for _ in range(2)]
    lengths=torch.full((n,),c['max_seq_len'],dtype=torch.int32,device=device)
    indices=torch.arange(n*pps,dtype=torch.int32,device=device).reshape(n,pps)
    cumulative=torch.arange(n+1,dtype=torch.int32,device=device)
    return queries,*kv,lengths,indices,cumulative

def workload(queries,k_pages,v_pages,kv_lens,page_indices,cu_q_lens):
    c=CONFIG;n,h,hkv,d=c['num_seqs'],c['num_query_heads'],c['num_kv_heads'],c['head_dim']
    length=c['pages_per_seq']*c['page_size'];groups=h//hkv
    positions=torch.arange(length,device=queries.device);outputs=[]
    for seq in range(n):
        start=cu_q_lens[seq].clamp(0,queries.shape[0]-1).to(torch.int64)
        q=queries.index_select(0,start.reshape(1))
        indices=page_indices[seq].to(torch.int64)
        k=k_pages.index_select(0,indices).reshape(length,hkv,d).repeat_interleave(groups,dim=1)
        v=v_pages.index_select(0,indices).reshape(length,hkv,d).repeat_interleave(groups,dim=1)
        scores=torch.einsum('qhd,khd->hqk',q,k)*(d**-0.5)
        scores=torch.where(positions<kv_lens[seq],scores,scores.new_tensor(-1e30))
        probabilities=torch.exp(scores-scores.amax(dim=-1,keepdim=True))
        probabilities=probabilities/probabilities.sum(dim=-1,keepdim=True)
        outputs.append(torch.einsum('hqk,khd->qhd',probabilities,v).squeeze(0))
    return torch.stack(outputs)
