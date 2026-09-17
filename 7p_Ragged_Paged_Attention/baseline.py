"""Independent PyTorch translation of the pinned JAX baseline; see SOURCE.md."""
import torch

CONFIG = {'name': 'ragged_paged_attention_llama70b',
 'model': 'Llama-3.1-70B',
 'operator': 'ragged_paged_attention',
 'max_num_batched_tokens': 4096,
 'max_num_seqs': 64,
 'num_q_heads': 64,
 'num_kv_heads': 8,
 'head_dim': 128,
 'page_size': 16,
 'pages_per_seq': 256}

ACTIVE_Q_LENS=(1,)*48+(512,)*7+(464,)
ACTIVE_KV_LENS=tuple(257+((i*73)%240)*16 for i in range(48))+(1023,1535,2047,2559,3071,3583,4095,4095)
DEFAULT_MASK_VALUE=-0.7*float(torch.finfo(torch.float32).max)

def create_inputs(dtype=torch.bfloat16,device='cpu'):
    g=torch.Generator(device=device).manual_seed(42);c=CONFIG
    t,n,h,hkv,d,p,pps=c['max_num_batched_tokens'],c['max_num_seqs'],c['num_q_heads'],c['num_kv_heads'],c['head_dim'],c['page_size'],c['pages_per_seq']
    q=torch.randn((t,h,d),dtype=dtype,device=device,generator=g)
    pages=torch.randn((n*pps,p,2*hkv,d),dtype=dtype,device=device,generator=g)
    kv_lens=torch.tensor(ACTIVE_KV_LENS+(0,)*(n-len(ACTIVE_Q_LENS)),dtype=torch.int32,device=device)
    lengths=torch.tensor(ACTIVE_Q_LENS,dtype=torch.int32,device=device)
    cumulative=torch.cat((torch.zeros(1,dtype=torch.int32,device=device),torch.cumsum(lengths,0,dtype=torch.int32),torch.full((n-len(ACTIVE_Q_LENS),),sum(ACTIVE_Q_LENS),dtype=torch.int32,device=device)))
    indices=torch.randperm(n*pps,generator=g,device=device,dtype=torch.int32).reshape(n,pps)
    return q,pages,kv_lens,indices,cumulative,torch.tensor([len(ACTIVE_Q_LENS)],dtype=torch.int32,device=device)

def workload(queries,kv_pages,kv_lens,page_indices,cu_q_lens,num_seqs,*,block_rows=256,head_block=8):
    if block_rows<=0 or head_block<=0:raise ValueError('Blocks must be positive')
    hkv=kv_pages.shape[2]//2;d=kv_pages.shape[3];h=queries.shape[1];groups=h//hkv
    output=torch.zeros_like(queries);device=queries.device
    for seq,capacity in enumerate(ACTIVE_Q_LENS):
        start=cu_q_lens[seq];qlen=cu_q_lens[seq+1]-start;klen=kv_lens[seq]
        indices=page_indices[seq].to(torch.int64)
        pages=kv_pages.index_select(0,indices)
        k=pages[:,:,0::2,:].reshape(-1,hkv,d).repeat_interleave(groups,dim=1)
        v=pages[:,:,1::2,:].reshape(-1,hkv,d).repeat_interleave(groups,dim=1)
        offsets=torch.arange(capacity,dtype=torch.int32,device=device)
        qpositions=(start+offsets).clamp_max(CONFIG['max_num_batched_tokens']-1).to(torch.int64)
        q=queries.index_select(0,qpositions)
        out=torch.empty_like(q)
        keys=torch.arange(k.shape[0],dtype=torch.int32,device=device)
        for first in range(0,h,head_block):
            last=min(first+head_block,h)
            for begin in range(0,capacity,block_rows):
                end=min(begin+block_rows,capacity)
                scores=torch.einsum('qhd,khd->hqk',q[begin:end,first:last].to(torch.float32),k[:,first:last].to(torch.float32))
                scores=scores*(CONFIG['head_dim']**-0.5)
                query_positions=klen-qlen+offsets[begin:end]
                masked=(query_positions[:,None]<keys[None,:])|(keys[None,:]>=klen)
                scores=torch.where(masked,scores.new_tensor(DEFAULT_MASK_VALUE),scores)
                probabilities=torch.softmax(scores,dim=-1).to(v.dtype)
                out[begin:end,first:last]=torch.einsum('hqk,khd->qhd',probabilities,v[:,first:last]).to(queries.dtype)
        valid=(offsets<qlen)&(seq<num_seqs[0])
        out=torch.where(valid[:,None,None],out,0.0)
        output.index_add_(0,qpositions,out)
    return output
