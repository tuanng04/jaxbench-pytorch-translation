"""Independent PyTorch translation of the pinned JAX baseline; see SOURCE.md."""
import torch

CONFIG = {'name': 'llama3_8b_cross_entropy',
 'model': 'Llama-3.1-8B',
 'operator': 'fused_cross_entropy',
 'batch_tokens': 8192,
 'hidden_dim': 4096,
 'vocab_size': 128256}

def create_inputs(dtype=torch.bfloat16, device='cpu'):
    g=torch.Generator(device=device).manual_seed(42)
    c=CONFIG
    hidden=torch.randn((c['batch_tokens'],c['hidden_dim']),dtype=dtype,device=device,generator=g)
    weight=torch.randn((c['hidden_dim'],c['vocab_size']),dtype=dtype,device=device,generator=g)*0.02
    labels=torch.randint(c['vocab_size'],(c['batch_tokens'],),dtype=torch.int32,device=device,generator=g)
    return hidden,weight,labels

def workload(hidden, weight, labels, *, block_rows=256):
    if block_rows <= 0: raise ValueError('block_rows must be positive')
    losses=[]
    classes=torch.arange(weight.shape[1],device=hidden.device)
    for start in range(0,hidden.shape[0],block_rows):
        logits=torch.matmul(hidden[start:start+block_rows],weight)
        shifted=logits-logits.amax(dim=-1,keepdim=True)
        log_probs=shifted-torch.log(torch.exp(shifted).sum(dim=-1,keepdim=True))
        # JAX one_hot defaults to float32, promoting the loss before reduction.
        one_hot=(labels[start:start+block_rows,None]==classes[None,:]).to(torch.float32)
        losses.append(-(one_hot*log_probs.to(torch.float32)).sum(dim=-1))
    return torch.cat(losses).mean()
