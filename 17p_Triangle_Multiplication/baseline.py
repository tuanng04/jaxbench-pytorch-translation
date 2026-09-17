"""Independent PyTorch translation of the pinned JAX baseline; see SOURCE.md."""
import torch

CONFIG = {'name': 'alphafold_768_triangle_mult',
 'model': 'AlphaFold2',
 'operator': 'triangle_mult_outgoing',
 'N': 1536,
 'C': 128,
 'direction': 'outgoing'}

def create_inputs(dtype=torch.bfloat16, device='cpu'):
    g=torch.Generator(device=device).manual_seed(42)
    n,c=CONFIG['N'],CONFIG['C']
    pair=torch.randn((n,n,c),dtype=dtype,device=device,generator=g)
    mask=torch.ones((n,n,1),dtype=dtype,device=device)
    weights=[torch.randn((c,c),dtype=dtype,device=device,generator=g)*0.02 for _ in range(4)]
    center=torch.randn((c,),dtype=dtype,device=device,generator=g)*0.1+1.0
    output=[torch.randn((c,c),dtype=dtype,device=device,generator=g)*0.02 for _ in range(2)]
    return (pair,mask,*weights,center,*output)

def workload(pair_act, mask, left_proj_w, right_proj_w, left_gate_w, right_gate_w, center_scale, out_proj_w, out_gate_w):
    act=pair_act*mask
    left=torch.matmul(act,left_proj_w)*torch.sigmoid(torch.matmul(act,left_gate_w))
    right=torch.matmul(act,right_proj_w)*torch.sigmoid(torch.matmul(act,right_gate_w))
    result=torch.einsum('ikc,jkc->ijc',left,right)
    rms=torch.sqrt((result*result).mean(dim=-1,keepdim=True)+1e-6)
    result=result/rms*center_scale
    output=torch.matmul(result,out_proj_w)
    # The source deliberately uses the unmasked pair for this final gate.
    gate=torch.sigmoid(torch.matmul(pair_act,out_gate_w))
    return output*gate
