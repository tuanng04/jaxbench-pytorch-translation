"""Independent PyTorch translation of the pinned JAX baseline; see SOURCE.md."""
import torch

CONFIG = {'name': 'mamba2_2_7b_ssd',
 'model': 'Mamba-2-2.7B',
 'operator': 'state_space_duality',
 'batch': 4,
 'seq_len': 4096,
 'num_heads': 64,
 'head_dim': 64,
 'd_state': 128,
 'd_model': 2560}

def create_inputs(dtype=torch.bfloat16,device='cpu'):
    g=torch.Generator(device=device).manual_seed(42);c=CONFIG
    shape=(c['batch'],c['num_heads'],c['seq_len'],c['head_dim'])
    values=tuple(torch.randn(shape,dtype=dtype,device=device,generator=g) for _ in range(3))
    return (*values,torch.randn(shape[:-1],dtype=torch.float32,device=device,generator=g)*0.5-4.0)


def _two_sum(a,b):
    total=a+b
    recovered=total-a
    error=(a-(total-recovered))+(b-recovered)
    return total,error

def _guard_add(a,b):
    hi,lo=_two_sum(a[0],b[0])
    lo=lo+(a[1]+b[1])
    return _two_sum(hi,lo)

def _product_residual(a,b,product):
    # Split binary32 significands into nonoverlapping 12-bit pieces. Every
    # partial product is exact in binary32; recover the rounded product's tail.
    ah=(a.contiguous().view(torch.int32)&-4096).view(torch.float32);al=a-ah
    bh=(b.contiguous().view(torch.int32)&-4096).view(torch.float32);bl=b-bh
    return (((ah*bh-product)+ah*bl)+al*bh)+al*bl

def _guard_multiply(a,b):
    product=a[0]*b[0]
    error=_product_residual(a[0],b[0],product)
    error=error+(a[0]*b[1]+a[1]*b[0])
    error=error+a[1]*b[1]
    return _two_sum(product,error)

def _guard_divide_integer(a,n):
    denominator=torch.full_like(a[0],n)
    quotient=a[0]/denominator
    product=quotient*denominator
    residual=((a[0]-product)-_product_residual(quotient,denominator,product))+a[1]
    return _two_sum(quotient,residual/denominator)

def _log2_toward_zero(value):
    """Finite positive float32 log2 with an explicit final toward-zero rounding."""
    assert value.dtype==torch.float32
    m,e=torch.frexp(value)
    shift=m<.707106781186547524
    m=torch.where(shift,m*2,m)
    e=e.to(torch.float32)-shift.to(torch.float32)
    one=torch.ones_like(m)
    denominator,den_error=_two_sum(m,one)
    numerator=m-one
    quotient=numerator/denominator
    product=quotient*denominator
    residual=((numerator-product)-_product_residual(quotient,denominator,product))-quotient*den_error
    t=_two_sum(quotient,residual/denominator)
    square=_guard_multiply(t,t)
    term=t
    total=t
    # |t| <= (sqrt(2)-1)/(sqrt(2)+1) < .172. The omitted terms after
    # degree 17 are below 4e-16; guard arithmetic decides F32 rounding.
    for n in range(3,18,2):
        term=_guard_multiply(term,square)
        total=_guard_add(total,_guard_divide_integer(term,n))
    natural=(total[0]*2,total[1]*2)
    reciprocal_ln2=(torch.full_like(m,1.4426950216293335),
                    torch.full_like(m,1.9259629911266175e-08))
    fraction=_guard_multiply(natural,reciprocal_ln2)
    hi,guard=_guard_add((e,torch.zeros_like(e)),fraction)
    rounded_away=((hi>0)&(guard<0))|((hi<0)&(guard>0))
    return torch.where(rounded_away,torch.nextafter(hi,torch.zeros_like(hi)),hi)

def _canonical_log(value):
    return _log2_toward_zero(value)*.6931471805599453


def _canonical_prefix(values):
    """Captured TPU prefix: sequential sums within and across 128-wide blocks."""
    shape=values.shape
    groups=values.reshape(*shape[:-1],shape[-1]//128,128)
    prefix=torch.empty_like(groups)
    value=groups[...,0]
    prefix[...,0]=value
    for position in range(1,128):
        value=value+groups[...,position]
        prefix[...,position]=value
    totals=prefix[...,-1]
    offsets=torch.zeros_like(totals)
    running=totals[...,0]
    for index in range(1,totals.shape[-1]):
        offsets[...,index]=running
        running=running+totals[...,index]
    return (prefix+offsets[...,None]).reshape(shape)

def _compute(query,key,value,A_log=None,*,block_rows=256,head_block=8):
    if block_rows<=0 or head_block<=0:raise ValueError('Block size must be positive')
    b,h,s,d=query.shape;device=query.device
    positions=torch.arange(s,dtype=torch.float32,device=device)
    output=torch.empty_like(query)
    if A_log is None:
        gammas=1.0-torch.exp2(-5.0-torch.arange(h,dtype=torch.float32,device=device))
        log_gamma=torch.log(gammas)
    else:
        canonical=(query.shape==key.shape==value.shape==(4,64,4096,64) and
                   A_log.shape==(4,64,4096) and A_log.dtype==torch.float32 and
                   all(t.dtype==torch.bfloat16 and t.device.type=='cuda' for t in (query,key,value)) and
                   A_log.device==query.device)
        if canonical:
            # The independent TPU pointwise grids support base-two exp and
            # toward-zero log2 followed by the float32 ln(2) product. Guard
            # terms implement that elementary-function rounding in float32.
            a=1/(1+torch.exp2(-A_log*1.4426950408889634))
            log_values=_canonical_log(a+1e-8)
        else:
            a=torch.sigmoid(A_log.to(torch.float32))
            log_values=torch.log(a+1e-8)
        cumulative=_canonical_prefix(log_values) if canonical else torch.cumsum(log_values,dim=-1)
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
def workload(query,key,value,A_log,*,block_rows=256,head_block=8):
    return _compute(query,key,value,A_log,block_rows=block_rows,head_block=head_block)
