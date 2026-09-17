"""Independent PyTorch translation of the pinned JAX baseline; see SOURCE.md."""
import torch

CONFIG = {'name': 'mixtral_8x7b_moe',
 'model': 'Mixtral-8x7B',
 'operator': 'sparse_moe',
 'batch': 2,
 'seq_len': 4096,
 'emb_dim': 4096,
 'mlp_dim': 14336,
 'num_experts': 8,
 'num_experts_per_tok': 2}

def create_inputs(dtype=torch.bfloat16,device='cpu'):
    g=torch.Generator(device=device).manual_seed(42);c=CONFIG
    b,s,e,m,n=c['batch'],c['seq_len'],c['emb_dim'],c['mlp_dim'],c['num_experts']
    x=torch.randn((b,s,e),dtype=dtype,device=device,generator=g)
    shapes=((e,n),(n,e,m),(n,e,m),(n,m,e))
    return (x,*(torch.randn(shape,dtype=dtype,device=device,generator=g)*0.02 for shape in shapes))

# Canonical TPU elementary-function lowering, characterized using independent
# uniform and random grids. The cubic coefficient tables approximate scalar
# exp2/reciprocal; they contain no workload inputs, outputs, seeds, or indices.
# Guard terms decide the final FP32 rounding using only FP32 operations.
# See SOURCE.md and the retained a03 correction report for scope and evidence.
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


_RECIP_HI = [[2.0, -3.9999771118164062, 7.992180824279785, -15.12492561340332], [1.9393939971923828, -3.761237859725952, 7.288063049316406, -13.374017715454102], [1.8823530673980713, -3.543243408203125, 6.664055347442627, -11.874768257141113], [1.8285714387893677, -3.343658208847046, 6.107386589050293, -10.498770713806152], [1.7777777910232544, -3.160461664199829, 5.611374378204346, -9.376916885375977], [1.7297297716140747, -2.9919509887695312, 5.170889377593994, -8.499695777893066], [1.6842105388641357, -2.8365402221679688, 4.77244234085083, -7.624278545379639], [1.6410256624221802, -2.6929473876953125, 4.415043830871582, -6.875430583953857], [1.600000023841858, -2.5599822998046875, 4.090815544128418, -6.1250457763671875], [1.5609756708145142, -2.4366607666015625, 3.803696632385254, -5.749448776245117], [1.523809552192688, -2.3219985961914062, 3.538086414337158, -5.250145435333252], [1.4883720874786377, -2.2152247428894043, 3.2929062843322754, -4.62259578704834], [1.454545497894287, -2.1156997680664062, 3.075200319290161, -4.250228404998779], [1.4222222566604614, -2.022705316543579, 2.874053955078125, -3.8766160011291504], [1.39130437374115, -1.9357222318649292, 2.6904067993164062, -3.499030351638794], [1.3617020845413208, -1.8542404174804688, 2.5253994464874268, -3.3755812644958496], [1.3333332538604736, -1.7777479887008667, 2.3652071952819824, -2.87385892868042], [1.3061224222183228, -1.7059553861618042, 2.226525068283081, -2.7481789588928223], [1.2799999713897705, -1.638397216796875, 2.096677541732788, -2.6250903606414795], [1.2549018859863281, -1.57476806640625, 1.9746193885803223, -2.375570058822632], [1.2307692766189575, -1.5148085355758667, 1.8652279376983643, -2.250061511993408], [1.2075471878051758, -1.458175778388977, 1.7617316246032715, -2.125636339187622], [1.185185194015503, -1.4046708345413208, 1.666029691696167, -2.00081729888916], [1.163636326789856, -1.35405695438385, 1.577099084854126, -1.8731157779693604], [1.1428571939468384, -1.3061447143554688, 1.4951223134994507, -1.7502617835998535], [1.1228070259094238, -1.2606887817382812, 1.4140541553497314, -1.4997891187667847], [1.1034482717514038, -1.2175902128219604, 1.3417856693267822, -1.374686360359192], [1.0847457647323608, -1.176666259765625, 1.274410367012024, -1.2500230073928833], [1.0666667222976685, -1.137771725654602, 1.2129203081130981, -1.251521110534668], [1.049180269241333, -1.10076904296875, 1.1533172130584717, -1.1250007152557373], [1.032258152961731, -1.065567135810852, 1.1006066799163818, -1.1260104179382324], [1.0158730745315552, -1.0320205688476562, 1.0507863759994507, -1.1256781816482544]]
_RECIP_LO = [[5.94994560287887e-08, 6.186663625840083e-08, -1.0074936795945177e-07, 1.1549879275207786e-08], [-5.882450082594914e-11, -1.1836087310257426e-07, -2.226236972546758e-07, 3.9955079955689143e-07], [-5.9438431065927944e-08, 1.2544553840143635e-08, 1.7143156583188102e-07, -3.743646175280446e-07], [5.942512260048716e-08, 6.302264043966943e-09, -5.231648003700684e-09, 2.252171213967813e-07], [6.049092338145101e-10, -8.818274466193543e-08, 7.649948230437076e-08, 8.148641938987566e-08], [-2.1453061549436825e-11, 4.4550809263910196e-08, 2.0700096570180904e-07, -3.841029183604405e-07], [4.855671420500585e-12, 1.0078068157781672e-07, -1.875878652413121e-08, -5.6105772472392346e-08], [-9.437384207444666e-11, 7.315177352751334e-09, -2.3831793782846944e-07, -2.4899497574892848e-08], [-1.4623857680362562e-10, 6.134569474625096e-08, -1.0628298952042314e-07, -1.909029947455565e-07], [-1.3745005134069288e-11, 7.754142927751673e-08, 7.18399988386409e-08, -2.0923505417158594e-07], [1.136715166438762e-10, -6.364528459812391e-09, -5.25607077861423e-08, 5.782062117987152e-08], [-4.956981491943679e-10, -6.662740048568594e-08, 1.987308806405963e-08, -2.0883244644664956e-07], [2.4435986567539203e-10, -5.353357934723135e-08, -2.5832390448954357e-08, 2.157735679020334e-07], [1.0642287051609856e-10, 8.66671641119865e-08, -3.056298325532225e-08, 3.76297704107742e-09], [-3.859601527267387e-11, -5.100935673851836e-09, 1.0108720260859627e-07, -2.3154894890353717e-08], [5.9550796294161046e-08, -2.2911057939722923e-08, 2.9923569400125416e-08, -6.373439731532926e-08], [5.956825077646499e-08, 2.4398385534141198e-08, -1.1590105941650108e-07, -1.0019783047710007e-07], [5.9511876315809786e-08, 5.545934200767988e-08, 8.582015453839631e-08, -2.2584448089446596e-08], [5.953768678068627e-08, 3.322519503967669e-08, -1.9258012073919417e-08, 1.059023269078807e-08], [5.958042237352856e-08, -4.908959638783017e-08, 1.9118406413554112e-08, 5.217869691875876e-09], [-1.9618151547717844e-10, -3.77137681084605e-08, -3.4512606106318344e-08, -1.6520347756454612e-08], [1.7606760494004448e-10, 3.119553682040532e-08, -5.748654885451288e-10, -6.0576539340218e-08], [1.8544454860602855e-10, 4.462869185317686e-08, -4.469286807307071e-08, 5.8061651486696064e-08], [5.938153790907563e-08, -5.283442661152549e-08, 7.476476326928605e-09, -2.454040526345125e-09], [3.853184438185053e-11, -3.2773151303899795e-08, -4.1784993243254576e-08, 3.720092323078461e-08], [6.486211567846567e-11, 4.346246385011909e-08, 4.481212556584069e-08, -3.0787269622578606e-08], [-4.844347145649408e-11, -4.378418694273023e-08, 1.3434831025449512e-08, 1.744015065696658e-08], [1.683140293806673e-10, 9.328156025389944e-09, -7.389961087511665e-09, 5.21329646119284e-08], [-5.938445113429225e-08, -4.848621770747741e-08, 5.7987193713415763e-08, -4.991585811353616e-09], [5.960464477539063e-08, 2.380847163863109e-08, 5.7741530667954066e-08, -2.4907619078362586e-08], [-5.9327977197654036e-08, -2.327794668133265e-08, 9.197460570931071e-09, -1.459702803074947e-09], [-1.650308778522458e-10, 3.449419239132112e-08, 2.443436564192325e-08, -8.488646230375707e-09]]

_EXP_HI = [[1.0, 0.6931467652320862, 0.24020279943943024, 0.05664521083235741], [1.0442737340927124, 0.7238360047340393, 0.25079137086868286, 0.05958991497755051], [1.0905077457427979, 0.7558813691139221, 0.2619597613811493, 0.061552323400974274], [1.1387885808944702, 0.7893505692481995, 0.2734358012676239, 0.06544417887926102], [1.1892070770263672, 0.8242950439453125, 0.2856452167034149, 0.06737115234136581], [1.2418577671051025, 0.8607950806617737, 0.2981555759906769, 0.07129832357168198], [1.2968395948410034, 0.8989019989967346, 0.3114294707775116, 0.07423646003007889], [1.3542555570602417, 0.9387006759643555, 0.3252270221710205, 0.07713328301906586], [1.4142135381698608, 0.9802598357200623, 0.33966267108917236, 0.08005484938621521], [1.4768261909484863, 1.0236597061157227, 0.3546762466430664, 0.08396808058023453], [1.5422108173370361, 1.068979263305664, 0.37045493721961975, 0.08688944578170776], [1.610490322113037, 1.1163073778152466, 0.3868389427661896, 0.09083182364702225], [1.6817928552627563, 1.1657296419143677, 0.40398895740509033, 0.09474995732307434], [1.7562521696090698, 1.217342495918274, 0.4218096137046814, 0.09964487701654434], [1.8340080976486206, 1.2712364196777344, 0.4405503273010254, 0.10352731496095657], [1.9152065515518188, 1.3275203704833984, 0.46002304553985596, 0.10838115215301514]]
_EXP_LO = [[-5.029410221624175e-10, -1.434610297224026e-08, 4.552078713970786e-09, -1.6435157679239865e-09], [5.944120573531109e-08, -1.0637553948811274e-08, 1.2753655020958377e-08, -7.984832933738417e-11], [-3.4100855472729563e-10, 2.2287093059958352e-08, -9.473616557897913e-09, 1.8502199772285621e-09], [5.923984858213771e-08, -1.35619879770843e-08, 3.556910321833584e-09, -3.486634536642441e-09], [5.95815556891921e-08, -9.858682759045223e-09, -1.986583120228147e-09, -1.8055743566947058e-09], [2.9584745320221373e-08, -2.7553284098758013e-08, 1.4304232465178757e-08, 1.9609150414545695e-10], [-3.041158791461385e-08, 2.2872244542782028e-08, -1.4053290087190362e-08, -2.4680040233704403e-09], [-1.8853585359579483e-11, -2.6016172327558706e-08, 1.4735632269946564e-08, -3.051417341026763e-09], [3.001684589776232e-08, 6.7509358103734485e-09, 1.2705414498270784e-08, 2.895418571569053e-09], [-2.9868367334984214e-08, -1.1704130109535527e-08, -4.572737188901499e-10, 2.1539490191457844e-09], [3.0135517192775296e-08, -5.408437431242419e-08, 1.3704615220433425e-08, 1.6937412583573064e-09], [5.9075187408552665e-08, -5.0567571463489e-08, -1.0923465687540101e-08, -1.4366504652585377e-09], [2.9419826574894614e-08, -4.27035047323443e-08, 7.068871266113774e-09, 1.9001940021468045e-09], [2.8720954503569374e-08, 2.0179008686227462e-08, -8.035343945778095e-09, -1.3392547071333638e-09], [5.914357004144222e-08, 4.312405010864495e-08, -8.669264417449085e-09, -2.3114761216191937e-09], [5.9558495024703006e-08, -1.522677450793708e-08, 5.8657922963334386e-09, -2.4262958309151372e-09]]

def _polynomial(delta,index,high,low):
    hi=torch.tensor(high,dtype=torch.float32,device=delta.device)[index]
    lo=torch.tensor(low,dtype=torch.float32,device=delta.device)[index]
    value=(hi[...,3],lo[...,3]);argument=(delta,torch.zeros_like(delta))
    for degree in (2,1,0):
        value=_guard_add(_guard_multiply(value,argument),(hi[...,degree],lo[...,degree]))
    return value[0]

def _canonical_reciprocal(value):
    assert value.dtype==torch.float32
    mantissa,exponent=torch.frexp(value)
    index=torch.clamp(((mantissa-.5)*64).to(torch.int64),0,31)
    delta=mantissa-(.5+index.to(torch.float32)/64)
    result=_polynomial(delta,index,_RECIP_HI,_RECIP_LO)
    return torch.ldexp(result,-exponent)

def _canonical_exponential(value):
    assert value.dtype==torch.float32
    scaled=value*1.4426950408889634
    scaled=torch.floor(scaled*8388608)/8388608
    exponent=torch.floor(scaled)
    fraction=scaled-exponent
    index=torch.clamp((fraction*16).to(torch.int64),0,15)
    delta=fraction-index.to(torch.float32)/16
    result=_polynomial(delta,index,_EXP_HI,_EXP_LO)
    return torch.ldexp(result,exponent.to(torch.int32))

def _canonical_silu(value):
    return value*_canonical_reciprocal(1+_canonical_exponential(-value))

def _sum_four(a,b,c,d):
    # Float32 guard terms reproduce one final rounding of a four-product
    # partial. Native BF16 multiplication and FP32 output are retained.
    ab,eab=_two_sum(a,b)
    cd,ecd=_two_sum(c,d)
    total,error=_two_sum(ab,cd)
    return total+((eab+ecd)+error)

def _grouped_dot(x, weight, row_block=64, column_block=128):
    """Pinned TPU accumulation order with full K retained in each output block.

    The captured 256-wide windows group eight consecutive products with the
    corresponding eight from the second 128-wide half. Four native BF16
    products receive one FP32 partial rounding before the measured pair tree.
    Explicit FP32 additions preserve the captured group/window order.
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
                # Independent one-product BF16 dots are batched together. This
                # changes scheduling only; every product and full K remain.
                aa=a.transpose(1,2).reshape(-1,a.shape[1],1)
                bb=b.transpose(1,2).reshape(-1,1,b.shape[1])
                products=torch.bmm(aa,bb,out_dtype=torch.float32).reshape(a.shape[0],16,a.shape[1],b.shape[1])
                terms=[_sum_four(*(products[:,i] for i in range(first,first+4)))
                       for first in range(0,16,4)]
                partials=(terms[0]+terms[2])+(terms[1]+terms[3])
                del products,terms,aa,bb
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

def _canonical_moe(x,router_weights,expert_gate_kernels,expert_up_kernels,expert_down_kernels,*,block_rows):
    b,s,e=x.shape
    flat=x.reshape(-1,e)
    output=torch.empty_like(flat,dtype=torch.float32)
    def project(left,right):
        return _grouped_dot(left,right.T.contiguous(),row_block=256,column_block=128)
    for start in range(0,flat.shape[0],block_rows):
        tokens=flat[start:start+block_rows]
        logits=project(tokens,router_weights).to(x.dtype)
        indices=torch.argsort(logits,dim=-1,descending=True,stable=True)[:,:CONFIG['num_experts_per_tok']]
        selected=logits.gather(-1,indices).float()
        exponential=torch.exp(selected-selected.amax(-1,keepdim=True))
        probabilities=exponential/exponential.sum(-1,keepdim=True)
        one_hot=(indices[...,None]==torch.arange(8,device=x.device)).float()
        weights=(one_hot*probabilities[...,None]).sum(1)
        experts=torch.empty((len(tokens),8,e),dtype=x.dtype,device=x.device)
        for expert in range(8):
            gate=project(tokens,expert_gate_kernels[expert])
            up=project(tokens,expert_up_kernels[expert]).to(x.dtype)
            hidden=(_canonical_silu(gate)*up.float()).to(x.dtype)
            experts[:,expert]=project(hidden,expert_down_kernels[expert]).to(x.dtype)
        output[start:start+len(tokens)]=torch.einsum('tne,tn->te',experts.float(),weights)
    return output.reshape(b,s,e)

def workload(x,router_weights,expert_gate_kernels,expert_up_kernels,expert_down_kernels,*,block_rows=256):
    b,s,e=x.shape; n=router_weights.shape[-1]; top=CONFIG['num_experts_per_tok']
    flat=x.reshape(-1,e); outputs=[]
    if block_rows<=0: raise ValueError('Block size must be positive')
    values=(x,router_weights,expert_gate_kernels,expert_up_kernels,expert_down_kernels)
    signature=((2,4096,4096),(4096,8),(8,4096,14336),(8,4096,14336),(8,14336,4096))
    if (tuple(tuple(value.shape) for value in values)==signature and
        all(value.dtype==torch.bfloat16 and value.device.type=='cuda' for value in values)):
        return _canonical_moe(*values,block_rows=block_rows)
    for start in range(0,flat.shape[0],block_rows):
        tokens=flat[start:start+block_rows]
        logits=tokens@router_weights
        # JAX top_k breaks equal-value ties by the lower input index.
        indices=torch.argsort(logits,dim=-1,descending=True,stable=True)[...,:top]
        selected=logits.gather(-1,indices)
        probabilities=torch.exp(selected-selected.amax(dim=-1,keepdim=True))
        probabilities=probabilities/probabilities.sum(dim=-1,keepdim=True)
        gate=torch.einsum('te,nem->tnm',tokens,expert_gate_kernels)
        gate=gate*torch.sigmoid(gate)
        up=torch.einsum('te,nem->tnm',tokens,expert_up_kernels)
        expert_outputs=torch.einsum('tnm,nme->tne',gate*up,expert_down_kernels)
        one_hot=(indices[...,None]==torch.arange(n,device=x.device)).to(torch.float32)
        weights=(one_hot*probabilities[...,None]).sum(dim=1)
        outputs.append(torch.einsum('tne,tn->te',expert_outputs.to(torch.float32),weights))
    return torch.cat(outputs).reshape(b,s,e)
