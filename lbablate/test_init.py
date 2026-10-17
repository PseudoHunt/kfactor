import torch, numpy as np, sys
sys.path.insert(0,'/home/LittleBit')
from transformers import AutoModelForCausalLM
from lbmods import *
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B", dtype=torch.float32)
r=120
for l in (2,12,23):
    kp=m.model.layers[l].self_attn.k_proj; W=kp.weight.data
    C=np.load(f'/home/kfactor/out_qwen1.5-0.5b_v2/C/layer_{l}.npy')
    rk=alloc_ranks(C,r); ek=even_ranks(rk)
    S=StructLinear(W,kp.bias.data,rk,'S'); Cc=StructLinear(W,kp.bias.data,ek,'C')
    Wn=W.norm()
    # plain truncated SVD (rank 120) reference
    U,s,Vh=torch.linalg.svd(W.double()); Wsvd=(U[:,:r]*s[:r])@Vh[:r]
    print(l,'ranks',rk.tolist(),'odd pairs',int((rk%2).sum()),'| rel err: svd120',f'{((W-Wsvd).norm()/Wn).item():.3f}',
          'S',f'{((S.reconstruct().cpu()-W).norm()/Wn).item():.3f}','C',f'{((Cc.reconstruct().cpu()-W).norm()/Wn).item():.3f}')
    # forward consistency with reconstruct
    x=torch.randn(5,1024)
    print('   fwd-vs-recon S',(S(x)-(x@S.reconstruct().T+S.bias)).abs().max().item(),' C',(Cc(x)-(x@Cc.reconstruct().T+Cc.bias)).abs().max().item())
