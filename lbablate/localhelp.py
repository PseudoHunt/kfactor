import torch, torch.nn as nn
from common import *
from quantization.modules import LittleBitLinear
from quantization.functions.binary import SmoothSign

def make(arm, l, W, b):
    if arm == "U":
        lin = nn.Linear(1024, 1024, bias=True); lin.weight.data = W.clone().cuda(); lin.bias.data = b.clone().cuda()
        lin.__class__ = LittleBitLinear
        lin.__quant_convert__(do_train=True, quant_func=SmoothSign, eff_bit=0.55, residual=True, split_dim=1024, min_split_dim=8)
        assert lin.split_dim == R_PATH
        m = lin
    elif arm in ("S", "C"):
        m = StructLinear(W, b, layer_ranks(l, arm), arm)
    else:                                   # "S2" / "S4" / "S8": pair-group structure, g adjacent pairs per block
        g = int(arm[1:]); m = StructLinear(W, b, layer_ranks(l, "S", g), "S", g=g)
    m = m.cuda()
    for n, p in m.named_parameters(): p.requires_grad = (n != "bias")
    return m

