import sys, json, math, os, types
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/LittleBit")
from transformers import AutoModelForCausalLM
from quantization.utils import apply_littlebit_patch
from lbmods import StructLinear, alloc_ranks, even_ranks, littlebit_bits

MID = "Qwen/Qwen1.5-0.5B"
R_PATH = 120          # their rank formula, 1024x1024 @0.55 BPW, residual: floor((1024^2*.55-32*2048)/(2*2064)/8)*8
CDIR = "/home/kfactor/out_qwen1.5-0.5b_v2/C"
D = "/home/kfactor/lbablate"

def lb_args():
    return types.SimpleNamespace(quant_func="SmoothSign", quant_mod="LittleBitLinear", residual=True, split_dim=1024,
                                 eff_bit=0.55, kv_factor=1.0, min_split_dim=8, use_itq=False, itq_n_iter=50, model_id=MID)

def load_fp(dtype=torch.float32):
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=dtype); m.config.use_cache = False; return m

def layer_ranks(l, kind, g=1):
    rk = alloc_ranks(np.load(f"{CDIR}/layer_{l}.npy"), R_PATH, g)
    return even_ranks(rk) if kind == "C" else rk

def build_student(arm, seed=0):
    """FP model -> LittleBit-patched student (published init for everything; S/C replace k_proj of layers 2-23)."""
    m = load_fp(torch.float32)
    kw = {l: (m.model.layers[l].self_attn.k_proj.weight.data.clone(), m.model.layers[l].self_attn.k_proj.bias.data.clone())
          for l in range(m.config.num_hidden_layers)}
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    for p in m.parameters(): p.requires_grad = False
    m = m.cuda()
    apply_littlebit_patch(m, lb_args(), do_train=True)
    if arm in ("S", "C"):
        for l in range(2, m.config.num_hidden_layers):
            W, b = kw[l]
            m.model.layers[l].self_attn.k_proj = StructLinear(W, b, layer_ranks(l, arm), arm).cuda()
    for n, p in m.named_parameters():
        p.requires_grad = not ("embed" in n or "lm_head" in n)
    return m.cuda()

def load_data():
    return np.load(f"{D}/train_seqs.npy"), np.load(f"{D}/test_windows.npy")

@torch.no_grad()
def eval_ppl(model, T, bs=4, dtype=torch.bfloat16):
    model.eval(); nll = 0.0; n = 0
    for i in range(0, len(T), bs):
        x = torch.from_numpy(T[i:i + bs]).long().cuda()
        with torch.autocast("cuda", dtype=dtype):
            lg = model(input_ids=x).logits.float()
        nll += F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="sum").item(); n += x[:, 1:].numel()
    return math.exp(nll / n)
