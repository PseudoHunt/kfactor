"""Local attention-output error: FP q and v, keys from (a) FP k_proj, (b) U local-QAT, (c) S local-QAT (+C for reference),
standard RoPE causal attention, 16 held-out seqs x 2048 (train_seqs[8:24]); MSE of attention output (pre-o_proj) vs (a)."""
import sys, csv, numpy as np, torch, torch.nn.functional as F
from common import *
from localhelp import make
torch.backends.cuda.matmul.allow_tf32 = False
NH, DH = 16, 64
ARMS = sys.argv[1].split(","); OUTCSV = sys.argv[2]
S, T = load_data(); fp = load_fp(torch.float32).cuda().eval()
cap = {l: [] for l in range(24)}
hs = [fp.model.layers[l].self_attn.k_proj.register_forward_hook(lambda m, i, o, l=l: cap[l].append(i[0].detach())) for l in range(24)]
with torch.no_grad():
    for i in range(8, 24): fp(input_ids=torch.from_numpy(S[i:i + 1]).long().cuda())
for h in hs: h.remove()
th = 1.0 / (1e6 ** (torch.arange(0, DH, 2, dtype=torch.float64) / DH)); ang = torch.arange(2048, dtype=torch.float64)[:, None] * th
cos = torch.cat([ang.cos()] * 2, -1).float().cuda(); sin = torch.cat([ang.sin()] * 2, -1).float().cuda()
rot = lambda t: torch.cat([-t[..., DH // 2:], t[..., :DH // 2]], -1)
def rope(t): return t * cos[:, None] + rot(t) * sin[:, None]
w = csv.writer(open(OUTCSV, "w")); w.writerow(["layer", "arm", "attn_mse", "attn_rel", "k_rel_mse"])
rows = {}
for l in range(2, 24):
    at = fp.model.layers[l].self_attn; k = at.k_proj
    mods = {"FP": None}
    for arm in ARMS:
        m = make(arm, l, k.weight.data.cpu(), k.bias.data.cpu())
        sd = torch.load(f"{D}/runs/local/{arm}_L{l}.pt"); named = dict(m.named_parameters())
        with torch.no_grad():
            for n, v in sd.items(): named[n].copy_(v.cuda())
        mods[arm] = m.eval()
    acc = {a: [0.0, 0.0, 0.0] for a in ARMS}; n_el = 0
    with torch.no_grad():
        for x in cap[l]:
            x = x[0] if x.dim() == 3 else x
            q = rope(at.q_proj(x).view(2048, NH, DH)); v = at.v_proj(x).view(2048, NH, DH)
            kf = k(x); out_f = F.scaled_dot_product_attention(q.transpose(0, 1)[None], rope(kf.view(2048, NH, DH)).transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True)
            for a in ARMS:
                ks = mods[a](x).float()
                out_s = F.scaled_dot_product_attention(q.transpose(0, 1)[None], rope(ks.view(2048, NH, DH)).transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True)
                acc[a][0] += ((out_s - out_f) ** 2).sum().item(); acc[a][1] += (out_f ** 2).sum().item(); acc[a][2] += ((ks - kf) ** 2).sum().item() / (kf ** 2).sum().item()
            n_el += out_f.numel()
    for a in ARMS:
        w.writerow([l, a, acc[a][0] / n_el, acc[a][0] / acc[a][1], acc[a][2] / len(cap[l])])
    print(l, {a: f"{acc[a][0] / acc[a][1]:.4f}" for a in ARMS}, flush=True)
