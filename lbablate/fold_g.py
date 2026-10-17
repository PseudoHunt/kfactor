"""Absorbed (one representative frequency per block) vs reconstructed RoPE score for grouped structured arm S<g>, fp32, n=4096.
Usage: fold_g.py <g> <out_json>.  Keys = local-QAT S<g> k_proj on FP-model input; q = FP q_proj; layers 2,12,23, all 16 heads.
Rep frequency of a block = arithmetic mean of theta_i over its g pairs; bias handled exactly (constant, per-pair frequencies)."""
import sys, json, numpy as np, torch
from common import *
from lbmods import NH, DH, HALF
from localhelp import make
torch.backends.cuda.matmul.allow_tf32 = False
g = int(sys.argv[1]); outp = sys.argv[2]; N = 4096; arm = "S" if g == 1 else f"S{g}"
fp = load_fp(torch.float32).cuda().eval()
th = 1.0 / (1e6 ** (torch.arange(0, DH, 2, dtype=torch.float64) / DH))
pos = torch.arange(N, dtype=torch.float64); ang = pos[:, None] * th
cos64 = torch.cat([ang.cos()] * 2, -1).float().cuda(); sin64 = torch.cat([ang.sin()] * 2, -1).float().cuda()
dl = (torch.arange(N)[:, None] - torch.arange(N)[None, :]).clamp(min=0).cuda()
causal = torch.ones(N, N, dtype=torch.bool, device="cuda").tril()
rot = lambda t: torch.cat([-t[..., HALF:], t[..., :HALF]], -1)
def cs_mat(theta):                                    # cos / sin(theta * Delta) as (n,n) from exact fp64 angles
    a = torch.arange(N, dtype=torch.float64) * theta
    return a.cos().float().cuda()[dl], a.sin().float().cuda()[dl]
out = []
for l in (2, 12, 23):
    kp = fp.model.layers[l].self_attn.k_proj; at = fp.model.layers[l].self_attn
    k = make(arm, l, kp.weight.data.cpu(), kp.bias.data.cpu())
    sd = torch.load(f"{D}/runs/local/{arm}_L{l}.pt"); named = dict(k.named_parameters())
    with torch.no_grad():
        for n, v in sd.items(): named[n].copy_(v.cuda())
    k.eval()
    x = torch.from_numpy(np.load(f"{D}/runs/local/Xho_{l}.npy")[:N]).cuda()           # 2 held-out seqs back to back (positions 0..4095)
    with torch.no_grad():
        q = at.q_proj(x).view(N, NH, DH); kk = k(x).view(N, NH, DH)
        qr = q * cos64[:, None] + rot(q) * sin64[:, None]; kr = kk * cos64[:, None] + rot(kk) * sin64[:, None]
        rho = torch.complex(q[..., :HALF], q[..., HALF:]); bias = k.bias.view(NH, DH); bc = torch.complex(bias[:, :HALF], bias[:, HALF:])
        S_abs = torch.zeros(NH, N, N, device="cuda"); S_rec = torch.zeros(NH, N, N, device="cuda")
        for h in range(NH): S_rec[h] = (qr[:, h] @ kr[:, h].t())
        ng = HALF // g; thg = [th[m * g:(m + 1) * g].mean().item() for m in range(ng)]
        xin = x
        for sfx in ("", "_R"):
            Uq, V, u1, lat, v2 = k._parts(sfx); Vq = k.quant_func(V)
            c = (xin * v2) @ Vq.t()                                                    # cache entries (n, r)
            Cre = (u1[0][:, None] * Uq).view(NH, DH, -1)[:, :HALF]; Cim = (u1[0][:, None] * Uq).view(NH, DH, -1)[:, HALF:]   # (16,32,r)
            for m in range(ng):
                if k.ranks[m] == 0: continue
                sl = slice(k.offs[m], k.offs[m + 1]); cm, sm = cs_mat(thg[m])
                e = 0
                for i in range(m * g, (m + 1) * g):                                    # sum the group's pairs: the fold at one frequency
                    e = e + rho[:, :, i, None] * torch.complex(Cre[:, i, sl], -Cim[:, i, sl])[None] * lat[0, sl]
                uc, us = e.real, -e.imag                                               # (n,16,d)
                for h in range(NH):
                    cc = c[:, sl]
                    S_abs[h] += cm * (uc[:, h] @ cc.t()) + sm * (us[:, h] @ cc.t())
        for i in range(HALF):                                                          # bias: constant latent, exact per-pair frequency
            cm, sm = cs_mat(th[i].item())
            for h in range(NH):
                pb = rho[:, h, i] * bc[h, i].conj(); S_abs[h] += cm * pb.real[:, None] - sm * pb.imag[:, None]
        S_abs = S_abs.masked_fill(~causal, 0); S_rec = S_rec.masked_fill(~causal, 0)
        d = S_abs - S_rec
        rel_fro = (d.flatten(1).norm(dim=1) / S_rec.flatten(1).norm(dim=1)); rel_max = d.abs().amax((1, 2)) / S_rec.abs().amax((1, 2))
        # error on the score *differences that survive softmax* is what matters: row-centred error vs row-centred score
        dc = (d - (d.sum(-1, keepdim=True) / causal.sum(-1)[None, :, None])).masked_fill(~causal, 0)
        sc = (S_rec - (S_rec.sum(-1, keepdim=True) / causal.sum(-1)[None, :, None])).masked_fill(~causal, 0)
        rel_c = dc.flatten(1).norm(dim=1) / sc.flatten(1).norm(dim=1)
    r = dict(layer=l, g=g, rel_fro_mean=rel_fro.mean().item(), rel_fro_max_head=rel_fro.max().item(), rel_max_worst=rel_max.max().item(),
             rel_rowcentred_mean=rel_c.mean().item(), rel_rowcentred_max_head=rel_c.max().item())
    out.append(r); print(r, flush=True)
json.dump(out, open(outp, "w"), indent=1)
