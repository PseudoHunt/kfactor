"""Absorbed (dual-channel fold) score vs reconstructed RoPE score, fp32, layers (2,12,23), all 16 heads, n=512.
Usage: verify_fold.py <S|C> <out_txt> [ckpt.pt]"""
import sys, json, torch
from common import *
from lbmods import NH, DH, HALF
arm, outp = sys.argv[1], sys.argv[2]
ck = sys.argv[3] if len(sys.argv) > 3 else None
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
S, T = load_data()
m = build_student(arm)
if ck:
    sd = torch.load(ck); named = dict(m.named_parameters())
    with torch.no_grad():
        for n, v in sd.items(): named[n].copy_(v.cuda())
m.eval(); N = 512; LAYERS = (2, 12, 23)
x_ids = torch.from_numpy(T[0:1, :N]).long().cuda()
cap = {}
hs = []
for l in LAYERS:
    hs.append(m.model.layers[l].self_attn.k_proj.register_forward_hook(lambda mod, i, o, l=l: cap.__setitem__(l, i[0][0].detach().double().float())))
with torch.no_grad(): m(input_ids=x_ids)
for h in hs: h.remove()
th = 1.0 / (1e6 ** (torch.arange(0, DH, 2, dtype=torch.float64) / DH))             # (32,) exact fp64 angles
pos = torch.arange(N, dtype=torch.float64)
ang = pos[:, None] * th[None]
cos64 = torch.cat([ang.cos()] * 2, -1).float().cuda(); sin64 = torch.cat([ang.sin()] * 2, -1).float().cuda()
dang = (pos[:, None] - pos[None, :])[..., None] * th                                  # (n,n,32) Delta*theta, fp64
cosD, sinD = dang.cos().float().cuda(), dang.sin().float().cuda()                     # cosD[m,n,i]
causal = torch.ones(N, N, dtype=torch.bool, device="cuda").tril()
rot = lambda t: torch.cat([-t[..., DH // 2:], t[..., :DH // 2]], -1)
fp32 = lambda t: t.float()

def path_terms(k, sfx):
    """per-path (latent c (n,r), coefficients Cre/Cim (NH,32,r) incl. h, lat scale folded separately (r,))"""
    if arm == "S":
        Uq, V, u1, lat, v2 = k._parts(sfx)
        Vq = k.quant_func(V)
        return Uq.detach(), Vq.detach(), u1.detach()[0], lat.detach()[0], v2.detach()[0]
    return None

report = []
for l in LAYERS:
    x = cap[l]; at = m.model.layers[l].self_attn; k = at.k_proj
    with torch.no_grad():
        q = at.q_proj(x); kk = k(x)
        qh, kh = q.view(N, NH, DH), kk.view(N, NH, DH)
        qr = qh * cos64[:, None] + rot(qh) * sin64[:, None]; kr = kh * cos64[:, None] + rot(kh) * sin64[:, None]
        S_rec = torch.einsum("mhd,nhd->hmn", qr, kr)                                    # reconstructed RoPE scores
        # --- absorbed ---------------------------------------------------------------------------------------------
        rho = torch.complex(qh[..., :HALF], qh[..., HALF:])                              # (n,NH,32) pre-RoPE complex query
        bias = k.bias.view(NH, DH); bc = torch.complex(bias[:, :HALF], bias[:, HALF:])   # (NH,32)
        S_abs = torch.zeros(NH, N, N, device="cuda")
        for sfx in ("", "_R"):
            Uq, Vq, u1, lat, v2 = path_terms(k, sfx) if arm == "S" else (None,) * 5
            if arm == "S":
                c = (x * v2) @ Vq.t()                                                    # cache entry (n, r), no scale applied
                Cre = (u1[:, None] * Uq).view(NH, DH, -1)[:, :HALF]                      # (NH,32,r): rows h*64+i
                Cim = (u1[:, None] * Uq).view(NH, DH, -1)[:, HALF:]
                e = rho[:, :, :, None] * torch.complex(Cre, -Cim)[None] * lat[None, None, None]  # (n,NH,32,r) = rho * conj(coef) * l
                uc, us = e.real, -e.imag                                                 # folded dual-channel queries
                for h in range(NH):
                    for i in range(HALF):
                        ri = k.ranks[i]
                        if ri == 0: continue
                        sl = slice(k.offs[i], k.offs[i + 1])
                        S_abs[h] += cosD[..., i] * (uc[:, h, i, sl] @ c[:, sl].t()) + sinD[..., i] * (us[:, h, i, sl] @ c[:, sl].t())
            else:
                # arm C: complex latent rotation, built from the raw (a,b) signs, not from the Uq matrix
                sa, sb = k.quant_func(getattr(k, "A" + sfx)), k.quant_func(getattr(k, "B" + sfx))      # (NH,K)
                Vq = k.quant_func(getattr(k, "V" + sfx)); v2_ = getattr(k, "v2" + sfx)[0]
                c = (x * v2_) @ Vq.t()                                                    # (n, r)
                zeta = torch.complex(c[:, 0::2], c[:, 1::2])                              # complex latents (n,K) -- block order = col0 order
                hh = getattr(k, "hp" + sfx)                                               # (NH*32,)
                lk = (getattr(k, "v1k" + sfx) * getattr(k, "l" + sfx))                    # (K,)
                ko = 0
                for i in range(HALF):
                    kdim = k.ranks[i] // 2
                    if kdim == 0: continue
                    sl = slice(ko, ko + kdim); ko += kdim
                    hi = hh[torch.arange(NH, device="cuda") * HALF + i]                   # (NH,)
                    E = rho[:, :, i, None] * (hi[None, :, None] * lk[None, None, sl]) * torch.complex(sa[:, sl], -sb[:, sl])[None]   # (n,NH,kd)
                    for h in range(NH):
                        P = E[:, h] @ zeta[:, sl].conj().t()                              # (n,n)
                        S_abs[h] += cosD[..., i] * P.real - sinD[..., i] * P.imag
        # constant (bias) latent: RoPE'd bias is position dependent but needs no cache
        for h in range(NH):
            for i in range(HALF):
                Pb = (rho[:, h, i] * bc[h, i].conj())[:, None]
                S_abs[h] += cosD[..., i] * Pb.real - sinD[..., i] * Pb.imag
        S_abs = S_abs.masked_fill(~causal, 0); S_rec = S_rec.masked_fill(~causal, 0)
        rel_max = ((S_abs - S_rec).abs().amax((1, 2)) / S_rec.abs().amax((1, 2)))        # per head
        rel_fro = ((S_abs - S_rec).flatten(1).norm(dim=1) / S_rec.flatten(1).norm(dim=1))
        # reference: native HF rotary angles (fp32 cos/sin) vs exact angles, same reconstruction
        report.append(dict(layer=l, rel_max_worst=rel_max.max().item(), rel_fro_worst=rel_fro.max().item(), rel_fro_mean=rel_fro.mean().item(),
                           score_scale=S_rec.abs().amax().item()))
        print(report[-1], flush=True)
ok = all(r["rel_fro_worst"] < 1e-5 and r["rel_max_worst"] < 1e-5 for r in report)
print(f"arm {arm}: fold check {'PASS' if ok else 'FAIL'} (threshold 1e-5 relative, fp32)")
open(outp, "w").write(json.dumps(dict(arm=arm, ckpt=ck, pass_=ok, layers=report), indent=1))
