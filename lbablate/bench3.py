"""Decode microbenchmark, layer 12 of arm S (weights from S_20M), fp16, batch 1, A30.
(a) reconstruct: cached latent c' (n x 240) -> dense fp16 stand-in for U_sign (scaled by h) -> +bias -> RoPE on keys -> q.k scores
(b) absorbed: fold q once per step into dual-channel queries per latent dim, then per token cos(th_g D)<u^c_g,c^(g)>+sin(th_g D)<u^s_g,c^(g)>
Cached latent already carries the latent scale (v1*u2) -- a per-latent constant folded at cache-write time (same cache width 240 in both)."""
import sys, json, numpy as np, torch
from common import *
from lbmods import NH, DH, HALF
torch.backends.cuda.matmul.allow_tf32 = False
L = 12; NMAX = 32768
fp = load_fp(torch.float32)
kp = fp.model.layers[L].self_attn.k_proj
k = StructLinear(kp.weight.data, kp.bias.data, layer_ranks(L, "S"), "S")
sd = torch.load(f"{D}/runs/S_20M/student.pt"); pre = f"model.layers.{L}.self_attn.k_proj."
with torch.no_grad():
    for n, p in k.named_parameters(): p.copy_(sd[pre + n])
k = k.cuda().eval(); r = k.r
# --- weights in the cache/compute layout ------------------------------------------------------------------------------------
with torch.no_grad():
    parts = [k._parts(s) for s in ("", "_R")]
    Uqs = torch.cat([u1.t() * Uq for Uq, V, u1, lat, v2 in parts], 1)                  # (1024, 240) = h (.) U_sign, both paths
    lat = torch.cat([p[3][0] for p in parts]); Vq = [k.quant_func(p[1]) for p in parts]; v2 = [p[4] for p in parts]
    blk = torch.cat([torch.repeat_interleave(torch.arange(HALF), torch.tensor(k.ranks))] * 2).cuda()   # pair index of each of the 240 dims
    Xh = torch.from_numpy(np.load(f"{D}/runs/local/Xho_{L}.npy")).cuda()
    xs = Xh[torch.arange(NMAX) % len(Xh)]
    cfull = torch.cat([(xs * v2[p]) @ Vq[p].t() for p in range(2)], 1) * lat           # realistic latent magnitudes (n, 240)
r = Uqs.shape[1]   # 240 = r_pri + r_res
bias = k.bias.detach()
th = 1.0 / (1e6 ** (torch.arange(0, DH, 2, dtype=torch.float64) / DH))
posn = torch.arange(NMAX + 1, dtype=torch.float64)
cosP = torch.cat([(posn[:, None] * th).cos()] * 2, -1); sinP = torch.cat([(posn[:, None] * th).sin()] * 2, -1)   # position tables (n, 64)
Dl = (NMAX - posn)[:, None] * th; cosR, sinR = Dl.cos(), Dl.sin()                                                 # Delta-tables, reversed index (n, 32)
import os
H = torch.float32 if os.environ.get("BENCH_FP32") else torch.float16
Uqs16, bias16, lat16 = Uqs.to(H), bias.to(H), lat.to(H)
cosP, sinP, cosR, sinR = [t.to(H).cuda() for t in (cosP, sinP, cosR, sinR)]
Cre = Uqs16.view(NH, DH, r)[:, :HALF]; Cim = Uqs16.view(NH, DH, r)[:, HALF:]            # (16,32,240)
blkmask = torch.zeros(HALF, r, device="cuda", dtype=H); blkmask[blk, torch.arange(r)] = 1   # (32,240) one-hot of the dim's pair
bre, bim = bias16.view(NH, DH)[:, :HALF], bias16.view(NH, DH)[:, HALF:]
rot = lambda t: torch.cat([-t[..., HALF:], t[..., :HALF]], -1)

def step_a(c, q, n):
    kk = c @ Uqs16.t() + bias16                                                         # (n,1024)
    kk = kk.view(n, NH, DH)
    kr = kk * cosP[:n, None] + rot(kk) * sinP[:n, None]
    qr = q * cosP[n] + rot(q) * sinP[n]                                                # query at position n, keys at 0..n-1
    return torch.einsum("hd,nhd->hn", qr, kr)

def step_b(c, q, n):
    a, b = q[:, :HALF], q[:, HALF:]                                                    # rho = a + i b (16,32)
    uc = (a[..., None] * Cre + b[..., None] * Cim)                                     # Re[rho conj(coef)]  (16,32,240)
    us = (a[..., None] * Cim - b[..., None] * Cre)                                     # -Im[rho conj(coef)]
    ucd = (uc * blkmask).sum(1)                                                        # (16,240): each dim has one pair
    usd = (us * blkmask).sum(1)
    # block-diagonal fold matrices (240, 16*32): column (h,i) holds u[h, j in block i]
    UC = (ucd.t()[:, :, None] * blkmask.t()[:, None, :]).reshape(r, NH * HALF)
    US = (usd.t()[:, :, None] * blkmask.t()[:, None, :]).reshape(r, NH * HALF)
    A = (c @ UC).view(n, NH, HALF); B = (c @ US).view(n, NH, HALF)
    cT, sT = cosR[NMAX - n:NMAX, None], sinR[NMAX - n:NMAX, None]                        # cos/sin(theta (n - t)), t = 0..n-1
    s = (A * cT + B * sT).sum(-1)                                                      # (n,16)
    # bias: constant latent, c = 1
    pb_c = a * bre + b * bim; pb_s = a * bim - b * bre                                  # Re/-Im of rho conj(b)
    sb = cT[:, 0] @ pb_c.t() + sT[:, 0] @ pb_s.t()                                      # (n,16)
    return (s + sb).t()

# NOTE: step_b's pair-dim contraction ((16,32,240) work) is part of the per-step fold cost and is timed.
def verify(n):
    c = cfull[:n].to(H); q = torch.randn(NH, DH, device="cuda").to(H)
    # fp32 reference of (a) with exact fp64 angles
    c32, q32 = cfull[:n].double(), q.double()
    Kd = c32 @ Uqs.double().t() + bias.double(); Kd = Kd.view(n, NH, DH)
    ang = torch.arange(n + 1, dtype=torch.float64, device="cuda")[:, None] * th.cuda()
    cs, sn = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    ref = torch.einsum("hd,nhd->hn", q32 * cs[n] + rot(q32) * sn[n], Kd * cs[:n, None] + rot(Kd) * sn[:n, None])
    sa, sb_ = step_a(c, q, n).double(), step_b(c, q, n).double()
    sc = ref.abs().max()
    return dict(n=n, score_scale=sc.item(), a_vs_ref=((sa - ref).abs().max() / sc).item(), b_vs_ref=((sb_ - ref).abs().max() / sc).item(), a_vs_b=((sa - sb_).abs().max() / sc).item())

def timeit(fn, c, n, steps=100, graph=False):
    qs = torch.randn(steps + 10, NH, DH, device="cuda").to(H)
    for i in range(5): fn(c, qs[i], n)
    torch.cuda.synchronize()
    if graph:
        qs_static = qs[0].clone(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out = fn(c, qs_static, n)
        for i in range(5): qs_static.copy_(qs[i]); g.replay()
        torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for i in range(steps): qs_static.copy_(qs[5 + i]); g.replay()
        e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / steps
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for i in range(steps): fn(c, qs[5 + i], n)
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / steps

res = []
if os.environ.get("BENCH_FP32"):
    for n in (4096, 16384, 32768): print("fp32 verify", json.dumps(verify(n)))
    sys.exit(0)
for n in (4096, 16384, 32768):
    v = verify(n); c = cfull[:n].to(H).contiguous()
    row = dict(v)
    for nm, fn in (("a", step_a), ("b", step_b)):
        row[f"{nm}_ms_eager"] = timeit(fn, c, n); row[f"{nm}_ms_graph"] = timeit(fn, c, n, graph=True)
    row["ratio_a_over_b_eager"] = row["a_ms_eager"] / row["b_ms_eager"]; row["ratio_a_over_b_graph"] = row["a_ms_graph"] / row["b_ms_graph"]
    row["cache_bytes"] = n * r * 2; row["table_bytes_a"] = n * DH * 2 * 2; row["table_bytes_b"] = n * HALF * 2 * 2
    res.append(row); print(json.dumps(row), flush=True)
print("weights read per step: dense stand-in U (1024x240 fp16) =", Uqs16.numel() * 2, "B; packed 1-bit would be", Uqs16.numel() // 8, "B (nonzero block entries only:", int(k.mask.sum() * 2 // 8), "B)")
json.dump(res, open("results/bench3.json", "w"), indent=1)
