"""Range-split test: long-range keys (Delta >= W, non-sink) drop the high-frequency RoPE pairs 0..J-1, optionally low-rank on the
kept band (activation-weighted, C restricted to band rows), optionally NoPE on the band (position-free), optionally value low-rank.
Window keys (Delta < W) and the first SINK tokens use full exact K,V with real RoPE. Layers 0-1 untouched.

Config grammar:  H | AW64 | GW64 | R<J> | R<J>L<r> | R<J>L<r>N | R<J>L<r>NV   [ _nw (no window, sinks kept) | _nws (no window, no sinks) ]
  AW64 = top-600 truncation of full C, exact RoPE; GW64 = same keys, pairs >=32 NoPE (old reference configs, recomputed here).
Modes:  ppl <model_id> <key> <ctx> <nwin> <configs> <out_csv>      (WikiText-2 test, same windows as fold_eval.py)
        probe <model_id> <key> <ctx> <nwin> <configs> <out_prefix>  (exact model; held-out train windows 64..; empirical werr etc.)"""
import sys, re, time, json, types
import numpy as np, torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.matmul.allow_tf32 = False
mode, model_id, key, ctx, nwin, cfgs, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6].split(","), sys.argv[7]
dev = "cuda"; FIRST = 2; SINK = 4; WIN = 64
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(dev).eval()
cfg = model.config; H = cfg.num_attention_heads; DH = cfg.hidden_size // H; HALF = DH // 2; NL = cfg.num_hidden_layers
theta = cfg.rope_parameters["rope_theta"] ** (-2.0 * np.arange(HALF) / DH)
layers = model.model.layers
SELV = {"llama2-7b": "fold/sel_llama2-7b_L2048", "qwen1.5-7b": "fold/sel_qwen1.5-7b_L2048"}[key]

split = "test" if mode == "ppl" else "train"
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
ids = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
off = 0 if mode == "ppl" else 64 * 2048          # probe windows are held out of the 64x2048 calibration tokens
bos = tok.bos_token_id
windows = []
for i in range(nwin):
    c = ids[off + i * ctx: off + (i + 1) * ctx]
    if mode == "probe" and bos is not None: c = torch.cat([torch.tensor([bos]), c[:-1]])
    windows.append(c)
assert all(len(w) == ctx for w in windows)

tpos = torch.arange(ctx, device=dev)
dmat = tpos[:, None] - tpos[None, :]
causal = torch.ones(ctx, ctx, dtype=torch.bool, device=dev).tril()
def phases(p):
    pos = torch.arange(ctx, dtype=torch.float64, device=dev) - ctx // 2
    pt = torch.as_tensor(p, dtype=torch.complex128, device=dev) * 1j
    return torch.exp(pos[:, None] * pt[None, :]).to(torch.complex64), torch.exp(-pos[:, None] * pt[None, :]).to(torch.complex64)
PH_ROPE = phases(theta)
PH_ZERO = None
def to_c(x): return torch.complex(x[..., :HALF], x[..., HALF:])
def re_mm(a, b): return a.real @ b.real.T - a.imag @ b.imag.T

def parse(name):
    d = dict(name=name, J=0, r=None, nope=False, vlr=False, win=WIN, sink=SINK, special=None)
    if name.endswith("_nws"): d["win"] = 0; d["sink"] = 0; name = name[:-4]
    elif name.endswith("_nw"): d["win"] = 0; name = name[:-3]
    if name in ("AW64", "GW64"):
        d["special"] = name; return d
    m = re.fullmatch(r"R(\d+)(?:L(\d+)(N)?(V)?)?", name)
    assert m, name
    d["J"] = int(m.group(1)); d["r"] = int(m.group(2)) if m.group(2) else None
    d["nope"] = bool(m.group(3)); d["vlr"] = bool(m.group(4))
    return d

_cache = {}
def getV(path):
    if path not in _cache: _cache[path] = torch.from_numpy(np.load(path)).to(dev)
    return _cache[path]
def band_idx(J):
    return torch.as_tensor([h * DH + j for h in range(H) for j in range(DH) if j % HALF >= J], device=dev)

def far_keys(d, l, k):
    """returns khat (n, 4096) fp32 used for long-range scores, and the pair-phase vector for rotation"""
    if d["special"]:
        Vl = getV(f"{SELV}/V/layer_{l}.npy")
        kh = (k @ Vl) @ Vl.T
        p = theta if d["special"] == "AW64" else np.where(np.arange(HALF) < 32, theta, 0.0)
        return kh, p
    J = d["J"]
    if J == 0: raise ValueError
    idx = band_idx(J)
    kb = k[:, idx]
    if d["r"]:
        Vb = getV(f"rsplit/V_{key}_J{J}_L{l}.npy")[:, :d["r"]]
        kb = (kb @ Vb) @ Vb.T
    kh = torch.zeros_like(k); kh[:, idx] = kb
    return kh, (np.zeros(HALF) if d["nope"] else theta)
_phc = {}
def ph_for(p):
    kk = p.tobytes()
    if kk not in _cache: _cache[kk] = phases(p)
    return _cache[kk]

def far_score(rho_h, kh, h, p):
    kap = to_c(kh[:, h * DH:(h + 1) * DH]); a, b = ph_for(np.asarray(p))
    return re_mm(rho_h * a, torch.conj(kap) * b)

def vhat_of(l, v):
    Vv = getV(f"rsplit/VV_{key}_L{l}.npy")
    return ((v.reshape(-1, H * DH) @ Vv) @ Vv.T).view(-1, H, DH)

def win_mask(d):
    return (dmat < d["win"]) | (tpos < d["sink"])[None, :]

# ---------------- ppl forward ----------------
def fold_forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kw):
    d = self._cfg; l = self._layer
    x = hidden_states[0]; n = x.shape[0]
    q = self.q_proj(x).float().view(n, H, DH); k = self.k_proj(x).float(); v = self.v_proj(x).float().view(n, H, DH)
    rho = to_c(q)
    kh, p = far_keys(d, l, k)
    vh = vhat_of(l, v) if d["vlr"] else None
    wm = win_mask(d)[:n, :n]; msk = causal[:n, :n]
    use_win = bool(wm.any())
    out = torch.empty(n, H, DH, device=dev)
    for h in range(H):
        s = far_score(rho[:, h], kh, h, p)
        if use_win:
            kx = to_c(k[:, h * DH:(h + 1) * DH])
            s_ex = re_mm(rho[:, h] * PH_ROPE[0][:n], torch.conj(kx) * PH_ROPE[1][:n])
            s = torch.where(wm, s_ex, s)
        a = torch.softmax((s * self.scaling).masked_fill(~msk, float("-inf")), -1)
        if vh is None: out[:, h] = a @ v[:, h]
        else: out[:, h] = (a * wm) @ v[:, h] + (a * ~wm) @ vh[:, h]
    return self.o_proj(out.reshape(1, n, H * DH).to(hidden_states.dtype)), None

def install(name):
    for l in range(NL):
        a = layers[l].self_attn
        if "forward" in a.__dict__: del a.forward
    if name == "H": return
    d = parse(name)
    for l in range(FIRST, NL):
        a = layers[l].self_attn; a._layer = l; a._cfg = d
        a.forward = types.MethodType(fold_forward, a)

@torch.no_grad()
def window_nll(w):
    x = w[None].to(dev)
    hs = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
    tot = 0.0
    for c0 in range(0, hs.shape[0] - 1, 1024):
        c1 = min(c0 + 1024, hs.shape[0] - 1)
        tot += F.cross_entropy(model.lm_head(hs[c0:c1]).float(), x[0, c0 + 1:c1 + 1], reduction="sum").item()
    return tot, hs.shape[0] - 1

import pandas as pd
if mode == "ppl":
    rows = []
    for name in cfgs:
        install(name); t0 = time.time(); tot = cnt = 0
        for w in windows:
            a, b = window_nll(w); tot += a; cnt += b
        install("H")
        p = float(np.exp(tot / cnt))
        print(f"{key} ctx={ctx} nwin={nwin} {name}: PPL {p:.4f} ({time.time()-t0:.0f}s)", flush=True)
        rows.append(dict(model=key, ctx=ctx, nwin=nwin, config=name, ppl=p))
        pd.DataFrame(rows).to_csv(out, index=False)
    sys.exit()

# ---------------- probe: exact model, empirical attention-weighted score error ----------------
BINS = [(0, 0), (1, 3), (4, 15), (16, 63), (64, 255), (256, 10 ** 9)]
PC = [parse(c) for c in cfgs]
nc = len(PC)
err = torch.zeros(nc, NL, H, dtype=torch.float64, device=dev)       # sum A (ds)^2 over far (non-window) pairs
den = torch.zeros(NL, H, dtype=torch.float64, device=dev)           # sum A s^2 over all causal pairs
farmass = torch.zeros(NL, H, dtype=torch.float64, device=dev)       # sum A over Delta>=64, non-sink
qsh = torch.zeros(NL, H, dtype=torch.float64, device=dev)           # pairs 0..15 share of query energy
nq = [0]
BL = 15
bnum = torch.zeros(nc, len(BINS), dtype=torch.float64, device=dev); bden = torch.zeros(len(BINS), dtype=torch.float64, device=dev)
bmask = [((dmat >= lo) & (dmat <= hi) & causal) for lo, hi in BINS]
exact_win = {c["name"]: win_mask(c) & causal for c in PC}

def probe_forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kw):
    l = self._layer
    x = hidden_states[0]; n = x.shape[0]
    q = self.q_proj(x).float().view(n, H, DH); k = self.k_proj(x).float(); v = self.v_proj(x).float().view(n, H, DH)
    rho = to_c(q)
    fk = [far_keys(c, l, k) for c in PC]
    out = torch.empty(n, H, DH, device=dev)
    wm0 = win_mask(dict(win=WIN, sink=SINK))
    for h in range(H):
        kx = to_c(k[:, h * DH:(h + 1) * DH])
        s_ex = re_mm(rho[:, h] * PH_ROPE[0], torch.conj(kx) * PH_ROPE[1]) * self.scaling
        a = torch.softmax(s_ex.masked_fill(~causal, float("-inf")), -1)
        out[:, h] = a @ v[:, h]
        den[l, h] += (a * s_ex ** 2).sum().double()
        farmass[l, h] += (a * (~wm0)).sum().double()
        e2 = (rho[:, h].abs() ** 2).sum(0); qsh[l, h] += (e2[:16].sum() / e2.sum()).double()
        for ci, c in enumerate(PC):
            s_c = far_score(rho[:, h], fk[ci][0], h, fk[ci][1]) * self.scaling
            ds2 = (s_c - s_ex) ** 2
            fm = ~exact_win[c["name"]] & causal
            err[ci, l, h] += (a * ds2 * fm).sum().double()
            if l == BL:
                for bi in range(len(BINS)):
                    bnum[ci, bi] += (ds2 * bmask[bi]).sum().double()
        if l == BL:
            for bi in range(len(BINS)): bden[bi] += (s_ex ** 2 * bmask[bi]).sum().double()
    if l == FIRST: nq[0] += 1
    return self.o_proj(out.reshape(1, n, H * DH).to(hidden_states.dtype)), None

for l in range(NL):
    a = layers[l].self_attn
    if "forward" in a.__dict__: del a.forward
for l in range(FIRST, NL):
    a = layers[l].self_attn; a._layer = l; a.forward = types.MethodType(probe_forward, a)
with torch.no_grad():
    for i, w in enumerate(windows):
        model.model(input_ids=w[None].to(dev), use_cache=False); print("probe window", i, flush=True)
nw = nq[0]
err, den, farmass, qsh = err.cpu().numpy(), den.cpu().numpy(), farmass.cpu().numpy() / (nw * ctx), qsh.cpu().numpy() / nw
np.savez(out + "_probe.npz", err=err, den=den, farmass=farmass, qsh=qsh, bnum=bnum.cpu().numpy(), bden=bden.cpu().numpy(),
         configs=np.array(cfgs), bins=np.array(BINS))
rows = []
for ci, c in enumerate(cfgs):
    for l in range(FIRST, NL):
        rows.append(dict(model=key, ctx=ctx, config=c, layer=l, werr=float(err[ci, l].sum() / den[l].sum())))
pd.DataFrame(rows).to_csv(out + "_werr_layer.csv", index=False)
print(pd.DataFrame(rows).groupby("config").werr.mean().to_string())
