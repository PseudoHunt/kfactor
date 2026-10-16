"""Teacher-forced WikiText-2 test perplexity with the RoPE-kernel fold implemented in attention.

Layers 0-1 always use the original attention. Layers 2..31 (configs A-G, X) use a custom fp32 attention:
  score_h(s,t) = Re sum_terms Qp[s,j] Kp[t,j] / sqrt(d_h), causal softmax, values unchanged.
Backends (both are the "atoms + P" form; f_k(Delta) = exp(p_k Delta) = exp(p_k (s-s0)) exp(-p_k (t-s0))):
  part  : P partitions pairs onto atoms (A, F, G, X):   Qp[s,i] = rho_si e^{p_k(i) s},  Kp[t,i] = conj(kappa_ti) e^{-p_k(i) t}
  latent: per (g,h) atoms and P (B-E, unit test):      Qp[s,j] = (rho_s . U[:,j]) e^{p_kj s}, Kp[t,j] = c_{t,g_j} e^{-p_kj t}
          with U[:,j] = P^{(g_j,h)}_{k_j,:} * conj(z_{g_j,h,:}),  c_t = k_t V_r.
H = unmodified model.

Usage: python fold_eval.py <hf_model_id> <model_key> <ctx> <n_windows> <sel_dir> <configs> <out_csv> [--unit-test]
"""
import sys, types, time, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ap = argparse.ArgumentParser()
for a_ in ("model_id", "model_key", "ctx", "nwin", "sel", "configs", "out"):
    ap.add_argument(a_)
ap.add_argument("--unit-test", action="store_true")
ap.add_argument("--unit-windows", type=int, default=4)
ap.add_argument("--unit-ppl-only", action="store_true")
args = ap.parse_args()
ctx, nwin = int(args.ctx), int(args.nwin)
dev = "cuda"
FIRST = 2

tok = AutoTokenizer.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(dev).eval()
cfg = model.config
H = cfg.num_attention_heads; DH = getattr(cfg, "head_dim", None) or cfg.hidden_size // H; HALF = DH // 2
NL = cfg.num_hidden_layers
assert cfg.num_key_value_heads == H
base = cfg.rope_parameters["rope_theta"]
theta = base ** (-2.0 * np.arange(HALF) / DH)
assert np.allclose(model.model.rotary_emb.inv_freq.double().cpu().numpy(), theta, rtol=1e-6)
for l in range(NL):
    assert getattr(model.model.layers[l].self_attn, "sliding_window", None) is None
layers = model.model.layers

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
assert ids.numel() >= ctx * nwin, (ids.numel(), ctx * nwin)
windows = [ids[i * ctx:(i + 1) * ctx] for i in range(nwin)]

V = {l: torch.from_numpy(np.load(f"{args.sel}/V/layer_{l}.npy")).to(dev) for l in range(FIRST, NL)}   # (out, r) fp32
causal = torch.ones(ctx, ctx, dtype=torch.bool, device=dev).tril()
SINK = 4
tpos = torch.arange(ctx, device=dev)
dmat = tpos[:, None] - tpos[None, :]

def phases(p, n):
    pos = torch.arange(n, dtype=torch.float64, device=dev) - n // 2
    pt = torch.as_tensor(p, dtype=torch.complex128, device=dev)
    return torch.exp(pos[:, None] * pt[None, :]).to(torch.complex64), torch.exp(-pos[:, None] * pt[None, :]).to(torch.complex64)

def re_mm(a, b):
    return a.real @ b.real.T - a.imag @ b.imag.T

def to_c(x):
    return torch.complex(x[..., :HALF], x[..., HALF:])

def attn_scores(self, rho, k, c, h, n):
    spec = self._fold
    if spec["kind"] == "part":
        kap = to_c((k if spec["keys"] == "exact" else self._khat)[:, h * DH:(h + 1) * DH])
        Qp = rho[:, h] * spec["ph"][0][:n]; Kp = torch.conj(kap) * spec["ph"][1][:n]
        return re_mm(Qp, Kp)
    hd = spec["heads"][h] if spec["kind"] == "latent" else unit_head(self._layer, h)
    Qp = (rho[:, h] @ hd["U"]) * spec["ph"][0][:n, hd["k"]]
    Kp = c[:, hd["g"]].to(torch.complex64) * spec["ph"][1][:n, hd["k"]]
    return re_mm(Qp, Kp)

def fold_forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kw):
    x = hidden_states[0]
    n = x.shape[0]
    q = self.q_proj(x).float().view(n, H, DH)
    k = self.k_proj(x).float()
    v = self.v_proj(x).float().view(n, H, DH)
    rho = to_c(q)
    Vl = V[self._layer]
    c = k @ Vl
    self._khat = c @ Vl.T
    out = torch.empty(n, H, DH, device=dev)
    msk = causal[:n, :n]
    W = self._fold.get("W", 0)
    if W:
        win = (dmat[:n, :n] < W) | (tpos[:n] < SINK)[None, :]
    for h in range(H):
        s = attn_scores(self, rho, k, c, h, n)
        if W:
            kx = to_c(k[:, h * DH:(h + 1) * DH])
            s_ex = re_mm(rho[:, h] * self._fold["ph_rope"][0][:n], torch.conj(kx) * self._fold["ph_rope"][1][:n])
            s = torch.where(win, s_ex, s)
        s = s * self.scaling
        s = s.masked_fill(~msk, float("-inf"))
        out[:, h] = torch.softmax(s, -1) @ v[:, h]
    self._khat = None
    return self.o_proj(out.reshape(1, n, H * DH).to(hidden_states.dtype)), None

def part_spec(p_pairs, keys):
    return {"kind": "part", "keys": keys, "ph": phases(p_pairs, ctx)}

def latent_spec(l, cfgname):
    z = np.load(f"{args.sel}/{cfgname}/layer_{l}.npz")
    heads = []
    for h in range(H):
        heads.append({"g": torch.from_numpy(z[f"h{h}_g"].astype(np.int64)).to(dev),
                      "k": torch.from_numpy(z[f"h{h}_k"].astype(np.int64)).to(dev),
                      "U": torch.from_numpy(z[f"h{h}_U"]).to(dev)})
    return {"kind": "latent", "heads": heads, "ph": phases(z["p"], ctx)}

def unit_head(l, h):
    """All 64 RoPE atoms for (g,h), P = I: U[i, j] = delta(i, k_j) conj(z_{g_j,h,i}); built on the fly per head."""
    Vl = V[l]
    r = Vl.shape[1]
    zc = torch.complex(Vl[h * DH:h * DH + HALF].T, Vl[h * DH + HALF:(h + 1) * DH].T)      # (r, 64)
    g = torch.arange(r, device=dev).repeat_interleave(HALF); kk = torch.arange(HALF, device=dev).repeat(r)
    U = torch.zeros(HALF, r * HALF, dtype=torch.complex64, device=dev)
    U[kk, torch.arange(r * HALF, device=dev)] = torch.conj(zc[g, kk])
    return {"g": g, "k": kk, "U": U}

def unit_spec(l):
    return {"kind": "unit", "ph": phases(1j * theta, ctx)}

def build(cfgname, l):
    W = 0
    if "W" in cfgname:
        W = int(cfgname.split("W")[1])
        base_ = cfgname.split("W")[0]
        spec = build(base_, l) if base_ in ("A", "F", "G") else latent_spec(l, cfgname)
        spec["W"] = W; spec["ph_rope"] = phases(1j * theta, ctx)
        return spec
    if cfgname in ("A", "X"):
        return part_spec(1j * theta, "hat" if cfgname == "A" else "exact")
    if cfgname == "F":
        return part_spec(1j * np.repeat(theta.reshape(-1, 8).mean(1), 8), "hat")
    if cfgname == "G":
        return part_spec(1j * np.where(np.arange(HALF) < 32, theta, 0.0), "hat")
    if cfgname == "U64":
        return unit_spec(l)
    return latent_spec(l, cfgname)

def install(cfgname):
    for l in range(NL):
        a = layers[l].self_attn
        if "forward" in a.__dict__:
            del a.forward
        a._fold = None
    if cfgname == "H":
        return
    for l in range(FIRST, NL):
        a = layers[l].self_attn
        a._layer = l
        a._fold = build(cfgname, l)
        a.forward = types.MethodType(fold_forward, a)

@torch.no_grad()
def window_nll(w):
    x = w[None].to(dev)
    hs = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
    tot = 0.0
    for c0 in range(0, hs.shape[0] - 1, 1024):
        c1 = min(c0 + 1024, hs.shape[0] - 1)
        logits = model.lm_head(hs[c0:c1]).float()
        tot += F.cross_entropy(logits, x[0, c0 + 1:c1 + 1], reduction="sum").item()
    return tot, hs.shape[0] - 1

def ppl(cfgname, wins):
    install(cfgname)
    t0 = time.time(); tot = cnt = 0
    nll = []
    for w in wins:
        a, b = window_nll(w); tot += a; cnt += b; nll.append(a / b)
    install("H")
    return float(np.exp(tot / cnt)), nll, time.time() - t0

results = []
if args.unit_test and not args.unit_ppl_only:
    # (1) score-level: capture layer inputs on window 0, compare A(real RoPE) vs A(part) vs U64(latent, P=I)
    from transformers.models.llama import modeling_llama as ml
    caps = {}
    hooks = [layers[l].self_attn.register_forward_pre_hook(
        (lambda l: lambda mod, a, kw: caps.__setitem__(l, kw["hidden_states"].detach()))(l), with_kwargs=True)
        for l in (2, 15, NL - 1)]
    with torch.no_grad():
        model.model(input_ids=windows[0][None].to(dev), use_cache=False)
    for hk in hooks:
        hk.remove()
    n = ctx
    pos = torch.arange(n, dtype=torch.float64, device=dev)
    ang = pos[:, None] * torch.as_tensor(theta, device=dev)[None]
    cos = torch.cat([ang.cos(), ang.cos()], -1).float()[None]; sin = torch.cat([ang.sin(), ang.sin()], -1).float()[None]
    for l, hsx in caps.items():
        a = layers[l].self_attn; a._layer = l
        x = hsx[0]
        q = a.q_proj(x).float().view(n, H, DH); k = a.k_proj(x).float(); rho = to_c(q)
        c = k @ V[l]; a._khat = c @ V[l].T
        qr, kr = ml.apply_rotary_pos_emb(q.transpose(0, 1)[None], a._khat.view(n, H, DH).transpose(0, 1)[None], cos, sin)
        worst = {"part": 0.0, "U64": 0.0}; scale = 0.0
        specA, specU = build("A", l), build("U64", l)
        for h in range(H):
            s_std = (qr[0, h] @ kr[0, h].T)
            a._fold = specA; s_part = attn_scores(a, rho, k, c, h, n)
            a._fold = specU; s_u = attn_scores(a, rho, k, c, h, n)
            m = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev))
            scale = max(scale, s_std[m].abs().max().item())
            worst["part"] = max(worst["part"], (s_part - s_std)[m].abs().max().item())
            worst["U64"] = max(worst["U64"], (s_u - s_std)[m].abs().max().item())
        a._fold = None; a._khat = None
        msg = (f"UNIT scores layer {l}: max|score| {scale:.3e}; max|part - realRoPE| {worst['part']:.3e}; "
               f"max|U64 - realRoPE| {worst['U64']:.3e} (rel {worst['U64'] / scale:.1e})")
        print(msg, flush=True)
        results.append(dict(config="unit_scores", layer=l, max_abs_score=scale, diff_part=worst["part"], diff_U64=worst["U64"]))
    del specA, specU
    torch.cuda.empty_cache()
if args.unit_test:
    # (2) PPL: U64 vs A on the first unit-windows windows
    wins = windows[:args.unit_windows]
    pA, nA, _ = ppl("A", wins)
    pU, nU, tU = ppl("U64", wins)
    print(f"UNIT ppl on {len(wins)} windows: A {pA:.6f}  U64 {pU:.6f}  rel diff {abs(pU - pA) / pA:.2e}; "
          f"max per-window NLL diff {max(abs(x - y) for x, y in zip(nA, nU)):.2e}  ({tU:.0f}s)", flush=True)
    results.append(dict(config="unit_ppl", ppl_A=pA, ppl_U64=pU, rel_diff=abs(pU - pA) / pA))
else:
    for cfgname in args.configs.split(","):
        p, nll, t = ppl(cfgname, windows)
        print(f"{args.model_key} ctx={ctx} nwin={nwin} config {cfgname}: PPL {p:.4f}  ({t:.0f}s)", flush=True)
        results.append(dict(model=args.model_key, ctx=ctx, nwin=nwin, config=cfgname, ppl=p, seconds=t,
                            window_nll=json.dumps([round(v, 6) for v in nll])))
import pandas as pd
pd.DataFrame(results).to_csv(args.out, index=False)
print("wrote", args.out)
