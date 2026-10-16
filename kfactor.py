"""Training-free test: per-RoPE-frequency-pair factorization of W_K vs unstructured low-rank.

Usage: python kfactor.py <hf_model_id> <out_dir>
"""
import os, sys, json
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

N_SEQ, SEQ_LEN = 64, 2048
BASE_RANKS = [300, 600, 1112, 1624]  # defined for out = 4096
F64 = torch.float64

model_id, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(f"{out_dir}/C", exist_ok=True)
dev = "cuda"

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(dev).eval()
cfg = model.config
n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
d_model = cfg.hidden_size
d_h = getattr(cfg, "head_dim", None) or d_model // n_h
n_layers = cfg.num_hidden_layers
out = n_kv * d_h
half = d_h // 2
print(f"{model_id}: layers={n_layers} d_model={d_model} n_h={n_h} n_kv={n_kv} d_h={d_h} out={out}")

pair_idx = [np.array([h * d_h + j for h in range(n_kv)] + [h * d_h + j + half for h in range(n_kv)])
            for j in range(half)]
inter_idx = [np.array([h * d_h + 2 * j for h in range(n_kv)] + [h * d_h + 2 * j + 1 for h in range(n_kv)])
             for j in range(half)]

# ---- Step 0: pairing check with the model's own rotary implementation ----
def pairing_check(pos=37, seed=0):
    rope_mod = sys.modules[type(model.model).__module__]
    apply_rope = rope_mod.apply_rotary_pos_emb
    remb = model.model.rotary_emb
    scale = float(getattr(remb, "attention_scaling", 1.0))
    k = torch.randn(1, n_kv, 1, d_h, dtype=F64, generator=torch.Generator().manual_seed(seed))
    pid = torch.tensor([[pos]])
    # fp64 cos/sin from the model's inv_freq (the native forward computes them in fp32)
    freqs = pos * remb.inv_freq.detach().cpu().double()
    emb = torch.cat([freqs, freqs])[None, None]
    cos64, sin64 = emb.cos() * scale, emb.sin() * scale
    cosn, sinn = remb(k.to(dev), pid.to(dev))  # native path, for information
    res = {}
    for name, (c, s) in {"fp64": (cos64, sin64), "native": (cosn.cpu().double(), sinn.cpu().double())}.items():
        _, kr = apply_rope(k, k, c, s)
        kf, krf = k.reshape(-1), kr.reshape(-1)
        for pname, P in (("rotate_half", pair_idx), ("interleaved", inter_idx)):
            res[(name, pname)] = max(abs((scale * kf[i].norm() - krf[i].norm()).item()) for i in P)
    return res

pc = pairing_check()
print("\n== pairing check (pos=37, max over pairs of | |k_pair| - |rot k_pair| |) ==")
for (cs, p), e in pc.items():
    print(f"  cos/sin={cs:6s} pairing={p:11s} max_err={e:.3e}  {'PASS' if e < 1e-9 else 'FAIL'} @1e-9")
if pc[("fp64", "rotate_half")] >= 1e-9:
    sys.exit("ABORT: rotate_half pairing failed the rotation check")
if pc[("fp64", "interleaved")] < 1e-9:
    sys.exit("ABORT: interleaved pairing unexpectedly passed; check is not discriminative")
print("  rotate_half pairing confirmed; interleaved fails as expected")

# ---- Step 1: calibration ----
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
ids = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
assert ids.numel() >= N_SEQ * SEQ_LEN, ids.numel()
bos = tok.bos_token_id

layers = model.model.layers
G = [torch.zeros(d_model, d_model, dtype=F64, device=dev) for _ in range(n_layers)]
S = [torch.zeros(d_model, dtype=F64, device=dev) for _ in range(n_layers)]
calls = [0] * n_layers

def mk_hook(l):
    def hook(mod, args):
        x = args[0].reshape(-1, d_model).float()
        G[l] += (x.T @ x).double()  # fp32 per-sequence product (one call per layer per sequence), fp64 accumulate
        S[l] += x.sum(0).double()
        calls[l] += 1
    return hook

hooks = [layers[l].self_attn.k_proj.register_forward_pre_hook(mk_hook(l)) for l in range(n_layers)]
N = 0
with torch.no_grad():
    for i in range(N_SEQ):
        chunk = ids[i * SEQ_LEN:(i + 1) * SEQ_LEN]
        if bos is not None:  # keep length 2048 incl. BOS
            chunk = torch.cat([torch.tensor([bos]), chunk[:-1]])
        model(chunk[None].to(dev), use_cache=False)
        N += chunk.numel()
for h in hooks:
    h.remove()
assert all(c == N_SEQ for c in calls), calls
print(f"calibrated on {N} tokens")

# ---- Step 2: per-layer spectra (GPU, fp64) ----
ranks = sorted({max(1, round(r * out / 4096)) for r in BASE_RANKS})
r_mid = max(1, round(600 * out / 4096))
G_SIZES = sorted({g for g in (1, 2, 4, 8, half) if g <= half and half % g == 0})
grp_index = {g: torch.tensor(np.stack([np.concatenate(pair_idx[g * m:g * m + g]) for m in range(half // g)]),
                             device=dev) for g in G_SIZES}

def eigdesc(M):
    return torch.linalg.eigvalsh(M).flip(-1)

def pooled(C, g):
    I = grp_index[g]
    e = torch.linalg.eigvalsh(C[I[:, :, None], I[:, None, :]])  # (n_groups, b)
    tags = torch.arange(I.shape[0], device=dev)[:, None].expand_as(e).reshape(-1)
    order = torch.argsort(e.reshape(-1), descending=True)
    return e.reshape(-1)[order], tags[order]

rows, grows, checks, bias_rows = [], [], [], []
alloc = {r: np.zeros((n_layers, half), int) for r in ranks}
for l in range(n_layers):
    Gl = G[l] / N
    asym = ((Gl - Gl.T).abs().max() / Gl.abs().max()).item()
    Gl = 0.5 * (Gl + Gl.T)
    gev = torch.linalg.eigvalsh(Gl)
    psd_ok = (gev.min() >= -1e-6 * gev.max()).item()

    kp = layers[l].self_attn.k_proj
    W = kp.weight.detach().to(F64)
    C = W @ Gl @ W.T
    C = 0.5 * (C + C.T)
    np.save(f"{out_dir}/C/layer_{l}.npy", C.float().cpu().numpy())
    tr = torch.trace(C).item()
    eig_u = eigdesc(C)
    cu = torch.cumsum(eig_u, 0)

    chk = dict(layer=l, G_asym=asym, G_min_over_max=(gev.min() / gev.max()).item(), G_psd=psd_ok,
               rel_err_unstruct_trace=abs(eig_u.sum().item() - tr) / tr)
    for g in G_SIZES:
        sv, st = pooled(C, g)
        cs = torch.cumsum(sv, 0)
        chk[f"rel_err_trace_g{g}"] = abs(sv.sum().item() - tr) / tr
        chk[f"ratio_full_g{g}"] = (cs[out - 1] / cu[out - 1]).item()
        for r in ranks:
            Eu, Es = cu[r - 1].item(), cs[r - 1].item()
            fu, fs = Eu / tr, Es / tr
            er = (1 - fs) / (1 - fu)
            grows.append(dict(layer=l, g=g, r=r, ratio=Es / Eu, err_ratio=er))
            if g == 1:
                rows.append(dict(layer=l, r=r, E_u=Eu, E_s=Es, ratio=Es / Eu, err_ratio=er, frac_u=fu, frac_s=fs))
                alloc[r][l] = np.bincount(st[:r].cpu().numpy(), minlength=half)
    checks.append(chk)

    # side check: include k_proj bias (second moment of Wx+b)
    if kp.bias is not None:
        b = kp.bias.detach().to(F64)
        Wmu = W @ (S[l] / N)
        Cb = C + torch.outer(Wmu, b) + torch.outer(b, Wmu) + torch.outer(b, b)
        Cb = 0.5 * (Cb + Cb.T)
        eu = torch.cumsum(eigdesc(Cb), 0)
        es = torch.cumsum(pooled(Cb, 1)[0], 0)
        for r in ranks:
            bias_rows.append(dict(layer=l, r=r, ratio_with_bias=(es[r - 1] / eu[r - 1]).item()))
    G[l] = None
    del W, C, Gl
    torch.cuda.empty_cache()

# ---- Step 3: outputs ----
df = pd.DataFrame(rows)
df[["layer", "r", "E_u", "E_s", "ratio", "err_ratio"]].to_csv(f"{out_dir}/ratios.csv", index=False)
df.to_csv(f"{out_dir}/ratios_extended.csv", index=False)
gdf = pd.DataFrame(grows)
gdf[["layer", "g", "r", "ratio", "err_ratio"]].to_csv(f"{out_dir}/groups.csv", index=False)
ck = pd.DataFrame(checks); ck.to_csv(f"{out_dir}/sanity.csv", index=False)
np.savez(f"{out_dir}/alloc.npz", **{f"r{r}": a for r, a in alloc.items()})

trace_cols = [c for c in ck if c.startswith("rel_err")]
print("\n== sanity ==")
print(f"G max rel asymmetry: {ck.G_asym.max():.2e}")
print(f"G PSD all layers: {ck.G_psd.all()}  (worst min/max eig = {ck.G_min_over_max.min():.2e})")
print(f"max rel |sum eigs - tr C| over unstructured and all g: {ck[trace_cols].max().max():.2e}")
for g in G_SIZES:
    c = ck[f"ratio_full_g{g}"]
    print(f"ratio at full rank r={out}, g={g}: min {c.min():.12f} max {c.max():.12f}")
gh = gdf[gdf.g == half]
print(f"g={half} (all pairs) ratio over all l,r: min {gh.ratio.min():.12f} max {gh.ratio.max():.12f}")
assert ck.G_psd.all() and ck[trace_cols].max().max() < 1e-3
assert all(np.allclose(ck[f"ratio_full_g{g}"], 1.0, atol=1e-9) for g in G_SIZES)
assert np.allclose(gh.ratio, 1.0, atol=1e-9)

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140)
fmt = lambda v: f"{v:.4f}"
summ = df.groupby("r").agg(mean_ratio=("ratio", "mean"), min_ratio=("ratio", "min"), max_ratio=("ratio", "max"),
                           mean_err_ratio=("err_ratio", "mean"), max_err_ratio=("err_ratio", "max"),
                           mean_frac_u=("frac_u", "mean"))
summ["layers_below_0.90"] = [df[(df.r == r) & (df.ratio < 0.90)].layer.tolist() for r in summ.index]
summ.to_csv(f"{out_dir}/summary.csv")
print("\n== summary, g=1 (ratio = E_s/E_u; err_ratio = (1-frac_s)/(1-frac_u)) ==")
print(summ.to_string(float_format=fmt))

gsum = gdf.groupby(["r", "g"]).agg(mean_ratio=("ratio", "mean"), mean_err_ratio=("err_ratio", "mean"),
                                   max_err_ratio=("err_ratio", "max"))
gsum.to_csv(f"{out_dir}/groups_summary.csv")
print("\n== group-size sweep (g adjacent pairs per group) ==")
print(gsum.to_string(float_format=fmt))

worst = df[df.r == r_mid].nsmallest(3, "ratio")
print(f"\n== three worst layers at r={r_mid} (600 x out/4096), g=1 ==")
print(worst[["layer", "ratio", "err_ratio"]].to_string(index=False, float_format=fmt))

A = alloc[r_mid]
print(f"\nallocation at r={r_mid}: share of kept rank in low-frequency half (j >= {half // 2}): "
      f"{A[:, half // 2:].sum() / A.sum():.3f}")

if bias_rows:
    bdf = pd.DataFrame(bias_rows); bdf.to_csv(f"{out_dir}/ratios_with_bias.csv", index=False)
    print("\n== side check: ratio incl. k_proj bias -- inflated by bias energy; "
          "not the comparison that matters ==")
    print(bdf.groupby("r").ratio_with_bias.agg(["mean", "min"]).to_string(float_format=fmt))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 0.25 * n_layers + 2))
im = ax.imshow(A, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0, vmax=A.max())
ax.set_xlabel("frequency-pair index j (0 = highest frequency, HF rotate_half)")
ax.set_ylabel("layer")
ax.set_title(f"{model_id}: rank kept per frequency pair, r={r_mid} (max {2 * n_kv})")
fig.colorbar(im, ax=ax, label="rank kept")
fig.tight_layout(); fig.savefig(f"{out_dir}/alloc.png", dpi=130)
json.dump(dict(model=model_id, n_kv=n_kv, n_h=n_h, d_h=d_h, out=out, ranks=ranks, group_sizes=G_SIZES,
               tokens=N, pairing_check={f"{a}/{b}": v for (a, b), v in pc.items()}),
          open(f"{out_dir}/meta.json", "w"), indent=1)
print(f"\nwrote {out_dir}")
