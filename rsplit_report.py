"""Range-split report: PPL table (+werr, cache), layer-15 Delta-bucket diagnostic, worst heads for R32, per-layer werr for R16/32/48."""
import re, os, numpy as np, pandas as pd
D = "/home/kfactor/rsplit"
SET = [("llama2-7b", 2048, "llama"), ("qwen1.5-7b", 2048, "qwen"), ("qwen1.5-7b", 8192, "qwen")]
def parse(c):
    base = re.sub(r"_nws?$", "", c)
    win = 0 if c.endswith("_nw") or c.endswith("_nws") else 64
    sink = 0 if c.endswith("_nws") else 4
    if base == "H": return dict(kd=4096, vd=4096, win=0, sink=0, full=True)
    if base in ("AW64", "GW64"): return dict(kd=600, vd=4096, win=64, sink=4)
    m = re.fullmatch(r"R(\d+)(?:L(\d+)(N)?(V)?)?", base)
    J = int(m.group(1)); r = int(m.group(2)) if m.group(2) else None
    return dict(kd=r if r else 64 * (64 - J), vd=600 if m.group(4) else 4096, win=win, sink=sink)
def cache(c, n):
    p = parse(c)
    fullK = 32 * 4096 * n
    if p.get("full"): return 4096.0, 8192.0, 1.0, 1.0
    ex = (p["win"] + p["sink"]) * 4096                      # exact window + sinks held in addition (same convention as phase 2)
    K = 2 * 4096 * n + 30 * (p["kd"] * n + ex)
    V = 2 * 4096 * n + 30 * (p["vd"] * n + ex)
    return K / n / 32, (K + V) / n / 32, fullK / K, 2 * fullK / (K + V)    # per-token per-layer averages (elements)
def load_ppl(model, ctx, tag):
    fs = [f"{D}/s1_{tag}_{ctx}.csv", f"{D}/s2_{tag}_{ctx}.csv"]
    return pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)]).drop_duplicates("config").set_index("config").ppl
ORDER = ["H", "AW64", "GW64", "R16", "R32", "R48"] + [f"R{j}L{r}{s}" for s in ("", "N", "NV") for j in (16, 32, 48) for r in (300, 600)] + ["R32L600N_nw", "R32L600N_nws"]
out = []
for model, ctx, tag in SET:
    p = load_ppl(model, ctx, tag)
    wf = f"{D}/p_{tag}_{ctx}_werr_layer.csv"
    we = pd.read_csv(wf).groupby("config").werr.mean() if os.path.exists(wf) else pd.Series(dtype=float)
    rows = []
    for c in ORDER:
        if c not in p.index: continue
        k, kv, rk, rkv = cache(c, ctx)
        rows.append(dict(config=c, PPL=p[c], dPPL=p[c] - p["H"], werr=we.get(c, np.nan), key_el_per_tok_layer=k, kv_el_per_tok_layer=kv,
                         key_ratio=rk, kv_ratio=rkv, key_ratio_32k=cache(c, 32768)[2], kv_ratio_32k=cache(c, 32768)[3]))
    t = pd.DataFrame(rows); t.to_csv(f"{D}/table_{tag}_{ctx}.csv", index=False)
    print(f"\n=== {model} @ {ctx} ===  (cache at context {ctx}; *_32k at 32768; layers 0-1 full; window+4 sinks exact, additive)")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
BINS = ["[0,0]", "[1,3]", "[4,15]", "[16,63]", "[64,255]", "[256,max]"]
for model, ctx, tag in SET[:2]:
    f = f"{D}/p_{tag}_{ctx}_probe.npz"
    if not os.path.exists(f): continue
    z = np.load(f); cf = list(z["configs"])
    print(f"\n=== layer-15 relative score error by Delta bin, no window substitution ({model} @ {ctx}) ===")
    t = pd.DataFrame({c: z["bnum"][cf.index(c)] / z["bden"] for c in ("R32", "R32L600", "R32L600N") if c in cf}, index=BINS)
    print(t.to_string(float_format=lambda v: f"{v:.4g}"))
for model, ctx, tag in SET:
    f = f"{D}/p_{tag}_{ctx}_probe.npz"
    if not os.path.exists(f): continue
    z = np.load(f); cf = list(z["configs"]); ci = cf.index("R32")
    w = z["err"][ci] / z["den"]                                     # (layer, head)
    rows = [dict(layer=l, head=h, werr=w[l, h], q_share_pairs0_15=z["qsh"][l, h], far_mass=z["farmass"][l, h]) for l in range(2, 32) for h in range(32)]
    t = pd.DataFrame(rows).sort_values("werr", ascending=False)
    t.to_csv(f"{D}/heads_R32_{tag}_{ctx}.csv", index=False)
    print(f"\n=== R32 per-head werr, 10 worst ({model} @ {ctx}); mean over heads: werr {t.werr.mean():.4g}, q_share {t.q_share_pairs0_15.mean():.3f}, far_mass {t.far_mass.mean():.3f} ===")
    print(t.head(10).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("Spearman(werr, far_mass) over heads:", round(pd.Series(t.werr.values).rank().corr(pd.Series(t.far_mass.values).rank()), 3),
          " Spearman(werr, q_share):", round(pd.Series(t.werr.values).rank().corr(pd.Series(t.q_share_pairs0_15.values).rank()), 3))
    print("top-10 far_mass (mean) vs all heads far_mass (mean):", round(t.head(10).far_mass.mean(), 3), round(t.far_mass.mean(), 3))
for model, ctx, tag in SET:
    f = f"{D}/p_{tag}_{ctx}_werr_layer.csv"
    if not os.path.exists(f): continue
    w = pd.read_csv(f); w = w[w.config.isin(["R16", "R32", "R48"])].pivot(index="layer", columns="config", values="werr")
    w.to_csv(f"{D}/layer_werr_{tag}_{ctx}.csv")
    print(f"\n=== per-layer werr R16/R32/R48 ({model} @ {ctx}) ===")
    print(w.to_string(float_format=lambda v: f"{v:.4g}"))
