"""Tables and figures for pencil_fit.py outputs."""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/kfactor/pencil_fit"
LM_DIR = "/home/kfactor/kernel_fit/lm_checks"
RATIOS = {"llama2-7b": "/home/kfactor/out_llama2-7b/ratios.csv", "qwen1.5-7b": "/home/kfactor/out_qwen1.5-7b/ratios.csv"}
LAM_FLOOR = 1e-7
pd.set_option("display.width", 220)
f4 = lambda v: f"{v:.4f}"

d = pd.read_csv(f"{OUT}/pencil_errors_all.csv")
lm_tag = sorted({m.rsplit("_", 1)[0] for m in d.method if m.startswith("pencil_lm")})

def channels(meth):
    k, v = meth.rsplit("_", 1)
    if meth == "nope_all": return 0
    if k == "freqfold": return 128 // int(v)
    if k == "nope_lowfreq": return 2 * int(v)
    if k.startswith("pencil"): return 2 * int(v)

MAIN = ([f"pencil_{m}" for m in (1, 2, 4, 8)] + [f"{t}_{m}" for t in lm_tag for m in (1, 2, 4, 8)]
        + ["freqfold_2", "freqfold_4", "freqfold_8", "nope_lowfreq_8", "nope_lowfreq_16", "nope_lowfreq_32", "nope_all"])
d = d[d.method.isin(MAIN + ["freqfold_1", "nope_lowfreq_64"])].copy()
d["channels"] = d.method.map(channels)
d[d.method.isin(MAIN)][["model", "layer", "window", "method", "channels", "err"]].to_csv(f"{OUT}/pencil_errors.csv", index=False)

# 2. channels table
tab = (d[d.method.isin(MAIN)].groupby(["model", "window", "method"])
       .agg(channels=("channels", "first"), mean_err=("err", "mean"), median_err=("err", "median"),
            layers_above_0_05=("err", lambda x: int((x > 0.05).sum()))).reset_index())
ex = tab.groupby(["model", "window"]).size().reset_index()[["model", "window"]]
ex = ex.assign(method="exact", channels=128, mean_err=0.0, median_err=0.0, layers_above_0_05=0)
tab = pd.concat([tab, ex]).sort_values(["model", "window", "channels", "method"], ascending=[True, False, True, True])
tab.to_csv(f"{OUT}/channels_table.csv", index=False)
for (m, L), g in tab.groupby(["model", "window"], sort=False):
    print(f"\n== {m}, window L={L} (r=600, omega-weighted per layer; stats over {d.layer.nunique()} layers) ==")
    print(g[["method", "channels", "mean_err", "median_err", "layers_above_0_05"]].to_string(index=False, float_format=f4))

# 3. cross-check vs LM-80
try:
    lm = pd.read_csv(f"{LM_DIR}/conv_80.csv"); lm = lm[lm.r == 600]
    p1 = d[d.method == "pencil_1"].pivot_table(index=["model", "layer"], columns="window", values="err").reset_index()
    cc = lm.merge(p1, on=["model", "layer"])
    cc = cc.rename(columns={"err_L4096": "lm80_L4096", "err_L512": "lm80_L512", 4096: "pencil1_L4096", 512: "pencil1_L512"})
    cc["ok_L4096"] = cc.pencil1_L4096 <= cc.lm80_L4096
    cc["ok_L512"] = cc.pencil1_L512 <= cc.lm80_L512
    print("\n== cross-check: pencil m=1 vs LM-80 (r=600). LM-80 was fit on L=4096 only; its L512 is that same fit ==")
    print(cc[["model", "layer", "lm80_L4096", "pencil1_L4096", "ok_L4096", "lm80_L512", "pencil1_L512", "ok_L512"]]
          .to_string(index=False, float_format=f4))
    cc.to_csv(f"{OUT}/crosscheck_lm80.csv", index=False)
except FileNotFoundError:
    pass

# 4. per-layer method choice
choice = []
for m in d.model.unique():
    rt = pd.read_csv(RATIOS[m]); rt = rt[rt.r == 600].set_index("layer").ratio
    pe = d[(d.model == m) & (d.window == 4096) & d.method.str.fullmatch(r"pencil_\d")]
    for l, g in pe.groupby("layer"):
        g = g.assign(mm=g.method.str[-1].astype(int)).sort_values("mm")
        ok = g[g.err < 0.05]
        choice.append(dict(model=m, layer=l, structured_ok=bool(rt[l] > 0.95), structured_ratio=rt[l],
                           min_m="none" if ok.empty else str(ok.mm.iloc[0])))
ch = pd.DataFrame(choice)
ch.to_csv(f"{OUT}/layer_choice.csv", index=False)
wide = ch.pivot(index="layer", columns="model", values=["structured_ok", "min_m"])
wide.columns = [f"{b}:{a}" for a, b in wide.columns]
print("\n== per-layer choice (structured_ok: pair-structured ratio at r=600 > 0.95; min_m: smallest pencil m with err < 0.05 at L=4096) ==")
print(wide[sorted(wide.columns)].to_string())

# 5. lambda distributions and figures
def wquant(x, w, qs):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) / w.sum()
    return [x[min(np.searchsorted(c, q), len(x) - 1)] for q in qs]

lam_rows = []
for m in d.model.unique():
    Z = np.load(f"{OUT}/poles_{m}.npz")
    layers = sorted({int(k.split("_")[0][1:]) for k in Z.files if k.startswith("L") and k.endswith("_omega")})
    for L in (4096, 512):
        for mm in (1, 2, 4, 8):
            th = np.concatenate([Z[f"L{l}_W{L}_m{mm}_th"].ravel() for l in layers]).astype(np.float64)
            lam = np.concatenate([Z[f"L{l}_W{L}_m{mm}_lam"].ravel() for l in layers]).astype(np.float64)
            wt = np.concatenate([(Z[f"L{l}_omega"][:, None] * Z[f"L{l}_W{L}_m{mm}_ew"]).ravel() for l in layers]).astype(np.float64)
            q = wquant(lam, wt, [0.1, 0.25, 0.5, 0.75, 0.9])
            lam_rows.append(dict(model=m, window=L, m=mm, q10=q[0], q25=q[1], q50=q[2], q75=q[3], q90=q[4],
                                 frac_lam0=wt[lam == 0].sum() / wt.sum(),
                                 frac_lamL4096_lt1=wt[lam * 4096 < 1].sum() / wt.sum(),
                                 frac_lamL512_lt1=wt[lam * 512 < 1].sum() / wt.sum(),
                                 frac_th_le0=wt[th <= 0].sum() / wt.sum()))
            if L == 4096 and mm == 4:
                pos = th > 0
                xb = np.logspace(np.log10(max(th[pos].min(), 1e-8)), np.log10(2.0), 120)
                yb = np.logspace(np.log10(LAM_FLOOR), np.log10(max(lam.max(), 1.0)) + 0.1, 100)
                fig, ax = plt.subplots(figsize=(10, 6))
                *_, im = ax.hist2d(th[pos], np.maximum(lam[pos], LAM_FLOOR), bins=[xb, yb], weights=wt[pos],
                                   norm=LogNorm(), cmap="Blues")
                ax.set_xscale("log"); ax.set_yscale("log")
                for t in Z["rope_theta"]:
                    ax.axvline(t, ymin=0.97, ymax=1.0, color="#333333", lw=0.8)
                ax.set_xlabel("pole th (rad / token)"); ax.set_ylabel(f"pole lam (floored at {LAM_FLOOR:g})")
                ax.set_title(f"{m}: pencil m=4 poles, L=4096, all layers, weight = omega x pole energy share; "
                             f"ticks = RoPE theta_i ({wt[~pos].sum() / wt.sum():.1%} weight at th<=0 not shown)", fontsize=9)
                fig.colorbar(im, ax=ax, label="weight (log)")
                fig.tight_layout(); fig.savefig(f"{OUT}/th_lam_hist_m4_{m}.png", dpi=130); plt.close(fig)
lr = pd.DataFrame(lam_rows); lr.to_csv(f"{OUT}/lambda_stats.csv", index=False)
print("\n== pole lambda distribution (weight = omega x pole energy share) ==")
print(lr.to_string(index=False, formatters={c: (lambda v: f"{v:.2e}") for c in ["q10", "q25", "q50", "q75", "q90"]},
                   float_format=lambda v: f"{v:.3f}"))

order = sorted([x for x in MAIN if x in set(d.method)], key=lambda x: (channels(x), x))
for m in d.model.unique():
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    for ax, L in zip(axes, (4096, 512)):
        piv = d[(d.model == m) & (d.window == L)].pivot(index="layer", columns="method", values="err")[order]
        im = ax.imshow(np.clip(piv.values, 1e-5, None), aspect="auto", cmap="Blues", norm=LogNorm(vmin=1e-5, vmax=10),
                       interpolation="nearest")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{o} ({channels(o)})" for o in order], rotation=55, ha="right", fontsize=8)
        ax.set_title(f"{m}, L={L}, r=600 (method (channels))")
    axes[0].set_ylabel("layer")
    fig.colorbar(im, ax=axes, label="relative kernel error (log)")
    fig.savefig(f"{OUT}/err_heatmap_{m}.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# 6. sanity
s = pd.read_csv(f"{OUT}/sanity.csv")
print("\n== sanity ==")
for L in (4096, 512):
    print(f"L={L}: max err freqfold_1 = {s[f'L{L}_max_freqfold_1'].max():.3e}; max err nope_lowfreq_64 = "
          f"{s[f'L{L}_max_nope_lowfreq_64'].max():.3e}; alias branch switches = {s[f'L{L}_alias_switches'].sum()}; "
          f"items non-monotone in m = {s[f'L{L}_items_nonmonotone'].sum()} of {s.n_items.sum()} "
          f"(max increase {s[f'L{L}_max_item_increase'].max():.2e})"
          + (f"; LM worse than pencil: {s[f'L{L}_lm_worse_than_pencil'].sum()}" if f"L{L}_lm_worse_than_pencil" in s else ""))
pe = d[d.method.str.fullmatch(r"pencil_\d")].assign(mm=lambda x: x.method.str[-1].astype(int)).sort_values("mm")
bad = [(m, L, l) for (m, L, l), g in pe.groupby(["model", "window", "layer"]) if (np.diff(g.err.values) > 0).any()]
print(f"layers where omega-weighted pencil error increases with m: {len(bad)} {bad[:20]}")
