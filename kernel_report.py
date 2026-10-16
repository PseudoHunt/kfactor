"""Summaries and figures for kernel_fit.py outputs."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = "/home/kfactor/kernel_fit"
METHODS = ["fitted_init", "fitted_refined", "freqfold_2", "freqfold_4", "freqfold_8",
           "nope_lowfreq_32", "nope_lowfreq_16", "nope_lowfreq_8", "nope_all"]
LAM_FLOOR = 1e-7

d = pd.read_csv(f"{OUT}/kernel_errors_all.csv")
d = d[d.method.isin(METHODS)]
d[["model", "layer", "r", "method", "err_L4096", "err_L512"]].to_csv(f"{OUT}/kernel_errors.csv", index=False)

pd.set_option("display.width", 220)
summ = d.groupby(["model", "r", "method"]).agg(mean_L4096=("err_L4096", "mean"), median_L4096=("err_L4096", "median"),
                                               mean_L512=("err_L512", "mean"), median_L512=("err_L512", "median"))
summ = summ.reindex(pd.MultiIndex.from_product([d.model.unique(), sorted(d.r.unique()), METHODS],
                                               names=summ.index.names))
summ.to_csv(f"{OUT}/summary.csv")
print("== per model, per r: error over layers ==")
print(summ.to_string(float_format=lambda v: f"{v:.4f}"))

print("\n== fitted_refined / freqfold_8 ==")
rat = []
for (m, r), g in d.groupby(["model", "r"]):
    p = g.pivot(index="layer", columns="method")
    for L in ("L4096", "L512"):
        f, ff = p[f"err_{L}"]["fitted_refined"], p[f"err_{L}"]["freqfold_8"]
        rat.append(dict(model=m, r=r, window=L, ratio_of_means=f.mean() / ff.mean(),
                        median_per_layer_ratio=(f / ff).median()))
rat = pd.DataFrame(rat); rat.to_csv(f"{OUT}/ratio_fitted_vs_freqfold8.csv", index=False)
print(rat.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n== layers with fitted_refined error > 0.05 (r=600) ==")
for m, g in d[(d.r == 600) & (d.method == "fitted_refined")].groupby("model"):
    for L in ("L4096", "L512"):
        bad = g[g[f"err_{L}"] > 0.05].layer.tolist()
        print(f"{m} {L}: {len(bad)}/32 -> {bad}")

def wquant(x, w, qs):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) / w.sum()
    return [x[np.searchsorted(c, q)] for q in qs]

print("\n== lambda distribution at r=600 (omega-weighted over all (g,h), all layers) ==")
for m in d.model.unique():
    P = np.load(f"{OUT}/fit_params_{m}.npz")
    om, th, lam = P["omega"].ravel(), P["th"].ravel(), P["lam"].ravel()
    ok = ~np.isnan(om); om, th, lam = om[ok], th[ok], lam[ok]
    q = wquant(lam, om, [0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"{m}: lam quantiles (10/25/50/75/90%): " + " ".join(f"{v:.2e}" for v in q))
    print(f"   weight with lam == 0 exactly: {om[lam == 0].sum() / om.sum():.3f}; "
          f"lam*4096 < 1: {om[lam * 4096 < 1].sum() / om.sum():.3f}; lam*512 < 1: {om[lam * 512 < 1].sum() / om.sum():.3f}; "
          f"lam*64 > 1 (gone within 64 tokens): {om[lam * 64 > 1].sum() / om.sum():.3f}")
    print(f"   weight with th <= 0: {om[th <= 0].sum() / om.sum():.3f};  "
          f"th quantiles (10/50/90%): " + " ".join(f"{v:.2e}" for v in wquant(th, om, [0.1, 0.5, 0.9])))

    # heatmap layers x methods
    piv = d[(d.model == m) & (d.r == 600)].pivot(index="layer", columns="method", values="err_L4096")[METHODS]
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(np.clip(piv.values, 1e-4, None), aspect="auto", cmap="Blues", norm=LogNorm(),
                   interpolation="nearest")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, rotation=40, ha="right")
    ax.set_ylabel("layer"); ax.set_title(f"{m}: kernel error, L=4096, r=600 (omega-weighted)")
    fig.colorbar(im, ax=ax, label="relative error (log)")
    fig.tight_layout(); fig.savefig(f"{OUT}/err_heatmap_{m}.png", dpi=130); plt.close(fig)

    # 2D histogram of (th, lam)
    pos = th > 0
    xb = np.logspace(np.log10(max(th[pos].min(), 1e-8)), np.log10(max(th.max(), 2.0)), 120)
    yb = np.logspace(np.log10(LAM_FLOOR), np.log10(max(lam.max(), 1.0)) + 0.1, 100)
    fig, ax = plt.subplots(figsize=(10, 6))
    H, xe, ye, im = ax.hist2d(th[pos], np.maximum(lam[pos], LAM_FLOOR), bins=[xb, yb], weights=om[pos],
                              norm=LogNorm(), cmap="Blues")
    ax.set_xscale("log"); ax.set_yscale("log")
    for t in P["rope_theta"]:
        ax.axvline(t, ymin=0.97, ymax=1.0, color="#333333", lw=0.8)
    ax.set_xlabel("fitted th (rad / token)"); ax.set_ylabel(f"fitted lam (floored at {LAM_FLOOR:g})")
    ax.set_title(f"{m}: (th, lam) over all layers and (g,h), r=600, weighted by omega; ticks = RoPE theta_i "
                 f"({om[~pos].sum() / om.sum():.1%} weight at th<=0 not shown)", fontsize=9)
    fig.colorbar(im, ax=ax, label="omega (log)")
    fig.tight_layout(); fig.savefig(f"{OUT}/th_lam_hist_{m}.png", dpi=130); plt.close(fig)

s = pd.read_csv(f"{OUT}/sanity.csv")
print("\n== sanity ==")
print(f"max err freqfold_1: {s.max_err_freqfold_1.max():.3e}; max err nope_lowfreq_64: {s.max_err_nope_lowfreq_64.max():.3e}; "
      f"items with refined > init: {s.n_refined_gt_init.sum()}; skipped (g,h): {s.n_skipped.sum()}")
