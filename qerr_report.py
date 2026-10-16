"""Tables and figures for qerr.py outputs."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = "/home/kfactor/qerr"
f4 = lambda v: f"{v:.4f}"
pd.set_option("display.width", 240); pd.set_option("display.max_colwidth", 200)

d = pd.read_csv(f"{OUT}/qerr_errors.csv")
CH = {"free_m1": 2, "free_m2": 4, "free_m4": 8, "dict_D8_m2": 4, "dict_D32_m2": 4, "dict_D64_m2": 4,
      "ext_energy_m2": 4, "ext_q_m2": 4, "ext_energy_m4": 8, "ext_q_m4": 8,
      "freqfold_8": 16, "freqfold_4": 32, "freqfold_2": 64,
      "nope_lowfreq_8": 16, "nope_lowfreq_16": 32, "nope_lowfreq_32": 64, "exact": 128}
main = d[d.method.isin(CH)].copy(); main["channels"] = main.method.map(CH)

# 1. err_q table
t = (main.groupby(["model", "method"]).agg(channels=("channels", "first"),
     energy_mean=("err_energy", "mean"), energy_median=("err_energy", "median"),
     energy_layers_gt_0_05=("err_energy", lambda x: int((x > 0.05).sum())),
     errq_mean=("err_q", "mean"), errq_median=("err_q", "median"),
     errq_layers_gt_0_05=("err_q", lambda x: int((x > 0.05).sum()))).reset_index())
order = list(CH)
t["o"] = t.method.map(order.index); t = t.sort_values(["model", "channels", "o"]).drop(columns="o")
t.to_csv(f"{OUT}/errq_table.csv", index=False)
for m, g in t.groupby("model"):
    print(f"\n== {m}: energy-weighted vs q-weighted kernel error, L=4096, r=600 (32 layers) ==")
    print(g.drop(columns="model").to_string(index=False, float_format=f4))

# 2. RoPE atom picks
pk = pd.read_csv(f"{OUT}/rope_picks.csv")
fcols = [f"f{i}" for i in range(64)]
print("\n== extended dictionary (8 learned + 64 RoPE atoms): omega-weighted fraction of selected atoms that are RoPE atoms ==")
by = pk.pivot_table(index=["model", "layer"], columns=["objective", "m"], values="frac_rope")
print(by.groupby("model").agg(["mean", "min", "max"]).T.to_string(float_format=f4))
by.to_csv(f"{OUT}/rope_picks_by_layer.csv")
for m in pk.model.unique():
    print(f"\n{m}: frac_rope by layer (q objective, m=2): " +
          " ".join(f"{l}:{v:.2f}" for l, v in by.loc[m][("q", 2)].items()))
    fr = pk[(pk.model == m)].groupby(["objective", "m"])[fcols].mean()
    for (obj, mm), row in fr.iterrows():
        v = row.values; top = np.argsort(-v)[:8]
        print(f"   {obj} m={mm}: share of all selected atoms by RoPE index -> pairs 0-15 {v[:16].sum():.3f}, "
              f"16-31 {v[16:32].sum():.3f}, 32-47 {v[32:48].sum():.3f}, 48-63 {v[48:].sum():.3f}; top: "
              + ", ".join(f"{i}:{v[i]:.3f}" for i in top))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, obj in zip(axes, ("energy", "q")):
        A = pk[(pk.model == m) & (pk.objective == obj) & (pk.m == 2)].sort_values("layer")[fcols].values
        im = ax.imshow(np.clip(A, 1e-5, None), aspect="auto", cmap="Blues", norm=LogNorm(vmin=1e-4, vmax=1),
                       interpolation="nearest")
        ax.set_title(f"{m}: RoPE-atom picks, {obj} objective, m=2"); ax.set_xlabel("RoPE frequency index i (0 = highest)")
    axes[0].set_ylabel("layer"); fig.colorbar(im, ax=axes, label="omega-weighted share of selected atoms")
    fig.savefig(f"{OUT}/rope_picks_{m}.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# 3 & 5. per-head
hd = pd.read_csv(f"{OUT}/qerr_heads.csv")
hd["errq_best_ext_m2"] = hd[["errq_ext_energy_m2", "errq_ext_q_m2"]].min(1)
hd.to_csv(f"{OUT}/qerr_heads.csv", index=False)
qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
print("\n== per-head err_q distribution (omega-weighted over the head's directions), quantiles over all layers x heads ==")
cols = ["errq_free_m2", "errq_dict_D8_m2", "errq_ext_energy_m2", "errq_ext_q_m2", "errq_ext_q_m4", "errq_best_ext_m2"]
print(hd.groupby("model")[cols].quantile(qs).to_string(float_format=f4))
print("\n== heads with err_q > 0.1 (best extended config, m=2) ==")
for m, g in hd.groupby("model"):
    bad = g[g.errq_best_ext_m2 > 0.1]
    print(f"{m}: {len(bad)} of {len(g)} heads")
    for l, gg in bad.groupby("layer"):
        print(f"   L{l} ({len(gg)}): " + " ".join(f"h{h}" for h in gg["head"]))
    good = g[g.errq_best_ext_m2 <= 0.1]
    rk = lambda x: x.rank().values
    rho = np.corrcoef(rk(g.errq_best_ext_m2), rk(g.q_share_pairs0_15))[0, 1]
    print(f"   query energy share in pairs 0-15: flagged heads mean {bad.q_share_pairs0_15.mean():.3f} "
          f"(median {bad.q_share_pairs0_15.median():.3f}) vs other heads mean {good.q_share_pairs0_15.mean():.3f} "
          f"(median {good.q_share_pairs0_15.median():.3f}); Spearman(err_q, share) over all heads = {rho:.3f}")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(g.q_share_pairs0_15, g.errq_best_ext_m2, s=8, color="#2a6fb0", alpha=0.6)
    ax.set_yscale("log"); ax.axhline(0.1, color="#888888", lw=1, ls="--")
    ax.set_xlabel("query energy share in pairs 0-15"); ax.set_ylabel("err_q, best extended config, m=2")
    ax.set_title(f"{m}: per-head err_q vs high-frequency query share ({len(g)} heads)")
    fig.tight_layout(); fig.savefig(f"{OUT}/head_errq_vs_qshare_{m}.png", dpi=130); plt.close(fig)

# 4. Qwen rebuild
rb = d[d.method.str.startswith("rebuild_")].copy()
if len(rb):
    rb["src_m"] = rb.method.str.extract(r"src(\d)").astype(int); rb["D"] = rb.method.str.extract(r"_D(\d+)_").astype(int)
    old = d[d.method.str.fullmatch(r"dict_D\d+_m2")].copy(); old["D"] = old.method.str.extract(r"_D(\d+)_").astype(int); old["src_m"] = 1
    fm = d[d.method == "free_m2"][["model", "layer", "err_energy", "err_q"]].rename(columns={"err_energy": "fe", "err_q": "fq"})
    allr = pd.concat([old[old.model == "qwen1.5-7b"], rb]).merge(fm, on=["model", "layer"])
    tb = allr.groupby(["src_m", "D"]).agg(energy_mean=("err_energy", "mean"), energy_median=("err_energy", "median"),
                                          energy_layers_gt_0_05=("err_energy", lambda x: int((x > 0.05).sum())),
                                          errq_mean=("err_q", "mean"), errq_median=("err_q", "median"),
                                          free_energy_mean=("fe", "mean"), free_errq_mean=("fq", "mean")).reset_index()
    tb["ratio_energy"] = tb.energy_mean / tb.free_energy_mean; tb["ratio_errq"] = tb.errq_mean / tb.free_errq_mean
    tb.to_csv(f"{OUT}/qwen_dict_rebuild.csv", index=False)
    print("\n== Qwen1.5-7B dictionary rebuilt from m=2 / m=4 free poles, OMP m=2 (src_m=1 = previous dictionary, D=8/32/64 only) ==")
    print(tb.to_string(index=False, float_format=f4))

s = pd.read_csv(f"{OUT}/sanity.csv")
print(f"\n== sanity == max |err_q| freqfold_1: {s.max_abs_freqfold_1.max():.1e}; nope_lowfreq_64: {s.max_abs_nope_64.max():.1e}; "
      f"most negative err_q over all items/methods: {s.max_neg_errq.min():.1e}")
