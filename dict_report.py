"""Tables for dict_fit.py outputs."""
import os
import numpy as np, pandas as pd

OUT = "/home/kfactor/dict_fit"
f4 = lambda v: f"{v:.4f}"
pd.set_option("display.width", 220)

free = pd.read_csv("/home/kfactor/pencil_fit/pencil_errors.csv")
free = free[(free.window == 4096) & free.method.str.fullmatch(r"pencil_\d")]
free = free.assign(m=free.method.str[-1]).rename(columns={"err": "free_err"})[["model", "layer", "m", "free_err"]]

if os.path.exists(f"{OUT}/dict_errors.csv"):
    d = pd.read_csv(f"{OUT}/dict_errors.csv", dtype={"m": str})
    d = d.merge(free, on=["model", "layer", "m"], how="left")
    d["ratio"] = d.err / d.free_err
    rows = []
    for (model, D, m), g in d.groupby(["model", "D", "m"]):
        rows.append(dict(model=model, D=D, m=m, mean_err=g.err.mean(), median_err=g.err.median(),
                         layers_above_0_05=int((g.err > 0.05).sum()),
                         mean_worst_pair=g.worst_pair.mean(), median_worst_pair=g.worst_pair.median(),
                         ratio_of_means=g.err.mean() / g.free_err.mean() if m != "all" else np.nan,
                         median_layer_ratio=g.ratio.median() if m != "all" else np.nan))
    t = pd.DataFrame(rows)
    t["m_order"] = t.m.map({"1": 1, "2": 2, "4": 4, "all": 99})
    t = t.sort_values(["model", "D", "m_order"]).drop(columns="m_order")
    t.to_csv(f"{OUT}/dict_table.csv", index=False)
    for model, g in t.groupby("model"):
        print(f"\n== {model}: dictionary-constrained OMP, L=4096, r=600 (stats over layers; worst_pair = omega-weighted "
              f"mean over (g,h) of max residual over pairs with >1% energy) ==")
        fr = free[free.model == model].groupby("m").free_err.agg(["mean", "median"])
        print("free-pole reference: " + "; ".join(f"m={m}: mean {r['mean']:.4f} median {r['median']:.4f}" for m, r in fr.iterrows()))
        print(g.drop(columns="model").to_string(index=False, float_format=f4))
        m2 = g[(g.m == "2")]
        ok = m2[m2.ratio_of_means < 1.5]
        ok2 = m2[m2.median_layer_ratio < 1.5]
        print(f"-> smallest D with ratio_of_means < 1.5 at m=2: {ok.D.min() if len(ok) else 'none'}; "
              f"with median per-layer ratio < 1.5: {ok2.D.min() if len(ok2) else 'none'}")
        # worst layers at m=2 for the flagged D
        if len(ok):
            Dsel = ok.D.min()
            w = d[(d.model == model) & (d.D == Dsel) & (d.m == "2")].nlargest(5, "ratio")[["layer", "err", "free_err", "ratio", "worst_pair"]]
            print(f"   worst 5 layers by ratio at D={Dsel}, m=2:")
            print(w.to_string(index=False, float_format=f4))

if os.path.exists(f"{OUT}/qwen_L32768_errors.csv"):
    q = pd.read_csv(f"{OUT}/qwen_L32768_errors.csv")
    def channels(meth):
        k, v = meth.rsplit("_", 1)
        return {"freqfold": lambda: 128 // int(v), "nope_lowfreq": lambda: 2 * int(v), "pencil": lambda: 2 * int(v)}.get(
            k, lambda: 0)()
    q = q[~q.method.isin(["freqfold_1", "nope_lowfreq_64"])]
    tab = (q.groupby("method").agg(mean_err=("err", "mean"), median_err=("err", "median"),
                                   layers_above_0_05=("err", lambda x: int((x > 0.05).sum()))).reset_index())
    tab["channels"] = tab.method.map(channels)
    tab = pd.concat([tab, pd.DataFrame([dict(method="exact", channels=128, mean_err=0.0, median_err=0.0, layers_above_0_05=0)])])
    tab = tab.sort_values(["channels", "method"])[["method", "channels", "mean_err", "median_err", "layers_above_0_05"]]
    tab.to_csv(f"{OUT}/qwen_L32768_channels_table.csv", index=False)
    print(f"\n== qwen1.5-7b, window L=32768 (stride-8 pencil, r=600, {q.layer.nunique()} layers) ==")
    print(tab.to_string(index=False, float_format=f4))
    s = pd.read_csv(f"{OUT}/qwen_L32768_errors.csv")
    print(f"sanity: max item err freqfold_1 = {s[s.method=='freqfold_1'].max_item_err.max():.2e}; "
          f"nope_lowfreq_64 = {s[s.method=='nope_lowfreq_64'].max_item_err.max():.2e}")
    p = q[q.method.str.startswith("pencil")].pivot(index="layer", columns="method", values="err")
    print("layers with pencil_2 err > 0.05:", p.index[p.pencil_2 > 0.05].tolist(),
          "| pencil_1 > 0.05:", p.index[p.pencil_1 > 0.05].tolist())
