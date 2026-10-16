"""Phase-2 report: PPL for all configs (old A-H + new weighted/windowed), profile-weighted score error, cache cost."""
import os, numpy as np, pandas as pd

D = "/home/kfactor/fold"
RUNS = [("llama2-7b", 2048, ["ppl_llama2_2048.csv", "ppl2_llama2_2048.csv"], "sel_llama2-7b_L2048"),
        ("qwen1.5-7b", 2048, ["ppl_qwen_2048.csv", "ppl2_qwen_2048.csv"], "sel_qwen1.5-7b_L2048"),
        ("qwen1.5-7b", 8192, ["ppl_qwen_8192.csv", "ppl2_qwen_8192.csv"], "sel_qwen1.5-7b_L8192")]
ORDER = ["H", "X", "A", "AW64", "B", "C", "D", "E", "F", "FW64", "G", "GW64", "B2", "C2", "D2", "C2W16", "C2W64", "D2W64"]
DESC = {"H": "full model", "X": "custom fp32 attn, exact keys", "A": "truncation only", "AW64": "truncation + exact window 64",
        "B": "RoPE atoms, uniform q-obj, m=2", "C": "RoPE atoms, uniform q-obj, m=4", "D": "8 poles+64 RoPE, uniform, m=4",
        "E": "as C, hard heads m=16", "F": "freqfold_8", "FW64": "freqfold_8 + window 64", "G": "nope_lowfreq_32",
        "GW64": "nope_lowfreq_32 + window 64", "B2": "RoPE atoms, weighted+constraint, m=2", "C2": "RoPE atoms, weighted+constraint, m=4",
        "D2": "8 poles+64 RoPE, weighted+constraint, m=4", "C2W16": "C2 + window 16", "C2W64": "C2 + window 64", "D2W64": "D2 + window 64"}
tab = pd.DataFrame({"config": ORDER, "description": [DESC[k] for k in ORDER]}).set_index("config")
corr = []
for model, ctx, files, sel in RUNS:
    col = f"{model}@{ctx}"
    ps = [pd.read_csv(f"{D}/{f}") for f in files if os.path.exists(f"{D}/{f}")]
    if not ps:
        continue
    p = pd.concat(ps).drop_duplicates("config", keep="last").set_index("config").ppl
    tab[f"{col} PPL"] = p.reindex(ORDER).round(4)
    tab[f"{col} dPPL"] = (p - p["H"]).reindex(ORDER).round(4)
    if os.path.exists(f"{D}/{sel}/werr.csv"):
        w = pd.read_csv(f"{D}/{sel}/werr.csv")
        we = {c[5:]: w[c].mean() for c in w if c.startswith("werr_")}
        we.update({"H": 0.0, "X": 0.0})
        tab[f"{col} werr"] = pd.Series(we).reindex(ORDER).round(5)
        both = pd.DataFrame({"ppl": p, "werr": pd.Series(we)}).dropna().drop(index=["H", "X"], errors="ignore")
        rk = lambda s: s.rank()
        corr.append(dict(setting=col, n_configs=len(both), spearman=np.corrcoef(rk(both.ppl), rk(both.werr))[0, 1]))
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
tab.to_csv(f"{D}/fold_ppl_table2.csv")
print(tab.to_string())
print("\nwerr = profile-weighted score error (fraction of attention-mass-weighted score energy lost, pooled over all key "
      "directions incl. truncation), mean over layers 2-31, at the eval window.")
if corr:
    print("\nSpearman(werr, PPL) across configs:\n" + pd.DataFrame(corr).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n== key-cache cost (keys only; layers 0-1 full; layers 2-31 latent r=600 + exact window W + 4 sink tokens) ==")
rows = []
for n in (2048, 8192, 32768):
    for W in (0, 16, 64):
        full = 32 * 4096 * n
        comp = 2 * 4096 * n + 30 * (600 * n + (4096 * (W + 4) if W else 0))
        rows.append(dict(context=n, W=W, key_cache_fraction=comp / full, compression=full / comp,
                         window_overhead_vs_latent=(4096 * (W + 4) / (600 * n)) if W else 0.0))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
