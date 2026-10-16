"""Final PPL table for the fold eval: configs A-H (+X), PPL and delta vs H, with mean err_q per config."""
import os
import pandas as pd

D = "/home/kfactor/fold"
RUNS = [("llama2-7b", 2048, "ppl_llama2_2048.csv", "sel_llama2-7b_L2048"),
        ("qwen1.5-7b", 2048, "ppl_qwen_2048.csv", "sel_qwen1.5-7b_L2048"),
        ("qwen1.5-7b", 8192, "ppl_qwen_8192.csv", "sel_qwen1.5-7b_L8192")]
NAMES = {"A": "A  truncation only (k_hat -> real RoPE)", "B": "B  RoPE atoms, q-obj, m=2", "C": "C  RoPE atoms, q-obj, m=4",
         "D": "D  8 poles + 64 RoPE atoms, q-obj, m=4", "E": "E  as C, hard heads m=16", "F": "F  freqfold_8",
         "G": "G  nope_lowfreq_32", "H": "H  full model (fp16 sdpa)", "X": "X  custom fp32 attention, exact keys"}
rows = {k: {"config": v} for k, v in NAMES.items()}
for model, ctx, f, sel in RUNS:
    col = f"{model}@{ctx}"
    if not os.path.exists(f"{D}/{f}"):
        continue
    p = pd.read_csv(f"{D}/{f}").set_index("config").ppl
    e = pd.read_csv(f"{D}/{sel}/errq.csv")
    errq = {k: e[f"errq_{k}"].mean() for k in "BCDEFG"}
    m16 = int(e.heads_m16.sum())
    for k in NAMES:
        if k in p:
            rows[k][f"{col} PPL"] = round(p[k], 4)
            rows[k][f"{col} dPPL"] = round(p[k] - p["H"], 4)
            rows[k][f"{col} err_q"] = round(errq[k], 4) if k in errq else (0.0 if k in "AHX" else None)
    rows["E"]["config"] = NAMES["E"] + f" ({m16} heads @ {col})" if rows["E"]["config"] == NAMES["E"] else rows["E"]["config"] + f", {m16} @ {col}"
t = pd.DataFrame([rows[k] for k in "ABCDEFGHX"])
t.to_csv(f"{D}/fold_ppl_table.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(t.to_string(index=False))
print("\nerr_q = omega-weighted q-weighted kernel error averaged over compressed layers 2-31, at the eval window L; "
      "A/H/X have no kernel approximation (A's error is key truncation only).")
