"""PPL (40 x 2048 WikiText-2 test windows) and per-layer k_proj output MSE vs FP for an arm, at init or from a checkpoint.
Usage: eval_arm.py <FP|U|S|C> <out_json> [ckpt.pt]"""
import sys, json, torch
from common import *
arm, outp = sys.argv[1], sys.argv[2]
ck = sys.argv[3] if len(sys.argv) > 3 else None
S, T = load_data()
fp = load_fp(torch.float32).cuda().eval()
res = {"arm": arm, "ckpt": ck}
if arm == "FP":
    res["ppl_bf16"] = eval_ppl(fp, T); res["ppl_fp32"] = eval_ppl(fp, T, dtype=torch.float32)
    fp16 = load_fp(torch.float16).cuda().eval(); res["ppl_fp16"] = eval_ppl(fp16, T, dtype=torch.float16)
else:
    m = build_student(arm)
    if ck:
        sd = torch.load(ck); named = dict(m.named_parameters())
        assert set(sd) == {n for n in named if "embed" not in n and "lm_head" not in n}, "param mismatch"
        with torch.no_grad():
            for n, v in sd.items(): named[n].copy_(v.cuda())
    res["ppl_bf16"] = eval_ppl(m, T)
    # layer-local k_proj error: same (FP-model) input to FP and student k_proj, fp32, 8 calibration sequences
    X = torch.from_numpy(S[:8]).long().cuda(); cap = {}
    hs = [fp.model.layers[l].self_attn.k_proj.register_forward_hook(lambda mod, i, o, l=l: cap.__setitem__(l, i[0].detach())) for l in range(24)]
    with torch.no_grad(): fp(input_ids=X)
    for h in hs: h.remove()
    mse, rel = [], []
    with torch.no_grad():
        for l in range(24):
            x = cap[l]; yf = fp.model.layers[l].self_attn.k_proj(x); ys = m.model.layers[l].self_attn.k_proj(x).float()
            mse.append(((ys - yf) ** 2).mean().item()); rel.append((((ys - yf) ** 2).mean() / (yf ** 2).mean()).item())
    res["kproj_mse"], res["kproj_relmse"] = mse, rel
    print("rel k_proj MSE by layer:", " ".join(f"{v:.3f}" for v in rel))
print(json.dumps({k: v for k, v in res.items() if not k.startswith("kproj")}))
json.dump(res, open(outp, "w"))
