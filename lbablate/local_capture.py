"""Capture FP-model k_proj inputs for layers 0-23: 64 calibration seqs (train_seqs[128:192]) and 8 held-out (train_seqs[:8]), fp32."""
import numpy as np, torch
from common import *
S, T = load_data(); fp = load_fp(torch.float32).cuda().eval()
torch.backends.cuda.matmul.allow_tf32 = False
for name, sl in (("tr", slice(128, 192)), ("ho", slice(0, 8))):
    cap = {l: [] for l in range(24)}
    hs = [fp.model.layers[l].self_attn.k_proj.register_forward_hook(lambda m, i, o, l=l: cap[l].append(i[0].detach().cpu())) for l in range(24)]
    with torch.no_grad():
        for i in range(sl.start, sl.stop):
            fp(input_ids=torch.from_numpy(S[i:i + 1]).long().cuda())
    for h in hs: h.remove()
    for l in range(24):
        np.save(f"{D}/runs/local/X{name}_{l}.npy", torch.cat(cap[l]).reshape(-1, 1024).numpy())
    print(name, "done", flush=True)
