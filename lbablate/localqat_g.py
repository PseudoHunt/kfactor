"""Layer-local k_proj QAT: minimise ||k_FP - k_student||^2 on the FP k_proj input (64 seqs x 2048), only the k_proj factors+scales
(both paths) train; bias frozen. Arms U/S/C from their Dual-SVID init; lr sweep; stop at plateau.
Usage: localqat.py <layers comma list> <out_csv>"""
import sys, time, csv, os
import numpy as np, torch, torch.nn as nn
from common import *
from quantization.modules import LittleBitLinear
from quantization.functions.binary import SmoothSign
torch.backends.cuda.matmul.allow_tf32 = False
LRS = [1e-4, 3e-4, 1e-3]; BATCH = 4096; EVAL_EVERY = 100; PATIENCE = 1000; MAXSTEP = 8000; TOL = 0.01
layers = [int(x) for x in sys.argv[1].split(",")]; outcsv = sys.argv[2]; ARMS = sys.argv[3].split(",")
fp = load_fp(torch.float32)
fcsv = open(outcsv, "a"); w = csv.writer(fcsv)
if os.path.getsize(outcsv) == 0: w.writerow(["layer", "arm", "lr", "init_mse", "best_mse", "ho_init_mse", "ho_best_mse", "var_y", "steps", "sec", "best_step"])

from localhelp import make

@torch.no_grad()
def mse(m, X, Y):
    s = 0.0
    for i in range(0, len(X), 32768):
        s += ((m(X[i:i + 32768]) - Y[i:i + 32768]) ** 2).sum().item()
    return s / Y.numel()

for l in layers:
    k = fp.model.layers[l].self_attn.k_proj; W, b = k.weight.data.cuda(), k.bias.data.cuda()
    Xtr = torch.from_numpy(np.load(f"{D}/runs/local/Xtr_{l}.npy")).cuda(); Xho = torch.from_numpy(np.load(f"{D}/runs/local/Xho_{l}.npy")).cuda()
    Ytr, Yho = Xtr @ W.t() + b, Xho @ W.t() + b
    vy = (Ytr ** 2).mean().item()
    for arm in ARMS:
        best_over_lr = None
        for lr in LRS:
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            m = make(arm, l, W.cpu(), b.cpu()); params = [p for p in m.parameters() if p.requires_grad]
            opt = torch.optim.Adam(params, lr=lr)
            m0 = mse(m, Xtr, Ytr); h0 = mse(m, Xho, Yho); best, best_step, best_sd = m0, 0, None
            t0 = time.time(); step = 0
            while step < MAXSTEP:
                for _ in range(EVAL_EVERY):
                    idx = torch.randint(0, len(Xtr), (BATCH,), device="cuda")
                    loss = ((m(Xtr[idx]) - Ytr[idx]) ** 2).mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
                cur = mse(m, Xtr, Ytr)
                if cur < best:
                    if cur < best * (1 - TOL): best_step = step
                    best = cur; best_sd = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
                if step - best_step >= PATIENCE: break
            sec = time.time() - t0
            if best_sd is not None:
                with torch.no_grad():
                    for n, p in m.named_parameters():
                        if n in best_sd: p.copy_(best_sd[n])
            hb = mse(m, Xho, Yho)
            w.writerow([l, arm, lr, m0, best, h0, hb, vy, step, sec, best_step]); fcsv.flush()
            print(f"L{l} {arm} lr {lr:g}: init {m0/vy:.4f} -> best {best/vy:.4f} rel (ho {hb/vy:.4f}) steps {step} {sec:.0f}s", flush=True)
            if best_over_lr is None or best < best_over_lr[0]:
                best_over_lr = (best, lr, {n: p.detach().cpu() for n, p in m.named_parameters() if n != "bias"} if best_sd is not None else None)
            del m, opt
        if best_over_lr[2] is not None: torch.save(best_over_lr[2], f"{D}/runs/local/{arm}_L{l}.pt")
        open(f"{D}/runs/local/best_lr_{arm}_L{l}.txt", "w").write(str(best_over_lr[1]))
