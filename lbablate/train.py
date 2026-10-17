"""LittleBit KD-QAT: loss = KL(student||teacher logits, 'batchmean' as in their KDTrainer) + 10 * sum_layers MSE(hidden).
Usage: train.py <arm U|S|C> <tokens> <lr> <tag> [--micro 2] [--accum 4]"""
import sys, time, csv, math, json
import numpy as np, torch, torch.nn.functional as F
from common import *

arm, tokens, lr, tag = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
micro = int(sys.argv[sys.argv.index("--micro") + 1]) if "--micro" in sys.argv else 2
accum = int(sys.argv[sys.argv.index("--accum") + 1]) if "--accum" in sys.argv else 4
maxsteps = int(sys.argv[sys.argv.index("--maxsteps") + 1]) if "--maxsteps" in sys.argv else None
SEQ = 2048; bsz = micro * accum
steps = int(tokens // (SEQ * bsz)); steps = min(steps, maxsteps) if maxsteps else steps
S, T = load_data()
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
if seed: S = S[np.random.RandomState(seed).permutation(len(S))]
assert steps * bsz <= len(S)
out = f"{D}/runs/{tag}"; os.makedirs(out, exist_ok=True)
torch.manual_seed(seed)
student = build_student(arm, seed=seed)
teacher = load_fp(torch.bfloat16).cuda().eval()
for p in teacher.parameters(): p.requires_grad = False
student.config.use_cache = False
student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
student.enable_input_require_grads()
params = [p for p in student.parameters() if p.requires_grad]
print(f"arm {arm}: {sum(p.numel() for p in params)/1e6:.1f}M trainable, steps {steps}, bsz {bsz} seqs ({bsz*SEQ} tok), lr {lr}", flush=True)
opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
warm = max(1, int(round(0.02 * steps)))
lr_at = lambda s: lr * (s + 1) / warm if s < warm else lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
log = open(f"{out}/loss.csv", "w"); w = csv.writer(log); w.writerow(["step", "tokens", "kd", "l2l", "lr", "gnorm", "sec"])
t0 = time.time()
for s in range(steps):
    for g in opt.param_groups: g["lr"] = lr_at(s)
    kd_s = l2_s = 0.0
    student.train()
    for a in range(accum):
        x = torch.from_numpy(S[s * bsz + a * micro: s * bsz + (a + 1) * micro]).long().cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            to = teacher(input_ids=x, output_hidden_states=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            so = student(input_ids=x, output_hidden_states=True)
        kd = F.kl_div(F.log_softmax(so.logits.float(), -1), F.softmax(to.logits.float(), -1), reduction="batchmean")
        l2 = 10.0 * sum(F.mse_loss(a_.float(), b_.float()) for a_, b_ in zip(so.hidden_states[1:], to.hidden_states[1:]))
        ((kd + l2) / accum).backward()
        kd_s += kd.item() / accum; l2_s += l2.item() / accum
        del to, so, kd, l2
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
    opt.step(); opt.zero_grad(set_to_none=True)
    w.writerow([s + 1, (s + 1) * bsz * SEQ, kd_s, l2_s, lr_at(s), gn, time.time() - t0]); log.flush()
    if s % 20 == 0 or s == steps - 1:
        print(f"step {s+1}/{steps} kd {kd_s:.4f} l2l {l2_s:.4f} gn {gn:.2f} lr {lr_at(s):.2e} {time.time()-t0:.0f}s", flush=True)
    if not math.isfinite(kd_s + l2_s): print("NONFINITE LOSS, abort"); sys.exit(2)
del teacher; torch.cuda.empty_cache()
ppl = eval_ppl(student, T)
print(f"FINAL {tag} PPL {ppl:.4f}", flush=True)
json.dump(dict(tag=tag, arm=arm, tokens=steps * bsz * SEQ, lr=lr, ppl=ppl, steps=steps, bsz=bsz), open(f"{out}/final.json", "w"))
torch.save({k: p.detach().cpu() for k, p in student.named_parameters() if "embed" not in k and "lm_head" not in k}, f"{out}/student.pt")
