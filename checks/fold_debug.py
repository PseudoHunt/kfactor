"""CPU fp64: fold scores for a selected-atom config vs real RoPE on k_hat (config A), one layer, short window."""
import sys
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(4)
model_id, sel, cfgname, layer, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
Wwin = int(cfgname.split("W")[1]) if "W" in cfgname else 0
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
cfg = model.config; H = cfg.num_attention_heads; DH = cfg.hidden_size // H; HALF = DH // 2
theta = cfg.rope_parameters["rope_theta"] ** (-2.0 * np.arange(HALF) / DH)
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0][:n]
cap = {}
hk = model.model.layers[layer].self_attn.register_forward_pre_hook(lambda m, a, kw: cap.update(x=kw["hidden_states"]), with_kwargs=True)
class Stop(Exception): pass
def stop(m, a, kw): raise Stop
hk2 = model.model.layers[layer].self_attn.register_forward_pre_hook(lambda m, a, kw: (_ for _ in ()).throw(Stop()) if "x" in cap else None, with_kwargs=True)
try:
    with torch.no_grad(): model.model(input_ids=ids[None], use_cache=False)
except Stop: pass
a = model.model.layers[layer].self_attn
with torch.no_grad():
    x = cap["x"][0]; q = a.q_proj(x).double().view(n, H, DH); k = a.k_proj(x).double()
V = torch.from_numpy(np.load(f"{sel}/V/layer_{layer}.npy")).double()
c = k @ V; khat = (c @ V.T).view(n, H, DH)
rho = torch.complex(q[..., :HALF], q[..., HALF:]); kap = torch.complex(khat[..., :HALF], khat[..., HALF:])
kx = k.view(n, H, DH); kapx = torch.complex(kx[..., :HALF], kx[..., HALF:])
pos = torch.arange(n, dtype=torch.float64)
z = np.load(f"{sel}/{cfgname}/layer_{layer}.npz")
p = torch.from_numpy(z["p"])
e_s = torch.exp(pos[:, None] * p[None]); e_t = torch.exp(-pos[:, None] * p[None])
eA = torch.exp(1j * pos[:, None] * torch.from_numpy(theta)[None])
m = torch.tril(torch.ones(n, n, dtype=torch.bool))
tot_err = tot_sig = 0.0
D = pos[:, None] - pos[None, :]
bins = [(0, 0), (1, 3), (4, 15), (16, 63), (64, 255), (256, 10**9)]
be = np.zeros(len(bins)); bs = np.zeros(len(bins)); bx = np.zeros(len(bins)); bsx = np.zeros(len(bins))
win = (D < Wwin) | (pos[None, :] < 4)
for h in range(H):
    sA = ((rho[:, h] * eA) @ (torch.conj(kap[:, h]) * torch.conj(eA)).T).real
    U = torch.from_numpy(z[f"h{h}_U"]).to(torch.complex128); g = torch.from_numpy(z[f"h{h}_g"]).long(); kk = torch.from_numpy(z[f"h{h}_k"]).long()
    sB = (((rho[:, h] @ U) * e_s[:, kk]) @ (c[:, g].to(torch.complex128) * e_t[:, kk]).T).real
    sX = ((rho[:, h] * eA) @ (torch.conj(kapx[:, h]) * torch.conj(eA)).T).real
    if Wwin:
        sB = torch.where(win, sX, sB)
    for bi, (lo, hi_) in enumerate(bins):
        mb = (D >= lo) & (D <= hi_)
        be[bi] += ((sB - sA)[mb] ** 2).sum().item(); bs[bi] += (sA[mb] ** 2).sum().item()
        bx[bi] += ((sB - sX)[mb] ** 2).sum().item(); bsx[bi] += (sX[mb] ** 2).sum().item()
    d = (sB - sA)[m]; tot_err += (d ** 2).sum().item(); tot_sig += (sA[m] ** 2).sum().item()
    if h < 4:
        print(f"h{h}: rel score err {((d**2).sum() / (sA[m]**2).sum()).item():.3e}; diag (Delta=0) rel err "
              f"{((torch.diagonal(sB - sA) ** 2).sum() / (torch.diagonal(sA) ** 2).sum()).item():.3e}; max|U| {U.abs().max().item():.2e}")
print("rel score error vs X (exact keys) by Delta bin: " + "  ".join(f"[{lo},{'max' if hi_ > 10**8 else hi_}] {e_ / s_:.3f}" for (lo, hi_), e_, s_ in zip(bins, bx, bsx)))
print("rel score error vs A (k_hat)     by Delta bin: " + "  ".join(f"[{lo},{'max' if hi_ > 10**8 else hi_}] {e_ / s_:.3f}" for (lo, hi_), e_, s_ in zip(bins, be, bs)))
print(f"{cfgname} layer {layer}: total relative score error (all heads, causal, n={n}) {tot_err / tot_sig:.4e}")
