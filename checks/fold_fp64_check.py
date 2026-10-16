"""fp64 check of the fold unit test: all-64-RoPE-atom latent fold (P = I) vs real RoPE, one window, a few heads.
If the fp32 gap in fold_eval's unit test is pure accumulation roundoff, it vanishes here."""
import sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import modeling_llama as ml

model_id, sel, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
n = 512
dev = "cuda"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(dev).eval()
cfg = model.config
H = cfg.num_attention_heads; DH = cfg.hidden_size // H; HALF = DH // 2
theta = cfg.rope_parameters["rope_theta"] ** (-2.0 * np.arange(HALF) / DH)
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0][:n]
cap = {}
hk = model.model.layers[layer].self_attn.register_forward_pre_hook(lambda m, a, kw: cap.update(x=kw["hidden_states"]), with_kwargs=True)
with torch.no_grad():
    model.model(input_ids=ids[None].to(dev), use_cache=False)
hk.remove()
a = model.model.layers[layer].self_attn
x = cap["x"][0]
with torch.no_grad():
    q = a.q_proj(x).double().view(n, H, DH); k = a.k_proj(x).double()
V = torch.from_numpy(np.load(f"{sel}/V/layer_{layer}.npy")).to(dev).double()
c = k @ V; khat = c @ V.T
pos = torch.arange(n, dtype=torch.float64, device=dev)
ang = pos[:, None] * torch.as_tensor(theta, device=dev)[None]
cos = torch.cat([ang.cos(), ang.cos()], -1)[None]; sin = torch.cat([ang.sin(), ang.sin()], -1)[None]
qr, kr = ml.apply_rotary_pos_emb(q.transpose(0, 1)[None], khat.view(n, H, DH).transpose(0, 1)[None], cos, sin)
rho = torch.complex(q[..., :HALF], q[..., HALF:])
r = V.shape[1]
ph = torch.exp(1j * ang - 1j * ang[n // 2:n // 2 + 1] * 0)          # e^{i theta s}
worst = scale = 0.0
for h in range(0, H, 8):
    zc = torch.complex(V[h * DH:h * DH + HALF].T, V[h * DH + HALF:(h + 1) * DH].T)            # (r, 64)
    Qp = (rho[:, h] * ph)[:, None, :] * torch.conj(zc)[None]                               # (n, r, 64): P = I
    Kp = c[:, :, None].to(torch.complex128) * torch.conj(ph)[:, None, :]
    s_fold = (Qp.reshape(n, -1) @ Kp.reshape(n, -1).T).real
    s_std = qr[0, h] @ kr[0, h].T
    m = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev))
    worst = max(worst, (s_fold - s_std)[m].abs().max().item()); scale = max(scale, s_std[m].abs().max().item())
print(f"fp64 layer {layer}, heads 0,8,16,24, n={n}: max|fold(P=I) - realRoPE| {worst:.2e} (max|score| {scale:.2e}, rel {worst / scale:.1e})")
