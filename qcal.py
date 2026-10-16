"""Query calibration: per layer, per head, M_h = E[rho rho^H] with rho_i = q[i] + 1j q[i+64] (pre-RoPE q_proj output).

Usage: python qcal.py <hf_model_id> <out_npz>
"""
import sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

N_SEQ, SEQ_LEN = 64, 2048
model_id, out_path = sys.argv[1], sys.argv[2]
dev = "cuda"

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(dev).eval()
cfg = model.config
n_h = cfg.num_attention_heads
d_h = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_h
half = d_h // 2
n_layers = cfg.num_hidden_layers
print(f"{model_id}: layers={n_layers} n_h={n_h} n_kv={cfg.num_key_value_heads} d_h={d_h}")

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
ids = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
bos = tok.bos_token_id

M = [torch.zeros(n_h, half, half, dtype=torch.complex128, device=dev) for _ in range(n_layers)]
calls = [0] * n_layers

def mk_hook(l):
    def hook(mod, args, out):
        q = out.reshape(-1, n_h, d_h).float()                 # q_proj output incl. bias, before RoPE
        rho = torch.complex(q[..., :half], q[..., half:])      # (T, n_h, 64)
        M[l] += torch.einsum("thi,thj->hij", rho, rho.conj()).to(torch.complex128)
        calls[l] += 1
    return hook

hooks = [model.model.layers[l].self_attn.q_proj.register_forward_hook(mk_hook(l)) for l in range(n_layers)]
N = 0
with torch.no_grad():
    for i in range(N_SEQ):
        chunk = ids[i * SEQ_LEN:(i + 1) * SEQ_LEN]
        if bos is not None:
            chunk = torch.cat([torch.tensor([bos]), chunk[:-1]])
        model(chunk[None].to(dev), use_cache=False)
        N += chunk.numel()
for h in hooks:
    h.remove()
assert all(c == N_SEQ for c in calls)
Mall = torch.stack(M).cpu().numpy() / N                        # (layers, heads, 64, 64)
herm = np.abs(Mall - np.conj(np.swapaxes(Mall, -1, -2))).max() / np.abs(Mall).max()
mineig = min(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min() / np.linalg.eigvalsh(0.5 * (m + m.conj().T)).max()
             for m in Mall.reshape(-1, half, half))
print(f"tokens={N}; max rel Hermitian asymmetry {herm:.2e}; worst min/max eig {mineig:.2e}")
np.savez(out_path, M=Mall, tokens=N)
print("wrote", out_path)
