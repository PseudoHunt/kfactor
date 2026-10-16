"""Attention-mass profile by offset: per layer and head, mass_h(Delta) = (1/N_queries) sum_s A_h[s, s-Delta]
(the first-order KL weight sum_{s,t} A dS^2 aggregated by offset), plus pair counts so the per-pair mean is
mass * N_queries / count. Computed with exact keys, fp32 softmax, on wikitext-2 train (calibration split).

Usage: python attn_profile.py <hf_model_id> <seq_len> <n_seq> <out_npz>
"""
import sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = False
model_id, n, nseq, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
dev = "cuda"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(dev).eval()
cfg = model.config
H = cfg.num_attention_heads; DH = getattr(cfg, "head_dim", None) or cfg.hidden_size // H; NL = cfg.num_hidden_layers
from transformers.models.llama import modeling_llama as ml

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
ids = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
bos = tok.bos_token_id
assert ids.numel() >= n * nseq
mass = torch.zeros(NL, H, n, dtype=torch.float64, device=dev)
mass_ns = torch.zeros(NL, H, n, dtype=torch.float64, device=dev)   # excluding sink keys t < 4
SINK = 4
s_idx = torch.arange(n, device=dev)
dmat = (s_idx[:, None] - s_idx[None, :]).clamp(min=0)            # offset matrix (masked entries have zero weight)
causal = torch.ones(n, n, dtype=torch.bool, device=dev).tril()

def prof_forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kw):
    x = hidden_states[0]; m = x.shape[0]
    q = self.q_proj(x).float().view(m, H, DH).transpose(0, 1)[None]
    k = self.k_proj(x).float().view(m, H, DH).transpose(0, 1)[None]
    v = self.v_proj(x).float().view(m, H, DH)
    cos, sin = position_embeddings
    q, k = ml.apply_rotary_pos_emb(q, k, cos.float(), sin.float())
    out = torch.empty(m, H, DH, device=dev)
    for h in range(H):
        s = (q[0, h] @ k[0, h].T) * self.scaling
        a = torch.softmax(s.masked_fill(~causal, float("-inf")), -1)
        mass[self._layer, h].index_add_(0, dmat.reshape(-1), a.reshape(-1).double())
        mass_ns[self._layer, h].index_add_(0, dmat[:, SINK:].reshape(-1), a[:, SINK:].reshape(-1).double())
        out[:, h] = a @ v[:, h]
    return self.o_proj(out.reshape(1, m, H * DH).to(hidden_states.dtype)), None

import types
for l in range(NL):
    a = model.model.layers[l].self_attn; a._layer = l; a.forward = types.MethodType(prof_forward, a)
with torch.no_grad():
    for i in range(nseq):
        chunk = ids[i * n:(i + 1) * n]
        if bos is not None:
            chunk = torch.cat([torch.tensor([bos]), chunk[:-1]])
        model.model(input_ids=chunk[None].to(dev), use_cache=False)
mass = (mass / (nseq * n)).cpu().numpy()                          # per query: sum_Delta mass = 1
mass_ns = (mass_ns / (nseq * n)).cpu().numpy()
count = (n - np.arange(n)).astype(np.float64) * nseq               # number of (s, s-Delta) pairs per offset
np.savez(out, mass=mass, mass_nosink=mass_ns, count=count, n=n, nseq=nseq)
tot = mass.sum(-1)
print(f"{model_id}: n={n} nseq={nseq}; mass sums per (layer,head) in [{tot.min():.6f}, {tot.max():.6f}]")
for l in range(NL):
    c = np.cumsum(mass[l].mean(0))
    print(f"  L{l:2d}: mass at Delta<16 {c[15]:.3f}  <64 {c[63]:.3f}  <256 {c[255]:.3f}  on sink tokens t<4 {1 - mass_ns[l].sum(-1).mean():.3f}")
print("wrote", out)
