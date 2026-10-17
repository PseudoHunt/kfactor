"""Token stream for the QAT ablation: WikiText-2 train (all) + C4 shard 0 (filling to 50M tokens), seq 2048, shuffled once.
The 20M-token run uses the first 9766 sequences of the same order. Also builds the WikiText-2 test windows (40 x 2048)."""
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer
SEQ, TOT = 2048, 50_000_000
tok = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
w_ids = tok("\n\n".join(wt["text"]), add_special_tokens=False).input_ids
print("wikitext train tokens", len(w_ids), flush=True)
c4 = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train")
need = TOT - len(w_ids); ids = []; n = 0
for i in range(0, len(c4), 2000):
    for t in tok(c4[i:i + 2000]["text"], add_special_tokens=False).input_ids:
        ids.extend(t); n += len(t)
    if n >= need: break
print("c4 tokens", n, flush=True)
def seqs(a): a = np.array(a[:len(a) // SEQ * SEQ], dtype=np.int32); return a.reshape(-1, SEQ)
S = np.concatenate([seqs(w_ids), seqs(ids[:need])])
S = S[np.random.RandomState(0).permutation(len(S))][:TOT // SEQ]
np.save("train_seqs.npy", S); print("train seqs", S.shape, flush=True)
te = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
t_ids = tok("\n\n".join(te["text"]), add_special_tokens=False).input_ids
T = seqs(t_ids)[:40]; print("test windows", T.shape, "of", len(t_ids) // SEQ, flush=True)
np.save("test_windows.npy", T)
