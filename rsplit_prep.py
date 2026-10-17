"""Band eigenvectors for the range-split test (CPU/GPU): for J in 16,32,48, top-600 eigenvectors of C restricted to rows
whose RoPE pair index >= J (row h*DH+j, pair = j mod HALF). Output rsplit/V_<model>_J<J>_L<l>.npy (n_band, 600) fp32.
With --vproj: second moment of v_proj outputs (calibration = first 64x2048 train tokens), top-600 eigvecs -> rsplit/VV_<model>_L<l>.npy.
Usage: python rsplit_prep.py <model_key> [--vproj <hf_id>]"""
import sys, numpy as np, torch
torch.set_num_threads(8)
key = sys.argv[1]
cdir = {"llama2-7b": "out_llama2-7b", "qwen1.5-7b": "out_qwen1.5-7b"}[key]
DH, H, HALF = 128, 32, 64
R = 600
if "--vproj" not in sys.argv:
    for l in range(2, 32):
        import os
        if all(os.path.exists(f'rsplit/V_{key}_J{J}_L{l}.npy') for J in (16,32,48)): continue
        C = torch.from_numpy(np.load(f"{cdir}/C/layer_{l}.npy").astype(np.float64)); C = 0.5 * (C + C.T)
        for J in (16, 32, 48):
            idx = np.array([h * DH + j for h in range(H) for j in range(DH) if j % HALF >= J])
            e, v = torch.linalg.eigh(C[idx][:, idx])
            np.save(f"rsplit/V_{key}_J{J}_L{l}.npy", torch.flip(v, [1])[:, :R].contiguous().float().numpy())
        print("layer", l, flush=True)
else:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = sys.argv[sys.argv.index("--vproj") + 1]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
    M = {l: torch.zeros(4096, 4096, dtype=torch.float32, device="cuda") for l in range(2, 32)}
    cnt = [0]
    def mk(l):
        def hook(mod, inp, out):
            o = out[0].float(); M[l] += o.T @ o
        return hook
    hs = [model.model.layers[l].self_attn.v_proj.register_forward_hook(mk(l)) for l in range(2, 32)]
    bos = tok.bos_token_id
    with torch.no_grad():
        for i in range(64):
            chunk = ids[i * 2048:(i + 1) * 2048]
            if bos is not None: chunk = torch.cat([torch.tensor([bos]), chunk[:-1]])
            model.model(input_ids=chunk[None].cuda(), use_cache=False)
    for l in range(2, 32):
        C = (M[l] / (64 * 2048)).double().cpu()
        e, v = torch.linalg.eigh(0.5 * (C + C.T))
        np.save(f"rsplit/VV_{key}_L{l}.npy", torch.flip(v, [1])[:, :R].contiguous().float().numpy())
        print("vproj layer", l, "energy kept", float(e[-R:].sum() / e.sum()), flush=True)
