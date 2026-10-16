# kfactor

Training-free probes of how much structure the RoPE key path of a transformer has, measured on the
calibrated key second moment `C = W_K G W_K^T` (pre-RoPE keys; `G = E[x x^T]` over 64 x 2048 wikitext-2 tokens).

Models: Llama-2-7b (`NousResearch/Llama-2-7b-hf`, RoPE base 1e4), Qwen1.5-7B (base 1e6), Qwen1.5-0.5B (pilot).
Both 7B models are MHA with 32 KV heads, `d_h = 128`, so `out = 4096` and 64 RoPE frequency pairs per head.
Everything after step 1 runs on CPU from the saved per-layer `C` matrices.

## Experiments

| step | script | question |
|---|---|---|
| 1 | `kfactor.py` | Can `W_K` be factorized per RoPE frequency pair (block-diagonal over pairs, across heads) without losing much energy vs an unstructured rank-r factorization? Also a group-size sweep (g adjacent pairs per block). |
| 2 | `kernel_fit.py` (+ `kernel_report.py`) | For the top-r eigendirections of `C`, approximate each head's positional kernel `K(Δ) = Σ_i w_i e^{iθ_i Δ}` by one damped exponential fitted with LM, vs frequency folding and NoPE-style baselines. **Superseded by step 3** (20 LM steps were not converged; see below). |
| 3 | `pencil_fit.py` (+ `pencil_report.py`) | Closed-form matrix-pencil (ESPRIT) fit of `K(Δ)` with m damped poles, m ∈ {1,2,4,8}, at L=4096 (stride 4, alias-branch choice) and L=512. |
| 4 | `dict_fit.py dict` (+ `dict_report.py`) | Per-layer shared pole dictionary (weighted k-means of the free poles, D ∈ {8..128}); per (g,h) OMP selection of m atoms; per-pair projection residuals. |
| 5 | `dict_fit.py long` | Qwen only: free-pole refit at L=32768 (stride 8). |

`checks/` holds the validation scripts (LM convergence check, closed-form vs explicit cross-checks).

## Headline results (r = 600, ω-weighted per-layer error, stats over 32 layers)

**Step 1 — pair-structured factorization** (ratio = structured / unstructured energy kept at the same rank):

| model | r=300 | r=600 | r=1112 | r=1624 |
|---|---|---|---|---|
| Llama-2-7b mean ratio | 0.873 | 0.900 | 0.928 | 0.949 |
| Qwen1.5-7B mean ratio | 0.817 | 0.854 | 0.891 | 0.919 |

Only Llama-2 layers 0–2 exceed 0.95 at r=600; no Qwen layer does. Allocation concentrates on low-frequency pairs
(76.7% / 68.9% of kept rank in the low-frequency half for Llama-2 / Qwen).

**Step 3 — matrix-pencil kernel fit, L=4096** (channels = real channels per direction per head):

| method | channels | Llama-2 mean / median | layers > 0.05 | Qwen mean / median | layers > 0.05 |
|---|---|---|---|---|---|
| pencil m=1 | 2 | 0.063 / 0.054 | 22 | 0.045 / 0.027 | 6 |
| pencil m=2 | 4 | 0.046 / 0.037 | 2 | 0.033 / 0.022 | 5 |
| pencil m=4 | 8 | 0.031 / 0.024 | 2 | 0.021 / 0.014 | 2 |
| pencil m=8 | 16 | 0.017 / 0.012 | 2 | 0.009 / 0.006 | 0 |
| freqfold_8 | 16 | 0.353 / 0.283 | 32 | 0.247 / 0.145 | 32 |
| freqfold_2 | 64 | 0.113 / 0.084 | 32 | 0.102 / 0.056 | 23 |
| nope_lowfreq_32 | 64 | 1.58 / 1.57 | 32 | 0.049 / 0.044 | 11 |
| exact | 128 | 0 | 0 | 0 | 0 |

The layers no pencil fits (Llama-2 layers 0–1) are exactly the layers where the pair-structured factorization
works (step 1 ratio > 0.95). At L=512, Qwen's `nope_lowfreq_32` is essentially exact (0.0008): with base 1e6 its
low-frequency half barely rotates in 512 tokens.

**Step 4 — shared pole dictionary + OMP, L=4096, m=2** (ratio = dictionary error / free-pole error):

| D | Llama-2 mean err | ratio | Qwen mean err | ratio |
|---|---|---|---|---|
| 8 | 0.061 | 1.35 | 0.044 | 1.32 |
| 32 | 0.051 | 1.11 | 0.042 | 1.26 |
| 128 | 0.042 | 0.93 | 0.039 | 1.17 |

A dictionary of 8 poles per layer is already within 1.5x of free poles at m=2 for both models.
**Caveat:** the selected atoms reproduce the ω-weighted kernel, not the individual RoPE pair rotations: the worst
significant pair (>1% energy) has projection residual ≈ 0.98–0.999 in every configuration. The kernel being fitted is
the q-aligned part; this does not by itself license replacing per-pair RoPE for arbitrary queries.

**Step 5 — Qwen1.5-7B free poles at L=32768** (stride-8 pencil with alias-branch choice):

| method | channels | L=4096 mean / median | L=32768 mean / median | layers > 0.05 at 32768 |
|---|---|---|---|---|
| pencil m=1 | 2 | 0.045 / 0.027 | 0.090 / 0.066 | 28 |
| pencil m=2 | 4 | 0.033 / 0.022 | 0.067 / 0.051 | 17 |
| freqfold_2 | 64 | 0.102 / 0.056 | 0.200 / 0.135 | 32 |
| nope_lowfreq_32 | 64 | 0.049 / 0.044 | 0.382 / 0.323 | 32 |

Errors roughly double going from 4k to 32k for the pencil; the NoPE-low-frequency shortcut that was nearly free for
Qwen at short range breaks down at 32k (its low-frequency pairs do rotate over that window).

**Step 6 — query-weighted kernel error** (`qcal.py`, `qerr.py`, `qerr_report.py`). Per-head query second moments
`M_h = E[ρρ^H]` (pre-RoPE `q_proj` output, 64 complex pairs) weight the kernel error:
`err_q = Σ Q∘R / Σ Q∘E`, `Q = M_h ∘ (conj(z) z^T)` (verified against HF `apply_rotary_pos_emb`). With real queries the
error is 2.5–3x the aligned-query error (free m=2 poles: 0.124 vs 0.046 on Llama-2). Hard heads (err_q > 0.1) carry
~2x more query energy in pairs 0–15 (Spearman 0.80 / 0.88).

**Step 7 — training-free perplexity with the fold implemented** (`fold_select.py`, `fold_eval.py`, `fold_report.py`).
WikiText-2 test, layers 0–1 uncompressed, keys of layers 2–31 replaced by the rank-600 latent `c_t = k_t V_r`.
Unit test: all 64 RoPE atoms with P = I reproduces real RoPE to 5e-14 (fp64) and PPL to 2.5e-5 (fp32).

| config | Llama-2 @2048 | Qwen @2048 | Qwen @8192 |
|---|---|---|---|
| H full model | 5.817 | 7.552 | 7.306 |
| A truncation only (k̂ → real RoPE) | 5.987 (+2.9%) | 8.249 (+9.2%) | 8.095 (+10.8%) |
| F freqfold_8 | 8.506 | 29.19 | 64.18 |
| G nope_lowfreq_32 | 21.07 | 8.303 | 8.866 |
| B RoPE atoms, uniform q-objective, m=2 | 121.4 | 233.1 | 229.2 |
| C same, m=4 | 95.7 | 217.0 | 174.9 |
| D 8 poles + 64 RoPE atoms, m=4 | 107.4 | 206.7 | 202.3 |
| E as C, hard heads m=16 | 93.5 | 181.9 | (running) |

The fitted configs B–E fail because the selection objective was uniform over Δ: total score error is 1.6–2.8%, but it
concentrates at small Δ (230–420% relative at Δ=0, 76–142% at Δ=1–3) where attention mass sits; the substitution
baselines F/G are exact at Δ=0 by construction. err_q (uniform over Δ) does not predict PPL.

**Step 8 — attention-mass-weighted, Δ=0-constrained fold, and an exact recent window** (`attn_profile.py`,
`fold_select2.py`, `fold_report2.py`). Selection and P minimize the attention-mass-weighted score error with
Σ_k P_ki = 1 (exact at Δ=0); window configs keep the last W keys and the first 4 (sink) tokens exact and fit the fold on
Δ ≥ W with sink-excluded mass. Sinks hold 40% (Llama-2) / 47% (Qwen) of attention mass in layers 2–31.

| config (ΔPPL vs H) | Llama-2 @2048 | Qwen @2048 | Qwen @8192 |
|---|---|---|---|
| A truncation only | +0.170 | +0.697 | +0.790 |
| **AW64 truncation + exact window 64 + sink** | **+0.030** | **+0.300** | **+0.460** |
| C2 weighted + constraint, m=4 | +1.43 | +34.3 | +105.1 |
| C2W16 | +1.09 | +1.48 | +7.90 |
| **C2W64** | **+0.206** | **+0.506** | **+4.86** |
| D2W64 (8 poles + 64 RoPE atoms) | +0.227 | +0.489 | +4.83 |
| FW64 freqfold_8 + window | +1.81 | +0.94 | +4.22 |
| GW64 nope_lowfreq_32 + window | +13.10 | +0.376 | **+1.12** |

Verdict against the pre-registered criteria: the weighting fixes the Δ=0 failure (Llama-2 C 95.7 → C2 7.25 PPL) but C2
alone is not within 2x of A; with an exact 64-token window the fold is within 1.2x of A on Llama-2 @2048 and beats
FW64/GW64 there by 9x/64x, but on Qwen it loses to GW64 at 2048 and to both FW64 and GW64 at 8192, where its error grows
with context. The fitted long-range kernel does not beat frequency substitution in the regime it was meant for. The
positive result is the simplest one: rank-600 key truncation with an exact 64-token window and 4 sink tokens costs
+0.5% (Llama-2 @2048), +4.0% (Qwen @2048) and +6.3% (Qwen @8192) PPL at 4.3–4.8x key-cache compression, training-free.
The profile-weighted score error tracks PPL across all 16 configs (Spearman 0.73 / 0.84 / 0.85).

## Important caveats

- `kernel_fit.py` used 20 LM iterations, which were **not converged** (80 and 200 iterations agree to 1e-4 and are
  2–5x better on the checked layers). Its fitted-method numbers are superseded by the matrix-pencil results.
- Pencil error is non-increasing in m at the layer level, but ~7–10% of individual (g,h) kernels get worse at some
  step up in m (pencil poles are not nested and Hankel truncation is not LS-optimal).
- Calibration is a single dataset (wikitext-2), 131k tokens, sequences of 2048.

## Reproducing

```
python kfactor.py NousResearch/Llama-2-7b-hf out_llama2-7b      # GPU, fp16; writes out_*/C/layer_{l}.npy
python kfactor.py Qwen/Qwen1.5-7B out_qwen1.5-7b
python pencil_fit.py --lm-steps 0 && python pencil_report.py    # CPU
python dict_fit.py dict && python dict_fit.py long && python dict_report.py
```

Paths are currently hard-coded to `/home/kfactor`. The per-layer `C` matrices (~2 GB per model) and the large
per-(g,h) parameter/pole archives are not committed; everything else under `out_*`, `kernel_fit/`, `pencil_fit/`,
`dict_fit/` is.
