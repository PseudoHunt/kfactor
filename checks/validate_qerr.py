"""Checks for qerr.py: HF RoPE convention, closed-form err_q vs explicit Delta sums, aligned-query reduction."""
import sys; sys.path.insert(0, "/home/kfactor")
import numpy as np, pandas as pd, torch
import qerr as Q, pencil_fit as pf, dict_fit as dfit
from transformers.models.llama import modeling_llama as ml

rng = np.random.default_rng(0)
theta = pf.thetas(10000.0)
# (a) HF rotate_half convention: q_m . k_n == Re sum_i rho_i conj(z_i) exp(1j theta_i (n - m))... with Delta = m - n
q, k = rng.standard_normal(128), rng.standard_normal(128)
def rot(x, pos):
    ang = pos * torch.tensor(theta)
    emb = torch.cat([ang, ang])[None, None]
    xt = torch.tensor(x)[None, None, None]
    return ml.apply_rotary_pos_emb(xt, xt, emb.cos(), emb.sin())[0][0, 0, 0].numpy()
rho, z = q[:64] + 1j * q[64:], k[:64] + 1j * k[64:]
mx = 0
for m_, n_ in [(37, 5), (100, 100), (900, 3), (4095, 0)]:
    s_hf = rot(q, m_) @ rot(k, n_)
    D = m_ - n_
    s_c = np.real(np.sum(rho * np.conj(z) * np.exp(1j * theta * D)))
    mx = max(mx, abs(s_hf - s_c))
print(f"(a) HF score vs Re sum rho conj(z) e^(+i theta Delta): max diff {mx:.1e}")

# (b),(c) on Llama-2 layer 15
z, OM, gi, hi, th = Q.load_layer_z("llama2-7b", 15)
Mq = np.load(Q.M_PATH["llama2-7b"])["M"][15]; Mq = 0.5 * (Mq + np.conj(np.swapaxes(Mq, -1, -2)))
P = np.load(Q.POLES.format("llama2-7b"))
Dl = np.arange(Q.L, dtype=float); E = np.exp(1j * np.outer(th, Dl))            # (64, L)
items = rng.choice(len(z), 6, replace=False)
mx_free = mx_fix = 0
for it in items:
    Qm = Mq[hi[it]] * np.outer(np.conj(z[it]), z[it])
    den_exp = np.real(np.einsum("ij,id,jd->", Qm, E, np.conj(E)))
    E0 = Q.Emat(th, th); den_cf = np.real((Qm * E0).sum())
    # free poles m=2: explicit projection of each e_i
    p = -P["L15_W4096_m2_lam"][it].astype(float) + 1j * P["L15_W4096_m2_th"][it].astype(float)
    B = np.exp(np.outer(Dl, p))
    eps = E.T - B @ np.linalg.lstsq(B, E.T, rcond=None)[0]                      # (L, 64)
    num_exp = np.real(np.einsum("ij,di,dj->", Qm, eps, np.conj(eps)))
    C, G = Q.atom_terms(p, th)
    num_cf = den_cf - Q.proj_gain(G, C @ Qm @ C.conj().T)
    mx_free = max(mx_free, abs(num_exp / den_exp - num_cf / den_cf))
    # freqfold_8 explicit
    t8 = np.repeat(th.reshape(-1, 8).mean(1), 8)
    eps8 = (E - np.exp(1j * np.outer(t8, Dl))).T
    n8 = np.real(np.einsum("ij,di,dj->", Qm, eps8, np.conj(eps8)))
    R8 = E0 - Q.Emat(th, t8) - Q.Emat(t8, th) + Q.Emat(t8, t8)
    mx_fix = max(mx_fix, abs(n8 / den_exp - np.real((Qm * R8).sum()) / den_cf))
print(f"(b) err_q closed form vs explicit Delta sums: free m=2 max diff {mx_free:.1e}; freqfold_8 max diff {mx_fix:.1e}")
# (c) aligned query (Q = w w^T) reproduces stored energy-weighted errors
w = np.abs(z) ** 2; KK = dfit.KK_of(w, th, Q.L)
p = -P["L15_W4096_m2_lam"].astype(float) + 1j * P["L15_W4096_m2_th"].astype(float)
C, G = Q.atom_terms(p, th)
Xw = np.einsum("nsi,ni,nj,ntj->nst", C, w, w, np.conj(C))
e_al = (OM * (KK - Q.proj_gain(G, Xw)) / KK).sum() / OM.sum()
st = pd.read_csv("/home/kfactor/pencil_fit/pencil_errors.csv")
ref = st[(st.model == "llama2-7b") & (st.layer == 15) & (st.window == 4096) & (st.method == "pencil_2")].err.item()
print(f"(c) aligned-query err for free m=2, layer 15: {e_al:.6f} vs stored energy err {ref:.6f}")
# (d) greedy_q picks the best single atom exactly (m=1 brute force)
cent = np.load(Q.CENTERS)["llama2-7b_L15_D8"]
pext = np.concatenate([-cent[:, 1] + 1j * cent[:, 0], 1j * th]); Ce, Ge = Q.atom_terms(pext, th)
Qs = Mq[hi[items]] * (np.conj(z[items])[:, :, None] * z[items][:, None, :])
X = Q.sandwich(Ce[None], Qs)
S1 = Q.greedy_q(Ge, X, 1)[:, 0]
brute = np.array([np.argmax([np.real(X[i, a, a] / Ge[a, a]) for a in range(len(pext))]) for i in range(len(items))])
print(f"(d) greedy m=1 == brute-force best atom: {np.all(S1 == brute)}")
