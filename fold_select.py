"""Atom selection + projection matrices for the fold PPL eval (CPU). Selection uses the calibration-set query
second moments M (wikitext train); window L should match the eval context.

Per layer (2..31), per (g,h) with g over the top-r eigenvectors of C:
  B: RoPE atoms only, q-greedy, m=2      C: RoPE atoms only, q-greedy, m=4
  D: 8 learned (D=8 dict) + 64 RoPE atoms, q-greedy, m=4
  E: as C, but heads with err_q(C) > 0.1 get m=16
  F/G (freqfold_8 / nope_lowfreq_32) need no selection; only their err_q is computed here.
Saves per config and layer: for each head the term list (g_j, atom_j) and U[:, j] = P_{atom_j,:} * conj(z_{g_j,h,:}),
plus the atom exponents p_k (f_k(Delta) = exp(p_k Delta)) and V_r.

Usage: python fold_select.py <model> <L> <out_dir> [--layers 2,..] [--workers 8]
"""
import os, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import pencil_fit as pf
import dict_fit as dfit
import qerr as qe

R, CHUNK, PINV, COLLINEAR = 600, 600, 1e-10, 1e-10
M_MAX_ROPE, M_MAX_EXT = 16, 4
HEAD_THRESH = 0.1
gsum = dfit.gsum

def emat(a, b, L):
    return gsum(1j * (a[:, None] - b[None, :]), L)

def atoms(p, theta, L):
    C = gsum(np.conj(p)[:, None] + 1j * theta[None, :], L)          # (A, 64)  b_k^H e_i
    G = gsum(np.conj(p)[:, None] + p[None, :], L)                   # (A, A)
    return C, G

def greedy(G, X, m_max):
    """Incremental greedy maximizing Tr(G_SS^+ X_SS); G (A,A) shared, X (n,A,A). Returns S (n, m_max)."""
    n, A = X.shape[:2]
    rows = np.arange(n)[:, None]
    gd = np.real(np.diag(G))
    S = np.zeros((n, 0), dtype=np.int64)
    for k in range(m_max):
        if k == 0:
            inc = np.real(np.einsum("naa->na", X)) / gd
        else:
            GSS = G[S[:, :, None], S[:, None, :]]
            W = np.linalg.pinv(GSS, rcond=PINV, hermitian=True) @ G[S]                  # (n, k, A)
            XSd = X[rows, S]                                                            # (n, k, A)  X[S_k, d]
            t2 = (np.conj(W) * XSd).sum(1)
            XSS = X[rows[:, :, None], S[:, :, None], S[:, None, :]]
            t3 = np.real((np.conj(W) * (XSS @ W)).sum(1))
            beta = gd - np.real((np.conj(G[S]) * W).sum(1))
            inc = (np.real(np.einsum("naa->na", X)) - 2 * np.real(t2) + t3) / np.maximum(beta, 1e-300)
            inc[beta < COLLINEAR * gd] = -np.inf
            inc[rows, S] = -np.inf
        S = np.concatenate([S, inc.argmax(1)[:, None]], 1)
    return S

def proj_terms(S, C, G, X, z):
    """P = G_SS^+ C_S per item; returns gain Re Tr(G^+ X_SS) and U (n, m, 64) = P * conj(z)."""
    rows = np.arange(len(S))[:, None, None]
    Gp = np.linalg.pinv(G[S[:, :, None], S[:, None, :]], rcond=PINV, hermitian=True)
    gain = np.real(np.einsum("nij,nji->n", Gp, X[rows, S[:, :, None], S[:, None, :]]))
    P = Gp @ C[S]
    return gain, P * np.conj(z)[:, None, :]

def run_layer(task):
    model, l, L, out = task
    z, OM, gi, hi, theta = qe.load_layer_z(model, l)
    # V_r exactly as used for z (saved for the GPU so eigenvector signs match)
    cdir, _ = pf.MODELS[model]
    Cm = np.load(f"{cdir}/layer_{l}.npy").astype(np.float64)
    e, V = np.linalg.eigh(0.5 * (Cm + Cm.T)); V = V[:, ::-1][:, :R]
    assert np.allclose(V.T.reshape(R, pf.N_KV, pf.D_H)[gi, hi, :pf.HALF] + 1j * V.T.reshape(R, pf.N_KV, pf.D_H)[gi, hi, pf.HALF:], z)
    Mq = np.load(qe.M_PATH[model])["M"][l]; Mq = 0.5 * (Mq + np.conj(np.swapaxes(Mq, -1, -2)))
    cent = np.load(qe.CENTERS)[f"{model}_L{l}_D8"]
    p_rope = 1j * theta
    p_ext = np.concatenate([-cent[:, 1] + 1j * cent[:, 0], p_rope])
    Cr, Gr = atoms(p_rope, theta, L); Ce, Ge = atoms(p_ext, theta, L)
    E0 = emat(theta, theta, L)
    fixed = {"F": np.repeat(theta.reshape(-1, 8).mean(1), 8), "G": np.where(np.arange(pf.HALF) < 32, theta, 0.0)}
    Rfix = {k: E0 - emat(theta, t, L) - emat(t, theta, L) + emat(t, t, L) for k, t in fixed.items()}
    n = len(z)
    err = {k: np.empty(n) for k in ("B", "C", "D", "C16", "F", "G")}
    Sr = np.empty((n, M_MAX_ROPE), dtype=np.int64); Se = np.empty((n, M_MAX_EXT), dtype=np.int64)
    U = {k: np.empty((n, m, pf.HALF), dtype=np.complex64) for k, m in (("B", 2), ("C", 4), ("D", 4), ("C16", 16))}
    for c0 in range(0, n, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        zc = z[sl]
        Q = Mq[hi[sl]] * (np.conj(zc)[:, :, None] * zc[:, None, :])
        den = np.real((Q * E0).sum((1, 2)))
        for k, Rm in Rfix.items():
            err[k][sl] = np.real((Q * Rm).sum((1, 2))) / den
        Xr = Cr[None] @ Q @ np.conj(Cr.T)[None]
        Xe = Ce[None] @ Q @ np.conj(Ce.T)[None]
        Sr[sl] = greedy(Gr, Xr, M_MAX_ROPE); Se[sl] = greedy(Ge, Xe, M_MAX_EXT)
        for k, S, Cc, Gc, X in (("B", Sr[sl, :2], Cr, Gr, Xr), ("C", Sr[sl, :4], Cr, Gr, Xr),
                                ("C16", Sr[sl], Cr, Gr, Xr), ("D", Se[sl], Ce, Ge, Xe)):
            gain, Uk = proj_terms(S, Cc, Gc, X, zc)
            err[k][sl] = 1 - gain / den
            U[k][sl] = Uk
    wm = lambda v, msk=slice(None): float((OM[msk] * v[msk]).sum() / OM[msk].sum())
    head_errC = np.array([wm(err["C"], hi == h) for h in range(pf.N_KV)])
    head_m = np.where(head_errC > HEAD_THRESH, 16, 4)
    errE = np.where(head_m[hi] == 16, err["C16"], err["C"])
    # save per config: per head the term arrays
    sel = {"B": (Sr[:, :2], U["B"], p_rope), "C": (Sr[:, :4], U["C"], p_rope), "D": (Se, U["D"], p_ext)}
    for cfg, (S, Uc, p) in sel.items():
        arrs = {"p": p.astype(np.complex128)}
        for h in range(pf.N_KV):
            msk = hi == h
            arrs[f"h{h}_g"] = np.repeat(gi[msk], S.shape[1]).astype(np.int32)
            arrs[f"h{h}_k"] = S[msk].ravel().astype(np.int32)
            arrs[f"h{h}_U"] = Uc[msk].reshape(-1, pf.HALF).T.copy()               # (64, J)
        np.savez(f"{out}/{cfg}/layer_{l}.npz", **arrs)
    arrs = {"p": p_rope.astype(np.complex128), "head_m": head_m}
    for h in range(pf.N_KV):
        msk = hi == h
        m = head_m[h]; Uc = (U["C16"] if m == 16 else U["C"])[msk]; S = Sr[msk, :m]
        arrs[f"h{h}_g"] = np.repeat(gi[msk], m).astype(np.int32)
        arrs[f"h{h}_k"] = S.ravel().astype(np.int32)
        arrs[f"h{h}_U"] = Uc.reshape(-1, pf.HALF).T.copy()
    np.savez(f"{out}/E/layer_{l}.npz", **arrs)
    np.save(f"{out}/V/layer_{l}.npy", V.astype(np.float32))
    row = dict(model=model, layer=l, L=L, **{f"errq_{k}": wm(err[k]) for k in ("B", "C", "D", "F", "G")},
               errq_E=wm(errE), errq_C16=wm(err["C16"]), heads_m16=int((head_m == 16).sum()))
    heads = [dict(model=model, layer=l, L=L, head=h, errq_C=head_errC[h], m_E=int(head_m[h])) for h in range(pf.N_KV)]
    print(f"{model} L{l} (L={L}) done: " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k.startswith("errq")) +
          f" m16 heads={row['heads_m16']}", flush=True)
    return row, heads

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("L", type=int); ap.add_argument("out")
    ap.add_argument("--layers", default=",".join(map(str, range(2, 32)))); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for d in ("B", "C", "D", "E", "V"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    tasks = [(a.model, int(l), a.L, a.out) for l in a.layers.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_layer, tasks))
    pd.DataFrame([r[0] for r in res]).to_csv(f"{a.out}/errq.csv", index=False)
    pd.DataFrame([h for r in res for h in r[1]]).to_csv(f"{a.out}/head_m.csv", index=False)
    print("wrote", a.out)
