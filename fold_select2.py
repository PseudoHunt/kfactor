"""Profile-weighted, Delta=0-constrained atom selection + projection (CPU), and the profile-weighted score-error
metric for every config (old and new).

Per head h, weights w_h(Delta) = attention-mass profile over the fit range [W, L). With e_i = exp(1j theta_i Delta),
atoms b_k = exp(p_k Delta), weighted sums  Ew = sum w e e^H-type (as Emat),  G = sum w conj(b) b,  C = sum w conj(b) e.
Constrained projection per pair i: min ||e_i - B P_i||_w^2 s.t. sum_k P_ki = 1. For query weighting Q:
  num(S) = den - Re Tr(G^-1 X) + (v^H X v - 2 Re v^H u + q0) / gamma,
  X = C_S Q C_S^H, u = C_S Q 1, q0 = 1^T Q 1, v = G^-1 1, gamma = 1^T v.
Metric (pooled over all 4096 key directions, profile-weighted over [0, L), exact for Delta < W):
  (sum_{g<=r} e_g num_g + sum_{g>r} e_g den_g[Delta>=W]) / sum_all e_g den_g.

Usage: python fold_select2.py <model> <L> <profile_npz> <sel_dir> [--layers ..] [--workers 8]
"""
import os, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import pencil_fit as pf
import qerr as qe

R, CHUNK, PINV, COLL = 600, 600, 1e-10, 1e-10
SINK = 4
NEW = {"B2": (0, "rope", 2), "C2": (0, "rope", 4), "D2": (0, "ext", 4),
       "C2W16": (16, "rope", 4), "C2W64": (64, "rope", 4), "D2W64": (64, "ext", 4)}
OLD = {"B": 2, "C": 4, "D": 4, "E": None}

def wsums(p_a, p_b, w, D):
    """sum_Delta w conj(exp(p_a D)) exp(p_b D) for D in range; p complex vectors."""
    ea = np.exp(np.outer(D, p_a)); eb = np.exp(np.outer(D, p_b))
    return (np.conj(ea) * w[:, None]).T @ eb

def constrained_num(G, X, u, q0, den):
    """num for constrained projection; G, X (..., m, m), u (..., m)."""
    Gp = np.linalg.pinv(G, rcond=PINV, hermitian=True)
    one = np.ones(G.shape[-1])
    v = Gp @ one
    gamma = np.real(v.sum(-1))
    tr = np.real(np.einsum("...ij,...ji->...", Gp, X))
    pen = (np.real(np.einsum("...i,...ij,...j->...", np.conj(v), X, v)) - 2 * np.real((np.conj(v) * u).sum(-1)) + q0) / gamma
    return den - tr + pen

def greedy_c(Gh, Xn, un, q0, den, m_max):
    """Greedy forward selection minimizing constrained num. Gh (n, A, A) per item (head's Gram)."""
    n, A = Xn.shape[:2]
    S = np.zeros((n, 0), dtype=np.int64)
    r1 = np.arange(n)[:, None, None, None]
    for k in range(m_max):
        Sc = np.concatenate([np.repeat(S[:, None, :], A, 1), np.broadcast_to(np.arange(A), (n, A))[:, :, None]], 2)
        G = Gh[r1, Sc[..., :, None], Sc[..., None, :]]
        X = Xn[r1, Sc[..., :, None], Sc[..., None, :]]
        u = un[np.arange(n)[:, None, None], Sc]
        val = constrained_num(G, X, u, q0[:, None], den[:, None])
        gd = np.real(np.einsum("naa->na", Gh))
        if k:
            # reject candidates (nearly) collinear with the selected set
            Gs = Gh[np.arange(n)[:, None, None], S[:, :, None], S[:, None, :]]
            Wt = np.linalg.pinv(Gs, rcond=PINV, hermitian=True) @ Gh[np.arange(n)[:, None], S]
            beta = gd - np.real((np.conj(Gh[np.arange(n)[:, None], S]) * Wt).sum(1))
            val[beta < COLL * gd] = np.inf
            val[np.arange(n)[:, None], S] = np.inf
        S = np.concatenate([S, val.argmin(1)[:, None]], 1)
    return S

def constrained_P(G, C):
    Gp = np.linalg.pinv(G, rcond=PINV, hermitian=True)
    v = Gp @ np.ones(G.shape[-1]); gamma = np.real(v.sum(-1))
    mu = ((np.conj(v)[..., None] * C).sum(-2) - 1) / gamma[..., None]           # (n, 64)  (v^H C - 1)/gamma
    return Gp @ C - v[..., :, None] * mu[..., None, :]

def run_layer(task):
    model, l, L, prof_path, sel = task
    cdir, base = pf.MODELS[model]
    theta = pf.thetas(base)
    Cm = np.load(f"{cdir}/layer_{l}.npy").astype(np.float64)
    e_all, V_all = np.linalg.eigh(0.5 * (Cm + Cm.T)); e_all, V_all = e_all[::-1], V_all[:, ::-1]
    Vs = np.load(f"{sel}/V/layer_{l}.npy").astype(np.float64)
    assert np.allclose(Vs, V_all[:, :R], atol=1e-6), "eigenvectors differ from the saved V_r"
    V_all[:, :R] = Vs
    zA = V_all.T.reshape(-1, pf.N_KV, pf.D_H); zA = zA[..., :pf.HALF] + 1j * zA[..., pf.HALF:]   # (4096, h, 64)
    Mq = np.load(qe.M_PATH[model])["M"][l]; Mq = 0.5 * (Mq + np.conj(np.swapaxes(Mq, -1, -2)))
    prof = np.load(prof_path)["mass"][l]                                                         # (h, n_prof)
    assert prof.shape[1] >= L
    D = np.arange(L, dtype=np.float64)
    cent = np.load(qe.CENTERS)[f"{model}_L{l}_D8"]
    p_rope = 1j * theta
    p_ext = np.concatenate([-cent[:, 1] + 1j * cent[:, 0], p_rope])
    pe = 1j * theta
    sub = {"F": 1j * np.repeat(theta.reshape(-1, 8).mean(1), 8), "G": 1j * np.where(np.arange(pf.HALF) < 32, theta, 0.0)}

    # per-head weighted sums over full range [0, L) and window ranges [W, L)
    def head_sums(W):
        Dw = D[W:]
        out = []
        for h in range(pf.N_KV):
            w = prof[h, W:L]
            d = {"Ew": wsums(pe, pe, w, Dw).T}                                  # Ew[i,i'] = sum w e_i conj(e_i')
            for nm, p in (("rope", p_rope), ("ext", p_ext)):
                d[f"G_{nm}"] = wsums(p, p, w, Dw); d[f"C_{nm}"] = wsums(p, pe, w, Dw)
            for nm, ps in sub.items():                                           # residual Gram of substitution
                Rm = wsums(pe, pe, w, Dw) - wsums(pe, ps, w, Dw) - wsums(ps, pe, w, Dw) + wsums(ps, ps, w, Dw)
                d[f"R_{nm}"] = Rm.T
            out.append(d)
        return out
    sums = {W: head_sums(W) for W in (0, 16, 64)}
    # uniform, unconstrained sums for the old configs' P
    unif = {"G_rope": wsums(p_rope, p_rope, np.ones(L), D), "C_rope": wsums(p_rope, pe, np.ones(L), D),
            "G_ext": wsums(p_ext, p_ext, np.ones(L), D), "C_ext": wsums(p_ext, pe, np.ones(L), D)}

    # T_h = M o Ew^T so that den_g = z^H T z  (full range and each window range)
    def dens(W):
        out = np.empty((zA.shape[0], pf.N_KV))
        for h in range(pf.N_KV):
            T = Mq[h] * sums[W][h]["Ew"].T
            out[:, h] = np.real(np.einsum("gi,ij,gj->g", np.conj(zA[:, h]), T, zA[:, h]))
        return out
    den_full, den_w = dens(0), {W: dens(W) for W in (16, 64)}
    tot = (e_all[:, None] * den_full).sum()
    trunc = {0: (e_all[R:, None] * den_full[R:]).sum(), **{W: (e_all[R:, None] * den_w[W][R:]).sum() for W in (16, 64)}}

    # per (g,h) for g < R
    gi, hi = np.meshgrid(np.arange(R), np.arange(pf.N_KV), indexing="ij"); gi, hi = gi.ravel(), hi.ravel()
    z = zA[gi, hi]; eg = e_all[gi]
    n = len(gi)
    num = {k: np.zeros(n) for k in list(NEW) + ["A", "AW64", "F", "FW64", "G", "GW64"] + list(OLD)}
    saves = {k: {"S": np.zeros((n, NEW[k][2]), dtype=np.int64), "U": np.zeros((n, NEW[k][2], pf.HALF), dtype=np.complex64)} for k in NEW}
    old_S = {}
    for k in OLD:
        zz = np.load(f"{sel}/{k}/layer_{l}.npz")
        old_S[k] = [zz[f"h{h}_k"] for h in range(pf.N_KV)]
    for h in range(pf.N_KV):
        it = np.nonzero(hi == h)[0]
        zh = z[it]
        Q = Mq[h] * (np.conj(zh)[:, :, None] * zh[:, None, :])
        q0 = np.real(Q.sum((1, 2)))
        Q1 = Q.sum(2)                                                               # Q 1
        for nm in ("F", "G"):
            num[nm][it] = np.real((Q * sums[0][h][f"R_{nm}"]).sum((1, 2)))
            num[nm + "W64"][it] = np.real((Q * sums[64][h][f"R_{nm}"]).sum((1, 2)))
        # old configs: uniform unconstrained P, evaluated under the profile-weighted metric (full range)
        Ew0 = sums[0][h]["Ew"]
        for k, mm in OLD.items():
            kk = old_S[k][h]
            m_h = len(kk) // len(it)
            S = kk.reshape(len(it), m_h)
            nmA = "ext" if k == "D" else "rope"
            Gu, Cu = unif[f"G_{nmA}"], unif[f"C_{nmA}"]
            P = np.linalg.pinv(Gu[S[:, :, None], S[:, None, :]], rcond=PINV, hermitian=True) @ Cu[S]   # (n, m, 64)
            Gw, Cw = sums[0][h][f"G_{nmA}"], sums[0][h][f"C_{nmA}"]
            Gs, Cs = Gw[S[:, :, None], S[:, None, :]], Cw[S]
            # K = Ew' - C^H P - P^H C + P^H G P  (in the Ew^T orientation used by Q o Ew)
            Kmat = Ew0.T - np.conj(np.swapaxes(Cs, 1, 2)) @ P - np.conj(np.swapaxes(P, 1, 2)) @ Cs + np.conj(np.swapaxes(P, 1, 2)) @ Gs @ P
            num[k][it] = np.real((Q * np.swapaxes(Kmat, 1, 2)).sum((1, 2)))
        for c0 in range(0, len(it), CHUNK):
            sl = slice(c0, c0 + CHUNK); Qc = Q[sl]
            for cfg, (W, nmA, m) in NEW.items():
                Gw, Cw = sums[W][h][f"G_{nmA}"], sums[W][h][f"C_{nmA}"]
                X = Cw[None] @ Qc @ np.conj(Cw.T)[None]
                u = (Cw[None] @ Q1[sl][:, :, None])[:, :, 0]
                denW = np.real((Qc * sums[W][h]["Ew"]).sum((1, 2)))
                Gh = np.broadcast_to(Gw, (len(Qc),) + Gw.shape)
                S = greedy_c(Gh, X, u, q0[sl], denW, m)
                Gs = Gw[S[:, :, None], S[:, None, :]]; Xs = X[np.arange(len(S))[:, None, None], S[:, :, None], S[:, None, :]]
                us = u[np.arange(len(S))[:, None], S]
                num[cfg][it[sl]] = constrained_num(Gs, Xs, us, q0[sl], denW)
                P = constrained_P(Gs, Cw[S])
                saves[cfg]["S"][it[sl]] = S
                saves[cfg]["U"][it[sl]] = P * np.conj(zh[sl])[:, None, :]
    # metric
    row = dict(model=model, layer=l, L=L, trunc_share=trunc[0] / tot)
    for k, v in num.items():
        W = 64 if k.endswith("W64") else 16 if k.endswith("W16") else 0
        row[f"werr_{k}"] = ((eg * v).sum() + trunc[W]) / tot
    row["werr_A"] = trunc[0] / tot
    row["werr_AW64"] = trunc[64] / tot
    for cfg, (W, nmA, m) in NEW.items():
        p = p_ext if nmA == "ext" else p_rope
        arrs = {"p": p.astype(np.complex128)}
        for h in range(pf.N_KV):
            it = np.nonzero(hi == h)[0]
            arrs[f"h{h}_g"] = np.repeat(gi[it], m).astype(np.int32)
            arrs[f"h{h}_k"] = saves[cfg]["S"][it].ravel().astype(np.int32)
            arrs[f"h{h}_U"] = saves[cfg]["U"][it].reshape(-1, pf.HALF).T.copy()
        os.makedirs(f"{sel}/{cfg}", exist_ok=True)
        np.savez(f"{sel}/{cfg}/layer_{l}.npz", **arrs)
    print(f"{model} L{l} (L={L}) done: " + " ".join(f"{k[5:]}={v:.4f}" for k, v in row.items() if k.startswith("werr")), flush=True)
    return row

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("L", type=int); ap.add_argument("profile"); ap.add_argument("sel")
    ap.add_argument("--layers", default=",".join(map(str, range(2, 32)))); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tasks = [(a.model, int(l), a.L, a.profile, a.sel) for l in a.layers.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run_layer, tasks))
    pd.DataFrame(rows).to_csv(f"{a.sel}/werr.csv", index=False)
    print("wrote", f"{a.sel}/werr.csv")
