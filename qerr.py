"""Query-weighted kernel error (err_q), extended RoPE+learned dictionary, Qwen dictionary rebuild, per-head view.

Score for query rho and key direction z (per head, +theta frame): s(Delta) = sum_i rho_i conj(z_i) e_i(Delta),
e_i = exp(1j theta_i Delta). An approximation replaces e_i by e_i - eps_i; with M = E[rho rho^H]:
    err_q = sum_{ii'} Q_{ii'} R_{ii'} / sum_{ii'} Q_{ii'} E_{ii'},   Q = M o (conj(z) z^T),
    E_{ii'} = sum_Delta e_i conj(e_i'),  R_{ii'} = sum_Delta eps_i conj(eps_i').
For projection onto atoms b_s = exp(p_s Delta): sum Q o R = sum Q o E - Re Tr(G_SS^+ (C Q C^H)_SS),
C_{si} = sum_Delta conj(b_s) e_i,  G_st = sum_Delta conj(b_s) b_t.  With Q = w w^T this is the energy-weighted error.

Usage: python qerr.py [--layers ..] [--models ..] [--workers 8] [--out DIR]
"""
import os, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import pencil_fit as pf
import dict_fit as dfit

L = 4096
R = 600
CHUNK = 1200
PINV = float(os.environ.get("QERR_PINV", "1e-10"))
M_PATH = {"llama2-7b": "/home/kfactor/out_llama2-7b/M_llama2-7b.npz", "qwen1.5-7b": "/home/kfactor/out_qwen1.5-7b/M_qwen1.5-7b.npz"}
POLES = "/home/kfactor/pencil_fit/poles_{}.npz"
CENTERS = "/home/kfactor/dict_fit/dict_centers.npz"
DICT_D_EVAL = (8, 32, 64)
EXT_D = 8
EXT_M = (2, 4)
REBUILD_SRC_M, REBUILD_D = (2, 4), (8, 16, 32, 64, 128)
gsum = dfit.gsum

def Emat(a, b):
    """sum_Delta exp(1j a_i Delta) exp(-1j b_j Delta)."""
    return gsum(1j * (a[:, None] - b[None, :]), L)

def load_layer_z(model, l):
    cdir, base = pf.MODELS[model]
    C = np.load(f"{cdir}/layer_{l}.npy").astype(np.float64)
    e, V = np.linalg.eigh(0.5 * (C + C.T))
    e, V = e[::-1][:R], V[:, ::-1][:, :R]
    Vt = V.T.reshape(R, pf.N_KV, pf.D_H)
    z = Vt[..., :pf.HALF] + 1j * Vt[..., pf.HALF:]
    sw = (np.abs(z) ** 2).sum(-1)
    gi, hi = np.nonzero(sw >= 1e-6)
    return z[gi, hi], (e[:, None] * sw)[gi, hi], gi, hi, pf.thetas(base)

def atom_terms(p, theta):
    """C (.., m, 64) and G (.., m, m) for atoms exp(p Delta); p (.., m)."""
    C = gsum(np.conj(p)[..., :, None] + 1j * theta, L)
    G = gsum(np.conj(p)[..., :, None] + p[..., None, :], L)
    return C, G

def proj_gain(G, X):
    """Re Tr(G^+ X) batched."""
    return np.real(np.einsum("...ij,...ji->...", np.linalg.pinv(G, rcond=PINV, hermitian=True), X))

def sandwich(C, Q):
    """C Q C^H, C (n or 1, m, 64), Q (n, 64, 64)."""
    return C @ Q @ np.conj(np.swapaxes(C, -1, -2))

def greedy_q(Gall, X, m_max):
    """Forward selection maximizing Re Tr(G_SS^+ X_SS) per item. Gall (A,A) shared, X (n,A,A). Returns S (n, m_max)."""
    n, A = X.shape[0], X.shape[1]
    S = np.zeros((n, 0), dtype=int)
    rows = np.arange(n)[:, None, None, None]
    for k in range(m_max):
        cand = np.broadcast_to(np.arange(A), (n, A))
        Sc = np.concatenate([np.repeat(S[:, None, :], A, 1), cand[:, :, None]], 2)       # (n, A, k+1)
        Gs = Gall[Sc[..., :, None], Sc[..., None, :]]
        Xs = X[rows, Sc[..., :, None], Sc[..., None, :]]                                  # (n, A, k+1, k+1)
        val = proj_gain(Gs, Xs)
        if k:
            val[np.arange(n)[:, None], S] = -np.inf
        S = np.concatenate([S, val.argmax(1)[:, None]], 1)
    return S

def run_layer(task):
    model, l = task
    z, OM, gi, hi, theta = load_layer_z(model, l)
    Mq = np.load(M_PATH[model])["M"][l]                                   # (heads, 64, 64)
    Mq = 0.5 * (Mq + np.conj(np.swapaxes(Mq, -1, -2)))
    n = len(z)
    w = np.abs(z) ** 2
    E0 = Emat(theta, theta)
    P = np.load(POLES.format(model))
    cent = np.load(CENTERS)
    q_share = np.real(np.einsum("hii->hi", Mq))
    q_share = q_share[:, :16].sum(1) / q_share.sum(1)

    # fixed-substitution residual Grams
    fixed = {f"freqfold_{g}": np.repeat(theta.reshape(-1, g).mean(1), g) for g in (1, 2, 4, 8)}
    for J in (8, 16, 32, 64):
        fixed[f"nope_lowfreq_{J}"] = np.where(np.arange(pf.HALF) < J, theta, 0.0)
    Rfix = {k: E0 - Emat(theta, t) - Emat(t, theta) + Emat(t, t) for k, t in fixed.items()}

    # shared dictionaries
    dicts = {f"dict_D{D}": cent[f"{model}_L{l}_D{D}"] for D in DICT_D_EVAL}
    c8 = cent[f"{model}_L{l}_D{EXT_D}"]
    p_ext = np.concatenate([-c8[:, 1] + 1j * c8[:, 0], 1j * theta])          # 8 learned + 64 RoPE atoms
    C_ext, G_ext = atom_terms(p_ext, theta)
    rebuild = {}
    if model == "qwen1.5-7b":
        rng = np.random.default_rng(7000 + l)
        for sm in REBUILD_SRC_M:
            th = P[f"L{l}_W4096_m{sm}_th"].astype(np.float64).ravel(); lam = P[f"L{l}_W4096_m{sm}_lam"].astype(np.float64).ravel()
            pw = (OM[:, None] * P[f"L{l}_W4096_m{sm}_ew"].astype(np.float64)).ravel()
            X = np.stack([dfit.symlog(th), np.log(lam + dfit.EPS)], 1)
            for D in REBUILD_D:
                Cc, _ = dfit.wkmeans(X, pw, D, rng)
                rebuild[(sm, D)] = np.stack([dfit.isymlog(Cc[:, 0]), np.maximum(np.exp(Cc[:, 1]) - dfit.EPS, 0.0)], 1)

    KKall = dfit.KK_of(w, theta, L)
    num = {}; den = np.empty(n); eng = {}
    picks = {}
    def put(d, k, sl, v):
        d.setdefault(k, np.full(n, np.nan))[sl] = v

    for c0 in range(0, n, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        zc, wc = z[sl], w[sl]
        Q = Mq[hi[sl]] * (np.conj(zc)[:, :, None] * zc[:, None, :])
        Qw = wc[:, :, None] * wc[:, None, :] + 0j                              # aligned query
        dq = np.real((Q * E0).sum((1, 2))); den[sl] = dq
        KK = KKall[sl]
        put(num, "exact", sl, np.zeros(len(zc))); put(eng, "exact", sl, np.zeros(len(zc)))
        for k, Rm in Rfix.items():
            put(num, k, sl, np.real((Q * Rm).sum((1, 2))) / dq)
            put(eng, k, sl, np.real((Qw * Rm).sum((1, 2))) / KK)

        def eval_sets(name, C, G):
            put(num, name, sl, (dq - proj_gain(G, sandwich(C, Q))) / dq)
            put(eng, name, sl, (KK - proj_gain(G, sandwich(C, Qw))) / KK)

        for m in (1, 2, 4):                                                      # free poles
            p = -P[f"L{l}_W4096_m{m}_lam"][sl].astype(np.float64) + 1j * P[f"L{l}_W4096_m{m}_th"][sl].astype(np.float64)
            eval_sets(f"free_m{m}", *atom_terms(p, theta))
        for name, cc in dicts.items():                                           # existing dictionaries, energy OMP m=2
            pd_ = -cc[:, 1] + 1j * cc[:, 0]
            S = dfit.omp_layer(wc, theta, pd_, KK)[2][2]
            Cd, Gd = atom_terms(pd_, theta)
            eval_sets(f"{name}_m2", Cd[S], Gd[S[:, :, None], S[:, None, :]])
        # extended dictionary: energy OMP and q-greedy
        Xq = sandwich(C_ext[None], Q)
        Sq = greedy_q(G_ext, Xq, max(EXT_M))
        oe = dfit.omp_layer(wc, theta, p_ext, KK)
        for m in EXT_M:
            for obj, S in (("energy", oe[m][2]), ("q", Sq[:, :m])):
                eval_sets(f"ext_{obj}_m{m}", C_ext[S], G_ext[S[:, :, None], S[:, None, :]])
                picks.setdefault((obj, m), np.zeros((n, m), dtype=int))[sl] = S
        for (sm, D), cc in rebuild.items():
            pd_ = -cc[:, 1] + 1j * cc[:, 0]
            S = dfit.omp_layer(wc, theta, pd_, KK)[2][2]
            Cd, Gd = atom_terms(pd_, theta)
            eval_sets(f"rebuild_src{sm}_D{D}_m2", Cd[S], Gd[S[:, :, None], S[:, None, :]])

    wmean = lambda v, msk=slice(None): (OM[msk] * v[msk]).sum() / OM[msk].sum()
    rows = [dict(model=model, layer=l, method=k, err_energy=wmean(eng[k]), err_q=wmean(num[k])) for k in num]
    heads = []
    for h in range(pf.N_KV):
        msk = hi == h
        rec = dict(model=model, layer=l, head=h, q_share_pairs0_15=q_share[h])
        for k in ("free_m2", "ext_energy_m2", "ext_q_m2", "ext_energy_m4", "ext_q_m4", "dict_D8_m2"):
            rec[f"errq_{k}"] = wmean(num[k], msk)
        heads.append(rec)
    pk = []
    for (obj, m), S in picks.items():
        is_rope = S >= EXT_D
        wt = np.repeat(OM[:, None], m, 1)
        pk.append(dict(model=model, layer=l, objective=obj, m=m, frac_rope=(wt * is_rope).sum() / wt.sum(),
                       **{f"f{i}": (wt * (S == EXT_D + i)).sum() / wt.sum() for i in range(pf.HALF)}))
    sanity = dict(model=model, layer=l, max_abs_freqfold_1=float(np.abs(num["freqfold_1"]).max()),
                  max_abs_nope_64=float(np.abs(num["nope_lowfreq_64"]).max()),
                  max_neg_errq=float(min(np.nanmin(v) for v in num.values())),
                  **{f"min_{k}": float(np.nanmin(v)) for k, v in num.items()},
                  **{f"minE_{k}": float(np.nanmin(v)) for k, v in eng.items()})
    print(f"{model} L{l} done: " + " ".join(f"{k}={wmean(num[k]):.4f}" for k in ("free_m2", "ext_energy_m2", "ext_q_m2")), flush=True)
    return rows, heads, pk, sanity

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default="llama2-7b,qwen1.5-7b"); ap.add_argument("--out", default="/home/kfactor/qerr")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(32))
    tasks = [(m, l) for m in a.models.split(",") for l in layers]
    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_layer, tasks))
    pd.DataFrame([r for x in res for r in x[0]]).to_csv(f"{a.out}/qerr_errors.csv", index=False)
    pd.DataFrame([r for x in res for r in x[1]]).to_csv(f"{a.out}/qerr_heads.csv", index=False)
    pd.DataFrame([r for x in res for r in x[2]]).to_csv(f"{a.out}/rope_picks.csv", index=False)
    pd.DataFrame([x[3] for x in res]).to_csv(f"{a.out}/sanity.csv", index=False)
    print("wrote", a.out)
