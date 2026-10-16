"""Dictionary-constrained pencil (OMP over a per-layer pole dictionary) and Qwen L=32768 free-pole refit.

All Gram matrices are closed-form geometric sums: sum_{Delta<L} exp(c Delta) = expm1(L c) / expm1(c).

Usage: python dict_fit.py dict  [--layers ..] [--models ..] [--workers 8] [--out DIR]
       python dict_fit.py long  [--layers ..] [--workers 8] [--out DIR]       (Qwen, L=32768, stride 8)
"""
import os, sys, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import pencil_fit as pf

R, L_DICT = 600, 4096
FREE_M = {"llama2-7b": 2, "qwen1.5-7b": 1}
DICT_D, OMP_M = (8, 16, 32, 64, 128), (1, 2, 4)
EPS = 1e-7
PAIR_SHARE = 0.01
PINV_RCOND = 1e-14
KM_RESTARTS, KM_ITERS = 3, 100
POLES = "/home/kfactor/pencil_fit/poles_{}.npz"

def gsum(c, L):
    c = np.asarray(c, dtype=np.complex128)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.expm1(L * c) / np.expm1(c)
    return np.where(c == 0, float(L), out)

def load_layer(model, l):
    cdir, base = pf.MODELS[model]
    C = np.load(f"{cdir}/layer_{l}.npy").astype(np.float64)
    e, V = np.linalg.eigh(0.5 * (C + C.T))
    e, V = e[::-1][:R], V[:, ::-1][:, :R]
    Vt = V.T.reshape(R, pf.N_KV, pf.D_H)
    w = Vt[..., :pf.HALF] ** 2 + Vt[..., pf.HALF:] ** 2
    sw = w.sum(-1)
    gi, hi = np.nonzero(sw >= 1e-6)
    return w[gi, hi], (e[:, None] * sw)[gi, hi], gi, hi, pf.thetas(base)

def KK_of(W, theta, L):
    GEE = np.real(gsum(1j * (theta[None, :] - theta[:, None]), L))     # (64,64): sum e^{-i th_i D} e^{i th_j D}
    return np.einsum("ni,ij,nj->n", W, GEE, W)

def err_free(W, th, lam, theta, L, KK=None):
    """Relative LS error of K by free poles (n, m) with complex amplitudes."""
    zl = -lam + 1j * th
    G = gsum(np.conj(zl)[:, :, None] + zl[:, None, :], L)
    b = (gsum(np.conj(zl)[:, :, None] + 1j * theta[None, None, :], L) @ W[:, :, None])[:, :, 0]
    A = (np.linalg.pinv(G, rcond=PINV_RCOND, hermitian=True) @ b[:, :, None])[:, :, 0]
    KK = KK_of(W, theta, L) if KK is None else KK
    return (KK - np.real((np.conj(b) * A).sum(1))) / KK

def resolve_gram(z, s, W, theta, L, KK):
    """Stride-s poles -> per-unit (th, lam); lam >= 0; every alias branch with |th| <= TH_MAX tried per pole."""
    lam = np.maximum(-np.log(np.abs(z)) / s, 0.0)
    th = np.angle(z) / s
    n_switch = 0
    if s > 1:
        cost = err_free(W, th, lam, theta, L, KK)
        for k in range(th.shape[1]):
            for j in (-2, -1, 1, 2):
                alt = th[:, k] + 2 * np.pi * j / s
                idx = np.nonzero(np.abs(alt) <= pf.TH_MAX)[0]
                if idx.size == 0:
                    continue
                th_try = th[idx].copy(); th_try[:, k] = alt[idx]
                c = err_free(W[idx], th_try, lam[idx], theta, L, KK[idx])
                better = c < cost[idx]
                th[idx[better], k] = alt[idx[better]]; cost[idx[better]] = c[better]
                n_switch += int(better.sum())
    return th, lam, n_switch

# ---------------- dictionary ----------------
def symlog(th):
    return np.sign(th) * np.log(np.maximum(np.abs(th), EPS) / EPS)

def isymlog(x):
    return np.sign(x) * EPS * np.exp(np.abs(x))

def wkmeans(X, w, k, rng):
    best = None
    for _ in range(KM_RESTARTS):
        C = X[[rng.choice(len(X), p=w / w.sum())]]
        for _ in range(1, k):                                   # weighted k-means++
            d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1).min(1)
            p = w * d2
            C = np.vstack([C, X[rng.choice(len(X), p=p / p.sum())]])
        for _ in range(KM_ITERS):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            Wc = np.bincount(lab, weights=w, minlength=k)
            newC = np.stack([np.bincount(lab, weights=w * X[:, j], minlength=k) for j in range(2)], 1)
            empty = Wc <= 0
            newC[~empty] /= Wc[~empty, None]
            if empty.any():                                     # reseed empty clusters at worst-fit points
                d2 = ((X - C[lab]) ** 2).sum(1) * w
                newC[empty] = X[np.argsort(-d2)[:empty.sum()]]
            if np.allclose(newC, C):
                C = newC; break
            C = newC
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        inertia = (w * ((X - C[lab]) ** 2).sum(1)).sum()
        if best is None or inertia < best[0]:
            best = (inertia, C)
    return best[1], best[0]

def omp_layer(W, theta, zd, KK):
    """OMP over dictionary atoms exp(zd * Delta). Returns per-m (err, worst-pair residual) and all-D err."""
    L = L_DICT
    GBB = gsum(np.conj(zd)[:, None] + zd[None, :], L)                  # (D, D)
    GBE = gsum(np.conj(zd)[:, None] + 1j * theta[None, :], L)          # (D, 64): b_d^H e_i
    c = W @ GBE.T                                                       # (n, D): b_d^H K
    nb = np.sqrt(np.real(np.diag(GBB)))
    n, D = c.shape
    rows = np.arange(n)[:, None]
    big = W / W.sum(1, keepdims=True) > PAIR_SHARE
    S = np.zeros((n, 0), dtype=int)
    out = {}
    for k in range(1, max(OMP_M) + 1):
        if k == 1:
            rc = c
        else:
            GS = GBB[S[:, :, None], S[:, None, :]]
            A = (np.linalg.pinv(GS, rcond=PINV_RCOND, hermitian=True) @ c[rows, S][:, :, None])[:, :, 0]
            rc = c - (GBB[:, S].transpose(1, 0, 2) @ A[:, :, None])[:, :, 0]
        score = np.abs(rc) / nb
        score[rows, S] = -np.inf
        S = np.concatenate([S, score.argmax(1)[:, None]], 1)
        if k in OMP_M:
            GS = GBB[S[:, :, None], S[:, None, :]]
            Gp = np.linalg.pinv(GS, rcond=PINV_RCOND, hermitian=True)
            cS = c[rows, S]
            err = (KK - np.real((np.conj(cS) * (Gp @ cS[:, :, None])[:, :, 0]).sum(1))) / KK
            g = GBE[S]                                                  # (n, k, 64)
            res = 1.0 - np.real((np.conj(g) * (Gp @ g)).sum(1)) / L     # per-pair projection residual
            worst = np.where(big, res, -np.inf).max(1)
            out[k] = (err, worst, S.copy())
    Gp = np.linalg.pinv(GBB, rcond=PINV_RCOND, hermitian=True)
    out["all"] = ((KK - np.real((np.conj(c) * (c @ Gp.T)).sum(1))) / KK, None, None)
    return out

def run_dict(task):
    model, l = task
    W, OM, gi, hi, theta = load_layer(model, l)
    P = np.load(POLES.format(model))
    m0 = FREE_M[model]
    th = P[f"L{l}_W4096_m{m0}_th"].astype(np.float64); lam = P[f"L{l}_W4096_m{m0}_lam"].astype(np.float64)
    ew = P[f"L{l}_W4096_m{m0}_ew"].astype(np.float64)
    assert th.shape[0] == len(gi)
    pw = (OM[:, None] * ew).ravel()
    X = np.stack([symlog(th.ravel()), np.log(lam.ravel() + EPS)], 1)
    KK = KK_of(W, theta, L_DICT)
    rng = np.random.default_rng(1000 * l + m0)
    rows, centers = [], {}
    for D in DICT_D:
        Cc, inertia = wkmeans(X, pw, D, rng)
        th_d, lam_d = isymlog(Cc[:, 0]), np.maximum(np.exp(Cc[:, 1]) - EPS, 0.0)
        centers[D] = np.stack([th_d, lam_d], 1)
        res = omp_layer(W, theta, -lam_d + 1j * th_d, KK)
        for m, (err, worst, _) in res.items():
            rows.append(dict(model=model, layer=l, D=D, m=m, err=(OM * err).sum() / OM.sum(),
                             worst_pair=np.nan if worst is None else (OM * worst).sum() / OM.sum(),
                             worst_pair_median=np.nan if worst is None else float(np.median(worst)),
                             inertia=inertia))
    print(f"{model} L{l} done: " + " ".join(f"D{r['D']}m{r['m']}={r['err']:.4f}" for r in rows if r["m"] == 2),
          flush=True)
    return rows, centers

# ---------------- Qwen long window ----------------
L_LONG, S_LONG = 32768, 8
LONG_MS = (1, 2)

def run_long(task):
    model, l = task
    W, OM, gi, hi, theta = load_layer(model, l)
    KK = KK_of(W, theta, L_LONG)
    pb = pf.pencil_basis(theta, L_LONG, S_LONG)
    rows, sw_total = [], 0
    errs = {m: np.empty(len(W)) for m in LONG_MS}
    for c0 in range(0, len(W), 2400):
        sl = slice(c0, c0 + 2400)
        Z = pf.esprit_poles(W[sl], pb, LONG_MS)
        for m in LONG_MS:
            th, lam, ns = resolve_gram(Z[m], S_LONG, W[sl], theta, L_LONG, KK[sl])
            sw_total += ns
            errs[m][sl] = err_free(W[sl], th, lam, theta, L_LONG, KK[sl])
    for m in LONG_MS:
        rows.append(dict(model=model, layer=l, window=L_LONG, method=f"pencil_{m}", err=(OM * errs[m]).sum() / OM.sum()))
    # fixed-frequency methods: ||K - Khat||^2 = w^T M w,  M = Re[(E - E')^H (E - E')] in closed form
    variants = {f"freqfold_{g}": np.repeat(theta.reshape(-1, g).mean(1), g) for g in pf.FF_G}
    for J in pf.NOPE_J:
        variants[f"nope_lowfreq_{J}"] = np.where(np.arange(pf.HALF) < J, theta, 0.0)
    variants["nope_all"] = np.zeros_like(theta)
    for name, th2 in variants.items():
        a, b = theta, th2
        M = (gsum(1j * (a[None, :] - a[:, None]), L_LONG) - gsum(1j * (b[None, :] - a[:, None]), L_LONG)
             - gsum(1j * (a[None, :] - b[:, None]), L_LONG) + gsum(1j * (b[None, :] - b[:, None]), L_LONG))
        e = np.einsum("ni,ij,nj->n", W, np.real(M), W) / KK
        rows.append(dict(model=model, layer=l, window=L_LONG, method=name, err=(OM * e).sum() / OM.sum(),
                         max_item_err=float(np.abs(e).max())))
    print(f"{model} L{l} long done: " + " ".join(f"{r['method']}={r['err']:.4f}" for r in rows[:2]), flush=True)
    return rows, sw_total

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dict", "long"])
    ap.add_argument("--layers", default=None); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default=None); ap.add_argument("--out", default="/home/kfactor/dict_fit")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(32))
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "dict":
        models = a.models.split(",") if a.models else list(FREE_M)
        tasks = [(m, l) for m in models for l in layers]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(run_dict, tasks))
        pd.DataFrame([r for x in res for r in x[0]]).to_csv(f"{a.out}/dict_errors.csv", index=False)
        np.savez(f"{a.out}/dict_centers.npz",
                 **{f"{t[0]}_L{t[1]}_D{D}": c for t, x in zip(tasks, res) for D, c in x[1].items()})
    else:
        tasks = [("qwen1.5-7b", l) for l in layers]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(run_long, tasks))
        pd.DataFrame([r for x in res for r in x[0]]).to_csv(f"{a.out}/qwen_L32768_errors.csv", index=False)
        print("alias branch switches at L=32768:", sum(x[1] for x in res))
    print("wrote", a.out)
