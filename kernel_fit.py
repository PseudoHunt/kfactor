"""Kernel-fit test on saved per-layer key second moments C (from kfactor.py). CPU, numpy, float64.

Usage: python kernel_fit.py [--layers 0,1] [--workers 8]
"""
import os, sys, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

MODELS = {  # name: (C dir, rope_theta from config)
    "llama2-7b": ("/home/kfactor/out_llama2-7b/C", 10000.0),
    "qwen1.5-7b": ("/home/kfactor/out_qwen1.5-7b/C", 1000000.0),
}
D_H, HALF, N_KV, OUT, L, L_SHORT = 128, 64, 32, 4096, 4096, 512
R_MAX, RANKS = 600, (300, 600)
N_LAYERS = 32
FF_G, NOPE_J = (1, 2, 4, 8), (8, 16, 32, 64)  # g=1 and J=64 are the sanity cases
LM_ITERS, CHUNK = 20, 2400
DELTA = np.arange(L, dtype=np.float64)

def thetas(base):
    return base ** (-2.0 * np.arange(HALF) / D_H)

def errs(K, Khat, n_full, n_short):
    d = np.abs(K - Khat) ** 2
    return d.sum(1) / n_full, d[:, :L_SHORT].sum(1) / n_short

def lm_fit(K, A, th, lam):
    """Batched Levenberg-Marquardt for Khat = A exp((-lam + i th) Delta), lam >= 0. Accepts only improving steps."""
    def model(A, th, lam):
        z = np.exp(np.outer(-lam, DELTA) + 1j * np.outer(th, DELTA))
        return z, A[:, None] * z
    z, f = model(A, th, lam)
    cost = (np.abs(K - f) ** 2).sum(1)
    mu = np.full(A.shape, 1e-3)
    for _ in range(LM_ITERS):
        r = K - f
        Df = DELTA * f
        J = (z, 1j * Df, -Df)
        JTJ = np.empty((A.size, 3, 3)); g = np.empty((A.size, 3))
        for a in range(3):
            g[:, a] = np.real((np.conj(J[a]) * r).sum(1))
            for b in range(a, 3):
                JTJ[:, a, b] = JTJ[:, b, a] = np.real((np.conj(J[a]) * J[b]).sum(1))
        dg = np.diagonal(JTJ, axis1=1, axis2=2)
        M = JTJ + (mu[:, None] * np.maximum(dg, 1e-300))[:, :, None] * np.eye(3)
        step = np.linalg.solve(M, g[:, :, None])[:, :, 0]
        A2, th2, lam2 = A + step[:, 0], th + step[:, 1], np.maximum(lam + step[:, 2], 0.0)
        z2, f2 = model(A2, th2, lam2)
        cost2 = (np.abs(K - f2) ** 2).sum(1)
        ok = cost2 < cost
        A, th, lam = np.where(ok, A2, A), np.where(ok, th2, th), np.where(ok, lam2, lam)
        z, f = np.where(ok[:, None], z2, z), np.where(ok[:, None], f2, f)
        cost = np.where(ok, cost2, cost)
        mu = np.where(ok, mu / 3, mu * 4)
    return A, th, lam, f

def run_layer(task):
    model_name, l = task
    cdir, base = MODELS[model_name]
    theta = thetas(base)
    C = np.load(f"{cdir}/layer_{l}.npy").astype(np.float64)
    C = 0.5 * (C + C.T)
    e, V = np.linalg.eigh(C)
    e, V = e[::-1][:R_MAX], V[:, ::-1][:, :R_MAX]
    Vt = V.T.reshape(R_MAX, N_KV, D_H)
    w = Vt[..., :HALF] ** 2 + Vt[..., HALF:] ** 2            # (g, h, i); |z|^2 == w
    sw = w.sum(-1)
    omega = e[:, None] * sw
    gi, hi = np.nonzero(sw >= 1e-6)
    W, OM = w[gi, hi], omega[gi, hi]

    E = np.exp(1j * np.outer(theta, DELTA))                  # (64, L)
    E_ff = {g: np.exp(1j * np.outer(np.repeat(theta.reshape(-1, g).mean(1), g), DELTA)) for g in FF_G}
    E_nj = {}
    for J in NOPE_J:
        E_nj[J] = E.copy(); E_nj[J][J:] = 1.0

    methods = (["fitted_init", "fitted_refined"] + [f"freqfold_{g}" for g in FF_G]
               + [f"nope_lowfreq_{J}" for J in NOPE_J] + ["nope_all"])
    res = {m: np.empty((len(gi), 2)) for m in methods}
    P = {k: np.empty(len(gi)) for k in ("A0", "th0", "lam0", "A", "th", "lam")}
    for s in range(0, len(gi), CHUNK):
        sl = slice(s, s + CHUNK)
        Wc = W[sl]; swc = Wc.sum(1)
        K = Wc @ E
        nf, ns = (np.abs(K) ** 2).sum(1), (np.abs(K[:, :L_SHORT]) ** 2).sum(1)
        for g in FF_G:
            res[f"freqfold_{g}"][sl] = np.stack(errs(K, Wc @ E_ff[g], nf, ns), 1)
        for J in NOPE_J:
            res[f"nope_lowfreq_{J}"][sl] = np.stack(errs(K, Wc @ E_nj[J], nf, ns), 1)
        res["nope_all"][sl] = np.stack(errs(K, swc[:, None] + 0j, nf, ns), 1)
        # fitted: init from energy-weighted circular mean / std of theta
        A0 = swc
        th0 = np.arctan2(Wc @ np.sin(theta), Wc @ np.cos(theta))
        lam0 = np.sqrt(np.maximum((Wc * (theta[None] - th0[:, None]) ** 2).sum(1) / swc, 0.0))
        f0 = A0[:, None] * np.exp(np.outer(-lam0, DELTA) + 1j * np.outer(th0, DELTA))
        res["fitted_init"][sl] = np.stack(errs(K, f0, nf, ns), 1)
        A, th, lam, f = lm_fit(K, A0, th0, lam0)
        res["fitted_refined"][sl] = np.stack(errs(K, f, nf, ns), 1)
        for k, v in zip(("A0", "th0", "lam0", "A", "th", "lam"), (A0, th0, lam0, A, th, lam)):
            P[k][sl] = v

    rows = []
    for r in RANKS:
        m = gi < r
        for meth, ev in res.items():
            rows.append(dict(model=model_name, layer=l, r=r, method=meth,
                             err_L4096=(OM[m] * ev[m, 0]).sum() / OM[m].sum(),
                             err_L512=(OM[m] * ev[m, 1]).sum() / OM[m].sum()))
    full = lambda x: _scatter(x, gi, hi)
    params = dict(omega=full(OM), A=full(P["A"]), th=full(P["th"]), lam=full(P["lam"]),
                  A_init=full(P["A0"]), th_init=full(P["th0"]), lam_init=full(P["lam0"]),
                  err_refined=full(res["fitted_refined"][:, 0]), err_init=full(res["fitted_init"][:, 0]))
    sanity = dict(model=model_name, layer=l, n_items=len(gi), n_skipped=R_MAX * N_KV - len(gi),
                  max_err_freqfold_1=res["freqfold_1"].max(), max_err_nope_lowfreq_64=res["nope_lowfreq_64"].max(),
                  n_refined_gt_init=int((res["fitted_refined"][:, 0] > res["fitted_init"][:, 0]).sum()))
    print(f"{model_name} L{l} done: fitted r600 L4096 = {rows[RANKS.index(600) * len(res) + 1]['err_L4096']:.4f}",
          flush=True)
    return rows, params, sanity

def _scatter(x, gi, hi):
    out = np.full((R_MAX, N_KV), np.nan); out[gi, hi] = x
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default=",".join(MODELS)); ap.add_argument("--out", default="/home/kfactor/kernel_fit")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(N_LAYERS))
    tasks = [(m, l) for m in a.models.split(",") for l in layers]
    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(run_layer, tasks))
    rows = [r for res in results for r in res[0]]
    pd.DataFrame(rows).to_csv(f"{a.out}/kernel_errors_all.csv", index=False)
    pd.DataFrame([res[2] for res in results]).to_csv(f"{a.out}/sanity.csv", index=False)
    for m in a.models.split(","):
        idx = [i for i, t in enumerate(tasks) if t[0] == m]
        P = {k: np.stack([results[i][1][k] for i in idx]) for k in results[idx[0]][1]}
        np.savez(f"{a.out}/fit_params_{m}.npz", layers=np.array([tasks[i][1] for i in idx]),
                 rope_theta=thetas(MODELS[m][1]), **P)
    print("wrote", a.out)
