"""Matrix-pencil (ESPRIT) fit of the per-(g,h) RoPE kernels, on saved C matrices. CPU, numpy, float64.

K(Delta) = sum_i w_i exp(1j Delta theta_i). The (stride-s) Hankel matrix H[a,b] = K(s(a+b)) factors exactly as
U_r diag(w) U_c^T with fixed Vandermonde U_r, U_c, so its SVD is computed from a 64x64 core per (g,h).

Usage: python pencil_fit.py [--layers 0,1] [--models llama2-7b] [--workers 8] [--lm-steps 5] [--out DIR]
"""
import os, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

MODELS = {
    "llama2-7b": ("/home/kfactor/out_llama2-7b/C", 10000.0),
    "qwen1.5-7b": ("/home/kfactor/out_qwen1.5-7b/C", 1000000.0),
}
D_H, HALF, N_KV, R = 128, 64, 32, 600
WINDOWS = {4096: 4, 512: 1}            # L -> pencil stride (N = L / stride samples)
MS = (1, 2, 4, 8)
FF_G, NOPE_J = (1, 2, 4, 8), (8, 16, 32, 64)
TH_MAX = 1.1                           # plausible |theta| range for alias branch choice (max theta_i = 1)
CHUNK = 600
LM_STEPS = 5

def thetas(base):
    return base ** (-2.0 * np.arange(HALF) / D_H)

def pencil_basis(theta, L, s):
    N = L // s
    P = int(round(N / 3))
    Ur = np.exp(1j * s * np.outer(np.arange(N - P), theta))
    Uc = np.exp(1j * s * np.outer(np.arange(P + 1), theta))
    Qr, Rr = np.linalg.qr(Ur)
    _, Rc = np.linalg.qr(Uc)
    return dict(Rr=Rr, Rc=Rc, G11=Qr[:-1].conj().T @ Qr[:-1], G12=Qr[:-1].conj().T @ Qr[1:], s=s, N=N, P=P)

def esprit_poles(W, pb, ms):
    """W: (n, 64) pair energies. Returns {m: complex poles of the stride-s signal, (n, m)}."""
    M = (pb["Rr"][None] * W[:, None, :]) @ pb["Rc"].T          # core of H = Q_r M Q_c^T
    U = np.linalg.svd(M)[0]
    out = {}
    for m in ms:
        Lc = U[:, :, :m]
        LcH = np.conj(np.swapaxes(Lc, 1, 2))
        psi = np.linalg.solve(LcH @ pb["G11"] @ Lc, LcH @ pb["G12"] @ Lc)
        out[m] = np.linalg.eigvals(psi)
    return out

def esprit_direct(Kfull, s, P, m):
    """Reference ESPRIT on the explicit Hankel matrix (one item), for verification."""
    x = Kfull[::s]; N = x.size
    H = np.array([x[a:a + P + 1] for a in range(N - P)])
    Us = np.linalg.svd(H)[0][:, :m]
    return np.linalg.eigvals(np.linalg.lstsq(Us[:-1], Us[1:], rcond=None)[0])

def basis(th, lam, delta):
    return np.exp(delta[None, :, None] * (-lam + 1j * th)[:, None, :])  # (n, L, m)

def ls_fit(K, th, lam, delta):
    B = basis(th, lam, delta)
    Bp = np.linalg.pinv(B, rcond=1e-12)
    A = (Bp @ K[:, :, None])[:, :, 0]
    r = K - (B @ A[:, :, None])[:, :, 0]
    return (np.abs(r) ** 2).sum(1), A, B, Bp, r

def lm_vp(K, th, lam, delta, steps):
    """Variable-projection LM on (th, lam) with A re-solved in closed form; accepts only improving steps."""
    cost, A, B, Bp, r = ls_fit(K, th, lam, delta)
    m = th.shape[1]
    mu = np.full(th.shape[0], 1e-3)
    for _ in range(steps):
        T = np.concatenate([1j * delta[None, :, None] * B * A[:, None, :],
                            -delta[None, :, None] * B * A[:, None, :]], axis=2)   # d(B A)/d(th_k, lam_k)
        J = T - B @ (Bp @ T)                                                       # (I - P_B) T  (Kaufman)
        JH = np.conj(np.swapaxes(J, 1, 2))
        JTJ = np.real(JH @ J); g = np.real((JH @ r[:, :, None])[:, :, 0])
        dg = np.maximum(np.diagonal(JTJ, axis1=1, axis2=2), 1e-300)
        step = np.linalg.solve(JTJ + (mu[:, None] * dg)[:, :, None] * np.eye(2 * m), g[:, :, None])[:, :, 0]
        th2, lam2 = th + step[:, :m], np.maximum(lam + step[:, m:], 0.0)
        c2, A2, B2, Bp2, r2 = ls_fit(K, th2, lam2, delta)
        ok = c2 < cost
        sel = lambda a, b: np.where(ok.reshape((-1,) + (1,) * (a.ndim - 1)), b, a)
        th, lam, cost, A, B, Bp, r = (sel(th, th2), sel(lam, lam2), sel(cost, c2), sel(A, A2),
                                      sel(B, B2), sel(Bp, Bp2), sel(r, r2))
        mu = np.where(ok, mu / 3, mu * 4)
    return th, lam, cost, A

def resolve_poles(z, s, K, delta):
    """Stride-s poles -> (th, lam) per unit Delta; lam projected >= 0; alias branch chosen by full-window error."""
    lam = np.maximum(-np.log(np.abs(z)) / s, 0.0)
    th = np.angle(z) / s
    n_switch = 0
    if s > 1:
        alt = th - np.sign(th) * 2 * np.pi / s
        cand = np.abs(alt) <= TH_MAX
        cost = ls_fit(K, th, lam, delta)[0]
        for k in range(th.shape[1]):
            idx = np.nonzero(cand[:, k])[0]
            if idx.size == 0:
                continue
            th_try = th[idx].copy(); th_try[:, k] = alt[idx, k]
            c = ls_fit(K[idx], th_try, lam[idx], delta)[0]
            better = c < cost[idx]
            th[idx[better], k] = alt[idx[better], k]; cost[idx[better]] = c[better]
            n_switch += int(better.sum())
    return th, lam, n_switch

def rel_err(K, Khat):
    return (np.abs(K - Khat) ** 2).sum(1) / (np.abs(K) ** 2).sum(1)

def run_layer(task):
    model_name, l, lm_steps = task
    cdir, base = MODELS[model_name]
    theta = thetas(base)
    C = np.load(f"{cdir}/layer_{l}.npy").astype(np.float64)
    e, V = np.linalg.eigh(0.5 * (C + C.T))
    e, V = e[::-1][:R], V[:, ::-1][:, :R]
    Vt = V.T.reshape(R, N_KV, D_H)
    w = Vt[..., :HALF] ** 2 + Vt[..., HALF:] ** 2
    sw = w.sum(-1)
    gi, hi = np.nonzero(sw >= 1e-6)
    W, OM = w[gi, hi], (e[:, None] * sw)[gi, hi]
    n = len(gi)

    rows, sanity, poles = [], dict(model=model_name, layer=l, n_items=n), {}
    for L, s in WINDOWS.items():
        delta = np.arange(L, dtype=np.float64)
        pb = pencil_basis(theta, L, s)
        E = np.exp(1j * np.outer(theta, delta))
        E_ff = {g: np.exp(1j * np.outer(np.repeat(theta.reshape(-1, g).mean(1), g), delta)) for g in FF_G}
        E_nj = {}
        for J in NOPE_J:
            E_nj[J] = E.copy(); E_nj[J][J:] = 1.0
        err = {k: np.empty(n) for k in [f"freqfold_{g}" for g in FF_G] + [f"nope_lowfreq_{J}" for J in NOPE_J]
               + ["nope_all"] + [f"pencil_{m}" for m in MS] + ([f"pencil_lm{lm_steps}_{m}" for m in MS] if lm_steps else [])}
        P = {m: dict(th=np.empty((n, m)), lam=np.empty((n, m)), ew=np.empty((n, m))) for m in MS}
        n_switch = 0
        for c0 in range(0, n, CHUNK):
            sl = slice(c0, c0 + CHUNK)
            Wc = W[sl]
            K = Wc @ E
            nK = (np.abs(K) ** 2).sum(1)
            for g in FF_G:
                err[f"freqfold_{g}"][sl] = rel_err(K, Wc @ E_ff[g])
            for J in NOPE_J:
                err[f"nope_lowfreq_{J}"][sl] = rel_err(K, Wc @ E_nj[J])
            err["nope_all"][sl] = rel_err(K, np.broadcast_to(Wc.sum(1)[:, None] + 0j, K.shape))
            Z = esprit_poles(Wc, pb, MS)
            for m in MS:
                th, lam, ns = resolve_poles(Z[m], s, K, delta)
                n_switch += ns
                cost, A, B, _, _ = ls_fit(K, th, lam, delta)
                err[f"pencil_{m}"][sl] = cost / nK
                ce = (np.abs(A) ** 2) * (np.abs(B) ** 2).sum(1)          # energy of each pencil pole component
                P[m]["th"][sl], P[m]["lam"][sl], P[m]["ew"][sl] = th, lam, ce / ce.sum(1, keepdims=True)
                if lm_steps:
                    err[f"pencil_lm{lm_steps}_{m}"][sl] = lm_vp(K, th, lam, delta, lm_steps)[2] / nK
        for meth, ev in err.items():
            rows.append(dict(model=model_name, layer=l, window=L, method=meth,
                             err=(OM * ev).sum() / OM.sum()))
        sanity[f"L{L}_max_freqfold_1"] = err["freqfold_1"].max()
        sanity[f"L{L}_max_nope_lowfreq_64"] = err["nope_lowfreq_64"].max()
        sanity[f"L{L}_alias_switches"] = n_switch
        pm = np.stack([err[f"pencil_{m}"] for m in MS], 1)
        sanity[f"L{L}_items_nonmonotone"] = int((np.diff(pm, axis=1) > 1e-12).any(1).sum())
        sanity[f"L{L}_max_item_increase"] = float(np.diff(pm, axis=1).max())
        if lm_steps:
            pl = np.stack([err[f"pencil_lm{lm_steps}_{m}"] for m in MS], 1)
            sanity[f"L{L}_lm_worse_than_pencil"] = int((pl > pm + 1e-15).sum())
        poles[L] = P
    print(f"{model_name} L{l} done: " + " ".join(f"{r['method']}@{r['window']}={r['err']:.4f}" for r in rows
                                                  if r["method"].startswith("pencil_") and r["window"] == 4096
                                                  and "lm" not in r["method"]), flush=True)
    return rows, sanity, dict(omega=OM.astype(np.float32), gi=gi.astype(np.int16), hi=hi.astype(np.int8),
                              poles={L: {m: {k: v.astype(np.float32) for k, v in d.items()} for m, d in P.items()}
                                     for L, P in poles.items()})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default=",".join(MODELS)); ap.add_argument("--lm-steps", type=int, default=LM_STEPS)
    ap.add_argument("--out", default="/home/kfactor/pencil_fit")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(32))
    tasks = [(m, l, a.lm_steps) for m in a.models.split(",") for l in layers]
    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(run_layer, tasks))
    pd.DataFrame([r for res in results for r in res[0]]).to_csv(f"{a.out}/pencil_errors_all.csv", index=False)
    pd.DataFrame([res[1] for res in results]).to_csv(f"{a.out}/sanity.csv", index=False)
    for mname in a.models.split(","):
        idx = [i for i, t in enumerate(tasks) if t[0] == mname]
        arrs = {}
        for i in idx:
            l = tasks[i][1]; d = results[i][2]
            arrs[f"L{l}_omega"], arrs[f"L{l}_gi"], arrs[f"L{l}_hi"] = d["omega"], d["gi"], d["hi"]
            for L, P in d["poles"].items():
                for m, pd_ in P.items():
                    for k, v in pd_.items():
                        arrs[f"L{l}_W{L}_m{m}_{k}"] = v
        np.savez_compressed(f"{a.out}/poles_{mname}.npz", rope_theta=thetas(MODELS[mname][1]), **arrs)
    print("wrote", a.out)
