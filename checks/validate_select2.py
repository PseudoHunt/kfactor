"""Explicit checks for fold_select2: weighted constrained projection num, P, and greedy vs brute force."""
import sys; sys.path.insert(0, "/home/kfactor")
import numpy as np, itertools
import fold_select2 as fs, pencil_fit as pf
rng = np.random.default_rng(0)
theta = pf.thetas(10000.0); L = 512; D = np.arange(L, dtype=float)
w = np.exp(-D / 20) + 0.01                                    # stand-in attention-mass profile
pe = 1j * theta
patoms = np.concatenate([-np.array([1e-3, 5e-3]) + 1j * np.array([2e-3, 0.3]), pe])
E = np.exp(np.outer(D, pe)); B = np.exp(np.outer(D, patoms))
Ew = fs.wsums(pe, pe, w, D).T; G = fs.wsums(patoms, patoms, w, D); C = fs.wsums(patoms, pe, w, D)
mx_num = mx_P = mx_c = 0
for trial in range(6):
    z = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    Ar = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64)); M = Ar @ Ar.conj().T / 64
    Q = M * np.outer(np.conj(z), z)
    S = rng.choice(len(patoms), 3, replace=False)
    # explicit constrained weighted LS per pair: min sum_D w |e_i - B_S P_i|^2 s.t. sum P_i = 1 (KKT)
    Bs = B[:, S]; Wd = np.sqrt(w)[:, None]
    K = np.block([[2 * (Bs * Wd).conj().T @ (Bs * Wd), np.ones((3, 1))], [np.ones((1, 3)), np.zeros((1, 1))]])
    P = np.zeros((3, 64), dtype=complex)
    for i in range(64):
        rhs = np.concatenate([2 * (Bs * Wd).conj().T @ (E[:, i] * Wd[:, 0]), [1.0]])
        P[:, i] = np.linalg.solve(K, rhs)[:3]
    mx_c = max(mx_c, np.abs(P.sum(0) - 1).max())
    Res = E - Bs @ P
    num_exp = np.real(np.einsum("ij,di,dj,d->", Q, Res, np.conj(Res), w))
    den = np.real((Q * Ew).sum())
    X = C[S] @ Q @ C[S].conj().T; u = C[S] @ Q.sum(1); q0 = np.real(Q.sum())
    num_cf = fs.constrained_num(G[np.ix_(S, S)], X, u, q0, den)
    mx_num = max(mx_num, abs(num_cf - num_exp) / den)
    mx_P = max(mx_P, np.abs(fs.constrained_P(G[np.ix_(S, S)], C[S]) - P).max())
print(f"constrained num: closed form vs explicit KKT, max rel diff {mx_num:.1e}; P max diff {mx_P:.1e}; constraint residual {mx_c:.1e}")
# greedy step 1 equals brute-force best single atom; greedy m=2 value <= best of greedy-extension (sanity)
n = 5
zs = rng.standard_normal((n, 64)) + 1j * rng.standard_normal((n, 64))
Qs = np.stack([M * np.outer(np.conj(zz), zz) for zz in zs])
Xn = C[None] @ Qs @ C.conj().T[None]; un = (C[None] @ Qs.sum(2)[:, :, None])[:, :, 0]
q0 = np.real(Qs.sum((1, 2))); den = np.real((Qs * Ew).sum((1, 2)))
Sg = fs.greedy_c(np.broadcast_to(G, (n,) + G.shape), Xn, un, q0, den, 2)
ok = True
for j in range(n):
    vals = [fs.constrained_num(G[np.ix_([a], [a])], Xn[j][np.ix_([a], [a])], un[j][[a]], q0[j], den[j]) for a in range(len(patoms))]
    ok &= int(np.argmin(vals)) == Sg[j, 0]
print("greedy first pick == brute-force best single atom:", ok)
