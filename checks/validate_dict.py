import numpy as np, pandas as pd, dict_fit as df_, pencil_fit as pf
stored = pd.read_csv("pencil_fit/pencil_errors_all.csv")
def ref(model, l, method):
    return stored[(stored.model == model) & (stored.layer == l) & (stored.window == 4096) & (stored.method == method)].err.item()
rng = np.random.default_rng(0)
for model, l in [("llama2-7b", 15), ("qwen1.5-7b", 14)]:
    W, OM, gi, hi, theta = df_.load_layer(model, l)
    L, s = 4096, 4
    KK = df_.KK_of(W, theta, L)
    pb = pf.pencil_basis(theta, L, s)
    Z = pf.esprit_poles(W, pb, (1, 2))
    for m in (1, 2):
        th, lam, _ = df_.resolve_gram(Z[m], s, W, theta, L, KK)
        e = (OM * df_.err_free(W, th, lam, theta, L, KK)).sum() / OM.sum(); r = ref(model, l, f"pencil_{m}")
        print(f"[1] {model} L{l} pencil_{m}@4096: gram {e:.6f} vs stored explicit {r:.6f}  rel diff {abs(e-r)/r:.1e}")
    for name, th2 in [("freqfold_8", np.repeat(theta.reshape(-1, 8).mean(1), 8)), ("nope_lowfreq_32", np.where(np.arange(64) < 32, theta, 0.0)),
                      ("freqfold_1", theta.copy()), ("nope_lowfreq_64", theta.copy())]:
        a, b = theta, th2; g = df_.gsum
        M = g(1j*(a[None]-a[:, None]), L) - g(1j*(b[None]-a[:, None]), L) - g(1j*(a[None]-b[:, None]), L) + g(1j*(b[None]-b[:, None]), L)
        e_it = np.einsum("ni,ij,nj->n", W, np.real(M), W) / KK
        e = (OM * e_it).sum() / OM.sum()
        print(f"[2] {model} L{l} {name}: closed-form {e:.6e} vs stored {ref(model, l, name):.6e}  max|item| {np.abs(e_it).max():.1e}")
    zd = -np.array([1e-4, 3e-4, 0, 2e-3, 5e-5]) + 1j * np.array([3e-4, 1e-3, 0.02, 0.3, 1e-5])
    out = df_.omp_layer(W[:200], theta, zd, KK[:200])
    D = np.arange(L, dtype=float); E = np.exp(1j * np.outer(theta, D))
    worst_err = worst_proj = worst_id = 0
    for it in rng.choice(200, 20, replace=False):
        K = W[it] @ E
        for m in (1, 2, 4):
            err, worst, S = out[m]
            B = np.exp(np.outer(D, zd[S[it]]))
            A = np.linalg.lstsq(B, K, rcond=None)[0]
            e_exp = np.sum(np.abs(K - B @ A) ** 2) / np.sum(np.abs(K) ** 2)
            worst_err = max(worst_err, abs(e_exp - err[it]))
            Pk = B @ np.linalg.lstsq(B, E.T, rcond=None)[0]
            r = np.sum(np.abs(E.T - Pk) ** 2, 0) / L
            big = W[it] / W[it].sum() > 0.01
            worst_proj = max(worst_proj, abs(r[big].max() - worst[it]))
            worst_id = max(worst_id, abs(np.sum(np.abs(K - Pk @ W[it]) ** 2) / np.sum(np.abs(K) ** 2) - e_exp))
    print(f"[3] {model} L{l} OMP: max |closed-form - explicit| err {worst_err:.1e}; worst-pair residual {worst_proj:.1e}; "
          f"|err(sum w_i P e_i) - err(LS on K)| {worst_id:.1e}")
    L2 = 32768; D2 = np.arange(L2, dtype=float)
    th = np.array([[3e-5, 2e-4]] * 5); lam = np.array([[1e-5, 0.0]] * 5)
    e_cf = df_.err_free(W[:5], th, lam, theta, L2)
    mx = 0
    for it in range(5):
        K = W[it] @ np.exp(1j * np.outer(theta, D2)); B = np.exp(np.outer(D2, -lam[it] + 1j * th[it]))
        A = np.linalg.lstsq(B, K, rcond=None)[0]
        mx = max(mx, abs(np.sum(np.abs(K - B @ A) ** 2) / np.sum(np.abs(K) ** 2) - e_cf[it]))
    print(f"[4] {model} L=32768 err_free closed form vs explicit: max diff {mx:.1e}")
theta = pf.thetas(1e6); L, s = 32768, 8; pb = pf.pencil_basis(theta, L, s)
for name, wd, m in [("pair 0", {0: 1.0}, 1), ("pair 1", {1: 1.0}, 1), ("pair 3", {3: 1.0}, 1), ("pair 6", {6: 1.0}, 1),
                    ("pairs 0+40", {0: .5, 40: .5}, 2), ("pairs 1+3", {1: .5, 3: .5}, 2)]:
    W = np.zeros((1, 64))
    for i, v in wd.items(): W[0, i] = v
    KK = df_.KK_of(W, theta, L)
    z = pf.esprit_poles(W, pb, (m,))[m]
    th, lam, nsw = df_.resolve_gram(z.copy(), s, W, theta, L, KK)
    got = np.sort(th[0])[::-1]; true = np.sort(theta[list(wd)])[::-1]
    d = np.abs(got - true).max()
    print(f"[5] stride 8 {name:10s}: true {np.round(true, 6)} raw {np.round(np.sort(np.angle(z[0])/s)[::-1], 6)} -> {np.round(got, 9)} "
          f"|dth| {d:.1e} |lam| {np.abs(lam).max():.1e} {'PASS' if d < 1e-6 and np.abs(lam).max() < 1e-6 else 'FAIL'}")
