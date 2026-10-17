"""Arms U / S / C for the LittleBit k_proj ablation on Qwen1.5-0.5B.

U: LittleBit as published (their LittleBitLinear, unmodified).
S: pair-structured k_proj: U_sign is block-sparse (block i = rows of RoPE pair i over all heads), V stacked per block.
C: as S, but inside a block the latent dims are paired and the up-projection is [a -b; b a] per (head, complex latent).

All use the same forward as LittleBit:  y = ((((x*v2) @ Vq^T) * (v1*u2)) @ Uq^T) * u1 (+ residual path) + bias.
"""
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/LittleBit")
from quantization.functions.binary import SmoothSign  # noqa: E402

NH, DH, HALF = 16, 64, 32
OUT = IN = 1024


def pair_rows(i):
    """rows of RoPE pair i over all heads: (re rows, im rows), HF rotate_half pairing (j, j+32)."""
    re = torch.tensor([h * DH + i for h in range(NH)])
    return re, re + HALF


def alloc_ranks(C, r):
    """greedy pooled-eigenvalue allocation: top-r eigenvalues pooled over the 32 pair slices of C (32x32 each)."""
    C = np.asarray(C, dtype=np.float64)
    ev, tag = [], []
    for i in range(HALF):
        re, im = pair_rows(i)
        idx = torch.cat([re, im]).numpy()
        e = np.linalg.eigvalsh(C[np.ix_(idx, idx)])
        ev.append(e); tag.append(np.full(len(e), i))
    ev, tag = np.concatenate(ev), np.concatenate(tag)
    order = np.argsort(-ev, kind="stable")[:r]
    return np.bincount(tag[order], minlength=HALF)


def even_ranks(rk):
    """round odd r_i down to even and give the leftover units back to the pairs with the largest dropped remainder
    (here: pairs whose r_i was odd, in order of increasing index among those with the most rank) keeping the total."""
    rk = np.array(rk); tot = rk.sum(); assert tot % 2 == 0
    out = rk // 2 * 2
    left = (tot - out.sum()) // 2
    odd = np.flatnonzero(rk % 2 == 1)
    # bump the 'left' odd pairs with the largest r_i (they lost the least relatively)
    for i in odd[np.argsort(-rk[odd], kind="stable")][:left]:
        out[i] += 2
    assert out.sum() == tot and (out % 2 == 0).all()
    return out


def _rank1_abs(M):
    """nonnegative rank-1 factors (a: rows, b: cols) with a b^T ~ M for M >= 0 (leading singular pair, sqrt split)."""
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    a, b = U[:, 0] * S[0].sqrt(), Vh[0] * S[0].sqrt()
    if a.sum() < 0:
        a, b = -a, -b
    return a, b


class StructLinear(nn.Module):
    """Pair-structured (kind 'S') or complex-latent ('C') LittleBit linear for a 1024x1024 k_proj, two paths."""

    def __init__(self, W, bias, ranks, kind, quant_func=SmoothSign):
        super().__init__()
        assert kind in ("S", "C")
        self.kind, self.quant_func = kind, quant_func
        self.ranks = [int(x) for x in ranks]
        self.r = int(sum(self.ranks))
        self.in_features, self.out_features = IN, OUT
        self.register_buffer("rank_vec", torch.tensor(self.ranks))
        self.bias = nn.Parameter(bias.detach().clone().float()) if bias is not None else None
        self.offs = np.concatenate([[0], np.cumsum(self.ranks)])
        if kind == "C":
            assert all(x % 2 == 0 for x in self.ranks), "complex arm needs even r_i"
            self.K = self.r // 2
        self._build_index()
        self.cuda()
        W = W.detach().float().cuda()
        self._init_path("", W)
        with torch.no_grad():
            pri = self._recon("").detach()
        self._init_path("_R", W - pri)
        self.to("cpu")

    # ---- index structure -------------------------------------------------------------------------------------------
    def _build_index(self):
        r = self.r
        if self.kind == "S":
            mask = torch.zeros(OUT, r)
            for i, ri in enumerate(self.ranks):
                re, im = pair_rows(i)
                rows = torch.cat([re, im])
                mask[rows[:, None], torch.arange(self.offs[i], self.offs[i] + ri)[None]] = 1
            self.register_buffer("mask", mask)
        else:
            K = self.K
            colblock = torch.cat([torch.full((self.ranks[i] // 2,), i) for i in range(HALF)])        # block of complex latent j
            col0 = torch.cat([torch.arange(self.offs[i], self.offs[i + 1], 2) for i in range(HALF)])  # first real dim of j
            h = torch.arange(NH)[:, None]
            row_re = h * DH + colblock[None]                    # (NH, K)
            self.register_buffer("c_col0", col0)
            self.register_buffer("c_rre", row_re)
            self.register_buffer("c_rim", row_re + HALF)
            # share h across the (re, im) rows of a head pair: row -> (h, i) index; l across the 2 real dims
            row2hp = torch.empty(OUT, dtype=torch.long)
            for hh in range(NH):
                for j in range(DH):
                    row2hp[hh * DH + j] = hh * HALF + (j % HALF)
            self.register_buffer("row2hp", row2hp)

    # ---- parameters ------------------------------------------------------------------------------------------------
    def _P(self, name, t):
        self.register_parameter(name, nn.Parameter(t.detach().clone().float()))

    def _init_path(self, sfx, T):
        """Dual-SVID of T (1024x1024) with this arm's block structure."""
        r = self.r
        T = T.double()
        if self.kind == "S":
            U = torch.zeros(OUT, r, dtype=torch.float64, device=T.device)
            V = torch.zeros(r, IN, dtype=torch.float64, device=T.device)
            u1 = torch.zeros(OUT, dtype=torch.float64, device=T.device)
            u2 = torch.zeros(r, dtype=torch.float64, device=T.device)
            for i, ri in enumerate(self.ranks):
                if ri == 0:
                    continue
                re, im = pair_rows(i)
                rows = torch.cat([re, im]).to(T.device)
                Uu, S, Vh = torch.linalg.svd(T[rows], full_matrices=False)
                s = S[:ri].sqrt()
                Ub, Vb = Uu[:, :ri] * s, s[:, None] * Vh[:ri]
                o = self.offs[i]
                U[rows[:, None], torch.arange(o, o + ri, device=T.device)[None]] = Ub
                V[o:o + ri] = Vb
                a, b = _rank1_abs(Ub.abs())
                u1[rows] = a; u2[o:o + ri] = b
            v1, v2 = _rank1_abs(V.abs())
            self._P("U" + sfx, U); self._P("V" + sfx, V)
            self._P("u1" + sfx, u1[None]); self._P("u2" + sfx, u2[None]); self._P("v1" + sfx, v1[None]); self._P("v2" + sfx, v2[None])
        else:
            K = self.K
            A = torch.zeros(NH, K, dtype=torch.float64, device=T.device); B = torch.zeros_like(A)
            V = torch.zeros(self.r, IN, dtype=torch.float64, device=T.device)
            hp = torch.zeros(NH * HALF, dtype=torch.float64, device=T.device)
            l = torch.zeros(K, dtype=torch.float64, device=T.device)
            Vmean = torch.zeros(K, IN, dtype=torch.float64, device=T.device)
            ko = 0
            for i, ri in enumerate(self.ranks):
                k = ri // 2
                if k == 0:
                    continue
                re, im = pair_rows(i)
                Tc = torch.complex(T[re.to(T.device)], T[im.to(T.device)])         # (NH, IN) complex
                Uu, S, Vh = torch.linalg.svd(Tc, full_matrices=False)
                s = S[:k].sqrt()
                Ac, Bc = Uu[:, :k] * s, s[:, None] * Vh[:k]                       # Tc ~ Ac @ Bc
                o = self.offs[i]
                A[:, ko:ko + k], B[:, ko:ko + k] = Ac.real, Ac.imag
                V[o:o + ri:2], V[o + 1:o + ri:2] = Bc.real, Bc.imag               # latent (2j, 2j+1) = (Re, Im) of B_j
                a, b = _rank1_abs((Ac.real.abs() + Ac.imag.abs()) / 2)             # tied model: rank-1 of the mean
                hp[torch.arange(NH, device=T.device) * HALF + i] = a
                l[ko:ko + k] = b
                Vmean[ko:ko + k] = (Bc.real.abs() + Bc.imag.abs()) / 2
                ko += k
            v1k, v2 = _rank1_abs(Vmean)                                            # v1 shared by the 2 real dims, v2 shared
            self._P("A" + sfx, A); self._P("B" + sfx, B); self._P("V" + sfx, V)
            self._P("hp" + sfx, hp); self._P("l" + sfx, l); self._P("v1k" + sfx, v1k); self._P("v2" + sfx, v2[None])

    # ---- forward ---------------------------------------------------------------------------------------------------
    def _parts(self, sfx):
        """(Uq (out x r) in +-1/0, V param, u1 (1,out), lat (1,r) = v1*u2, v2 (1,in))."""
        if self.kind == "S":
            Uq = self.quant_func(getattr(self, "U" + sfx)) * self.mask
            return Uq, getattr(self, "V" + sfx), getattr(self, "u1" + sfx), getattr(self, "v1" + sfx) * getattr(self, "u2" + sfx), getattr(self, "v2" + sfx)
        sa, sb = self.quant_func(getattr(self, "A" + sfx)), self.quant_func(getattr(self, "B" + sfx))
        Uq = torch.zeros(OUT, self.r, dtype=sa.dtype, device=sa.device)
        c0 = self.c_col0[None].expand_as(self.c_rre)
        Uq = Uq.index_put((self.c_rre, c0), sa).index_put((self.c_rre, c0 + 1), -sb)
        Uq = Uq.index_put((self.c_rim, c0), sb).index_put((self.c_rim, c0 + 1), sa)
        u1 = getattr(self, "hp" + sfx)[self.row2hp][None]
        lat = (getattr(self, "v1k" + sfx) * getattr(self, "l" + sfx)).repeat_interleave(2)[None]
        return Uq, getattr(self, "V" + sfx), u1, lat, getattr(self, "v2" + sfx)

    def _path(self, x, sfx):
        Uq, V, u1, lat, v2 = self._parts(sfx)
        Vq = self.quant_func(V.to(x.dtype))
        return ((((x * v2.to(x.dtype)) @ Vq.t()) * lat.to(x.dtype)) @ Uq.to(x.dtype).t()) * u1.to(x.dtype)

    def _recon(self, sfx):
        Uq, V, u1, lat, v2 = self._parts(sfx)
        Vq = self.quant_func(V)
        return (u1.t() * Uq) @ ((lat.t() * Vq) * v2)

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, shp[-1])
        y = self._path(x, "") + self._path(x, "_R")
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.reshape(*shp[:-1], OUT)

    def reconstruct(self):
        return self._recon("") + self._recon("_R")

    # ---- storage ---------------------------------------------------------------------------------------------------
    def stored_bits(self):
        """bits actually stored for the weight (bias excluded; scales fp16; sign bits only for non-structural entries)."""
        r = self.r
        if self.kind == "S":
            sign = 32 * r + r * IN                      # U blocks: 32 rows x r_i; V: r x in
            scales = 16 * (OUT + r + IN)                # h per row, l per latent, g per input
        else:
            sign = 2 * NH * self.K + r * IN             # a,b per (head, complex latent); V
            scales = 16 * (OUT // 2 + r // 2 + IN)      # h per head pair, l per complex latent, g per input
        return 2 * (sign + scales)                      # two paths


def littlebit_bits(m):
    """published LittleBit bits for a patched LittleBitLinear (their eff-bit formula incl. both paths)."""
    a, b, s = m.in_features, m.out_features, m.split_dim
    return (s * 2 * (a + b + 16) + 32 * (a + b)) if m.residual else (s * (a + b + 16) + 16 * (a + b))
