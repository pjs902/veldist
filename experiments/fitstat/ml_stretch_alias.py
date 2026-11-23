"""Cell-aliasing noise of the data-side stretch at production scale (docs/ml-reuse-investigation.md).

Under a stretch the cell edges s*e sweep across the stars. With errors well below the cell width (HST: median
1.5 km/s against 7.2 km/s cells) a flat-in-cell model gives each star a first-order error ~ (v - centre) f'/f that
sums like sqrt(N), and changes with s. A fresh library keeps the cells fixed relative to the stars, so its error
moves with the model only. Fixed model (no fitting), one axis of HST: N stars, the dump's per-star errors, h 7.2,
window +-82.8, a smooth model (two-Gaussian LOSVD). Reports, over s in [0.95, 1.05]:
  -2lnL(method) - -2lnL(exact)  detrended rms and peak-to-peak (the noise a Υ scan would see), for
  stretch and fresh, with and without sharpened rows.
    python ml_stretch_alias.py [N]
"""
import pickle
import sys

import numpy as np
from scipy.special import erf
from scipy.stats import norm

sys.path.insert(0, "/Users/pesmith/research/veldist/src")
from veldist.fitstat import sharpen_matrix  # noqa: E402

rng = np.random.default_rng(3)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 610846
d = pickle.load(open("work/hst_add_dump.pkl", "rb"))
err_all = np.concatenate([np.sqrt(b["cov"][:, 0, 0]) for b in d["bins"]])
err = rng.choice(err_all, N)
K, half = 23, 82.80638794003434
e = np.linspace(-half, half, K + 1)
h = e[1] - e[0]
# smooth model: 0.7 N(3, 14) + 0.3 N(-2, 22) (a rotating, radially broad LOSVD)
A_, M_, S_ = np.array([0.7, 0.3]), np.array([3.0, -2.0]), np.array([14.0, 22.0])
S_TRUE = 0.98
comp = rng.choice(2, N, p=A_)
v = rng.normal(S_TRUE * M_[comp], S_TRUE * S_[comp]) + rng.normal(0, err)
keep = np.abs(v) < half  # inside the window, as the export keeps stars with mass on the grid
v, err = v[keep], err[keep]
N = len(v)
SH = sharpen_matrix(K, ndim=1)


def rows(edges, vv, ee):
    z = (edges[None, :] - vv[:, None]) / (np.sqrt(2) * ee[:, None])
    return 0.5 * np.diff(erf(z), axis=1)


def masses(s):
    """cell masses on the native edges e of the model stretched by s"""
    return sum(a * np.diff(norm.cdf(e, s * m, s * sd)) for a, m, sd in zip(A_, M_, S_))


m1 = masses(1.0)
S = np.linspace(0.95, 1.05, 41)
res = {k: [] for k in ("exact", "fresh", "stretch", "fresh_sh", "stretch_sh")}
R_e = rows(e, v, err)
for s in S:
    tot = [np.hypot(s * sd, err) for sd in S_]
    res["exact"].append(-2 * np.sum(np.log(sum(a * norm.pdf(v, s * m, t) for a, m, t in zip(A_, M_, tot)))))
    ms = masses(s)
    R_s = rows(e, v / s, err / s)
    for sh, suf in ((False, ""), (True, "_sh")):
        Rf, Rs = (R_e @ SH, R_s @ SH) if sh else (R_e, R_s)
        res["fresh" + suf].append(-2 * np.sum(np.log(np.clip(Rf @ ms / h, 1e-300, None))))
        res["stretch" + suf].append(-2 * np.sum(np.log(np.clip(Rs @ m1 / h, 1e-300, None))) + 2 * N * np.log(s))
res = {k: np.array(x) for k, x in res.items()}
print(f"N = {N} stars (one axis), median err {np.median(err):.2f} km/s, h = {h:.2f}; s in [0.95, 1.05]")
print("method        offset@s=1   detrended rms   peak-to-peak (detrended)   linear slope per 0.01 in s")
for k in ("fresh", "stretch", "fresh_sh", "stretch_sh"):
    o = res[k] - res["exact"]
    p = np.polyfit(S, o, 1)
    r = o - np.polyval(p, S)
    print(f"{k:12s} {o[20]:+12.1f} {r.std():14.1f} {np.ptp(r):20.1f} {p[0] * 0.01:22.1f}")
a = res["exact"]
print(f"exact: argmin s = {S[np.argmin(a)]:.4f} (truth {S_TRUE}); curvature d2/ds2 ~ {np.polyfit(S, a, 2)[0] * 2:.3g}"
      f" -> Delta(-2lnL) per 0.01 in s near the minimum ~ {np.polyfit(S, a, 2)[0] * 1e-4:.1f}")
for k in ("fresh_sh", "stretch_sh"):
    print(f"{k}: argmin s = {S[np.argmin(res[k])]:.4f}")
