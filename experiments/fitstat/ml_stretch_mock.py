"""Can an orbit library be reused across M/L for the price of a weight solve? 1D mock (user 2026-11-23).

Scaling every mass by k = ml/ml0 keeps the orbits and multiplies their velocities by s = sqrt(k), so a library's cell
masses on its native edges e are exactly the new model's cell masses on the edges s*e. Four ways to score data at s:
  exact    continuous likelihood of the stretched components (the truth; no cells)
  fresh    a library integrated at s: cell masses of the stretched components on the native edges e
  stretch  data side: keep the library's masses, integrate each star's error kernel over s*e, density /(s h)
  spline / linear   model side (dynamite cdf_regrid_matrix): regrid the masses from s*e back to e
"Orbits" are Gaussian components; the data are drawn from a weighted mix at s_true. For each method and each s the
weights are fitted (EM, non-negative, sum 1) and -2lnL in data units is reported. A method is usable for reuse if its
-2lnL(s) curve tracks fresh: same minimum, offsets well below the model differences that matter (~100-700).
    python ml_stretch_mock.py [--plot work/ml_stretch_mock.png]
"""
import sys

import numpy as np
from scipy.special import erf
from scipy.stats import norm

sys.path.insert(0, "/Users/pesmith/research/dynamite-fitstat")
from dynamite.kinematics import cdf_regrid_matrix  # noqa: E402

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d  # noqa: E731
rng = np.random.default_rng(arg("--seed", 1))
H, NC = arg("--h", 5.0), arg("--nc", 40)            # cell width km/s (h/sigma ~ 0.3-0.7), window +-NC*H/2
e = (np.arange(NC + 1) - NC / 2) * H
if "--narrow" in sys.argv:  # orbit-like: 24 narrow components (sigma 1.5-3 km/s, below the cell width)
    mu = np.linspace(-36, 36, 24) + rng.uniform(-H / 2, H / 2, 24)  # random phase against the cells
    sg = np.tile([arg("--sn", 1.5), 2 * arg("--sn", 1.5)], 12)
else:  # 6 broad components: rigid enough that s is identifiable
    mu = np.repeat(np.linspace(-15, 15, 3), 2)
    sg = np.tile([7.0, 14.0], 3)
w_true = rng.dirichlet(np.ones(mu.size))
S_TRUE, N = np.sqrt(0.93), 40000                    # data at ml = 0.93 ml0
comp = rng.choice(mu.size, N, p=w_true)
err = rng.uniform(2, 8, N)
v = rng.normal(S_TRUE * mu[comp], S_TRUE * sg[comp]) + rng.normal(0, err)
keep = np.abs(v) < e[-1] + 3 * err                  # fixed star set for every s (as the export's row cut at s=1)
v, err = v[keep], err[keep]
if "--sharpen" in sys.argv:
    sys.path.insert(0, "/Users/pesmith/research/veldist/src")
    from veldist.fitstat import sharpen_matrix
    SH = sharpen_matrix(NC, ndim=1)
else:
    SH = None


def cellmass(m, s, edges):
    """(J, C) cell masses of N(m, s) on edges."""
    c = norm.cdf((edges[None, :] - m[:, None]) / s[:, None])
    return np.diff(c, axis=1)


def rows(edges):
    """(N, C) each star's error-kernel mass in each cell (veldist precompute_design_matrix)."""
    z = (edges[None, :] - v[:, None]) / (np.sqrt(2) * err[:, None])
    M = 0.5 * np.diff(erf(z), axis=1)
    return M @ SH if SH is not None else M  # production sharpens the rows (cell units: the same S at every s)


def em(A, it=arg("--it", 400)):
    A = 0.999 * A + 0.001 / (NC * H)  # per-star outlier floor (uniform over the window), the same for every method
    w = np.full(A.shape[1], 1 / A.shape[1])
    for _ in range(it):
        p = A @ w
        w = np.clip(w * (A / np.clip(p, 1e-300, None)[:, None]).mean(0), 0, None)  # clip: sharpened rows
    p = A @ w
    return -2 * np.sum(np.log(np.clip(p, 1e-300, None))), (p <= 0).sum()


m0 = cellmass(mu, sg, e)                            # the native library (s = 1)
R_e = rows(e)
out = {k: [] for k in ("exact", "fresh", "stretch", "spline", "linear")}
neg = []
S = np.linspace(0.92, 1.03, 23)
for s in S:
    tot = np.sqrt((s * sg[None, :]) ** 2 + err[:, None] ** 2)
    out["exact"].append(em(norm.pdf(v[:, None], s * mu[None, :], tot))[0])
    out["fresh"].append(em(R_e @ cellmass(s * mu, s * sg, e).T / H)[0])
    out["stretch"].append(em(rows(s * e) @ m0.T / (s * H))[0])
    for kind in ("spline", "linear"):
        Rg = cdf_regrid_matrix(s * e, e, kind=kind)
        mr = m0 @ Rg.T
        if kind == "spline":
            neg.append(int((mr < -1e-12).sum()))
        out[kind].append(em(R_e @ mr.T / H)[0])
out = {k: np.array(x) for k, x in out.items()}
print(f"{len(v)} stars, {mu.size} components, h = {H} km/s; truth s = {S_TRUE:.4f} (ml ratio 0.93)")
print("method   argmin s   -2lnL(s) - exact(s): at truth, range over s;  curve shape vs fresh (max |d - d_fresh|)")
d_f = out["fresh"] - out["fresh"].min()
for k, x in out.items():
    i = np.argmin(x)
    off = x - out["exact"]
    j = np.argmin(np.abs(S - S_TRUE))
    shape = np.max(np.abs((x - x.min()) - d_f))
    bend = np.ptp(off)  # how far the curve bends away from exact over the scan: the bias that moves the ml minimum
    print(f"{k:8s} {S[i]:.4f}   {off[j]:+9.1f}  [{off.min():+.1f}, {off.max():+.1f}]   {shape:8.1f}   bend {bend:6.1f}")
j1 = np.argmin(np.abs(S - 1))
print("TREND (offset from exact minus its value at s=1): s=" + " ".join(f"{S[i]:.3f}" for i in (0, 3, 6, 9, 22)))
for k, x in out.items():
    o = x - out["exact"]
    o = o - o[j1]
    print(f"TREND {k:8s} " + " ".join(f"{o[i]:+7.1f}" for i in (0, 3, 6, 9, 22)))
print("spline regrid: negative model cells (of J x C) per s:", neg)
if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    for k, x in out.items():
        ax[0].plot(S ** 2, x - out["exact"].min(), label=k)
        ax[1].plot(S ** 2, x - out["exact"], label=k)
    for a in ax:
        a.axvline(S_TRUE ** 2, color="0.6", lw=0.6)
        a.set_xlabel(r"$\Upsilon/\Upsilon_0$")
    ax[0].set_ylabel(r"$-2\ln\mathcal{L}$ - min(exact)")
    ax[1].set_ylabel("minus exact at the same $s$")
    ax[0].legend()
    fig.savefig(sys.argv[sys.argv.index("--plot") + 1])
