"""Paper figure m6 (appendix): the flat-in-cell bias of the per-star likelihood and its removal by sharpening.

usage: python m6_sharpening.py OUT.pdf
HST-like mocks (N = 430 stars, errors 1.5 km/s, sigma 18/16 km/s, grid +-82.8 km/s) on grids of K cells per
axis. For each mock the model family is a Gaussian of dispersions (s*18, s*16) in exact cell masses; s is fitted
by minimising the exact per-star likelihood (stat_exact) with the plain design matrix M and with the sharpened
M S. Points: mean bias of the fitted sigma_1 over the mocks, with its standard error; line: the second-order
prediction -Delta^2 / (12 sigma) for the plain matrix.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c1 import HST, gmass  # noqa: E402

from veldist.fitstat import sharpen_matrix, stat_exact  # noqa: E402
from veldist.veldist2d import KinematicSolver2D, precompute_design_matrix_2d  # noqa: E402

S1, S2, N, NMOCK = 18.0, 16.0, 430, 40
Ks = [11, 15, 19, 23, 31]
rng = np.random.default_rng(7)
res = {"plain": [], "sharp": []}
DATA = sys.argv[1].replace(".pdf", "_data.npz")
if Path(DATA).exists():  # replot only
    res = {k: list(v) for k, v in np.load(DATA).items()}
    Ks = []
for K in Ks:
    sol = KinematicSolver2D()
    sol.setup_grid(center=(0.0, 0.0), width=(2 * HST["half"],) * 2, n_bins=K)
    grid = sol.grid
    e = np.asarray(grid["edges_x"], float)
    Sh = sharpen_matrix(K)
    out = {"plain": [], "sharp": []}
    for _ in range(NMOCK):
        v = rng.normal(0, 1, (N, 2)) * [S1, S2] + rng.normal(0, HST["err"], (N, 2))
        cov = np.tile(np.eye(2) * HST["err"] ** 2, (N, 1, 1))
        M = np.asarray(precompute_design_matrix_2d(v[:, 0], v[:, 1], cov, grid), float)
        M = M[M.sum(1) >= 1e-3]
        for key, MM in (("plain", M), ("sharp", M @ Sh)):
            r = minimize_scalar(lambda s: stat_exact(gmass(e, S1 * s, S2 * s), MM, len(MM), 1.0),
                                bounds=(0.8, 1.2), method="bounded", options=dict(xatol=1e-5))
            out[key].append(S1 * r.x - S1)
    for key in res:
        b = np.array(out[key])
        res[key].append((2 * HST["half"] / K, b.mean(), b.std(ddof=1) / np.sqrt(len(b))))
    print(K, {k: tuple(round(x, 3) for x in res[k][-1]) for k in res}, flush=True)

np.savez(DATA, **{k: np.array(v) for k, v in res.items()})
w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(figsize=(w, 0.95 * h), constrained_layout=True)
# production ranges of Delta/sigma: cell width over the range of dispersions in each dataset's bins
for (lo, hi), lab, yy in (((7.2 / 20.0, 7.2 / 12.5), r"\textit{HST}", 0.96), ((8.0 / 19.0, 8.0 / 12.5), "MUSE", 0.88)):
    ax.plot([lo, hi], [yy, yy], transform=ax.get_xaxis_transform(), color="0.35", lw=3, solid_capstyle="butt")
    ax.text(hi + 0.01, yy, lab, transform=ax.get_xaxis_transform(), va="center", fontsize="small", color="0.3")
q = np.linspace(0.2, 0.9, 100)
ax.plot(q, -q ** 2 * S1 / 12, color="0.5", ls="--", lw=1, label=r"$-\Delta^2/(12\sigma)$")
for key, lab, c, mk in (("plain", r"exact, $\mathbf{M}$", "#D55E00", "s"),
                        ("sharp", r"exact, $\mathbf{M}\mathbf{S}$", "#0072B2", "o")):
    d, b, se = np.array(res[key]).T
    ax.errorbar(d / S1, b, se, color=c, marker=mk, ms=4, capsize=0, ls="none", label=lab)
ax.axhline(0, color="0.4", lw=0.6)
ax.set_xlabel(r"cell width over dispersion, $\Delta/\sigma$")
ax.set_ylabel(r"$\langle \sigma_{\rm fit} - \sigma_{\rm true} \rangle$ [km s$^{-1}$]")
ax.set_ylim(-1.1, 1.0)
ax.legend(frameon=False, loc="lower left", fontsize="small")
fig.savefig(sys.argv[1])
fig.savefig(sys.argv[1].replace(".pdf", ".png"), dpi=200)
