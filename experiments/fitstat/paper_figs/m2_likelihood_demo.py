"""Paper figure m2: what the per-star likelihood keeps, on one HST-like mock bin.

usage: python m2_likelihood_demo.py OUT.pdf
One mock bin (C1 'calib' set-up: N = 430, errors 1.5 km/s, sigma 18/16, K = 23 grid),
one veldist fit (NUTS, ~2 min). Panels:
  a  per-cell weights of the diagonal chi2, 1/sigma_k^2 (veldist half-width, floored)
  b  the per-star likelihood's information per cell, F_kk, on the same scale
  c  each statistic along a family of model histograms (both dispersions scaled by s):
     exact per-star (sharpened), Laplace about the veldist mean, Laplace re-expanded at
     its own minimum, and the diagonal chi2; each minus its own minimum.
"""
import sys
import zlib
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c1 import HST, gmass, simulate  # noqa: E402

from veldist.fitstat import (laplace_rows, laplace_terms, sharpen_matrix, stat_diag,  # noqa: E402
                             stat_exact, stat_laplace)
from veldist.veldist import _cell_mass_correction  # noqa: E402
from veldist.veldist2d import KinematicSolver2D, precompute_design_matrix_2d  # noqa: E402

S1, S2, N, SEED = 18.0, 16.0, 430, 3
rng = np.random.default_rng([SEED, zlib.crc32(b"calib")])
v = simulate(rng, HST, N, S1, S2, 0.0, 0.0, 0.0)
cov = np.tile(np.eye(2) * HST["err"] ** 2, (len(v), 1, 1))
sol = KinematicSolver2D()
sol.setup_grid(center=(0.0, 0.0), width=(2 * HST["half"],) * 2, n_bins=HST["K"])
sol.add_data(v[:, 0], v[:, 1], cov)
sol.run(num_samples=3000, seed=SEED)
sol.clip_uncertainties()
K = HST["K"]
e = np.asarray(sol.grid["edges_x"], float)
c = 0.5 * (e[1:] + e[:-1])
med, sig = sol.clipped_samples["pdf_median"], sol.clipped_samples["pdf_uncertainty"]
d, sd = med / med.sum(), sig / med.sum()
mean = _cell_mass_correction(np.asarray(sol.samples["intrinsic_pdf"], float), sol.grid["shape"]).mean(0)
h0 = mean / mean.sum()
M = precompute_design_matrix_2d(v[:, 0], v[:, 1], cov, sol.grid).astype(float)
M = M[M.sum(1) >= 1e-3] @ sharpen_matrix(K)
n = len(M)


def lap(h_ref):
    F, g = laplace_terms(M, h_ref, n, 1.0)
    Lt, tg = laplace_rows(F, g, h_ref, n_stars=n)
    return (lambda h: stat_laplace(h, Lt, tg, h0=h_ref)), F


family = lambda s: gmass(e, S1 * s, S2 * s)
sgrid = np.linspace(0.9, 1.1, 81)
lap0, F0 = lap(h0)
stats = {"exact": lambda h: stat_exact(h, M, n, 1.0), "lap0": lap0,
         "diag": lambda h: stat_diag(h, d, sd)}
curves = {k: np.array([f(family(s)) for s in sgrid]) for k, f in stats.items()}
s_min = sgrid[np.argmin(curves["lap0"])]
lap1, _ = lap(0.999 * family(s_min) + 0.001 / K ** 2)
curves["lap1"] = np.array([lap1(family(s)) for s in sgrid])

fig, ax = plt.subplots(1, 3, figsize=(3.3 * plt.rcParams["figure.figsize"][0], plt.rcParams["figure.figsize"][1]), constrained_layout=True)
ext = [e[0], e[-1], e[0], e[-1]]
# relative information per cell = effective number of stars the statistic credits to it:
# diagonal chi2 (d_k / sigma_k)^2 against the per-star likelihood's h0_k^2 F_kk
wa = ((d / sd) ** 2).reshape(K, K)
wb = (h0 ** 2 * np.diag(F0)).reshape(K, K)
lo, hi = 0.1, max(wa.max(), wb.max())
for a, z, lab, ttl in ((ax[0], wa, r"$(d_k/\sigma_k)^2$", r"diagonal $\chi^2$"),
                       (ax[1], wb, r"$(h^0_k)^2 F_{kk}$", "per-star likelihood")):
    a.set_title(ttl)
    im = a.imshow(np.clip(z, lo, None).T, origin="lower", extent=ext, cmap="Blues",
                  norm=mcolors.LogNorm(vmin=lo, vmax=hi))
    fig.colorbar(im, ax=a, shrink=0.85, label=lab, pad=0.02)
print("effective stars: diag sum %.0f, per-star sum %.0f, N %d" % (wa.sum(), wb.sum(), n))
for a in ax[:2]:
    a.set_xlim(-60, 60)
    a.set_ylim(-60, 60)
    a.set_xlabel(r"$v_1$ [km s$^{-1}$]")
    a.set_ylabel(r"$v_2$ [km s$^{-1}$]")
sty = {"exact": ("per-star, exact", "#009E73", "-"), "lap0": ("Laplace about veldist", "#0072B2", "--"),
       "lap1": ("Laplace, re-expanded", "#0072B2", "-"), "diag": (r"diagonal $\chi^2$", "#D55E00", "-")}
for k in ("diag", "lap0", "lap1", "exact"):
    lab, col, ls = sty[k]
    y = curves[k] - curves[k].min()
    ax[2].plot(sgrid * S1, y, color=col, ls=ls, lw=1.6 if k == "exact" else 1.0, label=lab)
ax[2].axhline(1, color="0.5", lw=0.6, ls=":")
ax[2].axvline(S1, color="k", lw=0.6)
ax[2].set_ylim(0, 12)
ax[2].set_xlabel(r"model $\sigma_1$ [km s$^{-1}$]")
ax[2].set_ylabel(r"$\Delta$ statistic")
ax[2].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.savefig(sys.argv[1])
fig.savefig(sys.argv[1].replace(".pdf", ".png"), dpi=200)
mins = {k: sgrid[np.argmin(v)] * S1 for k, v in curves.items()}
print("saved", sys.argv[1], "minima (km/s):", {k: round(float(x), 2) for k, x in mins.items()})
