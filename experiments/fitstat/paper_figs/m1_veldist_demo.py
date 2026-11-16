"""Paper figure m1: veldist on one real MUSE bin (1D) and one real HST bin (2D).

usage: python m1_veldist_demo.py MERGED_PARQUET OUT.pdf
A MUSE LOS bin with ~150 high-quality stars and an HST PM bin with ~430 (bins from
the catalogue's own Voronoi ids; the demo does not depend on the production binning).
Panels:
  a  MUSE: measured velocities (grey histogram), the deconvolved distribution
     (median and 68% of the cell masses, as a density), and the Gaussian core
  b  rows of the design matrix M for four stars of different error: each star's
     error kernel integrated over the cells
  c  HST: the deconvolved 2D proper-motion distribution (median cell masses) with
     contours of the bivariate Gaussian of the same mean and covariance
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

from veldist import KinematicSolver
from veldist.veldist import precompute_design_matrix
from veldist.veldist2d import KinematicSolver2D

KMS_PER_MASYR = 4.74047 * 5.494  # d = 5494 pc (Haberle+25)
d = pd.read_parquet(sys.argv[1], columns=[
    "vlos", "vlos_err", "voronoi_bin_ids_los", "selection_hq_los", "pmra_corrected", "pmdec_corrected",
    "pmra_corrected_err", "pmdec_corrected_err", "voronoi_bin_ids_pm", "selection_hq_astrometry_and_membership"])

# --- MUSE bin: the one whose star count is closest to 152
los = d[d.selection_hq_los == True]  # noqa: E712
n = los.groupby("voronoi_bin_ids_los").size()
b = (n - 152).abs().idxmin()
s = los[los.voronoi_bin_ids_los == b]
v, e = s.vlos.to_numpy() - 232.7, s.vlos_err.to_numpy()
width, K = 8.018, 21  # the production MUSE grid (2 x the dataset's median error; 3.5 sigma_max + rotation span)
sol = KinematicSolver()
sol.setup_grid(center=0.0, width=K * width, n_bins=K)
sol.add_data(v, e)
sol.run(num_samples=3000, seed=0)
pdf = np.asarray(sol.samples["intrinsic_pdf"], float)
edges = np.linspace(-K * width / 2, K * width / 2, K + 1)  # the grid set up above
cen = 0.5 * (edges[1:] + edges[:-1])
dv = np.diff(edges)
lo, med, hi = np.percentile(pdf / dv, [16, 50, 84], axis=0)
mu, sd = np.average(cen, weights=med), np.sqrt(np.average((cen - np.average(cen, weights=med)) ** 2, weights=med))
core = np.diff(ndtr((edges - mu) / sd)) / dv

# --- HST bin: star count closest to 430
pm = d[d.selection_hq_astrometry_and_membership == True]  # noqa: E712
n2 = pm.groupby("voronoi_bin_ids_pm").size()
b2 = (n2 - 430).abs().idxmin()
p = pm[pm.voronoi_bin_ids_pm == b2]
v1, v2 = (p[c].to_numpy() * KMS_PER_MASYR for c in ("pmra_corrected", "pmdec_corrected"))
e1, e2 = (p[c].to_numpy() * KMS_PER_MASYR for c in ("pmra_corrected_err", "pmdec_corrected_err"))
cov = np.zeros((len(p), 2, 2))
cov[:, 0, 0], cov[:, 1, 1] = e1 ** 2, e2 ** 2
sol2 = KinematicSolver2D()
sol2.setup_grid(center=(0.0, 0.0), width=(165.6, 165.6), n_bins=23)
sol2.add_data(v1, v2, cov)
sol2.run(num_samples=3000, seed=0)
h2 = np.median(np.asarray(sol2.samples["intrinsic_pdf"], float), axis=0).reshape(23, 23)
e2d = np.asarray(sol2.grid["edges_x"], float)
c2d = 0.5 * (e2d[1:] + e2d[:-1])
X, Y = np.meshgrid(c2d, c2d, indexing="ij")
w = h2 / h2.sum()
m1, m2 = (w * X).sum(), (w * Y).sum()
C = np.array([[(w * (X - m1) ** 2).sum(), (w * (X - m1) * (Y - m2)).sum()],
              [(w * (X - m1) * (Y - m2)).sum(), (w * (Y - m2) ** 2).sum()]])
Ci = np.linalg.inv(C)
r2 = Ci[0, 0] * (X - m1) ** 2 + 2 * Ci[0, 1] * (X - m1) * (Y - m2) + Ci[1, 1] * (Y - m2) ** 2

fig, ax = plt.subplots(1, 3, figsize=(3 * plt.rcParams["figure.figsize"][0], plt.rcParams["figure.figsize"][1]),
                       constrained_layout=True)
ax[0].hist(v, bins=edges, density=True, histtype="step", color="0.55", label="measured")
ax[0].fill_between(cen, lo, hi, step="mid", color="#0072B2", alpha=0.3, lw=0)
ax[0].step(cen, med, where="mid", color="#0072B2", label="deconvolved")
ax[0].plot(cen, core, color="k", ls="--", label="Gaussian core")
ax[0].set_xlabel(r"$v_{\rm LOS} - v_{\rm sys}$ [km s$^{-1}$]")
ax[0].set_ylabel(r"density [(km s$^{-1}$)$^{-1}$]")
ax[0].legend(frameon=False, loc="upper left")
ax[0].set_title(f"MUSE bin, {len(v)} stars")
M = np.asarray(precompute_design_matrix(v, e, cen, width))
order = np.argsort(e)
pick = [order[0], order[len(e) // 3], order[2 * len(e) // 3], order[-1]]
for k, i in enumerate(pick):
    ax[1].step(cen, M[i], where="mid", color=plt.cm.Blues(0.45 + 0.15 * k),
               label=rf"$\epsilon = {e[i]:.1f}$ km s$^{{-1}}$")
    ax[1].axvline(v[i], color=plt.cm.Blues(0.45 + 0.15 * k), lw=0.6, ls=":")
ax[1].set_xlabel(r"cell centre [km s$^{-1}$]")
ax[1].set_ylabel(r"$M_{ik}$")
ax[1].legend(frameon=False, loc="upper left")
ax[1].set_title("design-matrix rows")
im = ax[2].pcolormesh(e2d, e2d, (h2 / h2.sum()).T, cmap="Blues", shading="flat")
ax[2].contour(c2d, c2d, r2.T, levels=[1, 4, 9], colors="k", linewidths=0.8)
fig.colorbar(im, ax=ax[2], label="cell mass")
ax[2].set_xlim(-60, 60)
ax[2].set_ylim(-60, 60)
ax[2].set_aspect("equal")
ax[2].set_xlabel(r"$\mu_{\alpha*}$ [km s$^{-1}$]")
ax[2].set_ylabel(r"$\mu_\delta$ [km s$^{-1}$]")
ax[2].set_title(rf"\textit{{HST}} bin, {len(p)} stars")
fig.savefig(sys.argv[2])
fig.savefig(sys.argv[2].replace(".pdf", ".png"), dpi=200)
print("saved", sys.argv[2], "MUSE bin", b, len(v), "K", K, "HST bin", b2, len(p))
