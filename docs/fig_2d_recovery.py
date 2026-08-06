"""
fig_2d_recovery.py — 2D (proper-motion) Bayesian deconvolution, end to end.

Mirrors fig_deconvolution.py for the 2D solver: a true 2D velocity
distribution, the observed proper motions (true convolved with per-star
covariance), and the posterior recovered by KinematicSolver2D, run for
real (not a schematic).

Uses HST_FAINT from calibration2d.py, not HST_BRIGHT: HST_BRIGHT's real
err/sigma (~0.014) is calibrated so small that there is almost nothing to
deconvolve, which makes it a poor demo of what deconvolution buys you.
HST_FAINT (err/sigma ~0.147, validated PASS in validation.md) is close to
the 1D line-of-sight regime and is where the comparison to a naive
(non-deconvolved) estimate is actually informative. An earlier version of
this script invented its own error scale (a *30.0 fudge on err_median)
instead of calling ObservingProfile2D.draw_errors, which put it ~30x
outside any validated regime and produced a demo whose recovered cov_xx
was worse than a naive sample covariance -- always compare against
draw_errors()'s calibrated distribution, not an ad hoc one.

The figure reports both the posterior uncertainty on the recovered
mean/covariance (from the per-sample PMF, not just the point estimate) and
a naive (no-deconvolution) sample mean/covariance of the observed data
side by side, so "does this beat the naive estimator" is answered by the
figure itself rather than left to eyeballing contours.

Requires JAX, NumPyro, and a working KinematicSolver2D installation.
Run from the docs/ directory:
    python fig_2d_recovery.py
"""

import sys
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
# Ignore any user matplotlibrc (e.g. a personal MNRAS print style) so this
# figure renders identically regardless of whose machine builds the docs.
matplotlib.rcdefaults()
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from veldist.veldist2d import KinematicSolver2D
from veldist.calibration2d import HST_FAINT

rng = np.random.default_rng(11)

# --------------------------------------------------------------------------
# True 2D velocity distribution: a mildly anisotropic, correlated Gaussian
# scaled to the profile's own sigma_ref, so the grid (also derived from
# sigma_ref) comfortably encloses it.
# --------------------------------------------------------------------------
sr = HST_FAINT.sigma_ref
true_mean = np.array([0.3 * sr, -0.2 * sr])
true_cov = np.array([[(1.1 * sr) ** 2, 0.35 * (1.1 * sr) * (0.7 * sr)], [0.35 * (1.1 * sr) * (0.7 * sr), (0.7 * sr) ** 2]])

N = HST_FAINT.n_stars
v_int = rng.multivariate_normal(true_mean, true_cov, size=N)

# Per-star errors from the profile's own calibrated distribution (log-normal
# about err_median, truncated at the quality cut) -- not an invented scale.
sigma_err = HST_FAINT.draw_errors(N, rng)
cov_err = np.zeros((N, 2, 2))
cov_err[:, 0, 0] = sigma_err**2
cov_err[:, 1, 1] = sigma_err**2

pm1 = v_int[:, 0] + rng.normal(0.0, sigma_err)
pm2 = v_int[:, 1] + rng.normal(0.0, sigma_err)

# --------------------------------------------------------------------------
# Naive (no deconvolution) estimate: the observed sample mean/covariance.
# This is what a plain 2D KDE's first/second moments give you too --
# bandwidth changes the smoothing, not the empirical mean/covariance.
# --------------------------------------------------------------------------
naive_mean = np.array([pm1.mean(), pm2.mean()])
naive_cov = np.cov(np.vstack([pm1, pm2]))

# --------------------------------------------------------------------------
# Run inference
# --------------------------------------------------------------------------
solver = KinematicSolver2D()
solver.setup_grid(center=(0.0, 0.0), width=(HST_FAINT.grid_width,) * 2, n_bins=27)
solver.add_data(pm1=pm1, pm2=pm2, cov=cov_err)
solver.run(num_warmup=400, num_samples=800, gpu=False)
solver.clip_uncertainties()

pdf_mass = np.asarray(solver.samples["intrinsic_pdf"])  # (n_samples, K**2)
centers_2d = solver.grid["centers_2d"]

# Per-posterior-sample mean/covariance (not just the median PMF), so the
# recovered mean/covariance carry their own posterior uncertainty rather
# than being reported as bare point estimates.
means = np.einsum("sk,kd->sd", pdf_mass, centers_2d)
d = centers_2d[None, :, :] - means[:, None, :]
covs = np.einsum("sk,skd,ske->sde", pdf_mass, d, d)
rec_mean, rec_mean_unc = means.mean(axis=0), means.std(axis=0)
rec_cov, rec_cov_unc = covs.mean(axis=0), covs.std(axis=0)

med = np.median(pdf_mass, axis=0)
med /= med.sum()
shape = solver.grid["shape"]
# centers_2d flattens as m = ix*K + iy (x-major; see setup_grid_2d), so
# med.reshape(shape) is med_grid[ix, iy]. contourf/pcolormesh expect
# Z[iy, ix], so transpose before plotting or the recovered density comes
# out visibly rotated/sheared relative to the true contours even though
# the moments above (computed directly from centers_2d) are unaffected.
med_grid = med.reshape(shape).T
cx, cy = solver.grid["centers_x"], solver.grid["centers_y"]

# --------------------------------------------------------------------------
# Plot: observed scatter (a) vs. recovered contours against true contours
# on the same footing (b), so the comparison doesn't rely on reading a
# heatmap against smooth lines.
# --------------------------------------------------------------------------
from scipy.stats import multivariate_normal

xx, yy = np.meshgrid(cx, cy)
true_density = multivariate_normal(true_mean, true_cov).pdf(np.dstack([xx, yy]))

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))

ax = axes[0]
ax.scatter(pm1, pm2, s=6, alpha=0.35, color="gray", label=f"Observed ($N={N}$ stars)")
ax.contour(cx, cy, true_density, levels=5, colors="black", linewidths=1.2)
ax.set_title("(a) Observed proper motions\n(true contours overlaid)", fontsize=9)
ax.set_xlabel(r"$v_{\mathrm{pm},1}$ (km s$^{-1}$)", fontsize=8)
ax.set_ylabel(r"$v_{\mathrm{pm},2}$ (km s$^{-1}$)", fontsize=8)
ax.legend(fontsize=7, loc="upper right")

ax = axes[1]
ax.contour(cx, cy, true_density, levels=5, colors="black", linewidths=1.4, linestyles="--", zorder=3)
ax.contourf(cx, cy, med_grid, levels=8, cmap="Blues", zorder=1)
ax.contour(cx, cy, med_grid, levels=5, colors="#1f77b4", linewidths=1.0, alpha=0.6, zorder=2)
ax.set_title("(b) Recovered (blue) vs. true (dashed black)", fontsize=9)
ax.set_xlabel(r"$v_{\mathrm{pm},1}$ (km s$^{-1}$)", fontsize=8)
ax.set_ylabel(r"$v_{\mathrm{pm},2}$ (km s$^{-1}$)", fontsize=8)


def fmt_cov(c):
    return f"[{c[0, 0]:.0f}, {c[0, 1]:.0f}; {c[1, 1]:.0f}]"


info = (
    f"true mean:      ({true_mean[0]:+.1f}, {true_mean[1]:+.1f})\n"
    f"recovered mean: ({rec_mean[0]:+.1f}, {rec_mean[1]:+.1f}) "
    f"+/- ({rec_mean_unc[0]:.1f}, {rec_mean_unc[1]:.1f})\n"
    f"naive mean:     ({naive_mean[0]:+.1f}, {naive_mean[1]:+.1f})\n"
    f"true cov:       {fmt_cov(true_cov)}\n"
    f"recovered cov:  {fmt_cov(rec_cov)}\n"
    f"  posterior unc. on diag: ({rec_cov_unc[0, 0]:.0f}, {rec_cov_unc[1, 1]:.0f})\n"
    f"naive cov:      {fmt_cov(naive_cov)}"
)
for ax in axes:
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

fig.subplots_adjust(bottom=0.3, wspace=0.3)
fig.text(
    0.5,
    0.02,
    info,
    fontsize=7,
    family="monospace",
    va="bottom",
    ha="center",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
)
fig.savefig("images/fig_2d_recovery.png", dpi=150, bbox_inches="tight")
print("Saved: images/fig_2d_recovery.png")

print("\n--- naive vs deconvolved |bias| ---")
print("mean:  naive", np.abs(naive_mean - true_mean), " deconv", np.abs(rec_mean - true_mean))
print("cov_xx: naive", abs(naive_cov[0, 0] - true_cov[0, 0]), " deconv", abs(rec_cov[0, 0] - true_cov[0, 0]), " unc", rec_cov_unc[0, 0])
print("cov_yy: naive", abs(naive_cov[1, 1] - true_cov[1, 1]), " deconv", abs(rec_cov[1, 1] - true_cov[1, 1]), " unc", rec_cov_unc[1, 1])
print("cov_xy: naive", abs(naive_cov[0, 1] - true_cov[0, 1]), " deconv", abs(rec_cov[0, 1] - true_cov[0, 1]), " unc", rec_cov_unc[0, 1])
