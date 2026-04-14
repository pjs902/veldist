"""
fig_design_matrix.py — Design matrix illustration.

Shows the N x K design matrix M for a small synthetic dataset and highlights
three individual rows (stars with different measurement errors) to illustrate
how each star's error kernel is integrated over the velocity bins.

Does not require JAX or a running KinematicSolver.  The design matrix is
computed here with the same CDF-difference formula used internally by
precompute_design_matrix().

Run from the docs/ directory:
    python fig_design_matrix.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng(99)

# --------------------------------------------------------------------------
# Synthetic dataset: N stars on a K-bin velocity grid
# --------------------------------------------------------------------------
K = 50
v_edges = np.linspace(-150, 150, K + 1)
v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
dv = v_edges[1] - v_edges[0]

N = 60
v_true = rng.normal(0, 30, N)
errors = rng.uniform(5, 40, N)
v_obs = v_true + rng.normal(0, errors)

# Sort stars by measurement error for a cleaner matrix image
order = np.argsort(errors)
v_obs = v_obs[order]
errors = errors[order]

# --------------------------------------------------------------------------
# Build the design matrix M[i, j] = CDF(edge_{j+1}) - CDF(edge_j)
# --------------------------------------------------------------------------
M = norm.cdf(v_edges[1:][np.newaxis, :], loc=v_obs[:, np.newaxis], scale=errors[:, np.newaxis]) \
  - norm.cdf(v_edges[:-1][np.newaxis, :], loc=v_obs[:, np.newaxis], scale=errors[:, np.newaxis])

# --------------------------------------------------------------------------
# Pick three representative rows: small, medium, large error
# --------------------------------------------------------------------------
idx_small  = np.argmin(errors)
idx_medium = np.argmin(np.abs(errors - np.median(errors)))
idx_large  = np.argmax(errors)
highlight  = [idx_small, idx_medium, idx_large]
colors     = ["#2166ac", "#f4a582", "#d6604d"]
row_labels = [
    rf"Star {idx_small+1:02d}: $\varepsilon = {errors[idx_small]:.1f}$ km s$^{{-1}}$",
    rf"Star {idx_medium+1:02d}: $\varepsilon = {errors[idx_medium]:.1f}$ km s$^{{-1}}$",
    rf"Star {idx_large+1:02d}: $\varepsilon = {errors[idx_large]:.1f}$ km s$^{{-1}}$",
]

# --------------------------------------------------------------------------
# Figure: two-panel layout
# --------------------------------------------------------------------------
fig, (ax_mat, ax_rows) = plt.subplots(
    1, 2, figsize=(10, 4),
    gridspec_kw={"width_ratios": [1.6, 1]}
)

# Left panel: the full M matrix
im = ax_mat.imshow(
    M,
    aspect="auto",
    origin="upper",
    cmap="Blues",
    extent=[v_centers[0], v_centers[-1], N + 0.5, 0.5],
    interpolation="nearest",
)
# Mark the three highlighted rows
for idx, col in zip(highlight, colors):
    ax_mat.axhline(idx + 1, color=col, linewidth=1.8, alpha=0.85)

cbar = fig.colorbar(im, ax=ax_mat, pad=0.02)
cbar.set_label(r"$M_{ij}$", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax_mat.set_xlabel(r"Velocity bin centre (km s$^{-1}$)", fontsize=9)
ax_mat.set_ylabel("Star index (sorted by $\\varepsilon$)", fontsize=9)
ax_mat.set_title("Design matrix $\\mathbf{M}$", fontsize=10)
ax_mat.tick_params(labelsize=8)

# Add annotation: small errors at bottom of matrix (after sort)
ax_mat.text(0.02, 0.96, "small $\\varepsilon$", transform=ax_mat.transAxes,
            fontsize=7, va="top", color="gray")
ax_mat.text(0.02, 0.04, "large $\\varepsilon$", transform=ax_mat.transAxes,
            fontsize=7, va="bottom", color="gray")

# Right panel: three individual rows
for idx, col, lbl in zip(highlight, colors, row_labels):
    ax_rows.plot(v_centers, M[idx], color=col, linewidth=1.8, label=lbl)
    ax_rows.axvline(v_obs[idx], color=col, linewidth=0.8, linestyle=":")

ax_rows.set_xlabel(r"Velocity bin centre (km s$^{-1}$)", fontsize=9)
ax_rows.set_ylabel(r"$M_{ij}$", fontsize=9)
ax_rows.set_title("Individual rows of $\\mathbf{M}$", fontsize=10)
ax_rows.legend(fontsize=7.5, loc="upper right")
ax_rows.tick_params(labelsize=8)
ax_rows.set_xlim(v_centers[0], v_centers[-1])
ax_rows.set_ylim(bottom=0)

# Dotted vertical lines at observed velocities (all, faint)
for i in range(N):
    ax_rows.axvline(v_obs[i], color="gray", linewidth=0.3, alpha=0.2, zorder=0)

fig.tight_layout()
fig.savefig("images/fig_design_matrix.png", dpi=150, bbox_inches="tight")
print("Saved: images/fig_design_matrix.png")
