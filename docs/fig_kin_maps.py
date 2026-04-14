"""
fig_kin_maps.py — Kinematic maps from a synthetic rotating cluster.

Generates V, sigma, skewness (gamma_1), and excess kurtosis (kappa) maps
from a synthetic globular cluster with a solid-body rotating core and a
non-rotating outer halo.  Used in examples.md Example 3.

Requires JAX, NumPyro, and a working KinematicSolver installation.

Run from the docs/ directory:
    python fig_kin_maps.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

rng = np.random.default_rng(2025)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# --------------------------------------------------------------------------
# Synthetic cluster: solid-body rotating core + non-rotating halo
# --------------------------------------------------------------------------
# Spatial layout: 7x7 Voronoi-like grid of bins
nx, ny = 7, 7
x_arcsec = np.linspace(-18, 18, nx)
y_arcsec = np.linspace(-18, 18, ny)
xbin, ybin = np.meshgrid(x_arcsec, y_arcsec)
xbin = xbin.ravel()
ybin = ybin.ravel()
n_bins = len(xbin)

# Rotation: V_rot = A * x for |r| < r_core, falling off outside
r = np.sqrt(xbin**2 + ybin**2)
r_core = 12.0       # arcsec
A_rot  = 2.5        # km/s per arcsec within core
v_sys  = 0.0

r_safe = np.where(r > 0, r, 1.0)   # avoid division by zero at origin
v_rot = np.where(r < r_core, A_rot * xbin, A_rot * r_core * xbin / r_safe)
v_rot = np.where(r == 0, 0.0, v_rot)
sigma_profile = 20.0 + 5.0 * np.exp(-(r / 8.0)**2)   # central peak

# Stars per bin (more in the centre)
n_stars_per_bin = np.clip(
    (80 * np.exp(-(r / 10.0)**2) + 30).astype(int), 20, 120
)

print(f"Synthetic cluster: {n_bins} bins, {n_stars_per_bin.sum()} total stars")

# --------------------------------------------------------------------------
# Build bin_data_list
# --------------------------------------------------------------------------
bin_data_list = []
for i in range(n_bins):
    n  = n_stars_per_bin[i]
    vi = rng.normal(v_rot[i], sigma_profile[i], n)
    ei = rng.uniform(4.0, 10.0, n)
    vi_obs = vi + rng.normal(0, ei)
    bin_data_list.append({"vel": vi_obs, "err": ei})

# --------------------------------------------------------------------------
# Run batch inference
# --------------------------------------------------------------------------
try:
    from veldist import fit_all_bins
    from veldist.analysis import compute_summary_maps

    print("Running fit_all_bins ...")
    solvers = fit_all_bins(
        bin_data_list,
        grid_kwargs={"center": 0.0, "width": 200.0, "n_bins": 50},
        run_kwargs={"num_warmup": 300, "num_samples": 600, "gpu": False, "seed": 42},
        min_stars=15,
    )

    print("Computing summary maps ...")
    maps = compute_summary_maps(solvers)

except Exception as e:
    print(f"\nInference failed: {type(e).__name__}: {e}")
    print("Saving placeholder figure; re-run once JAX/NumPy are available.")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.text(
        0.5, 0.6,
        "fig_kin_maps — inference not available",
        ha="center", va="center", fontsize=13, color="#555555",
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.38,
        f"{type(e).__name__}: {e}",
        ha="center", va="center", fontsize=8, color="#888888",
        transform=ax.transAxes, wrap=True,
    )
    fig.savefig("images/fig_kin_maps.png", dpi=150, bbox_inches="tight")
    print("Saved placeholder: images/fig_kin_maps.png")
    sys.exit(0)

# --------------------------------------------------------------------------
# Plot four kinematic maps
# --------------------------------------------------------------------------
metric_cfg = [
    ("v_mean",   r"Mean velocity $V$ (km s$^{-1}$)",   "RdBu_r",  True),
    ("sigma",    r"Dispersion $\sigma$ (km s$^{-1}$)",  "viridis", False),
    ("skewness", r"Skewness $\gamma_1$",                "PuOr",    True),
    ("kurtosis", r"Excess kurtosis $\kappa$",           "PuOr",    True),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
fig.subplots_adjust(wspace=0.05)

for ax, (key, label, cmap, symmetric) in zip(axes, metric_cfg):
    vals = maps[key]["median"]
    finite = vals[np.isfinite(vals)]

    if symmetric:
        vmax = np.percentile(np.abs(finite), 95) if len(finite) else 1.0
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    else:
        vmin, vmax = np.percentile(finite, [5, 95]) if len(finite) else (0, 1)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(xbin, ybin, c=vals, cmap=cmap, norm=norm,
                    s=200, marker="s", edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax, pad=0.03, shrink=0.85)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_aspect("equal")
    ax.set_xlim(x_arcsec[0] - 3, x_arcsec[-1] + 3)
    ax.set_ylim(y_arcsec[0] - 3, y_arcsec[-1] + 3)
    ax.set_xlabel(r"$x$ (arcsec)", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel(r"$y$ (arcsec)", fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

fig.savefig("images/fig_kin_maps.png", dpi=150, bbox_inches="tight")
print("Saved: images/fig_kin_maps.png")
