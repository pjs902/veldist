"""
fig_kin_maps.py — Kinematic maps: recovered rotation, and naive vs.
deconvolved dispersion bias against known ground truth.

Used in examples.md Example 3.

Two things changed from an earlier version of this script:

1. Speed: n_stars_per_bin used to vary per bin (20-120), which forces XLA
   to recompile the NUTS kernel for every distinct star count -- with ~40
   distinct counts across 49 bins, that's ~40 JIT compiles instead of 1,
   which is the actual cost (compile time dominates sampling time for
   small per-bin problems; see docs/validation.md). Fixing n_stars per
   bin and cutting the grid to 5x5=25 bins takes this from >20 minutes to
   well under a minute.

2. Usefulness: skewness/kurtosis maps are dropped. veldist's own
   documented acceptance criterion (TASKS.md) is "recover v_mean and sigma
   well, with calibrated uncertainty; h3/h4 are not required" -- so a map
   of noisy, unreliable kurtosis values was never a fair demonstration of
   the method. Instead, since this is synthetic data with a *known* true
   sigma(r), the figure plots naive (no-deconvolution) sigma bias against
   deconvolved sigma bias, on the same color scale, which is where
   measurement error provably biases a naive estimator (variances add:
   naive var = true var + error var) and answers "does this beat the
   naive approach" directly rather than by eye.

Requires JAX, NumPyro, and a working KinematicSolver installation.

Run from the docs/ directory:
    python fig_kin_maps.py
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
import matplotlib.colors as mcolors

rng = np.random.default_rng(2025)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# --------------------------------------------------------------------------
# Synthetic cluster: solid-body rotating core + non-rotating halo
# --------------------------------------------------------------------------
nx, ny = 5, 5
x_arcsec = np.linspace(-18, 18, nx)
y_arcsec = np.linspace(-18, 18, ny)
xbin, ybin = np.meshgrid(x_arcsec, y_arcsec)
xbin = xbin.ravel()
ybin = ybin.ravel()
n_bins = len(xbin)

r = np.sqrt(xbin**2 + ybin**2)
r_core = 12.0  # arcsec
A_rot = 2.5  # km/s per arcsec within core

r_safe = np.where(r > 0, r, 1.0)
v_rot = np.where(r < r_core, A_rot * xbin, A_rot * r_core * xbin / r_safe)
v_rot = np.where(r == 0, 0.0, v_rot)
sigma_true = 20.0 + 5.0 * np.exp(-((r / 8.0) ** 2))  # central peak

# Same star count in every bin: the one thing that makes fit_all_bins take
# a single JIT compile instead of one per distinct shape (see module
# docstring). n=50, err in [6, 12] km/s puts err/sigma ~0.3-0.6, large
# enough that the naive-vs-deconvolved sigma bias is a real, visible
# effect rather than noise.
n_per_bin = 50

print(f"Synthetic cluster: {n_bins} bins, {n_per_bin} stars/bin (fixed)")

bin_data_list = []
naive_sigma = np.empty(n_bins)
for i in range(n_bins):
    ei = rng.uniform(6.0, 12.0, n_per_bin)
    vi = rng.normal(v_rot[i], sigma_true[i], n_per_bin)
    vi_obs = vi + rng.normal(0, ei)
    bin_data_list.append({"vel": vi_obs, "err": ei})
    naive_sigma[i] = np.std(vi_obs)  # no deconvolution: raw sample sigma

# --------------------------------------------------------------------------
# Run batch inference
# --------------------------------------------------------------------------
try:
    from veldist import fit_all_bins
    from veldist.analysis import compute_summary_maps

    print("Running fit_all_bins ...")
    solvers = fit_all_bins(
        bin_data_list,
        grid_kwargs={"center": 0.0, "width": 160.0, "n_bins": 40},
        run_kwargs={"num_warmup": 300, "num_samples": 500, "gpu": False, "seed": 42},
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
        0.5,
        0.6,
        "fig_kin_maps — inference not available",
        ha="center",
        va="center",
        fontsize=13,
        color="#555555",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.38,
        f"{type(e).__name__}: {e}",
        ha="center",
        va="center",
        fontsize=8,
        color="#888888",
        transform=ax.transAxes,
        wrap=True,
    )
    fig.savefig("images/fig_kin_maps.png", dpi=150, bbox_inches="tight")
    print("Saved placeholder: images/fig_kin_maps.png")
    sys.exit(0)

deconv_sigma = maps["sigma"]["median"]
naive_bias = naive_sigma - sigma_true
deconv_bias = deconv_sigma - sigma_true

print(f"mean |sigma bias|: naive={np.mean(np.abs(naive_bias)):.2f} km/s, "
      f"deconvolved={np.mean(np.abs(deconv_bias)):.2f} km/s")

# --------------------------------------------------------------------------
# Plot: recovered rotation (a); naive vs. deconvolved sigma bias against
# the known truth, on a shared color scale (b, c).
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
fig.subplots_adjust(wspace=0.4)

ax = axes[0]
vals = maps["v_mean"]["median"]
vmax = np.nanpercentile(np.abs(vals), 95)
norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
sc = ax.scatter(xbin, ybin, c=vals, cmap="RdBu_r", norm=norm, s=800, marker="s", edgecolors="none")
fig.colorbar(sc, ax=ax, pad=0.03, shrink=0.85, label=r"Recovered $V$ (km s$^{-1}$)")
ax.set_title("(a) Recovered rotation", fontsize=9)

bias_vmax = np.nanpercentile(np.abs(np.concatenate([naive_bias, deconv_bias])), 95)
bias_norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-bias_vmax, vmax=bias_vmax)

ax = axes[1]
sc = ax.scatter(xbin, ybin, c=naive_bias, cmap="RdBu_r", norm=bias_norm, s=800, marker="s", edgecolors="none")
fig.colorbar(sc, ax=ax, pad=0.03, shrink=0.85, label=r"$\sigma_{\rm naive}-\sigma_{\rm true}$ (km s$^{-1}$)")
ax.set_title(f"(b) Naive sigma bias\nmean |bias| = {np.mean(np.abs(naive_bias)):.1f} km/s", fontsize=9)

ax = axes[2]
sc = ax.scatter(xbin, ybin, c=deconv_bias, cmap="RdBu_r", norm=bias_norm, s=800, marker="s", edgecolors="none")
fig.colorbar(sc, ax=ax, pad=0.03, shrink=0.85, label=r"$\sigma_{\rm veldist}-\sigma_{\rm true}$ (km s$^{-1}$)")
ax.set_title(f"(c) veldist sigma bias\nmean |bias| = {np.mean(np.abs(deconv_bias)):.1f} km/s", fontsize=9)

for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(x_arcsec[0] - 4, x_arcsec[-1] + 4)
    ax.set_ylim(y_arcsec[0] - 4, y_arcsec[-1] + 4)
    ax.set_xlabel(r"$x$ (arcsec)", fontsize=9)
    ax.set_ylabel(r"$y$ (arcsec)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

fig.savefig("images/fig_kin_maps.png", dpi=150, bbox_inches="tight")
print("Saved: images/fig_kin_maps.png")
