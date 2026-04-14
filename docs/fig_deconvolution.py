"""
fig_deconvolution.py — Core physics of Bayesian LOSVD deconvolution.

Three-panel figure illustrating the fundamental problem:
  (a) The intrinsic LOSVD and a sample of individual star error kernels.
  (b) The observed velocity distribution (convolution of intrinsic with errors).
  (c) The recovered LOSVD posterior (requires a live KinematicSolver).

Panels (a) and (b) are generated from pure numpy/scipy.  Panel (c) requires
JAX and NumPyro.  Run with --no-inference to generate only panels (a) and (b).

Usage:
    python fig_deconvolution.py              # full figure with inference
    python fig_deconvolution.py --no-inference  # panels a/b only (no JAX needed)

Run from the docs/ directory.
"""

import sys
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

# --------------------------------------------------------------------------
# Parse arguments
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--no-inference", action="store_true",
                    help="Skip panel (c); do not require JAX/NumPyro")
args = parser.parse_args()

rng = np.random.default_rng(2024)

# --------------------------------------------------------------------------
# True intrinsic LOSVD: slightly asymmetric (rotation-like)
# --------------------------------------------------------------------------
v_grid = np.linspace(-120, 120, 300)

# Gaussian core with a faint low-velocity tail
core  = np.exp(-0.5 * ((v_grid - 15.0) / 28.0)**2)
tail  = 0.25 * np.exp(-0.5 * ((v_grid + 50.0) / 35.0)**2) * (v_grid < 15)
true_pdf = core + tail
true_pdf /= np.trapz(true_pdf, v_grid)

# --------------------------------------------------------------------------
# Synthetic dataset: N stars
# --------------------------------------------------------------------------
N = 300

# Sample intrinsic velocities from the true PDF (rejection sampling)
v_range = v_grid[-1] - v_grid[0]
max_pdf = true_pdf.max()
v_int = []
while len(v_int) < N:
    v_try = rng.uniform(v_grid[0], v_grid[-1], N * 3)
    p_try = np.interp(v_try, v_grid, true_pdf) / max_pdf
    accept = rng.uniform(0, 1, len(v_try)) < p_try
    v_int.extend(v_try[accept].tolist())
v_int = np.array(v_int[:N])

# Heteroscedastic errors: larger near the wings
errors = 8.0 + 4.0 * np.abs(v_int) / 50.0 + rng.uniform(0, 4, N)
v_obs  = v_int + rng.normal(0.0, errors)

# --------------------------------------------------------------------------
# Build figure panels
# --------------------------------------------------------------------------
n_panels = 2 if args.no_inference else 3
fig_width = 7.5 if args.no_inference else 11.0
fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 3.8))
fig.subplots_adjust(wspace=0.38)

# ---- Panel (a): True LOSVD + individual error kernels -------------------
ax = axes[0]
ax.plot(v_grid, true_pdf, color="black", linewidth=2.0,
        label="True LOSVD", zorder=5)

# Plot a random subsample of individual error kernels (rescaled for visibility)
n_show = 30
idx_show = rng.choice(N, n_show, replace=False)
kernel_scale = true_pdf.max() * 0.18   # rescale so kernels are visible but not dominant
for i in idx_show:
    k = norm.pdf(v_grid, loc=v_obs[i], scale=errors[i])
    k /= k.max()
    k *= kernel_scale
    ax.plot(v_grid, k, color="#aec7e8", linewidth=0.6, alpha=0.5, zorder=2)

# One highlighted kernel to call out
i_ex = idx_show[0]
k_ex = norm.pdf(v_grid, loc=v_obs[i_ex], scale=errors[i_ex])
k_ex /= k_ex.max()
k_ex *= kernel_scale
ax.plot(v_grid, k_ex, color="#1f77b4", linewidth=1.4, alpha=0.9, zorder=3,
        label=fr"Error kernel, $\varepsilon = {errors[i_ex]:.1f}$ km/s")
ax.axvline(v_obs[i_ex], color="#1f77b4", linewidth=0.8, linestyle=":",
           alpha=0.7, zorder=3)

ax.set_xlim(v_grid[0], v_grid[-1])
ax.set_ylim(bottom=0)
ax.set_xlabel(r"$v_\mathrm{los}$ (km s$^{-1}$)", fontsize=9)
ax.set_ylabel("Probability density", fontsize=9)
ax.set_title("(a) Intrinsic LOSVD\nand measurement error kernels", fontsize=9)
ax.legend(fontsize=8, loc="upper left")
ax.tick_params(labelsize=8)
ax.spines[["top", "right"]].set_visible(False)

# ---- Panel (b): Observed velocity distribution --------------------------
ax = axes[1]

# Histogram of observed velocities
ax.hist(v_obs, bins=25, density=True, color="gray", alpha=0.55,
        label="Observed ($N={:d}$ stars)".format(N), zorder=2)

# Naive Gaussian fit to observed
mu_obs, sig_obs = np.mean(v_obs), np.std(v_obs)
v_fit = np.linspace(v_grid[0], v_grid[-1], 300)
ax.plot(v_fit,
        norm.pdf(v_fit, mu_obs, sig_obs),
        color="#d62728", linewidth=1.8, linestyle="--",
        label=fr"Gaussian fit to obs. ($\sigma = {sig_obs:.0f}$ km/s)", zorder=4)

# True LOSVD (for reference)
ax.plot(v_grid, true_pdf, color="black", linewidth=1.8, linestyle="-",
        label=fr"True LOSVD ($\sigma = 28$ km/s)", zorder=5)

ax.set_xlim(v_grid[0], v_grid[-1])
ax.set_ylim(bottom=0)
ax.set_xlabel(r"$v_\mathrm{los}$ (km s$^{-1}$)", fontsize=9)
ax.set_ylabel("Probability density", fontsize=9)
ax.set_title("(b) Observed distribution\n(intrinsic $\\otimes$ errors)", fontsize=9)
ax.legend(fontsize=8, loc="upper left")
ax.tick_params(labelsize=8)
ax.spines[["top", "right"]].set_visible(False)

# ---- Panel (c): Recovered LOSVD (requires inference) --------------------
if not args.no_inference:
    ax = axes[2]

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from veldist import KinematicSolver

        solver = KinematicSolver()
        solver.setup_grid(center=0.0, width=240.0, n_bins=55)
        solver.add_data(vel=v_obs, err=errors)
        solver.run(num_warmup=500, num_samples=1000, gpu=False)

        pdf_mass = solver.samples["intrinsic_pdf"]
        # Convert to density for plotting
        dv_solver = solver.grid["width"] / solver.grid["n_bins"]
        pdf_density = pdf_mass / dv_solver
        vc = solver.grid["centers"]

        p16 = np.percentile(pdf_density, 16, axis=0)
        p84 = np.percentile(pdf_density, 84, axis=0)
        med = np.percentile(pdf_density, 50, axis=0)

        ax.fill_between(vc, p16, p84, color="steelblue", alpha=0.30,
                        label="68% credible interval")
        ax.plot(vc, med, color="steelblue", linewidth=2.0,
                label="Posterior median")
        ax.plot(v_grid, true_pdf, color="black", linewidth=1.8, linestyle="--",
                label="True LOSVD")

        ax.set_xlim(v_grid[0], v_grid[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$v_\mathrm{los}$ (km s$^{-1}$)", fontsize=9)
        ax.set_title("(c) Recovered LOSVD\n(Bayesian deconvolution)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    except Exception as e:
        # Common causes:
        #   ImportError  — JAX or NumPyro not installed
        #   TypeError: asarray() got an unexpected keyword argument 'copy'
        #               — JAX version requires NumPy >= 2.0; upgrade with
        #                 `pip install "numpy>=2.0"`
        import numpy as _np
        np_ver = _np.__version__
        msg = (
            f"Inference unavailable\n"
            f"({type(e).__name__})\n\n"
            f"NumPy {np_ver} detected.\n"
            f"JAX requires NumPy >= 2.0.\n"
            f"Run: pip install 'numpy>=2.0'"
            if isinstance(e, TypeError) and "copy" in str(e)
            else f"Inference unavailable:\n{e}"
        )
        ax.text(0.5, 0.5, msg,
                ha="center", va="center", transform=ax.transAxes, fontsize=8,
                color="gray", wrap=True)
        ax.set_title("(c) Recovered LOSVD\n(requires JAX/NumPyro + NumPy ≥ 2.0)", fontsize=9)
        print(f"\nWarning: panel (c) skipped — {type(e).__name__}: {e}")
        if isinstance(e, TypeError) and "copy" in str(e):
            print("  JAX requires NumPy >= 2.0. Fix: pip install 'numpy>=2.0'")

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
suffix = "_no_inference" if args.no_inference else ""
outfile = f"images/fig_deconvolution{suffix}.png"
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved: {outfile}")
