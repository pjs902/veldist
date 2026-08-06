"""
fig_prior_gaussian_core.py — gaussian_core prior realisations (default prior).

Shows how sigma3 controls departure from the Gaussian null space: at
sigma3 -> 0 every draw is an exact Gaussian (v0, s0); larger sigma3 lets the
penalised deviation add non-Gaussian structure on top of that core.

Does not require JAX or a running KinematicSolver.
Run from the docs/ directory:
    python fig_prior_gaussian_core.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
# Ignore any user matplotlibrc (e.g. a personal MNRAS print style) so this
# figure renders identically regardless of whose machine builds the docs.
matplotlib.rcdefaults()
import matplotlib.pyplot as plt
from scipy.special import softmax

rng = np.random.default_rng(12345)

K = 60
v = np.linspace(-300, 300, K)
n_draws = 20
sigma3_vals = [0.0, 1.0, 4.0]
labels = [
    r"$\sigma_3 = 0$  (exact Gaussian)",
    r"$\sigma_3 = 1$  (default prior scale)",
    r"$\sigma_3 = 4$  (strongly non-Gaussian)",
]


def null_space_basis(centers, rw_order):
    u = (centers - centers.mean()) / (centers.max() - centers.min())
    vander = np.vander(u, N=rw_order, increasing=True)
    q, _ = np.linalg.qr(vander)
    return q


def rw_deviation_scale(n_bins, rw_order):
    # Matches _rw_deviation_scale's generalised-variance standardisation,
    # evaluated once by Monte Carlo rather than importing JAX.
    centers = np.linspace(-1, 1, n_bins)
    q = null_space_basis(centers, rw_order)
    n_mc = 4000
    d = rng.normal(size=(n_mc, n_bins))
    w = d.copy()
    for _ in range(rw_order):
        w = np.cumsum(w, axis=1)
    dev = w - (w @ q) @ q.T
    var = dev.var(axis=0)
    log_gm = np.mean(np.log(np.clip(var, 1e-12, None)))
    return 1.0 / np.sqrt(np.exp(log_gm))


rw_order = 3
scale = rw_deviation_scale(K, rw_order)
q = null_space_basis(v, rw_order)

# Draw each curve's (v0, s0, d3) once and reuse it across all three panels,
# scaling only sigma3. This isolates what sigma3 actually changes: at
# sigma3=0 every curve is exactly its own Gaussian core; increasing sigma3
# perturbs that *same* core rather than resampling a new, unrelated one.
# Tighter v0/s0 priors than the model's actual (deliberately vague) ones,
# purely so the panel of draws reads clearly at this figure size instead of
# scattering across the whole grid.
v0s = rng.normal(0.0, 50.0, n_draws)
s0s = rng.lognormal(np.log((v.max() - v.min()) / 10.0), 0.4, n_draws)
d3s = rng.normal(size=(n_draws, K))

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
fig.subplots_adjust(wspace=0.15)

for ax, sigma3, label in zip(axes, sigma3_vals, labels):
    for v0, s0, d3 in zip(v0s, s0s, d3s):
        core = -0.5 * ((v - v0) / max(s0, 1e-3)) ** 2

        w = (d3 * sigma3 * scale).copy()
        for _ in range(rw_order):
            w = np.cumsum(w)
        deviation = w - q @ (q.T @ w)

        curve = softmax(core + deviation)
        ax.plot(v, curve, color="steelblue", alpha=0.45, linewidth=1.0)

    ax.set_title(label, fontsize=9)
    ax.set_xlabel(r"Velocity (km s$^{-1}$)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(v[0], v[-1])
    ax.set_ylim(bottom=0)

axes[0].set_ylabel("Probability mass", fontsize=9)

fig.savefig("images/fig_prior_gaussian_core.png", dpi=150, bbox_inches="tight")
print("Saved: images/fig_prior_gaussian_core.png")
