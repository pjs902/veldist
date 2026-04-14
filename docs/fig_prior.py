"""
fig_prior.py — Gaussian random walk prior realisations.

Shows how the smoothing scale sigma_smooth controls the expected shape of
the LOSVD prior.  Three values are shown; in practice the sampler
marginalises over this parameter.

Does not require JAX or a running KinematicSolver.
Run from the docs/ directory:
    python fig_prior.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax

rng = np.random.default_rng(12345)

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
K = 60                                # number of velocity bins
v = np.linspace(-300, 300, K)         # velocity grid (km/s)
n_draws = 20                          # prior draws per panel
sigma_smooth_vals = [0.02, 0.10, 0.50]
labels = [
    r"$\sigma_\mathrm{smooth} = 0.02$  (smooth)",
    r"$\sigma_\mathrm{smooth} = 0.10$  (intermediate)",
    r"$\sigma_\mathrm{smooth} = 0.50$  (flexible)",
]

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)
fig.subplots_adjust(wspace=0.35)

for ax, sigma, label in zip(axes, sigma_smooth_vals, labels):
    for _ in range(n_draws):
        # Sample a Gaussian random walk latent vector u
        u = np.zeros(K)
        u[0] = rng.normal(0, sigma)
        for k in range(1, K):
            u[k] = rng.normal(u[k - 1], sigma)
        # Convert to probability masses via softmax
        w = softmax(u)
        ax.plot(v, w, color="steelblue", alpha=0.35, linewidth=0.9)

    # Overlay the prior mean (uniform distribution)
    ax.axhline(1.0 / K, color="black", linewidth=1.2, linestyle="--",
               label="uniform (mean)")
    ax.set_title(label, fontsize=9)
    ax.set_xlabel(r"Velocity (km s$^{-1}$)", fontsize=9)
    ax.set_ylabel("Probability mass", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(v[0], v[-1])
    ax.set_ylim(bottom=0)

axes[0].legend(fontsize=8, loc="upper right")

fig.savefig("images/fig_prior.png", dpi=150, bbox_inches="tight")
print("Saved: images/fig_prior.png")
