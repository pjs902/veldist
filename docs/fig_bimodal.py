"""
fig_bimodal.py — Posterior LOSVD for a two-component system.

Illustrates recovery of a bimodal LOSVD (prograde + secondary component)
with the 68% credible interval shown.  Used in examples.md Example 1.

Does not require JAX or a running KinematicSolver — the posterior is
simulated with a Dirichlet distribution around the known true PMF.

Run from the docs/ directory:
    python fig_bimodal.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(404)

# --------------------------------------------------------------------------
# Velocity grid
# --------------------------------------------------------------------------
K = 60
v = np.linspace(-120, 150, K)

# --------------------------------------------------------------------------
# True two-component PMF
# --------------------------------------------------------------------------
n1, n2 = 150, 100
mu1, sig1 = -30.0, 12.0
mu2, sig2 = +50.0, 15.0
w1, w2 = n1 / (n1 + n2), n2 / (n1 + n2)

pmf = (w1 * np.exp(-0.5 * ((v - mu1) / sig1)**2) / sig1
     + w2 * np.exp(-0.5 * ((v - mu2) / sig2)**2) / sig2)
pmf /= pmf.sum()

# --------------------------------------------------------------------------
# Simulate posterior by Dirichlet draws (concentration ≈ effective N)
# --------------------------------------------------------------------------
n_samples = 800
concentration = (n1 + n2) * 0.6   # slightly underdetermined — moderate width
samples = rng.dirichlet(pmf * concentration, size=n_samples)

p16  = np.percentile(samples, 16, axis=0)
p84  = np.percentile(samples, 84, axis=0)
med  = np.percentile(samples, 50, axis=0)

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.5))

ax.fill_between(v, p16, p84, color="steelblue", alpha=0.30,
                label="68% credible interval")
ax.plot(v, med, color="steelblue", linewidth=2.0, label="Posterior median")
ax.plot(v, pmf, color="black", linewidth=1.8, linestyle="--",
        label="True LOSVD")

# Mark individual components for reference
c1 = w1 * np.exp(-0.5 * ((v - mu1) / sig1)**2) / sig1
c2 = w2 * np.exp(-0.5 * ((v - mu2) / sig2)**2) / sig2
c1 /= pmf.sum()   # already normalised via pmf, just rescale for display
c2 /= pmf.sum()
ax.plot(v, c1, color="#888888", linewidth=1.0, linestyle=":",
        label=fr"Component A ($\mu={mu1:.0f}$, $\sigma={sig1:.0f}$ km/s)")
ax.plot(v, c2, color="#888888", linewidth=1.0, linestyle="-.",
        label=fr"Component B ($\mu={mu2:.0f}$, $\sigma={sig2:.0f}$ km/s)")

ax.set_xlim(v[0], v[-1])
ax.set_ylim(bottom=0)
ax.set_xlabel(r"$v_\mathrm{los}$ (km s$^{-1}$)", fontsize=10)
ax.set_ylabel("Probability mass", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.tick_params(labelsize=9)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("images/fig_bimodal.png", dpi=150, bbox_inches="tight")
print("Saved: images/fig_bimodal.png")
