"""
fig_projection.py — What "project out the quadratic trend" actually does.

`generate_gaussian_core_curve` builds a wiggly curve `w` by cumulatively
summing (integrating) white noise three times, then subtracts its own
least-squares quadratic fit before adding it to the Gaussian core. This
figure makes that subtraction concrete: one raw `w`, its quadratic fit,
and the residual that's actually used as the "deviation" term.

Pure NumPy. Run from the docs/ directory.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
# Ignore any user matplotlibrc (e.g. a personal MNRAS print style) so this
# figure renders identically regardless of whose machine builds the docs.
matplotlib.rcdefaults()
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

n_bins = 60
idx = np.arange(n_bins, dtype=float)
centers = idx - idx.mean()  # arbitrary units, just for the x-axis

# --- Build w: a triple-integrated random walk, same construction as
# generate_gaussian_core_curve (cumsum of cumsum of cumsum of white noise) ---
d3 = rng.normal(size=n_bins)
w = d3.copy()
for _ in range(3):
    w = np.cumsum(w)
w /= np.std(w)  # arbitrary normalisation, purely for a readable y-axis

# --- The quadratic fit is exactly Q @ (Q.T @ w) for an orthonormal basis Q
# spanning {1, u, u^2}; np.polyfit's least-squares fit is the same
# projection expressed differently, used here because it needs no QR ---
quad_coeffs = np.polyfit(idx, w, deg=2)
quad_fit = np.polyval(quad_coeffs, idx)
residual = w - quad_fit

fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
fig.subplots_adjust(wspace=0.35)

ax = axes[0]
ax.plot(centers, w, color="#1f77b4", linewidth=1.6)
ax.set_title("(a) Raw triple-integrated\nrandom walk $w$", fontsize=9)
ax.axhline(0, color="gray", linewidth=0.6)

ax = axes[1]
ax.plot(centers, w, color="#1f77b4", linewidth=1.2, alpha=0.5, label="$w$")
ax.plot(centers, quad_fit, color="#d62728", linewidth=1.8, label=r"$QQ^\top w$ (best-fit quadratic)")
ax.set_title("(b) the parabola hidden\ninside the wiggles", fontsize=9)
ax.legend(fontsize=7, loc="best")

ax = axes[2]
ax.plot(centers, residual, color="#2ca02c", linewidth=1.6)
ax.axhline(0, color="gray", linewidth=0.6)
ax.set_title(
    r"(c) what is left once it is removed"
    + "\n(all the deviation is allowed to be)",
    fontsize=9,
)

for ax in axes:
    ax.set_xlabel("bin index (centred)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("log-density units", fontsize=8)

fig.savefig("images/fig_projection.png", dpi=150, bbox_inches="tight")
print("Saved: images/fig_projection.png")
