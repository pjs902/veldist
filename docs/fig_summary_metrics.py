"""
fig_summary_metrics.py — Summary metrics illustrated on two example LOSVDs.

Shows a symmetric leptokurtic LOSVD (heavy tails, radial anisotropy analogue)
and an asymmetric LOSVD (skewed, rotation-like) side by side, with scalar
summary metrics annotated on each panel.

Does not require JAX or a running KinematicSolver — the "posterior" here is
simulated with a Dirichlet distribution for illustration purposes.

Run from the docs/ directory:
    python fig_summary_metrics.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax

rng = np.random.default_rng(77)

# --------------------------------------------------------------------------
# Velocity grid
# --------------------------------------------------------------------------
K = 80
v = np.linspace(-250, 250, K)
dv = v[1] - v[0]

# --------------------------------------------------------------------------
# Helper: simulate a "posterior" by Dirichlet draws around a true PMF
# --------------------------------------------------------------------------
def sim_posterior(pmf, n_samples=600, concentration=400):
    alpha = pmf * concentration
    return rng.dirichlet(alpha, size=n_samples)

# --------------------------------------------------------------------------
# Helper: compute median and IQR from posterior samples via CDF
# --------------------------------------------------------------------------
def cdf_percentile(pdf_samples, vc, p):
    cdf = np.cumsum(pdf_samples, axis=1)
    p_arr = np.atleast_1d(np.asarray(p, dtype=float))
    result = np.array([np.interp(p_arr, cdf[s], vc) for s in range(len(pdf_samples))])
    return result[:, 0] if np.ndim(p) == 0 else result

# --------------------------------------------------------------------------
# LOSVD 1: symmetric, leptokurtic (heavy tails — radial anisotropy)
# --------------------------------------------------------------------------
# A Gaussian core plus Lorentzian-like heavy tails
core  = np.exp(-0.5 * (v / 40.0)**2)
tails = 1.0 / (1.0 + (v / 70.0)**2)
pmf1  = 0.7 * core + 0.3 * tails
pmf1  = pmf1 / pmf1.sum()

samples1 = sim_posterior(pmf1)
means1   = samples1 @ v
stds1    = np.sqrt(np.einsum("ij,ij->i", samples1, (v - means1[:, None])**2))
safe1    = np.where(stds1 > 0, stds1, 1.0)
kurts1   = np.einsum("ij,ij->i", samples1, (v - means1[:, None])**4) / safe1**4 - 3.0
pctls1   = cdf_percentile(samples1, v, np.array([0.25, 0.50, 0.75]))
q25_1, med1, q75_1 = pctls1[:, 0], pctls1[:, 1], pctls1[:, 2]
iqr1     = q75_1 - q25_1
tw_mask1 = np.abs(v[None, :] - means1[:, None]) > stds1[:, None]
tw1      = np.sum(samples1 * tw_mask1, axis=1)

# --------------------------------------------------------------------------
# LOSVD 2: asymmetric (left-skewed, trailing low-velocity tail — h3 > 0 side)
# --------------------------------------------------------------------------
# Skewed distribution: narrow core + low-velocity tail
core2  = np.exp(-0.5 * ((v - 20.0) / 35.0)**2)
tail2  = np.exp(-0.5 * ((v + 80.0) / 60.0)**2) * (v < 0)
pmf2   = core2 + 0.35 * tail2
pmf2   = pmf2 / pmf2.sum()

samples2 = sim_posterior(pmf2)
means2   = samples2 @ v
stds2    = np.sqrt(np.einsum("ij,ij->i", samples2, (v - means2[:, None])**2))
safe2    = np.where(stds2 > 0, stds2, 1.0)
skews2   = np.einsum("ij,ij->i", samples2, (v - means2[:, None])**3) / safe2**3
pctls2   = cdf_percentile(samples2, v, np.array([0.25, 0.50, 0.75]))
q25_2, med2, q75_2 = pctls2[:, 0], pctls2[:, 1], pctls2[:, 2]
iqr2     = q75_2 - q25_2

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(wspace=0.38)

losvd_kw   = dict(color="steelblue", linewidth=2.0)
ci_kw      = dict(color="steelblue", alpha=0.20)
annot_kw   = dict(fontsize=8.5, color="#333333")
bracket_kw = dict(color="#d62728", linewidth=1.3)

# ---- Panel 1: leptokurtic -----------------------------------------------
p16_1 = np.percentile(samples1, 16, axis=0)
p84_1 = np.percentile(samples1, 84, axis=0)
median_pmf1 = np.percentile(samples1, 50, axis=0)

ax1.fill_between(v, p16_1, p84_1, **ci_kw)
ax1.plot(v, median_pmf1, **losvd_kw)
ax1.set_xlim(v[0], v[-1])
ax1.set_ylim(bottom=0)
ax1.set_xlabel(r"$v_\mathrm{los}$ (km s$^{-1}$)", fontsize=10)
ax1.set_ylabel("Probability mass", fontsize=10)
ax1.set_title("Leptokurtic LOSVD (heavy tails)", fontsize=10)

# Annotate mean and ±1σ
vm1  = float(np.median(means1))
sm1  = float(np.median(stds1))
ku1  = float(np.median(kurts1))
tw1m = float(np.median(tw1))
ax1.axvline(vm1, color="#d62728", linewidth=1.3, linestyle="--", label=fr"$V = {vm1:.0f}$ km/s")
ax1.axvspan(vm1 - sm1, vm1 + sm1, alpha=0.08, color="#d62728")
# σ bracket
y_br = median_pmf1.max() * 0.88
ax1.annotate("", xy=(vm1 + sm1, y_br), xytext=(vm1 - sm1, y_br),
             arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.3))
ax1.text(vm1, y_br * 1.06, fr"$\sigma = {sm1:.0f}$ km/s", ha="center", **annot_kw)

# κ and tail_weight annotation
ytext = median_pmf1.max() * 0.55
ax1.text(0.97, 0.97,
         fr"$\kappa = {ku1:+.2f}$" + "\n" + fr"$T = {tw1m:.3f}$  (Gauss: 0.317)",
         transform=ax1.transAxes, ha="right", va="top", **annot_kw,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", lw=0.8))

# ---- Panel 2: asymmetric -----------------------------------------------
p16_2 = np.percentile(samples2, 16, axis=0)
p84_2 = np.percentile(samples2, 84, axis=0)
median_pmf2 = np.percentile(samples2, 50, axis=0)

ax2.fill_between(v, p16_2, p84_2, **ci_kw)
ax2.plot(v, median_pmf2, **losvd_kw)
ax2.set_xlim(v[0], v[-1])
ax2.set_ylim(bottom=0)
ax2.set_xlabel(r"$v_\mathrm{los}$ (km s$^{-1}$)", fontsize=10)
ax2.set_title("Asymmetric LOSVD (trailing tail)", fontsize=10)

vm2  = float(np.median(means2))
med2m= float(np.median(med2))
sk2  = float(np.median(skews2))
iq2  = float(np.median(iqr2))

# Mean and median markers
ax2.axvline(vm2,   color="#d62728",   linewidth=1.3, linestyle="--",
            label=fr"mean $V = {vm2:.0f}$ km/s")
ax2.axvline(med2m, color="#2ca02c", linewidth=1.3, linestyle=":",
            label=fr"median $= {med2m:.0f}$ km/s")
# IQR bracket
y_iq = median_pmf2.max() * 0.82
q25m = float(np.median(q25_2))
q75m = float(np.median(q75_2))
ax2.annotate("", xy=(q75m, y_iq), xytext=(q25m, y_iq),
             arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.3))
ax2.text(0.5 * (q25m + q75m), y_iq * 1.07,
         fr"IQR $= {iq2:.0f}$ km/s", ha="center", **annot_kw)

ax2.legend(fontsize=8, loc="upper right")
ax2.text(0.03, 0.97,
         fr"$\gamma_1 = {sk2:+.2f}$" + "\n"
         + fr"$V - v_* = {vm2 - med2m:+.1f}$ km/s",
         transform=ax2.transAxes, ha="left", va="top", **annot_kw,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", lw=0.8))

for ax in (ax1, ax2):
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)

fig.savefig("images/fig_summary_metrics.png", dpi=150, bbox_inches="tight")
print("Saved: images/fig_summary_metrics.png")
