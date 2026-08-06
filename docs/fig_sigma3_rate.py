"""
fig_sigma3_rate.py — SIGMA3_RATE trade-off: per-bin coverage vs. wall time.

Plots the table already in validation.md's "Default prior and
regularisation" section: per-bin LOSVD coverage (gaussian/skew/student-t)
and h3+h4 moment coverage, both against a nominal 0.68 target, across the
SIGMA3_RATE values that were actually measured. Makes visible why
SIGMA3_RATE=0.35 (the adopted default) was kept over tighter rates that
look identical in the moment-only metrics but quietly lose per-bin
calibration.

Pure matplotlib, no library import required. These are the literal
numbers from docs/validation.md; if that table changes, update both.

Run from the docs/ directory:
    python fig_sigma3_rate.py
"""

import matplotlib

matplotlib.use("Agg")
# Ignore any user matplotlibrc (e.g. a personal MNRAS print style) so this
# figure renders identically regardless of whose machine builds the docs.
matplotlib.rcdefaults()
import matplotlib.pyplot as plt

# From docs/validation.md, "Default prior and regularisation" table.
rates = [0.35, 1.0, 5.0, 10.0]
gaussian = [0.724, 0.730, 0.716, None]
skew = [0.710, 0.680, 0.609, None]
student_t = [0.709, 0.687, 0.646, None]
h3h4 = [0.603, 0.570, 0.393, 0.312]

fig, ax = plt.subplots(figsize=(6.5, 4.2))

for ys, label, marker in [
    (gaussian, "gaussian", "o"),
    (skew, "skew-normal", "s"),
    (student_t, "Student-$t$", "^"),
]:
    xs = [r for r, y in zip(rates, ys) if y is not None]
    yv = [y for y in ys if y is not None]
    ax.plot(xs, yv, marker=marker, linewidth=1.6, label=f"per-bin coverage ({label})")

ax.plot(rates, h3h4, marker="D", linewidth=2.0, color="black", label="h3+h4 mean coverage")

ax.axhline(0.68, color="gray", linewidth=1.0, linestyle="--", label="nominal target (0.68)")
ax.axvline(0.35, color="#d62728", linewidth=1.0, linestyle=":", alpha=0.7)
ax.text(0.37, 0.76, "adopted default\n(SIGMA3_RATE=0.35)", fontsize=8, color="#d62728", va="top")

ax.set_xscale("log")
ax.set_xlabel("SIGMA3_RATE (Exponential prior rate on $\\sigma_3$)")
ax.set_ylabel("Coverage (nominal 0.68)")
ax.set_title("Tightening SIGMA3_RATE trades away per-bin calibration\nwhile moment coverage looks flat")
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim(0.25, 0.78)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("images/fig_sigma3_rate.png", dpi=150, bbox_inches="tight")
print("Saved: images/fig_sigma3_rate.png")
