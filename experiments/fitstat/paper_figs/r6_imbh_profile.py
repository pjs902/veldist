"""Paper figure r6: the IMBH-mass profile of the fit statistic, diagonal chi2 against the per-star likelihood.

usage: python r6_imbh_profile.py RESCORE_DIR OUT.pdf
RESCORE_DIR holds rescore_v4.py outputs (one npz per finished v4 model, fixed v4 weights). For each model:
  diagonal  = kinchi2                                   (the v4 fit statistic, all datasets)
  per-star  = kinchi2 - sum(HST diag) + sum(HST term)   (HST scored by the Laplace approximation or exact)
so the two differ only in how the HST proper motions are scored. Weights stay at their diagonal-chi2 solution,
so the per-star values are upper bounds on what a re-solve would reach. The other six parameters differ between
models (Bayesian-optimization samples), so this is not a profile; each statistic is shown minus its own minimum.
"""
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np

fs = sorted(glob.glob(f"{sys.argv[1]}/rescore_*.npz"))
rows = []
for f in fs:
    z = np.load(f, allow_pickle=True)
    k, d = float(z["kinchi2"]), z["diag"].sum()
    rows.append((float(z["m_bh"]), k, k - d + z["laplace"].sum(), k - d + z["exact"].sum()))
m, base, lap, ex = np.array(rows).T
print(f"{len(fs)} models; M_IMBH range {m.min():.3g}-{m.max():.3g}")


w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(2, 1, figsize=(w, 1.7 * h), constrained_layout=True, sharex=True)
for a, y, lab, c in ((ax[0], base, r"diagonal $\chi^2$", "#D55E00"),
                     (ax[1], lap, r"per-star likelihood", "#0072B2")):
    dy = y - y.min()
    a.scatter(m / 1e3, dy, s=12, color=c, lw=0, alpha=0.7)
    a.text(0.03, 0.95, lab, transform=a.transAxes, va="top")
    a.set_xscale("log")
    a.set_xticks([5, 10, 20, 50], ["5", "10", "20", "50"])
    a.set_xticks([], minor=True)
    a.set_ylabel(r"$\Delta$ statistic")
ax[1].scatter(m / 1e3, ex - ex.min(), s=14, marker="^", facecolor="none", edgecolor="#009E73", lw=0.7,
              label="exact")
ax[1].scatter([], [], s=14, color="#0072B2", label="Laplace")
ax[1].set_xlabel(r"$M_{\rm IMBH}$ [$10^3\,M_\odot$]")
ax[1].legend(frameon=False, loc="upper right", ncol=2)
ax[1].set_ylim(top=1.22 * (lap.max() - lap.min()))
fig.savefig(sys.argv[2])
fig.savefig(sys.argv[2].replace(".pdf", ".png"), dpi=200)
for lab, y in (("diag", base), ("laplace", lap), ("exact", ex)):
    print(lab, "best M_IMBH %.0f" % m[np.argmin(y)], "range of Delta %.0f" % (y.max() - y.min()))
