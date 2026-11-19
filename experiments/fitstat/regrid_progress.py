"""Regrid progress: the v4 grid's diagonal chi2 against the full-covariance re-solve, per model, against M_IMBH.

usage: python regrid_progress.py ALL_MODELS.ecsv RESULTS_DIR OUT.png
ALL_MODELS.ecsv: $W/regrid/NGC5139_output/all_models.ecsv (74 models; kinchi2 = v4 diagonal chi2 at v4 weights, NaN for
the 12 profile-fill models that never got v4 weights). RESULTS_DIR: regrid_one.py outputs (chi2_kin = per-star Laplace
on MUSE, HST, Gaia + LVM chi2 at the re-solved weights).
Top: each model's move from its old Delta (v4 kinchi2) to its new Delta (chi2_kin), old relative to the v4 minimum over
all 62 models with a v4 chi2, new relative to the minimum over the re-solved models; models not re-solved yet in grey. Bottom: new Delta with the soft lower envelope of r6.
"""
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).parent / "paper_figs"))
from r6_imbh_profile_env import softmin_curve  # noqa: E402

t = Table.read(sys.argv[1])
d = [str(x).strip("/") for x in t["directory"]]
m = np.asarray(t["m-bh"], float) / 1e3
old = np.asarray(t["kinchi2"], float)
new = np.full(len(t), np.nan)
for f in glob.glob(f"{sys.argv[2]}/*.npz"):
    z = np.load(f)
    new[d.index(str(z["directory"]).strip("/"))] = float(z["chi2_kin"])
done = np.isfinite(new)
both = done & np.isfinite(old)
print(f"{done.sum()}/{len(t)} re-solved ({both.sum()} with a v4 chi2)")

o0 = np.nanmin(old)
n0 = new[done].min() if done.any() else 0.0
w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(2, 1, figsize=(w, 1.7 * h), sharex=True, constrained_layout=True)
a = ax[0]
a.scatter(m[~done], old[~done] - o0, s=10, facecolor="none", edgecolor="0.7", lw=0.6, label="v4, pending")
for i in np.flatnonzero(both):
    a.annotate("", (m[i], new[i] - n0), (m[i], old[i] - o0),
               arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6, shrinkA=2, shrinkB=2))
a.scatter(m[both], old[both] - o0, s=12, color="#D55E00", lw=0, label=r"v4 diag. $\chi^2$")
a.scatter(m[done], new[done] - n0, s=12, color="#0072B2", lw=0, label="re-solve")
a.set_yscale("symlog", linthresh=100)
a.set_ylim(-20, 40 * np.nanmax(old - o0))  # headroom for the legend
a.set_ylabel(r"$\Delta$ statistic")
a.legend(frameon=False, fontsize="small", loc="upper center", ncol=3,
         title=f"{done.sum()}/{len(t)} models re-solved", title_fontsize="small")

b = ax[1]
if done.any():
    b.scatter(m[done], new[done] - n0, s=14, color="#0072B2", lw=0)
    if done.sum() >= 4:
        lm = np.log10(m[done])
        xs = np.linspace(lm.min(), lm.max(), 300)
        ys = softmin_curve(lm, new[done], xs)
        b.plot(10 ** xs, ys - n0, color="#0072B2", lw=1)
        print("re-solve envelope min at %.1fk" % 10 ** xs[np.argmin(ys)])
    print("best re-solved model:", d[int(np.nanargmin(new))], "M_IMBH %.1fk" % m[np.nanargmin(new)])
b.set_ylabel(r"$\Delta(-2\ln\mathcal{L})$, re-solve")
b.set_xlabel(r"$M_{\rm IMBH}$ [$10^3\,M_\odot$]")
for a in ax:
    a.set_xscale("log")
    a.set_xticks([5, 10, 20, 50], ["5", "10", "20", "50"])
    a.set_xticks([], minor=True)
fig.savefig(sys.argv[3], dpi=200)
print("saved", sys.argv[3])
