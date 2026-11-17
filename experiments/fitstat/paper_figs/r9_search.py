"""Paper figure r9 (appendix): convergence and coverage of the Bayesian-optimization parameter search.

usage: python r9_search.py ALL_MODELS.ecsv RESCORE_DIR OUT.pdf
Finished models of the v4 grid in the order they were computed (iteration, then time). First panel: each
model's per-star -2 ln L relative to the best (points) and the running best (line). Other panels: each
parameter against model number, coloured by -2 ln L, within its search range (Table 2).
"""
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

t = Table.read(sys.argv[1])
t = t[np.asarray(t["all_done"], bool)]
t.sort(["which_iter", "time_modified"])
n = np.arange(1, len(t) + 1)
k = np.asarray(t["kinchi2"], float)
resc = {}
for f in glob.glob(f"{sys.argv[2]}/rescore_*.npz"):
    z = np.load(f, allow_pickle=True)
    resc[str(z["directory"]).strip("/")] = float(z["kinchi2"]) - z["diag"].sum() + z["laplace"].sum()
ps = np.array([resc.get(str(d).strip("/"), np.nan) for d in t["directory"]])

P = [("m-bh", r"$M_{\rm IMBH}$ [$M_\odot$]", (10 ** 3.5, 1e5), True),
     ("mscale-sbh", r"$f_{M,\rm sBH}$", (0.5, 2), True), ("rscale-sbh", r"$f_{r,\rm sBH}$", (0.5, 2), True),
     ("q-stars", r"$q$", (0.40, 0.83), False), ("p-stars", r"$p$", (0.90, 0.999), False),
     ("u-stars", r"$u$", (0.95, 0.9999), False), ("ml", r"$\Upsilon$", (2.5, 3.5), False)]
w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(2, 4, figsize=(3 * w, 1.5 * h), constrained_layout=True, sharex=True)
ax = ax.ravel()
a = ax[0]
d = ps - np.nanmin(ps)
a.scatter(n, d, s=10, color="0.5", lw=0)
a.plot(n, np.fmin.accumulate(d), color="#0072B2")
a.set_yscale("symlog", linthresh=1000)
a.set_ylabel(r"$\Delta(-2\ln\mathcal{L})$")
it = np.asarray(t["which_iter"])
for j in np.flatnonzero(np.diff(it)):
    for b in ax:
        b.axvline(j + 1.5, color="0.85", lw=0.6, zorder=0)
for b, (col, lab, rng, lg) in zip(ax[1:], P):
    sc = b.scatter(n, np.asarray(t[col], float), c=d, cmap="viridis_r", s=12, lw=0,
                   vmin=0, vmax=np.nanpercentile(d, 90))
    b.set_ylim(*rng)
    if lg:
        b.set_yscale("log")
        if rng[1] <= 2:
            b.set_yticks([0.5, 0.7, 1, 1.4, 2], ["0.5", "0.7", "1", "1.4", "2"])
            b.set_yticks([], minor=True)
    b.set_ylabel(lab)
for b in ax[4:]:
    b.set_xlabel("model number")
fig.colorbar(sc, ax=ax[1:].tolist(), label=r"$\Delta(-2\ln\mathcal{L})$", shrink=0.6, pad=0.01)
fig.savefig(sys.argv[3])
fig.savefig(sys.argv[3].replace(".pdf", ".png"), dpi=200)
print(len(t), "models; iterations", np.unique(it))
