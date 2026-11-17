"""Paper figure r7: rotation of the core along the bulk axis, from the raw Nitschai+2023 MUSE catalogue.

usage: python r7_core_rotation.py OUT.pdf
Uses omegaCen/counter_rotation/nitschai_core.py (raw MUSE table, RV membership, three quality cuts, vdMA10
centre). In each annulus: the dipole amplitude along the bulk rotation axis (fixed from 40-400"), negative =
counter-rotating, from nitschai_core.proj_amp; error bar = sd of a within-annulus permutation null (the
amplitude that velocity noise alone produces); grey band = +-2 sd of that null for the middle cut.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/pesmith/research/omegaCen/counter_rotation")
from nitschai_core import CUTS, load, pa_profile, proj_amp, xyv  # noqa: E402

EDGES = np.array([0, 5, 10, 15, 20, 30, 45, 70, 110, 170, 260], float)
NPERM = 1000
t = load()
w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(figsize=(1.0 * w, 0.9 * h), constrained_layout=True)
rng = np.random.default_rng(0)
cols = {"strict": "#0072B2", "pechetti_like": "#D55E00", "magacc": "#009E73"}
labs = {"strict": "strict", "pechetti_like": "intermediate", "magacc": "magnitude only"}
for k, cut in enumerate(("strict", "pechetti_like", "magacc")):
    x, y, v, e = xyv(t, CUTS[cut](t))
    r = np.hypot(x, y)
    th = np.deg2rad(pa_profile(x, y, v, e, edges=np.array([40, 400.0]), min_points=30)["pa_deg"][0])
    rc, a, s = [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() < 50:
            continue
        o = proj_amp(x[m], y[m], v[m], e[m], th)
        null = [proj_amp(x[m], y[m], v[m][rng.permutation(m.sum())], e[m], th) for _ in range(NPERM)]
        rc.append(np.median(r[m]))
        a.append(o)
        s.append(np.std(null))
    rc, a, s = map(np.array, (rc, a, s))
    if cut == "pechetti_like":
        ax.fill_between(rc, -2 * s, 2 * s, color="0.88", lw=0, zorder=0)
    ax.errorbar(rc * (1 + 0.04 * (k - 1)), a, s, color=cols[cut], marker="o", ms=3.5, capsize=0, lw=0.9,
                label=f"{labs[cut]} ({int((r < 20).sum())} stars inside 20\\arcsec)".replace("\\arcsec", "$''$"))
    print(cut, " ".join(f"{q:.0f}:{b:+.2f}+-{c:.2f}" for q, b, c in zip(rc, a, s)))
ax.axhline(0, color="0.4", lw=0.6)
ax.axvspan(0.1, 20, color="#E69F00", alpha=0.08, lw=0, zorder=0)
ax.set_xscale("log")
ax.set_xlim(1.5, 300)
ax.set_ylim(-4.5, 9.5)
ax.set_xlabel("projected radius [arcsec]")
ax.set_ylabel(r"rotation along the bulk axis [km s$^{-1}$]")
ax.legend(frameon=False, loc="upper left", fontsize="small")
fig.savefig(sys.argv[1])
fig.savefig(sys.argv[1].replace(".pdf", ".png"), dpi=200)
