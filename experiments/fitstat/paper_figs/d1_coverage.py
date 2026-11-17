"""Paper figure d1: spatial coverage of the four kinematic datasets (bin centres) and their radial ranges.

usage: python d1_coverage.py PACK_DIR OUT.pdf [OUTER_DIR]   (x-East packs: *_pm_2dhist.npz, veldist/lvm ecsv;
OUTER_DIR holds the outer relative-PM sets ziliotto/scalco_f2f3/bellini_f1_veldist_pm_2dhist.npz)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

P = Path(sys.argv[1])
hst, gaia = (np.load(P / f) for f in ("hst_veldist_pm_2dhist.npz", "gaia_veldist_pm_2dhist.npz"))
muse = Table.read(P / "veldist_kinematics.ecsv")
lvm = Table.read(P / "lvm_kins.ecsv")
lx, ly = (lvm["x"], lvm["y"]) if "x" in lvm.colnames else (lvm["xbin"], lvm["ybin"])
sets = [(r"\textit{Gaia} PM", gaia["xbin"], gaia["ybin"], "#E69F00", "s"),
        ("LVM LOS", np.asarray(lx), np.asarray(ly), "#009E73", "D"),
        (r"\textit{HST} PM", hst["xbin"], hst["ybin"], "#D55E00", "."),
        ("MUSE LOS", np.asarray(muse["xbin"]), np.asarray(muse["ybin"]), "#0072B2", "o")]
OUTER = [("ziliotto", "Ziliotto+26 PM", "#CC79A7", "v"), ("scalco_f2f3", "Scalco+21 PM", "#56B4E9", "^"),
         ("bellini_f1", "Bellini+18 PM", "0.15", "P")]
if len(sys.argv) > 3:
    for stem, lab, c, m in OUTER:
        z = np.load(Path(sys.argv[3]) / f"{stem}_veldist_pm_2dhist.npz")
        sets.insert(2, (lab, z["xbin"], z["ybin"], c, m))
fig, ax = plt.subplots(1, 2, figsize=(2 * plt.rcParams["figure.figsize"][0], plt.rcParams["figure.figsize"][1]),
                       constrained_layout=True, width_ratios=[1, 1])
for lab, x, y, c, m in sets:
    ax[0].scatter(x, y, s=4 if m == "." else 10, c=c, marker="o" if m == "." else m, lw=0, label=lab)
ax[0].set_aspect("equal")
ax[0].invert_xaxis()
ax[0].set_xlabel(r"$x$ [arcsec, East]")
ax[0].set_ylabel(r"$y$ [arcsec, North]")
for k, (lab, x, y, c, m) in enumerate(sets[::-1]):
    r = np.hypot(x, y)
    ax[1].plot([r.min(), r.max()], [k, k], color=c, lw=6, solid_capstyle="butt")
    ax[1].text(r.max() * 1.15, k, f"{lab}: {len(r)} bins", va="center")
ax[1].axvline(287, color="0.5", ls=":", lw=1)
ax[1].text(287 * 1.05, -0.6, r"$r_{\rm HL}$", color="0.4")
ax[1].set_xscale("log")
ax[1].set_xlim(1, 2e5)
ax[1].set_ylim(-1, len(sets))
ax[1].set_yticks([])
ax[1].set_xlabel("projected radius of bin centres [arcsec]")
fig.savefig(sys.argv[2])
fig.savefig(sys.argv[2].replace(".pdf", ".png"), dpi=200)
print("saved", sys.argv[2], {s[0]: len(s[1]) for s in sets})
