"""Paper figure d4: every relative proper-motion dataset tied to one absolute rotation curve by Gaia.

usage: python d4_rotation_tie.py DATAPREP_DIR PACK_DIR OUT.pdf [--slides]   (--slides: 16:9)
DATAPREP_DIR = omegaCen/dynamite_dataprep: NGC5139_relative_input_xeast/hst_pm_rot_rows.ecsv (Gaia - HST rings,
0-300"), relative_pm/outer_fields_gaia.ecsv (field radii) and relative_pm/outer_rows/*_pm_rot_rows.ecsv (one Gaia
row per outer field). PACK_DIR: x-East pack with gaia_veldist_pm_2dhist.npz (Gaia bins: annulus means of mu_T, as
d3_profiles). Plus the galaxy-anchored absolute PM of Bellini F1 (Libralato et al. 2018) minus the Gaia bulk motion
and the perspective expansion. Rotation is plotted positive (x-East mu_T is negative for omega Cen).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from d2_maps import pm_moments
from d3_profiles import annuli, mean_err

KMS = 26.0441  # km/s per mas/yr at 5494 pc
RA0, DEC0 = 201.696833, -47.479569  # AvdM10
BULK = np.array([-3.2474, -6.7761])  # VB21 members, perspective removed (outer_fields_gaia.txt)
LIB, ELIB = np.array([-3.341, -6.557]), np.array([0.028, 0.043])  # Libralato+18 F1, galaxies
F1 = (201.405500, -47.666722)  # 13:25:37.32 -47:40:00.2

D, P = Path(sys.argv[1]), Path(sys.argv[2])
rows = Table.read(D / "NGC5139_relative_input_xeast/hst_pm_rot_rows.ecsv")
edges = np.arange(0.0, 300.1, 30.0)
rc = 0.5 * (edges[1:] + edges[:-1])
G = Table.read(D / "relative_pm/outer_fields_gaia.ecsv")
Rf = {str(r["field"]): float(r["R_arcsec"]) for r in G}
outer = []
for stem, names in (("ziliotto", ["Ziliotto patch 0", "Ziliotto patch 1"]), ("scalco_f2f3", ["Scalco F2", "Scalco F3"]),
                    ("bellini_f1", ["Bellini F1"])):
    t = Table.read(D / f"relative_pm/outer_rows/{stem}_pm_rot_rows.ecsv")
    for k, n in enumerate(names):
        outer.append((stem, n, Rf[n], -float(t["mu"][k]), float(t["dmu"][k])))
gaia = pm_moments(P / "gaia_veldist_pm_2dhist.npz")
rg, mg, eg = mean_err(gaia["R"], gaia["mt"], annuli(gaia["R"], 15))
# Libralato F1: tangent-plane position (x East, y North, arcsec), perspective, bulk -> mu_T (positive = rotation)
x = (F1[0] - RA0) * np.cos(np.radians(DEC0)) * 3600
y = (F1[1] - DEC0) * 3600
r = np.hypot(x, y)
persp = -6.1363e-5 * (r / 60) * 232.9 / 5.494
mE, mN = LIB[0] - BULK[0] - persp * x / r, LIB[1] - BULK[1] - persp * y / r
lib_T = -(x * mN - y * mE) / r * KMS
lib_e = np.hypot(*(ELIB * np.array([-y, x]) / r)) * KMS

C = {"ziliotto": "#CC79A7", "scalco_f2f3": "#56B4E9", "bellini_f1": "0.15"}
MK = {"ziliotto": "v", "scalco_f2f3": "^", "bellini_f1": "P"}
LAB = {"ziliotto": "Ziliotto+26", "scalco_f2f3": "Scalco+21", "bellini_f1": "Bellini+18"}
w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(figsize=(1.6 * h * 16 / 9, 1.6 * h) if "--slides" in sys.argv else (1.3 * w, h), constrained_layout=True)
ax.errorbar(rc, -np.asarray(rows["mu"]), np.asarray(rows["dmu"]), color="#D55E00", marker="s", ms=4, ls="none",
            label=r"\textit{Gaia} $-$ \textit{HST}, rings")
ax.errorbar(rg, mg, eg, color="#E69F00", marker="s", ms=4, mfc="w", ls="none", label=r"\textit{Gaia} bins")
seen = set()
for stem, n, R, mu, e in outer:
    ax.errorbar(R, mu, e, color=C[stem], marker=MK[stem], ms=6, ls="none",
                label=None if stem in seen else rf"{LAB[stem]} (\textit{{Gaia}} tie)")
    seen.add(stem)
ax.errorbar(r * 1.04, lib_T, lib_e, color="0.15", marker="*", ms=8, mfc="w", ls="none",
            label="Bellini+18 F1, galaxy-anchored")
ax.set_xscale("log")
ax.set_xlabel("projected radius [arcsec]")
ax.set_ylabel(r"$\langle\mu_T\rangle$ [km s$^{-1}$]")
ax.set_ylim(0, 11)
ax.legend(frameon=False, fontsize="x-small", loc="upper right", ncol=2)
fig.savefig(sys.argv[3])
fig.savefig(sys.argv[3].replace(".pdf", ".png"), dpi=200)
print(f"Libralato F1: R {r:.0f}\", mu_T {lib_T:+.2f} +- {lib_e:.2f} km/s; outer:",
      " ".join(f"{n}:{mu:.2f}+-{e:.2f}" for _, n, _, mu, e in outer))
