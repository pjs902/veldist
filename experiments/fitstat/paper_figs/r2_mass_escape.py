"""Paper figure r2: mass model of the best-fit model -- enclosed mass by component, and escape speed with the fast stars.

usage: python r2_mass_escape.py R2_MASS.npz R3_ESCAPE.npz OUT.pdf
Inputs from veldist-fitstat/paperfig/extract_potential.py on the node (each npz carries a 'desc' array).
Escape speed is shown cumulatively (stars; + sBH; + IMBH), since component escape speeds do not add.
Fast stars: projected radius and speed are both lower limits on the 3D values (arrows); filled = 3D speed
(PM + preliminary LOS velocity), open = plane-of-sky speed only.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

m, e = (np.load(f, allow_pickle=True) for f in sys.argv[1:3])
C = {"stars": "#0072B2", "sbh": "#E69F00", "imbh": "#D55E00", "total": "k"}
PC = 0.026635  # pc per arcsec at 5494 pc
w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(1, 2, figsize=(2 * w, h), constrained_layout=True)

r = m["r_arcsec"]
for key, lab in (("M_stars", r"stars ($\Upsilon = %.2f$)" % float(m["ml"])),
                 ("M_sbh", r"stellar-mass black holes ($%.1f \times 10^4\,M_\odot$)" % (float(m["m_sbh_tot"]) / 1e4)),
                 ("M_imbh", r"IMBH ($%.1f \times 10^3\,M_\odot$)" % (float(m["m_imbh"]) / 1e3)),
                 ("M_total", "total")):
    c = C[key.split("_")[1]]
    ax[0].plot(r, m[key], color=c, lw=1.6 if key == "M_total" else 1.0, label=lab)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_ylim(1e2, 1e7)
ax[0].set_xlim(0.05, 3000)
ax[0].set_xlabel(r"3D radius $r$ [arcsec]")
ax[0].set_ylabel(r"$M(<r)$ [$M_\odot$]")
ax[0].legend(frameon=False, loc="upper left")

r = e["r_arcsec"]
cum = [("stars", e["psi_stars"]), ("+ black holes", e["psi_stars"] + e["psi_sbh"]),
       ("+ IMBH (total)", e["psi_stars"] + e["psi_sbh"] + e["psi_imbh"] + e["psi_halo"])]
for (lab, psi), c in zip(cum, (C["stars"], C["sbh"], "k")):
    ax[1].plot(r, np.sqrt(2 * psi), color=c, lw=1.6 if c == "k" else 1.0, label=lab)
has_los = np.isfinite(e["fs_vlos"])
for i in range(len(e["fs_v3d"])):
    x, y = e["fs_r_proj_arcsec"][i], e["fs_v3d"][i]
    ax[1].errorbar(x, y, e["fs_v3d_err"][i], color=C["imbh"], marker="o", ms=4, mfc=C["imbh"] if has_los[i] else "w",
                   capsize=0, lw=0.8)
    ax[1].annotate("", xy=(x * 1.6, y), xytext=(x, y), arrowprops=dict(arrowstyle="->", lw=0.6, color=C["imbh"]))
ax[1].plot([], [], "o", color=C["imbh"], ms=4, label="fast stars (lower limits)")
ax[1].set_xscale("log")
ax[1].set_xlim(0.05, 3000)
ax[1].set_ylim(0, 1.15 * max(np.sqrt(2 * cum[-1][1][r > 0.1]).max(), np.nanmax(e["fs_v3d"] + e["fs_v3d_err"])))
ax[1].set_xlabel(r"3D radius $r$ [arcsec]")
ax[1].set_ylabel(r"$v_{\rm esc}$ [km s$^{-1}$]")
ax[1].legend(frameon=False, loc="upper right")
for a in ax:
    sec = a.secondary_xaxis("top", functions=(lambda x: x * PC, lambda x: x / PC))
    sec.set_xlabel("[pc]")
fig.savefig(sys.argv[3])
fig.savefig(sys.argv[3].replace(".pdf", ".png"), dpi=200)
i5 = np.argmin(abs(m["r_arcsec"] - 5))
print("M(<5\") stars %.3g sbh %.3g imbh %.3g; vesc total at 1\" %.1f" % (
    m["M_stars"][i5], m["M_sbh"][i5], m["M_imbh"][i5], np.interp(1, r, np.sqrt(2 * cum[-1][1]))))
