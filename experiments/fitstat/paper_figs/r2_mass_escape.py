"""Paper figure r2: mass model of the best-fit model -- enclosed mass by component, and escape speed with the fast stars.

usage: python r2_mass_escape.py R2_MASS.npz R3_ESCAPE.npz OUT.pdf [NIRSPEC_RV.fits]
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
                 ("M_sbh", r"black holes ($%.1f \times 10^4\,M_\odot$)" % (float(m["m_sbh_tot"]) / 1e4)),
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
ax[0].legend(frameon=False, loc="upper left", fontsize="small")

r = e["r_arcsec"]
cum = [("stars", e["psi_stars"]), ("+ black holes", e["psi_stars"] + e["psi_sbh"]),
       ("+ IMBH (total)", e["psi_stars"] + e["psi_sbh"] + e["psi_imbh"] + e["psi_halo"])]
for (lab, psi), c in zip(cum, (C["stars"], C["sbh"], "k")):
    ax[1].plot(r, np.sqrt(2 * psi), color=c, lw=1.6 if c == "k" else 1.0, label=lab)
# NIRSpec fast stars (Goold et al. in prep; good_data & usable_scores): 3D speed at our distance from the
# corrected PMs and the RV (systemic 233 km/s); radius projected from the AvdM10 centre. Stars above V3D_MIN.
# Filled circles: fast in proper motion alone (v2d > 62 km/s); diamonds: fast only with the RV.
# F, G of the escape table (no NIRSpec spectrum) as plane-of-sky lower limits.
from astropy.table import Table
D_KPC, VSYS, RA0, DE0, V3D_MIN = 5.494, 233.0, 201.696833, -47.479569, 60.0
done_ids = set()
if len(sys.argv) > 4:
    g = Table.read(sys.argv[4]).to_pandas()
    g = g.loc[g.good_data & g.usable_scores]
    k = 4.74047 * D_KPC
    v2 = k * np.hypot(g.pmRA_mas_per_yr, g.pmDEC_mas_per_yr)
    v2e = k * np.hypot(g.pmRA_mas_per_yr * g.pmRA_err_mas_per_yr, g.pmDEC_mas_per_yr * g.pmDEC_err_mas_per_yr) / \
        np.maximum(np.hypot(g.pmRA_mas_per_yr, g.pmDEC_mas_per_yr), 1e-9)
    vl = g.RV_kms - VSYS
    v3 = np.hypot(v2, vl)
    v3e = np.hypot(v2 * v2e, vl * g.RV_err_kms) / v3
    R = np.hypot((g.RA_deg - RA0) * np.cos(np.radians(DE0)), g.DEC_deg - DE0) * 3600
    sel = v3 > V3D_MIN
    print(f"NIRSpec: {len(g)} good stars, {sel.sum()} with v3d > {V3D_MIN:.0f} km/s at {D_KPC} kpc")
    for x, y, dy, pmfast in zip(R[sel], v3[sel], v3e[sel], (v2 > 62)[sel]):
        mk, col = ("o", C["imbh"]) if pmfast else ("D", "#CC79A7")
        ax[1].errorbar(x, y, dy, color=col, marker=mk, ms=4, capsize=0, lw=0.8)
        ax[1].annotate("", xy=(x * 1.6, y), xytext=(x, y), arrowprops=dict(arrowstyle="->", lw=0.6, color=col))
    done_ids = {"A", "B", "C", "D", "E"}
for i, lab in enumerate(e["fs_label"]):
    if str(lab) in done_ids:
        continue
    x, y = e["fs_r_proj_arcsec"][i], e["fs_v2d"][i]
    ax[1].errorbar(x, y, e["fs_v2d_err"][i], color=C["imbh"], marker="o", ms=4, mfc="w", capsize=0, lw=0.8)
    ax[1].annotate("", xy=(x * 1.6, y), xytext=(x, y), arrowprops=dict(arrowstyle="->", lw=0.6, color=C["imbh"]))
hs = [ax[1].plot([], [], "o", color=C["imbh"], ms=4)[0], ax[1].plot([], [], "D", color="#CC79A7", ms=4)[0],
      ax[1].plot([], [], "o", color=C["imbh"], mfc="w", ms=4)[0]]
leg2 = ax[1].legend(hs, ["fast in PM (3D speed)", "fast only with RV", "PM only"], frameon=False, loc="lower left",
                    ncol=2, fontsize="small", columnspacing=0.8, handletextpad=0.2, borderaxespad=0.3)
ax[1].add_artist(leg2)
ax[1].set_xscale("log")
ax[1].set_xlim(0.05, 1000)
ax[1].set_ylim(0, 1.15 * max(np.sqrt(2 * cum[-1][1][r > 0.1]).max(), 140))
ax[1].set_xlabel(r"3D radius $r$ [arcsec]")
ax[1].set_ylabel(r"$v_{\rm esc}$ [km s$^{-1}$]")
ax[1].legend(handles=ax[1].lines[:3], frameon=False, loc="upper right")
for a in ax:
    sec = a.secondary_xaxis("top", functions=(lambda x: x * PC, lambda x: x / PC))
    sec.set_xlabel("[pc]")
fig.savefig(sys.argv[3])  # noqa
fig.savefig(sys.argv[3].replace(".pdf", ".png"), dpi=200)
i5 = np.argmin(abs(m["r_arcsec"] - 5))
print("M(<5\") stars %.3g sbh %.3g imbh %.3g; vesc total at 1\" %.1f" % (
    m["M_stars"][i5], m["M_sbh"][i5], m["M_imbh"][i5], np.interp(1, r, np.sqrt(2 * cum[-1][1]))))
