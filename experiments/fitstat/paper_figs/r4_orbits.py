"""Paper figure r4: orbital structure of the best-fit model -- anisotropy and rotation.

usage: python r4_orbits.py R5_BETA.npz R4_LAMBDAZ.npz OUT.pdf
Inputs from veldist-fitstat/paperfig/extract_orbits.py (each npz carries a 'desc' array).
a  beta_r(r) = 1 - (sigma_th^2 + sigma_ph^2)/(2 sigma_r^2), shell mass-weighted, mean motion removed per cell
b  beta_r in the meridional plane (R, z), cells below 1e-3 of the peak mass masked
c  fraction of the tube-orbit weight co- and counter-rotating with the bulk (defined beyond 180"), and the
   box-orbit fraction, in radial bins of the orbits' mean radius
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

b, o = (np.load(f, allow_pickle=True) for f in sys.argv[1:3])
w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(1, 3, figsize=(3 * w, h), constrained_layout=True)

ok = (b["r_prof_mass"] < 1500) & (b["r_prof_mass"] > 1)  # few orbits inside 1"
ax[0].plot(b["r_prof_mass"][ok], b["beta_r"][ok], "o-", color="#0072B2", ms=3)
ax[0].axhline(0, color="0.6", lw=0.5)
ax[0].axvline(287, color="0.5", ls=":", lw=0.8)
ax[0].set_xscale("log")
ax[0].set_xlabel(r"$r$ [arcsec]")
ax[0].set_ylabel(r"$\beta_r$")
ax[0].text(0.03, 0.95, "radial", transform=ax[0].transAxes, va="top")
ax[0].text(0.03, 0.05, "tangential", transform=ax[0].transAxes, va="bottom")

R, z, beta, mass = b["R_mer"], b["z_mer"], b["beta_mer"], b["mass_mer"]
good = np.isfinite(beta) & (mass / np.nanmax(mass) > 1e-3) & (np.hypot(R, z) < 1500)
sc = ax[1].scatter(np.log10(np.hypot(R, z)[good]), np.degrees(np.arctan2(z, R))[good], c=beta[good], cmap="RdBu_r",
                   vmin=-0.6, vmax=0.6, s=75, marker="s", lw=0)
cb = fig.colorbar(sc, ax=ax[1], pad=0.02)
cb.ax.set_title(r"$\beta_r$")
ax[1].set_xlabel(r"$\log_{10}\, r$ [arcsec]")
ax[1].set_ylabel(r"latitude [deg]")

edges = o["r_edges"]
rc = np.sqrt(edges[1:] * edges[:-1])
tot = o["w_co"] + o["w_counter"] + o["w_box"]
okb = tot > 0
for key, lab, c in (("w_co", "co-rotating tubes", "#0072B2"), ("w_counter", "counter-rotating tubes", "#D55E00"),
                    ("w_box", "boxes", "0.4")):
    ax[2].plot(rc[okb], (o[key] / np.where(okb, tot, 1))[okb], "o-", ms=3, color=c, label=lab)
ax[2].axvline(287, color="0.5", ls=":", lw=0.8)
ax[2].set_xscale("log")
ax[2].set_ylim(0, 1)
ax[2].set_xlabel(r"mean orbital radius [arcsec]")
ax[2].set_ylabel("fraction of orbit weight")
ax[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.56, 1.0))
fig.savefig(sys.argv[3])
fig.savefig(sys.argv[3].replace(".pdf", ".png"), dpi=200)
fc = o["frac_counter_of_tubes"]
print("counter fraction of tubes by bin:", " ".join(f"{r:.0f}:{f:.2f}" for r, f in zip(rc, fc) if np.isfinite(f)))
print("beta:", " ".join(f"{r:.0f}:{v:.2f}" for r, v in zip(b["r_prof_mass"][ok], b["beta_r"][ok])))
