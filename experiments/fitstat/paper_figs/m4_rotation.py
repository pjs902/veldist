"""Paper figure m4: rotation under the two treatments of the HST proper motions (relative vs addback).

usage: python m4_rotation.py M4_ROTATION.npz OUT.pdf
Input from veldist-fitstat/paperfig/m4/make_m4.py (i = 44 deg library; 'desc' inside). mu_T is flipped so the
rotation is positive. Left: binned mean tangential proper motion -- data (HST addback means, Gaia, and the
oMEGACat absolute rings) and the two fits at Upsilon = 3.08 (solid) and 2.75 (dotted). Right: model - data.
Colour = HST treatment. Grey band: no proper-motion data.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
C = {"relative": "#0072B2", "addback": "#D55E00"}
w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(1, 2, figsize=(2.8 * w, 1.15 * h), constrained_layout=True, sharex=True)
for a in ax:
    a.axvspan(*z["nodata_gap"], color="0.9", lw=0, zorder=0)
    a.set_xscale("log")
    a.set_xlabel("projected radius [arcsec]")

# data
for s, mk, lab in (("hst_pm", "o", r"\textit{HST} (addback)"), ("gaia_pm", "s", r"\textit{Gaia}")):
    r, d, e = (z[f"ml308_{s}_muT_{k}"] for k in ("r", "data", "data_err"))
    ax[0].errorbar(r, -d, e, color="k", marker=mk, ms=4, mfc="w" if s == "gaia_pm" else "k", ls="none",
                   capsize=0, lw=0.8, label=lab, zorder=4)
ax[0].errorbar(z["omegacat_ring_r"], -z["omegacat_ring_muT"], z["omegacat_ring_err"], color="0.4", marker="D",
               ms=3.5, mfc="w", ls="none", capsize=0, lw=0.8, label="oMEGACat rings", zorder=4)
# models
for tag, ls in (("ml308", "-"), ("ml275", ":")):
    for fit in ("relative", "addback"):
        for s in ("hst_pm", "gaia_pm"):
            k = f"{tag}_{s}_muT"
            if f"{k}_r" not in z.files:
                continue
            r, d, m = z[f"{k}_r"], z[f"{k}_data"], z[f"{k}_model_{fit}"]
            ax[0].plot(r, -m, color=C[fit], ls=ls, lw=1.2)
            ax[1].plot(r, -(m - d), color=C[fit], ls=ls, lw=1.2, marker="o" if s == "hst_pm" else "s", ms=3)
ax[0].set_ylabel(r"$\langle \mu_T \rangle$ [km s$^{-1}$]")
ax[1].set_ylabel(r"model $-$ data [km s$^{-1}$]")
ax[1].axhline(0, color="0.5", lw=0.6)
from matplotlib.lines import Line2D
h1 = [Line2D([], [], color=C[f], label=f"{f} fit") for f in ("relative", "addback")]
h2 = [Line2D([], [], color="0.3", ls=ls, label=lab) for ls, lab in (("-", r"$\Upsilon = 3.08$"),
                                                                    (":", r"$\Upsilon = 2.75$"))]
ax[0].legend(frameon=False, loc="upper left", fontsize="small")
ax[1].legend(handles=h1 + h2, frameon=False, loc="upper left", fontsize="small", ncol=2)
fig.savefig(sys.argv[2])
fig.savefig(sys.argv[2].replace(".pdf", ".png"), dpi=200)
