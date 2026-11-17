"""Paper figure r10: the fitting statistic changes the inferred anisotropy. sigma_T/sigma_R against R for the data and
for one model (orbit library orblib_001_001) solved under two statistics: diagonal chi2 on every set (v4) and the
per-star HST likelihood (E1; MUSE and Gaia still diagonal).

usage: python r10_aniso.py PACK_DIR MODEL_ANISO.npz TAG OUT.pdf [--slides]
MODEL_ANISO.npz from experiments/fitstat/model_aniso.py (keys '{set}|{tag}|{sR,sT}'). Data: moments of the
deconvolved 2D histograms (d2_maps.pm_moments). Model: moments of the model histograms on the same cells; both
with the cell-width (h^2/12) correction. Annuli of equal aperture count per set; data mean +- error of the mean.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from d2_maps import pm_moments
from d3_profiles import annuli, mean_err

P, Z, TAG, OUT = Path(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3], sys.argv[4]
SLIDES = "--slides" in sys.argv
COL = {"hst_pm": "#0072B2", "gaia_pm": "#D55E00"}
LAB = {"hst_pm": "HST", "gaia_pm": "Gaia"}

w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(figsize=(1.6 * h * 16 / 9, 1.6 * h) if SLIDES else (w, h), constrained_layout=True)
for s, f in (("hst_pm", "hst"), ("gaia_pm", "gaia")):
    z = np.load(P / f"{f}_veldist_pm_2dhist.npz")
    d = pm_moments(P / f"{f}_veldist_pm_2dhist.npz")
    for t in ("v4", TAG):
        assert np.allclose(d["x"], Z[f"{s}|{t}|x"]) and np.allclose(d["y"], Z[f"{s}|{t}|y"])
    c2 = (2 * float(z["vxrange"]) / z["PM_2dhist"].shape[1]) ** 2 / 12
    ratio = lambda t: np.sqrt((Z[f"{s}|{t}|sT"] ** 2 - c2) / (Z[f"{s}|{t}|sR"] ** 2 - c2))
    g = annuli(d["R"], nper=len(d["R"]) // 8)
    r, m, e = mean_err(d["R"], d["st"] / d["sr"], g)
    ax.errorbar(r, m, e, ls="none", marker="o", color=COL[s], mec=COL[s], mfc="w", label=f"{LAB[s]} data", zorder=3)
    ax.plot(r, mean_err(d["R"], ratio("v4"), g)[1], color="0.4", ls="--",
            label=r"model, diagonal $\chi^2$" if s == "hst_pm" else None)
    ax.plot(r, mean_err(d["R"], ratio(TAG), g)[1], color="k",
            label="model, per-star HST" if s == "hst_pm" else None)
ax.axhline(1, color="0.7", lw=0.5)
ax.set_xscale("log")
ax.set_xlabel(r"$R$ [arcsec]")
ax.set_ylabel(r"$\sigma_T/\sigma_R$")
ax.legend(frameon=False, fontsize="small", loc="lower left", ncol=2 if SLIDES else 1)
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
