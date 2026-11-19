"""Paper figure r12: the proper motions predicted by fits that leave them out. sigma_T/sigma_R and mean mu_T against R
for HST and Gaia: data, the joint fit (all four sets), the fit without HST (MUSE+LVM+Gaia) and the LOS-only fit
(MUSE+LVM), all on orblib_001_001 at the production ridge.

usage: python r12_holdout.py PACK_DIR MODEL_HISTS_LOS.npz MODEL_HISTS_NOHST.npz OUT.pdf
MODEL_HISTS_*.npz from experiments/fitstat/model_hists.py (keys '{set}|data', '{set}|model|{tag}'); the joint fit's tag
starts with 'prod', the left-out fit's with 'los' / 'nohst'. Pack: work/v4pack (1406 apertures).
Moments: d2_maps.pm_moments (cell-width corrected); annuli of equal aperture count, data mean +- error of the mean.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from d2_maps import pm_moments
from d3_profiles import annuli, mean_err

P, ZL, ZH, OUT = Path(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3]), sys.argv[4]


def tag(Z, s, start):
    return next(k for k in Z.files if k.startswith(f"{s}|model|{start}"))


w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(2, 2, figsize=(2 * w, 1.6 * h), sharex="row", constrained_layout=True)
for r, (s, f, nm) in enumerate((("hst_pm", "hst", "HST"), ("gaia_pm", "gaia", "Gaia"))):
    path = P / f"{f}_veldist_pm_2dhist.npz"
    d = pm_moments(path)
    g = annuli(d["R"], nper=len(d["R"]) // 8)
    curves = [("joint fit", pm_moments(path, ZL[tag(ZL, s, "prod")]), dict(color="k")),
              ("without HST", pm_moments(path, ZH[tag(ZH, s, "nohst")]), dict(color="#0072B2")),
              ("LOS only", pm_moments(path, ZL[tag(ZL, s, "los")]), dict(color="#D55E00", ls="--"))]
    for c, (q, ql) in enumerate(((lambda m: m["st"] / m["sr"], r"$\sigma_T/\sigma_R$"),
                                 (lambda m: m["mt"], r"$\langle\mu_T\rangle$ [km s$^{-1}$]"))):
        rr, mm, ee = mean_err(d["R"], q(d), g)
        ax[r, c].errorbar(rr, mm, ee, ls="none", marker="o", color="0.3", mfc="w", label="data", zorder=3)
        for lab, m, kw in curves:
            ax[r, c].plot(rr, mean_err(d["R"], q(m), g)[1], label=lab, **kw)
            print(nm, lab, ql[:12], np.round(mean_err(d["R"], q(m), g)[1], 3))
        print(nm, "data", ql[:12], np.round(mm, 3), "R", np.round(rr).astype(int))
        ax[r, c].set_xscale("log")
        ax[r, c].set_ylabel(f"{nm} {ql}")
        ax[r, c].set_xlabel(r"$R$ [arcsec]")
    ax[r, 0].axhline(1, color="0.7", lw=0.5)
ax[0, 0].legend(frameon=False, fontsize="small")
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print("saved", OUT)
