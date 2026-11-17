"""Paper figure r8: where in the HST field the two statistics decide between a low- and a high-mass IMBH model.

usage: python r8_bin_delta.py PACK_DIR RESCORE_LOW.npz RESCORE_HIGH.npz OUT.pdf
Two rescored v4 models (fixed v4 weights): the best diagonal-chi2 model (low IMBH mass) and the best per-star
model (high IMBH mass). Per HST bin, Delta = statistic(high) - statistic(low), so Delta > 0 favours the low-mass
model. Left/middle: Delta per bin for the diagonal chi2 and the per-star likelihood (Laplace), on the bins and
radial stretch of figure d2. Right: Delta accumulated outward in radius for the diagonal chi2, Laplace and exact.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from d2_maps import binmap, panel, srad

P = Path(sys.argv[1])
lo, hi = (np.load(f, allow_pickle=True) for f in sys.argv[2:4])
z = np.load(P / "hst_veldist_pm_2dhist.npz")
R = np.hypot(z["xbin"], z["ybin"])
G = binmap(P, "hst_veldist")
D = {k: hi[k] - lo[k] for k in ("diag", "laplace", "exact")}
mlo, mhi = float(lo["m_bh"]) / 1e3, float(hi["m_bh"]) / 1e3

w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(1, 3, figsize=(3 * w, h), constrained_layout=True)
for a, k, lab in ((ax[0], "diag", r"diagonal $\chi^2$"), (ax[1], "laplace", "per-star likelihood")):
    lim = np.percentile(np.abs(D[k]), 95)
    panel(a, [(*G, D[k])], "RdBu_r", -lim, lim, r"$\Delta$ per bin")
    a.set_title(f"{lab}: total {D[k].sum():+.0f}")
    e = float(srad(400))
    a.set_xlim(e, -e)
    a.set_ylim(-e, e)
    for t in a.texts:
        t.set_clip_on(True)
o = np.argsort(R)
for k, lab, c, ls in (("diag", r"diagonal $\chi^2$", "#D55E00", "-"), ("laplace", "per-star, Laplace", "#0072B2", "-"),
                      ("exact", "per-star, exact", "#009E73", "--")):
    ax[2].plot(R[o], np.cumsum(D[k][o]), color=c, ls=ls, label=lab)
ax[2].axhline(0, color="0.5", lw=0.6)
ax[2].set_xscale("log")
ax[2].set_xlabel("projected radius [arcsec]")
ax[2].set_ylabel(r"cumulative $\Delta$ (inside $R$)")
ax[2].legend(frameon=False, loc="upper left", fontsize="small")
ax[2].set_title(rf"$\Delta > 0$: favours {mlo:.1f}k over {mhi:.0f}k")
fig.savefig(sys.argv[4])
fig.savefig(sys.argv[4].replace(".pdf", ".png"), dpi=200)
for k in D:
    c = np.cumsum(D[k][o])
    print(k, "total %.0f; inside 30\" %.0f, 100\" %.0f, 200\" %.0f" % (c[-1], c[R[o] < 30][-1], c[R[o] < 100][-1],
                                                                    c[R[o] < 200][-1]))
