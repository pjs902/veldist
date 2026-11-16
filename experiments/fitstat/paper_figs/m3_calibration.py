"""Paper figure m3: calibration and bias of the fit statistics on mock bins (C1c).

usage: python m3_calibration.py OUT.pdf
Reads results/raw/c1c/*.jsonl (per-seed fits; truth sigma 18/16 or 17/17 km/s).
Versions shown (addback mode): diagonal chi2 (diag), per-star likelihood with the
Laplace approximation + sharpening + one re-expansion (lap_S_it1), exact per-star
likelihood with sharpening (exact_S).
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RAW = Path(__file__).resolve().parents[1] / "results" / "raw" / "c1c"
TESTS = [("calib", r"$HST$-like"), ("gaia", r"$Gaia$-like"), ("addback", "mean offset"),
         ("tail", "1% tail"), ("edge", "1% edge ring")]
VERS = [("diag", r"diagonal $\chi^2$", "#D55E00", "s"),
        ("lap_S_it1", "per-star, Laplace", "#0072B2", "o"),
        ("exact_S", "per-star, exact", "#009E73", "^")]
key = lambda r: (r["truth"]["N"], r["truth"]["delta"], r["truth"]["f"])


def load(t):
    return [r for r in map(json.loads, open(RAW / f"{t}.jsonl")) if "fits" in r]


def ratio(rs, v):
    """scatter/claimed for sigma (both axes), pooled over groups after removing each group's mean."""
    z = []
    for g in {key(r) for r in rs}:
        s = [r for r in rs if key(r) == g]
        if len(s) < 3:
            continue
        for i in (0, 1):
            x = np.array([r["fits"][f"addback_{v}"]["x"][i] for r in s])
            e = np.nanmedian([r["fits"][f"addback_{v}"]["err"][i] for r in s])
            z += list((x - x.mean()) / e)
    return np.std(z, ddof=1), np.std(z, ddof=1) / np.sqrt(2 * len(z))


def bias(rs, v):
    d = np.array([r["fits"][f"addback_{v}"]["x"][i] - r["truth"][p] for r in rs for i, p in ((0, "s1"), (1, "s2"))])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


plt.style.use("default")
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "text.usetex": False, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.7), constrained_layout=True)
x = np.arange(len(TESTS))
off = {v: o for (v, *_), o in zip(VERS, (-0.2, 0.0, 0.2))}
for v, lab, c, m in VERS:
    R = [ratio(load(t), v) for t, _ in TESTS]
    B = [bias(load(t), v) for t, _ in TESTS]
    ax[0].errorbar(x + off[v], [r[0] for r in R], [r[1] for r in R], fmt=m, color=c, ms=4, lw=0.8, label=lab)
    ax[1].errorbar(x + off[v], [b[0] for b in B], [b[1] for b in B], fmt=m, color=c, ms=4, lw=0.8)
ax[0].axhline(1, color="0.5", lw=0.6, ls=":")
ax[0].set_yscale("log")
ax[0].set_yticks([0.7, 1, 2, 4, 8], ["0.7", "1", "2", "4", "8"])
ax[0].yaxis.set_minor_formatter(plt.NullFormatter())
ax[0].set_ylabel("scatter / claimed")
ax[1].axhline(0, color="0.5", lw=0.6, ls=":")
ax[1].set_ylabel(r"$\langle \sigma_{\rm fit} - \sigma_{\rm true} \rangle$ [km s$^{-1}$]")
for a in ax[:2]:
    a.set_xticks(x, [t for _, t in TESTS], rotation=30, ha="right")
# tail fraction recovery
rs = load("tail") + load("edge")
for (v, lab, c, m), o in zip(VERS, (-0.2, 0.0, 0.2)):
    for j, (t, f0, name) in enumerate((("tail", 0.0, "tail, none"), ("tail", 0.01, "tail, 1%"), ("edge", 0.01, "edge, 1%"))):
        s = [r for r in load(t) if r["truth"]["f"] == f0]
        f = np.array([r["fits"][f"addback_{v}"]["x"][-1] for r in s])
        ax[2].errorbar(j + o, f.mean() * 100, f.std(ddof=1) / np.sqrt(len(f)) * 100, fmt=m, color=c, ms=4, lw=0.8)
for j, f0 in enumerate((0.0, 1.0, 1.0)):
    ax[2].plot([j - 0.35, j + 0.35], [f0, f0], color="k", lw=0.8)
ax[2].set_xticks([0, 1, 2], ["no tail", "1% tail", "1% edge ring"], rotation=30, ha="right")
ax[2].set_ylabel("recovered fraction (%)")
fig.legend(*ax[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
fig.savefig(sys.argv[1])
fig.savefig(sys.argv[1].replace(".pdf", ".png"), dpi=200)
print("saved", sys.argv[1])
