"""Paper figure r11: data, model and data - model maps for one dataset, on the sky without radial stretch.

usage: python r11_dataset_maps.py PACK_DIR MODEL_HISTS.npz TAG SET OUT.pdf      SET = veldist | hst_pm | gaia_pm
MODEL_HISTS.npz from experiments/fitstat/model_hists.py ('{set}|data', '{set}|model|{TAG}', n_ap x cells). Data and
model moments are computed the same way from the histograms (unit-sum per aperture; Sheppard-corrected dispersions):
MUSE v_LOS and sigma_LOS from the LOSVD cells (pack ecsv meta 'vcent'); PM mu_T (positive along the rotation), mu_R,
sigma_T, sigma_R as in d2_maps.pm_moments. Each dataset's DYNAMITE bins at true sky position (x East to the left,
y North, arcsec), axes limited to the dataset's own footprint. Rows: quantities; columns: data, model (same colour
scale), data - model (symmetric scale); residual panel titles give median and rms over the bins.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from d2_maps import binmap, fill_holes, pm_moments

P, Z, TAG, SET, OUT = Path(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
STEM = {"veldist": "veldist", "hst_pm": "hst_veldist", "gaia_pm": "gaia_veldist"}[SET]
NAME = {"veldist": "MUSE", "hst_pm": r"\textit{HST}", "gaia_pm": r"\textit{Gaia}"}[SET]


def muse_moments(H):
    vc = np.asarray(Table.read(P / "veldist_kinematics.ecsv").meta["vcent"], float)
    h = np.median(np.diff(vc))
    w = H / H.sum(1, keepdims=True)
    m = w @ vc
    s = np.sqrt(np.maximum((w * (vc[None] - m[:, None]) ** 2).sum(1) - h ** 2 / 12, 0))
    return {"v": m, "s": s}


def moments(H):
    if SET == "veldist":
        return muse_moments(H)
    return pm_moments(P / f"{STEM}_pm_2dhist.npz", H)


D, M = moments(Z[f"{SET}|data"]), moments(Z[f"{SET}|model|{TAG}"])
Q = ([("v", r"v_{\rm LOS}"), ("s", r"\sigma_{\rm LOS}")] if SET == "veldist" else
     [("mt", r"\mu_T"), ("mr", r"\mu_R"), ("st", r"\sigma_T"), ("sr", r"\sigma_R")])
xe, ye, b = binmap(P, STEM)
b = fill_holes(b)
used = np.argwhere(b > 0)
pad = 0.03 * max(xe[-1] - xe[0], ye[-1] - ye[0])
xlo, xhi = xe[used[:, 0].min()] - pad, xe[used[:, 0].max() + 1] + pad
ylo, yhi = ye[used[:, 1].min()] - pad, ye[used[:, 1].max() + 1] + pad


def draw(ax, val, cmap, lo, hi, label):
    v = np.full(b.shape, np.nan)
    v[b > 0] = val[b[b > 0] - 1]
    im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(v.T), cmap=cmap, vmin=lo, vmax=hi, shading="flat",
                       rasterized=True)
    ax.set_xlim(xhi, xlo)  # East to the left
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label=label, shrink=0.85, pad=0.02)


w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(len(Q), 3, figsize=(3 * w, len(Q) * h * 0.95), constrained_layout=True, squeeze=False)
unit = r" [km s$^{-1}$]"
for r, (q, lab) in enumerate(Q):
    d, m = D[q], M[q]
    if q.startswith("s"):
        lo, hi, cm = *np.nanpercentile(d, [2, 98]), "viridis"
    else:
        a = np.nanpercentile(np.abs(d), 98)
        lo, hi, cm = -a, a, "RdBu_r"
    draw(ax[r, 0], d, cm, lo, hi, rf"${lab}$" + unit)
    draw(ax[r, 1], m, cm, lo, hi, rf"${lab}$" + unit)
    res = d - m
    lim = np.nanpercentile(np.abs(res), 98)
    draw(ax[r, 2], res, "RdBu_r", -lim, lim, rf"$\Delta {lab}$" + unit)
    ax[r, 2].set_title(rf"median {np.nanmedian(res):+.2f}, rms {np.sqrt(np.nanmean(res ** 2)):.2f}" + r" km s$^{-1}$",
                       fontsize="small")
    ax[r, 0].set_ylabel(r"$y$ (North) [arcsec]")
for j, t in enumerate(["data", "model", "data $-$ model"]):
    ax[0, j].set_title(f"{NAME} {t}" + ("\n" + ax[0, j].get_title() if ax[0, j].get_title() else ""))
for a in ax[-1]:
    a.set_xlabel(r"$x$ (East) [arcsec]")
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print("saved", OUT, {q: (float(np.nanmedian(D[q] - M[q])), float(np.sqrt(np.nanmean((D[q] - M[q]) ** 2)))) for q, _ in Q})
