"""Paper figure r1: data minus best-fit model, on the same maps as figure d2.

usage: python r1_residuals.py PACK_DIR R1_RESID.npz OUT.pdf
R1_RESID.npz from veldist-fitstat/paperfig/extract_resid.py (per-aperture data and model moments; 'desc' inside).
Layout mirrors d2_maps: top v_LOS (MUSE + LVM), mu_T, mu_R; bottom sigma_LOS (MUSE), sigma_T, sigma_R; each panel
is data - model in km/s on the DYNAMITE bins, radius stretched as sqrt(R). mu_T is positive along the rotation
(the extract's T is counter-clockwise E->N, opposite to the cluster's sense, so its residual is negated).
Panels whose set is missing from the npz are left empty.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from d2_maps import binmap, panel

P = Path(sys.argv[1])
z = np.load(sys.argv[2], allow_pickle=True)


def res(s, q, sign=1):
    if f"{s}_data_{q}" not in z.files:
        return None
    return sign * (z[f"{s}_data_{q}"] - z[f"{s}_model_{q}"])


G = {k: binmap(P, k) for k in ("hst_veldist", "gaia_veldist", "veldist", "lvm")}
SET = {"hst_veldist": "hst_pm", "gaia_veldist": "gaia_pm", "veldist": "veldist", "lvm": "lvm"}


def layers(keys, q, sign=1):
    out = []
    for k in keys:
        r = res(SET[k], q, sign)
        if r is not None:
            out.append((*G[k], r))
    return out


w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(2, 3, figsize=(3 * w, 2 * h), constrained_layout=True)
lim = 3
spec = [(ax[0, 0], ["lvm", "veldist"], "v", 1, r"$\Delta v_{\rm LOS}$"),
        (ax[0, 1], ["gaia_veldist", "hst_veldist"], "pmT", -1, r"$\Delta \mu_T$"),
        (ax[0, 2], ["gaia_veldist", "hst_veldist"], "pmR", 1, r"$\Delta \mu_R$"),
        (ax[1, 0], ["veldist"], "sigma", 1, r"$\Delta \sigma_{\rm LOS}$"),
        (ax[1, 1], ["gaia_veldist", "hst_veldist"], "sigma_T", 1, r"$\Delta \sigma_T$"),
        (ax[1, 2], ["gaia_veldist", "hst_veldist"], "sigma_R", 1, r"$\Delta \sigma_R$")]
for a, keys, q, sign, lab in spec:
    L = layers(keys, q, sign)
    if L:
        panel(a, L, "RdBu_r", -lim, lim, lab + r" [km s$^{-1}$]")
        vals = np.concatenate([l[3] for l in L])
        a.set_title(f"median {np.nanmedian(vals):+.2f}, rms {np.sqrt(np.nanmean(vals ** 2)):.2f}" + r" km s$^{-1}$")
    else:
        a.set_axis_off()
fig.savefig(sys.argv[3])
fig.savefig(sys.argv[3].replace(".pdf", ".png"), dpi=200)
print("saved", sys.argv[3])
