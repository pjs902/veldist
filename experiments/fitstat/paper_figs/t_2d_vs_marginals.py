"""Talk figure: fitting v_R and v_T separately loses how rotation and anisotropy are shared between orbits.

usage: python t_2d_vs_marginals.py STARS_PACK_DIR MODEL_HISTS.npz TAG OUT.pdf [RMIN RMAX]
Gaia bins with RMIN <= R < RMAX (default 462-942") stacked in each bin's (v_R, v_T) frame (v_T positive along the
rotation, v_R centred per bin). Stars: centroid of each star's error-kernel row (pack PM_star_*), per-axis error from
the row variance (+ h^2/12 for the sharpened cell-integrated kernel). Model: each star replaced by NDRAW draws from its
bin's model histogram (uniform within the cell) plus that star's errors, i.e. the model in observed space.
Uncoupled: the model draws with v_T shuffled within the bin: identical v_R and v_T distributions, no coupling.
Left: the v_R and v_T distributions (what separate R and T fits see). Right: rotation <v_T> minus each sample's own mean (the model's overall offset is not the point) and tangential
dispersion sigma_T for stars grouped by radial speed |v_R| (slow < 0.5 sigma_R <= medium < 1.2 sigma_R <= fast).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from d2_maps import PA_KIN

D, Z, TAG, OUT = Path(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3], sys.argv[4]
RMIN, RMAX = (float(sys.argv[5]), float(sys.argv[6])) if len(sys.argv) > 6 else (462.0, 942.0)
NDRAW = 20
rng = np.random.default_rng(1)
z = np.load(D / "gaia_veldist_pm_2dhist.npz", allow_pickle=True)
K, half = z["PM_2dhist"].shape[1], float(z["vxrange"])
h = 2 * half / K
c = np.linspace(-half + h / 2, half - h / 2, K)
G1, G2 = [g.ravel() for g in np.meshgrid(c, c, indexing="ij")]
pa = np.radians(PA_KIN)
MAJ, MNR = np.array([np.sin(pa), np.cos(pa)]), np.array([-np.cos(pa), np.sin(pa)])
ip, cell, mass = z["PM_star_indptr"], z["PM_star_cell"], z["PM_star_mass"].astype(float)
ap = z["PM_star_aperture"] - z["PM_star_aperture"].min()
seg = np.repeat(np.arange(len(ap)), np.diff(ip))
m0 = np.bincount(seg, mass)
v1, v2 = np.bincount(seg, mass * G1[cell]) / m0, np.bincount(seg, mass * G2[cell]) / m0
e1 = np.sqrt(np.maximum(np.bincount(seg, mass * G1[cell] ** 2) / m0 - v1 ** 2 + h ** 2 / 12, 0))
e2 = np.sqrt(np.maximum(np.bincount(seg, mass * G2[cell] ** 2) / m0 - v2 ** 2 + h ** 2 / 12, 0))
x, y = np.asarray(z["xbin"], float), np.asarray(z["ybin"], float)
R = np.hypot(x, y)
H = np.asarray(Z[f"gaia_pm|model|{TAG}"], float)
H = np.clip(H, 0, None) / np.clip(H, 0, None).sum(1, keepdims=True)


def rt(i, a, b):
    rh = np.array([x[i], y[i]]) / R[i]
    th = -np.array([-rh[1], rh[0]])  # along the rotation (x-East mu_T negative for omega Cen)
    return (rh @ MAJ) * a + (rh @ MNR) * b, (th @ MAJ) * a + (th @ MNR) * b


obs, mod, sep = [], [], []
for i in np.flatnonzero((R >= RMIN) & (R < RMAX)):
    s = ap == i
    n = s.sum()
    vr, vt = rt(i, v1[s], v2[s])
    obs.append(np.c_[vr - vr.mean(), vt])
    k = rng.choice(K * K, size=(NDRAW, n), p=H[i])
    a = G1[k] + h * (rng.random(k.shape) - 0.5)
    b = G2[k] + h * (rng.random(k.shape) - 0.5)
    vr_i, vt_i = rt(i, a, b)  # intrinsic draws in (R, T)
    nr, nt = rt(i, rng.standard_normal(k.shape) * e1[s], rng.standard_normal(k.shape) * e2[s])  # star's own errors
    mr, mt_ = (vr_i + nr).ravel() - vr_i.mean(), (vt_i + nt).ravel()
    mod.append(np.c_[mr, mt_])
    sep.append(np.c_[mr, rng.permutation(mt_)])
obs, mod, sep = (np.concatenate(v) for v in (obs, mod, sep))

COL = {"stars": "0.1", "model": "#0072B2", "sep": "#D55E00"}
LAB = {"stars": r"\textit{Gaia} stars", "model": "dynamical model", "sep": "same model, R and T uncoupled"}
fig, ax = plt.subplots(2, 2, figsize=(1.35 * 2.9 * plt.rcParams["figure.figsize"][1], 2.9 * plt.rcParams["figure.figsize"][1]),
                       constrained_layout=True)
edges = np.linspace(-50, 50, 41)
cen = 0.5 * (edges[1:] + edges[:-1])
for a, j, lab in ((ax[0, 0], 0, r"$v_R$"), (ax[1, 0], 1, r"$v_T$")):
    n_o = np.histogram(obs[:, j], edges)[0]
    a.errorbar(cen, n_o / n_o.sum(), np.sqrt(n_o) / n_o.sum(), ls="none", marker="o", ms=3, color=COL["stars"],
               label=LAB["stars"])
    for v, k, ls in ((mod, "model", "-"), (sep, "sep", "--")):
        n_m = np.histogram(v[:, j], edges)[0]
        a.plot(cen, n_m / n_m.sum(), color=COL[k], ls=ls, lw=2, label=LAB[k])
    a.set_xlabel(lab + r" [km s$^{-1}$]")
    a.set_ylabel("fraction of stars")
    a.set_ylim(0, 1.45 * n_o.max() / n_o.sum())
ax[0, 0].set_title(r"\textbf{Fit $v_R$ and $v_T$ separately}" + "\nthe two models are indistinguishable", fontsize="medium")
ax[0, 0].legend(frameon=False, fontsize="small", loc="upper left")
sR = obs[:, 0].std()
grp = [(0, 0.5), (0.5, 1.2), (1.2, np.inf)]
xg = np.arange(3)
names = [rf"slow\\$|v_R|<{0.5 * sR:.0f}$", rf"medium", rf"fast\\$|v_R|>{1.2 * sR:.0f}$"]
for v, k in ((obs, "stars"), (mod, "model"), (sep, "sep")):
    a_ = np.abs(v[:, 0])
    ms = [v[(a_ >= lo * sR) & (a_ < hi * sR), 1] for lo, hi in grp]
    mu = np.array([t.mean() for t in ms]); sg = np.array([t.std() for t in ms])
    if k == "stars":
        n = np.array([len(t) for t in ms])
        ax[0, 1].errorbar(xg, mu - v[:, 1].mean(), sg / np.sqrt(n), ls="none", marker="o", ms=7, color=COL[k], zorder=5)
        ax[1, 1].errorbar(xg, sg, sg / np.sqrt(2 * n), ls="none", marker="o", ms=7, color=COL[k], zorder=5)
    else:
        for a, y in ((ax[0, 1], mu - v[:, 1].mean()), (ax[1, 1], sg)):
            a.plot(xg, y, color=COL[k], ls="-" if k == "model" else "--", lw=2, marker="s" if k == "model" else None)
ax[0, 1].set_ylabel(r"rotation relative to average [km s$^{-1}$]")
ax[0, 1].axhline(0, color="0.6", lw=0.6)
ax[1, 1].set_ylabel(r"tangential dispersion $\sigma_T$ [km s$^{-1}$]")
ax[0, 1].set_title(r"\textbf{Fit in 2D}" + "\nradially moving stars rotate less and are hotter", fontsize="medium")
for a in ax[:, 1]:
    a.set_xticks(xg, ["slow", "medium", "fast"])
    a.set_xlim(-0.4, 2.4)
ax[1, 1].set_xlabel(rf"radial speed $|v_R|$ (slow $<{0.5 * sR:.0f}$, fast $>{1.2 * sR:.0f}$ km s$^{{-1}}$)")
fig.suptitle(rf"\textit{{Gaia}}, ${RMIN:.0f}''$--${RMAX:.0f}''$, {len(obs)} stars", fontsize="small")
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print(f"{len(obs)} stars, sigma_R {sR:.1f} km/s")
