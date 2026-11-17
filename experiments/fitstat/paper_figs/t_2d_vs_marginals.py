"""Talk figure: why fit the 2D proper-motion distribution rather than its R and T marginals.

usage: python t_2d_vs_marginals.py STARS_PACK_DIR MODEL_HISTS.npz TAG OUT.pdf [RMIN RMAX]
Gaia bins with RMIN <= R < RMAX (default 462-942") are stacked in each bin's (v_R, v_T) frame (v_T positive along the
rotation). Observed: each star's velocity is the centroid of its error-kernel row (pack PM_star_*), its per-axis error
from the row variance (+ h^2/12 for the sharpened cell-integrated kernel). Model: every star is replaced by draws from
its bin's model histogram (uniform within the cell) plus Gaussian noise with that star's errors, i.e. the model in
observed space. Separable: the same draws with v_T shuffled within the bin, which keeps both marginals and removes
any R-T coupling. Left: the two marginals (model and separable identical by construction). Right: <v_T> in bins of
|v_R|: the coupling that only the 2D distribution carries.
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
    na = rng.standard_normal(k.shape) * e1[s]
    nb = rng.standard_normal(k.shape) * e2[s]
    nr, nt = rt(i, na, nb)  # each star's own errors, rotated to (R, T)
    vt_sh = np.take_along_axis(vt_i, rng.permuted(np.tile(np.arange(n), (NDRAW, 1)), axis=1), 1)
    for out, t in ((mod, vt_i), (sep, vt_sh)):
        out.append(np.c_[(vr_i + nr).ravel() - vr.mean(), (t + nt).ravel()])
obs, mod, sep = (np.concatenate(v) for v in (obs, mod, sep))

COL = {"stars": "0.15", "model": "#0072B2", "sep": "#D55E00"}
w, hh = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(1, 3, figsize=(1.3 * 2.9 * hh, 0.42 * 2.9 * hh), constrained_layout=True,
                       subplot_kw=dict(box_aspect=1))
edges = np.linspace(-50, 50, 41)
cen = 0.5 * (edges[1:] + edges[:-1])
for a, j, lab in ((ax[0], 0, r"$v_R$"), (ax[1], 1, r"$v_T$")):
    n_o = np.histogram(obs[:, j], edges)[0]
    a.errorbar(cen, n_o / n_o.sum(), np.sqrt(n_o) / n_o.sum(), ls="none", marker="o", ms=3, color=COL["stars"],
               label="stars")
    for v, k, ls in ((mod, "model", "-"), (sep, "sep", "--")):
        n_m = np.histogram(v[:, j], edges)[0]
        a.plot(cen, n_m / n_m.sum(), color=COL[k], ls=ls, lw=1.5,
               label={"model": "model", "sep": "separable: same marginals, no R-T coupling"}[k])
    a.set_xlabel(lab + r" [km s$^{-1}$]")
    a.set_title(lab + " marginal: model = separable", fontsize="small")
ax[0].set_ylabel("fraction of stars")
ax[0].legend(frameon=False, fontsize="x-small", loc="upper left")
ax[0].set_ylim(top=1.4 * ax[0].get_ylim()[1])
ax[1].set_ylim(ax[0].get_ylim())
ed = np.arange(0, 41, 5.0)
mid = 0.5 * (ed[1:] + ed[:-1])
for v, k, mk in ((obs, "stars", "o"), (mod, "model", None), (sep, "sep", None)):
    b = np.digitize(np.abs(v[:, 0]), ed) - 1
    ok = (b >= 0) & (b < len(mid))
    mt = np.array([v[ok & (b == q), 1].mean() for q in range(len(mid))])
    if k == "stars":
        et = np.array([v[ok & (b == q), 1].std() / np.sqrt(np.sum(ok & (b == q))) for q in range(len(mid))])
        ax[2].errorbar(mid, mt, et, ls="none", marker=mk, ms=4, color=COL[k])
    else:
        ax[2].plot(mid, mt, color=COL[k], ls="-" if k == "model" else "--", lw=1.5)
ax[2].set_xlabel(r"$|v_R|$ [km s$^{-1}$]")
ax[2].set_ylabel(r"$\langle v_T\rangle$ [km s$^{-1}$]")
ax[2].set_title(r"$\langle v_T\rangle$ against $|v_R|$: only in 2D", fontsize="small")
fig.suptitle(rf"\textit{{Gaia}}, {RMIN:.0f}--{RMAX:.0f}$''$ ({len(obs)} stars)", fontsize="small")
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print(f"{len(obs)} stars; <v_T> by |v_R| bins {mid}:")
for v, k in ((obs, "stars"), (mod, "model"), (sep, "separable")):
    b = np.digitize(np.abs(v[:, 0]), ed) - 1
    print(f"  {k:9s}", " ".join(f"{v[b == q, 1].mean():+5.2f}" for q in range(len(mid))))
