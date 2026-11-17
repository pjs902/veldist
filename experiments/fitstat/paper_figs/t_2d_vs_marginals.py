"""Talk figure: why fit the 2D proper-motion distribution rather than its R and T marginals.

usage: python t_2d_vs_marginals.py STARS_PACK_DIR MODEL_HISTS.npz TAG OUT.pdf [RMIN RMAX]
Gaia bins with RMIN <= R < RMAX (default 462-942") are stacked in each bin's (v_R, v_T) frame (v_T positive along the
rotation). Observed: each star's velocity is the centroid of its error-kernel row (pack PM_star_*), its per-axis error
from the row variance (+ h^2/12 for the sharpened cell-integrated kernel). Model: every star is replaced by draws from
its bin's model histogram (uniform within the cell) plus Gaussian noise with that star's errors, i.e. the model in
observed space. Separable: the same draws with v_T shuffled within the bin, which keeps both marginals and removes
any R-T coupling. Stars are made separable the same way (v_T shuffled among the bin's stars, 20 shuffles), so each 2D map is a
distribution minus its own product of marginals: the part that a fit to the R and T marginals cannot see.
A: stars, B: model (both as significance for the star count), C: <v_T> minus its mean in bins of |v_R|.
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


obs, obs_sep, mod, sep = [], [], [], []


def shuffle_t(vr, vt, n_rep):
    """(v_R, v_T) pairs with v_T permuted among the bin's entries: same marginals, no R-T coupling."""
    return np.c_[np.tile(vr, n_rep), np.concatenate([rng.permutation(vt) for _ in range(n_rep)])]


for i in np.flatnonzero((R >= RMIN) & (R < RMAX)):
    s = ap == i
    n = s.sum()
    vr, vt = rt(i, v1[s], v2[s])
    vr = vr - vr.mean()
    obs.append(np.c_[vr, vt])
    obs_sep.append(shuffle_t(vr, vt, NDRAW))
    k = rng.choice(K * K, size=(NDRAW, n), p=H[i])
    a = G1[k] + h * (rng.random(k.shape) - 0.5)
    b = G2[k] + h * (rng.random(k.shape) - 0.5)
    vr_i, vt_i = rt(i, a, b)  # intrinsic draws in (R, T)
    nr, nt = rt(i, rng.standard_normal(k.shape) * e1[s], rng.standard_normal(k.shape) * e2[s])  # star's own errors
    mr, mt_ = (vr_i + nr).ravel() - vr_i.mean(), (vt_i + nt).ravel()
    mod.append(np.c_[mr, mt_])
    sep.append(shuffle_t(mr, mt_, 1))
obs, obs_sep, mod, sep = (np.concatenate(v) for v in (obs, obs_sep, mod, sep))

ex, ey = np.arange(0, 42.1, 6.0), np.arange(-42, 42.1, 6.0)  # folded in v_R: the coupling is symmetric
H2 = lambda v: np.histogram2d(np.abs(v[:, 0]), v[:, 1], [ex, ey])[0]
sig_d = (H2(obs) - H2(obs_sep) / NDRAW) / np.sqrt(np.maximum(H2(obs_sep) / NDRAW, 1))
f = len(obs) / len(mod)  # model expectation scaled to the star count
sig_m = f * (H2(mod) - H2(sep)) / np.sqrt(np.maximum(f * H2(sep), 1))
w, hh = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(1, 3, figsize=(1.3 * 2.9 * hh, 0.45 * 2.9 * hh), constrained_layout=True,
                       subplot_kw=dict(box_aspect=1.2))
for a, S_, t in ((ax[0], sig_d, "stars"), (ax[1], sig_m, "model")):
    im = a.pcolormesh(ex, ey, S_.T, cmap="RdBu_r", vmin=-4, vmax=4)
    a.axhline(0, color="0.5", lw=0.5)
    a.set_xlabel(r"$|v_R|$ [km s$^{-1}$]")
    a.set_title(t + r": 2D $-$ (R marginal $\times$ T marginal)", fontsize="small")
ax[0].set_ylabel(r"$v_T$ [km s$^{-1}$] (+ = along rotation)")
fig.colorbar(im, ax=ax[:2], shrink=0.6, location="bottom", label=r"excess over uncoupled [$\sigma$]")
ed = np.arange(0, 41, 5.0)
mid = 0.5 * (ed[1:] + ed[:-1])
COL = {"stars": "0.15", "model": "#0072B2", "sep": "#D55E00"}
for v, k in ((obs, "stars"), (mod, "model"), (sep, "sep")):
    b = np.digitize(np.abs(v[:, 0]), ed) - 1
    mt = np.array([v[b == q, 1].mean() for q in range(len(mid))]) - v[(b >= 0) & (b < len(mid)), 1].mean()
    if k == "stars":
        et = np.array([v[b == q, 1].std() / np.sqrt(np.sum(b == q)) for q in range(len(mid))])
        ax[2].errorbar(mid, mt, et, ls="none", marker="o", ms=4, color=COL[k], label="stars")
    else:
        ax[2].plot(mid, mt, color=COL[k], ls="-" if k == "model" else "--", lw=1.5,
                   label="model" if k == "model" else "same marginals, uncoupled")
ax[2].axhline(0, color="0.5", lw=0.5)
ax[2].set_xlabel(r"$|v_R|$ [km s$^{-1}$]")
ax[2].set_ylabel(r"$\langle v_T\rangle - \overline{v_T}$ [km s$^{-1}$]")
ax[2].set_title("stars on radial orbits rotate less", fontsize="small")
ax[2].legend(frameon=False, fontsize="x-small", loc="upper right")
fig.suptitle(rf"\textit{{Gaia}}, {RMIN:.0f}--{RMAX:.0f}$''$, {len(obs)} stars: the R$-$T coupling is invisible in the "
             r"R and T marginals", fontsize="small")
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print(f"{len(obs)} stars; max |sig| data {np.abs(sig_d).max():.1f}, model {np.abs(sig_m).max():.1f}; "
      f"corr(data map, model map) {np.corrcoef(sig_d.ravel(), sig_m.ravel())[0, 1]:.2f}")
