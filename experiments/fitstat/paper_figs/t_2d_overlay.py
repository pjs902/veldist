"""Talk figure from real data: the Gaia stars' (v_R, v_T) distribution against the same stars with v_T shuffled
within each bin: identical v_R and v_T distributions, different 2D distribution.

usage: python t_2d_overlay.py STARS_PACK_DIR MODEL_HISTS.npz TAG OUT.pdf [ERR_PERCENTILE]
Stars and bins as t_2d_vs_marginals.py (Gaia 462-942", each bin's (v_R, v_T) frame, v_T along the rotation), keeping
only stars whose per-star error is below the given percentile (default 50: ~7.7 km/s). Bright stars only because
heteroscedastic errors also couple |v_R| and |v_T| (a large-error star is broad in both): over all stars that
accounts for most of the difference, below the median error for ~0.26 of the 0.73 rms (noise floor 0.375).
Shuffling within bins keeps each bin's v_R and v_T distributions exactly, so stacking bins with different rotation
and dispersion adds no coupling. Left: contours (smoothed 2 km/s histogram) of the stars and of the shuffled stars;
below it the two marginals (identical). Middle: stars minus shuffled; right: the model (each star replaced by model draws plus its errors, same\nselection) minus its shuffle. Maps folded in v_R (f even in v_R in equilibrium; the odd part is consistent with noise).
"""
import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
PCT = float(sys.argv[5]) if len(sys.argv) > 5 else 50.0
out = sys.argv[4]
sys.argv = sys.argv[:4] + [str(HERE / "_unused.pdf")]  # reuse the star extraction of t_2d_vs_marginals.py
sys.path.insert(0, str(HERE))
g = runpy.run_path(str(HERE / "t_2d_vs_marginals.py"))
(HERE / "_unused.pdf").unlink(missing_ok=True)
(HERE / "_unused.png").unlink(missing_ok=True)
obs, ap, e1, e2 = g["obs"], g["ap"], g["e1"], g["e2"]
z = np.load(Path(sys.argv[1]) / "gaia_veldist_pm_2dhist.npz")
R = np.hypot(z["xbin"].astype(float), z["ybin"].astype(float))
sel = np.flatnonzero((R >= g["RMIN"]) & (R < g["RMAX"]))
lab = np.concatenate([np.full((ap == i).sum(), i) for i in sel])  # obs is stacked in this bin order
err = np.concatenate([np.hypot(e1[ap == i], e2[ap == i]) / np.sqrt(2) for i in sel])
cut = np.percentile(err, PCT)
keep = err < cut
NDRAW = g["NDRAW"]
labm = np.concatenate([np.full(NDRAW * (ap == i).sum(), i) for i in sel])
errm = np.concatenate([np.tile(np.hypot(e1[ap == i], e2[ap == i]) / np.sqrt(2), NDRAW) for i in sel])
mod, labm = g["mod"][errm < cut], labm[errm < cut]
obs, lab = obs[keep], lab[keep]
rng = np.random.default_rng(4)


def shuffle_within(v, l, nrep):
    """v_T permuted among each bin's entries: every bin keeps its v_R and v_T distributions, coupling removed."""
    out = []
    for _ in range(nrep):
        t = v[:, 1].copy()
        for i in np.unique(l):
            m = l == i
            t[m] = rng.permutation(t[m])
        out.append(np.c_[v[:, 0], t])
    return np.concatenate(out)


mirror = lambda v: np.r_[v, np.c_[-v[:, 0], v[:, 1]]]  # fold in v_R (equilibrium: f even in v_R); halves the noise
NREP = 20
sh = shuffle_within(obs, lab, NREP)
msh = shuffle_within(mod, labm, 1)
e = np.arange(-44, 44.1, 2.0)
c = 0.5 * (e[1:] + e[:-1])
Hh = lambda v, n: gaussian_filter(np.histogram2d(v[:, 0], v[:, 1], [e, e])[0] / n, 1.5)
Ho, Hs = Hh(mirror(obs), 2), Hh(mirror(sh), 2 * NREP)
f = len(mod) / len(obs)  # model scaled to the star count
Dm = Hh(mirror(mod), 2 * f) - Hh(mirror(msh), 2 * f)
D = Ho - Hs

COL = {"stars": "k", "sh": "#D55E00"}
W, Hf = 13.33, 7.5  # 16:9 slide
fig = plt.figure(figsize=(W, Hf))
sq = 0.25  # panel width as a fraction of the figure width; height follows for square panels
hq = sq * W / Hf
y0 = 0.14
a0 = fig.add_axes([0.06, y0, sq, hq])
top = fig.add_axes([0.06, y0 + hq + 0.01, sq, 0.11], sharex=a0)
rgt = fig.add_axes([0.06 + sq + 0.005, y0, 0.11 * Hf / W, hq], sharey=a0)
a1 = fig.add_axes([0.435, y0, sq, hq])
a2 = fig.add_axes([0.700, y0, sq, hq], sharey=a1)
cax = fig.add_axes([0.962, y0 + 0.1 * hq, 0.008, 0.8 * hq])
lev = Ho.max() * np.array([0.1, 0.3, 0.6, 0.9])
a0.contour(c, c, Hs.T, levels=lev, colors=COL["sh"], linewidths=2.2, linestyles="--")
a0.contour(c, c, Ho.T, levels=lev, colors=COL["stars"], linewidths=1.5)
edges = np.arange(-40, 40.1, 2.5)
cen = 0.5 * (edges[1:] + edges[:-1])
for j, ax_, orient in ((0, top, "x"), (1, rgt, "y")):
    n_o = np.histogram(mirror(obs)[:, j], edges)[0] / 2
    n_s = np.histogram(mirror(sh)[:, j], edges)[0] / (2 * NREP)
    if orient == "x":
        ax_.step(cen, n_o, where="mid", color=COL["stars"], lw=1.5)
        ax_.plot(cen, n_s, color=COL["sh"], lw=2.2, ls="--")
        ax_.set_ylim(0, 1.15 * n_o.max())
    else:
        ax_.step(n_o, cen, where="mid", color=COL["stars"], lw=1.5)
        ax_.plot(n_s, cen, color=COL["sh"], lw=2.2, ls="--")
        ax_.set_xlim(0, 1.15 * n_o.max())
    ax_.set_axis_off()
fig.text(0.06 + sq / 2, y0 + hq + 0.175, r"same $v_R$ and $v_T$ histograms", ha="center", fontsize="x-large")
fig.text(0.06 + sq / 2, y0 + hq + 0.135, r"black: \textit{Gaia} stars; orange dashed: same stars, $v_T$ shuffled",
         ha="center", fontsize="small")
vm = max(np.abs(D).max(), np.abs(Dm).max())
for a, M, t in ((a1, D, r"\textit{Gaia} stars $-$ shuffled"), (a2, Dm, r"dynamical model $-$ shuffled")):
    im = a.pcolormesh(e, e, M.T, cmap="RdBu_r", vmin=-vm, vmax=vm)
    a.contour(c, c, Hs.T, levels=lev, colors="0.55", linewidths=0.6)
    a.set_title(t, fontsize="large")
fig.colorbar(im, cax=cax, label="excess stars per cell")
fig.text(0.6925, y0 + hq + 0.105, r"what fitting $v_R$ and $v_T$ separately cannot see", ha="center",
         fontsize="x-large")
box = dict(boxstyle="round,pad=0.25", fc="w", ec="none", alpha=0.85)
a1.text(0, 33, r"near-circular orbits (small $|v_R|$):" + "\nmore rotation", ha="center", va="center", fontsize="small", bbox=box)
a1.text(-24, -22, "radial orbits:\nless rotation", ha="center", va="center", fontsize="small", bbox=box)
a1.text(24, -22, "radial orbits:\nless rotation", ha="center", va="center", fontsize="small", bbox=box)
for a in (a0, a1, a2):
    a.set_xlim(-40, 40)
    a.set_ylim(-40, 40)
    a.axhline(0, color="0.7", lw=0.5)
    a.axvline(0, color="0.7", lw=0.5)
    a.set_xlabel(r"$v_R$ [km s$^{-1}$]")
    a.set_xticks([-20, 0, 20])
    a.set_yticks([-20, 0, 20])
a0.set_ylabel(r"$v_T$ [km s$^{-1}$] (+ along rotation)")
plt.setp(a2.get_yticklabels(), visible=False)
fig.text(0.5, 0.015, rf"\textit{{Gaia}}, ${g['RMIN']:.0f}''$--${g['RMAX']:.0f}''$, {len(obs)} stars with errors "
         rf"$<{cut:.1f}$ km s$^{{-1}}$; maps folded in $v_R$; model: each star replaced by model draws plus its errors",
         ha="center", fontsize="small", color="0.3")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print(f"{len(obs)} stars; corr(stars map, model map) {np.corrcoef(D.ravel(), Dm.ravel())[0, 1]:.2f}")
