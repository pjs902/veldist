"""Paper figure r_fits: model against data velocity distributions for a selection of apertures of every veldist set.

usage: python r_fits.py PACK_DIR MODEL_HISTS.npz TAG OUT.pdf [--slides]
MODEL_HISTS.npz from experiments/fitstat/model_hists.py ('{set}|data', '{set}|err', '{set}|model|{TAG}', each
n_ap x cells, scaled by the aperture's projected mass; each is renormalised here to its own unit sum).
Columns: four demonstration bins per set, ordered by radius (demo_pm, demo_muse): centre, strongest rotation,\nstrongest radial anisotropy, outermost (MUSE: centre, most receding, most approaching, outermost). 2D panels carry\nthe bin's radial (R, dashed) and rotation (T, dotted) directions.
Paper rows (marginals along the bin's R (filled, solid) and T (open, dashed) directions, linear scale, each cell
split into 5x5 sub-points before binning the projection; contours at 1e-3, 1e-2, 0.1, 0.5 of the model peak; slides at 0.05-0.8): MUSE LOSVD (data +- error, model line); HST data (grey, log scale down to 1e-3 of the peak) with log-spaced model contours; HST marginals along
v_maj (circles) and v_min (squares, x 1e-2) on a log axis, cells with data < error omitted (errors summed in quadrature over cells, ignoring cell correlations: display only);
the same two rows for Gaia. --slides: landscape (~4:3, square panels), data and model only (MUSE, HST 2D, Gaia 2D).
PM axes are the pack's (pm1 along the kinematic major axis PA 104.3, pm2 along the minor), km/s.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.colors import ListedColormap, LogNorm

from d2_maps import PA_KIN, pm_moments

P, Z, TAG, OUT = Path(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3], sys.argv[4]
SLIDES = "--slides" in sys.argv
CM = "#D55E00"
GREY = ListedColormap(plt.cm.Greys(np.linspace(0, 0.7, 256)))  # darkest = mid grey, so contours stay visible


def demo_pm(m):
    """Columns as (index, role, note), ordered by radius: centre, strongest rotation (mu_T over its error),
    strongest radial anisotropy (lowest sigma_T/sigma_R), outermost; rotation and anisotropy among bins with at
    least the median star count. A centre bin within 0.8 of the rotation bin's radius is replaced by the median-R bin."""
    big, ratio = m["n"] >= np.median(m["n"]), m["st"] / m["sr"]
    rot = int(np.argmax(np.where(big, m["mt"] * np.sqrt(m["n"]) / m["st"], -np.inf)))
    ani = int(np.argmin(np.where(big, ratio, np.inf)))
    cen = int(np.argmin(m["R"]))
    cen = (cen, "centre") if m["R"][cen] < 0.8 * m["R"][rot] else (int(np.argsort(m["R"])[len(m["R"]) // 2]), "intermediate")
    cols = [(cen[0], cen[1], rf"$\sigma_T/\sigma_R={ratio[cen[0]]:.2f}$"),
            (rot, "rotation", rf"$\langle v_T\rangle={m['mt'][rot]:+.1f}$ km s$^{{-1}}$"),
            (ani, "radial anisotropy", rf"$\sigma_T/\sigma_R={ratio[ani]:.2f}$"),
            (int(np.argmax(m["R"])), "outermost", rf"$\sigma_T/\sigma_R={ratio[int(np.argmax(m['R']))]:.2f}$")]
    return sorted(cols, key=lambda c: m["R"][c[0]])


def dirs(S, i):
    """Unit radial vector and the unit vector along the rotation (x-East mu_T is negative for omega Cen), (x, y)."""
    rh = np.array([S["x"][i], S["y"][i]]) / S["R"][i]
    return rh, -np.array([-rh[1], rh[0]])


def demo_muse(R, v):
    cols = [(int(np.argmin(R)), "centre"), (int(np.argmax(v)), "receding"), (int(np.argmin(v)), "approaching"),
            (int(np.argmax(R)), "outermost")]
    return [(i, role, rf"$\bar v={v[i]:+.1f}$") for i, role in sorted(cols, key=lambda c: R[c[0]])]


def unit(s):
    d, e, m = (np.asarray(Z[f"{s}|{k}"], float) for k in ("data", "err", f"model|{TAG}"))
    n = d.sum(1, keepdims=True)
    return d / n, e / n, m / m.sum(1, keepdims=True)


T = Table.read(P / "veldist_kinematics.ecsv")
assert (np.asarray(T["binID_dynamite"]) == np.arange(1, len(T) + 1)).all()
vc = np.asarray(T.meta["vcent"])
dM, eM, mM = unit("veldist")
L = np.array([T[f"losvd_{j}"] for j in range(len(vc))]).T
assert dM.shape == L.shape and np.corrcoef((L / L.sum(1, keepdims=True)).ravel(), dM.ravel())[0, 1] > 0.99
RM = np.hypot(T["xbin"], T["ybin"])
COLS = {"veldist": demo_muse(RM, np.asarray(T["v"]))}
PA = np.radians(PA_KIN)
MAJ, MNR = np.array([np.sin(PA), np.cos(PA)]), np.array([-np.cos(PA), np.sin(PA)])

PM = {}
for s, f in (("hst_pm", "hst_veldist_pm_2dhist.npz"), ("gaia_pm", "gaia_veldist_pm_2dhist.npz")):
    z = np.load(P / f)
    K, half = z["PM_2dhist"].shape[1], float(z["vxrange"])
    d, e, m = unit(s)
    H = z["PM_2dhist"].reshape(len(d), -1)
    assert d.shape == H.shape and np.corrcoef((H / H.sum(1, keepdims=True)).ravel(), d.ravel())[0, 1] > 0.99
    cc = np.linspace(-half, half, K + 1)
    PM[s] = dict(K=K, half=half, cc=0.5 * (cc[1:] + cc[:-1]), d=d, e=e, m=m, R=np.hypot(z["xbin"], z["ybin"]),
                 N=z["nstarbin"], x=np.asarray(z["xbin"], float), y=np.asarray(z["ybin"], float))
    COLS[s] = demo_pm(pm_moments(P / f))

rows = ["veldist", "hst_pm", "gaia_pm"] if SLIDES else ["veldist", "hst_pm", "hst_marg", "gaia_pm", "gaia_marg"]
w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(len(rows), 4, figsize=(1.3 * 2.9 * h, 2.9 * h) if SLIDES else (2.2 * w, 1.4 * 2.2 * w),
                       constrained_layout=True, subplot_kw=dict(box_aspect=1))  # every panel square, same size
LAB = {"hst_pm": "HST", "gaia_pm": "Gaia"}
for r, kind in enumerate(rows):
    if kind == "veldist":
        for j, (i, role, note) in enumerate(COLS["veldist"]):
            a = ax[r, j]
            a.errorbar(vc, dM[i], eM[i], ls="none", marker="o", ms=3, color="0.2", label="data")
            a.plot(vc, mM[i], color=CM, label="model")
            a.set_title(rf"MUSE, {role}: $R={RM[i]:.0f}''$" + "\n" + note + r" km s$^{-1}$", fontsize="small")
            a.set_ylim(top=1.35 * max((dM[i] + eM[i]).max(), mM[i].max()))  # headroom for the legend
            a.set_xlabel(r"$v_{\rm LOS}$ [km s$^{-1}$]")
        ax[r, 0].set_ylabel("fraction per bin")
        ax[r, 0].legend(frameon=False, fontsize="x-small", loc="upper left")
        continue
    s = kind.replace("_marg", "_pm")
    S = PM[s]
    K, cc, half = S["K"], S["cc"], S["half"]
    for j, (i, role, note) in enumerate(COLS[s]):
        a = ax[r, j]
        img = lambda v: v[i].reshape(K, K).T  # rows = pm2 (minor), columns = pm1 (major)
        if kind.endswith("_marg"):
            rh, th = dirs(S, i)
            h = 2 * half / K
            sub = (np.arange(5) + 0.5) / 5 - 0.5  # 5x5 sub-points per cell: no aliasing when binning projections
            G1, G2 = (np.add.outer(g.ravel(), h * o.ravel()) for g, o in
                      zip(np.meshgrid(cc, cc, indexing="ij"), np.meshgrid(sub, sub, indexing="ij")))
            edges = np.linspace(-half, half, K + 1)
            for u, mk, mfc, ls, lab in ((rh, "o", "0.2", "-", "R"), (th, "s", "w", "--", "T")):
                pu = G1 * (u @ MAJ) + G2 * (u @ MNR)  # sub-points (cell, 25) projected on R or T
                F = np.zeros((K * K, K))  # fraction of each cell falling in each projected bin
                b = np.clip(np.digitize(pu, edges) - 1, 0, K - 1)
                np.add.at(F, (np.repeat(np.arange(K * K), 25), b.ravel()), 1 / 25)
                D, M = S["d"][i] @ F, S["m"][i] @ F
                E = np.sqrt(S["e"][i] ** 2 @ F ** 2)  # a cell's sub-points are fully correlated
                a.errorbar(cc, D, E, ls="none", marker=mk, ms=3, mfc=mfc, color="0.2", label=f"data, {lab}")
                a.plot(cc, M, color=CM, ls=ls, label=f"model, {lab}")
            a.set_ylim(top=1.45 * a.get_ylim()[1])  # headroom for the legend
            if j == 0:
                a.legend(frameon=False, fontsize="xx-small", loc="upper left", ncol=2)
            a.set_xlabel(r"$v_R$, $v_T$ [km s$^{-1}$]")
        else:
            a.imshow(img(S["d"]), origin="lower", extent=[-half, half, -half, half], cmap=GREY,
                     norm=None if SLIDES else LogNorm(img(S["d"]).max() * 1e-3, img(S["d"]).max()))
            a.contour(cc, cc, img(S["m"]), levels=img(S["m"]).max() * np.array([0.05, 0.2, 0.5, 0.8] if SLIDES else [1e-3, 1e-2, 0.1, 0.5]),
                      colors=CM, linewidths=0.8)
            rh, th = dirs(S, i)
            L = 0.85 * half
            for u, lab, ls in ((rh, "R", "--"), (th, "T", ":")):
                pu = np.array([u @ MAJ, u @ MNR])  # direction in (v_maj, v_min)
                a.plot([-L * pu[0], L * pu[0]], [-L * pu[1], L * pu[1]], color="0.35", ls=ls, lw=0.7)
                a.text(*(1.08 * L * pu), lab, color="0.35", fontsize="x-small", ha="center", va="center")
            a.set_title(rf"{LAB[s]}, {role}: $R={S['R'][i]:.0f}''$" + "\n" + note, fontsize="small")
            a.set_xlabel(r"$v_{\rm maj}$ [km s$^{-1}$]")
    ax[r, 0].set_ylabel("fraction per bin" if kind.endswith("_marg") else r"$v_{\rm min}$ [km s$^{-1}$]")
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print({s: [(role, int(i)) for i, role, _ in c] for s, c in COLS.items()})
