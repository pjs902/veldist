"""Paper figure r_fits: model against data velocity distributions for a selection of apertures of every veldist set.

usage: python r_fits.py PACK_DIR MODEL_HISTS.npz TAG OUT.pdf [--slides]
MODEL_HISTS.npz from experiments/fitstat/model_hists.py ('{set}|data', '{set}|err', '{set}|model|{TAG}', each
n_ap x cells, scaled by the aperture's projected mass; each is renormalised here to its own unit sum).
Columns: four apertures per set, the innermost and then radius quantiles 35, 65 and 95 per cent.
Paper rows (contours at 1e-3, 1e-2, 0.1, 0.5 of the model peak; slides at 0.05-0.8): MUSE LOSVD (data +- error, model line); HST data (grey) with log-spaced model contours; HST marginals along
v_maj and v_min on a log axis (errors summed in quadrature over cells, ignoring cell correlations: display only);
the same two rows for Gaia. --slides: landscape, data and model only (MUSE, HST 2D, Gaia 2D).
PM axes are the pack's (pm1 along the kinematic major axis PA 104.3, pm2 along the minor), km/s.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

P, Z, TAG, OUT = Path(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3], sys.argv[4]
SLIDES = "--slides" in sys.argv
QS = (0.35, 0.65, 0.95)
CM = "#D55E00"


def pick(R):
    o = np.argsort(R)
    return [o[0]] + [o[int(q * (len(R) - 1))] for q in QS]


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

PM = {}
for s, f in (("hst_pm", "hst_veldist_pm_2dhist.npz"), ("gaia_pm", "gaia_veldist_pm_2dhist.npz")):
    z = np.load(P / f)
    K, half = z["PM_2dhist"].shape[1], float(z["vxrange"])
    d, e, m = unit(s)
    H = z["PM_2dhist"].reshape(len(d), -1)
    assert d.shape == H.shape and np.corrcoef((H / H.sum(1, keepdims=True)).ravel(), d.ravel())[0, 1] > 0.99
    cc = np.linspace(-half, half, K + 1)
    PM[s] = dict(K=K, half=half, cc=0.5 * (cc[1:] + cc[:-1]), d=d, e=e, m=m, R=np.hypot(z["xbin"], z["ybin"]),
                 N=z["nstarbin"])

rows = ["veldist", "hst_pm", "gaia_pm"] if SLIDES else ["veldist", "hst_pm", "hst_marg", "gaia_pm", "gaia_marg"]
w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(len(rows), 4, figsize=(16 / 9 * 2.9 * h, 2.9 * h) if SLIDES else (2.2 * w, 2.9 * h),
                       constrained_layout=True)
LAB = {"hst_pm": "HST", "gaia_pm": "Gaia"}
for r, kind in enumerate(rows):
    if kind == "veldist":
        for j, i in enumerate(pick(RM)):
            a = ax[r, j]
            a.errorbar(vc, dM[i], eM[i], ls="none", marker="o", ms=3, color="0.2", label="data")
            a.plot(vc, mM[i], color=CM, label="model")
            a.set_title(rf"MUSE: $R={RM[i]:.0f}''$", fontsize="small")
            a.set_xlabel(r"$v_{\rm LOS}$ [km s$^{-1}$]")
        ax[r, 0].set_ylabel("fraction per bin")
        ax[r, -1].legend(frameon=False, fontsize="x-small", loc="upper left")
        continue
    s = kind.replace("_marg", "_pm")
    S = PM[s]
    K, cc, half = S["K"], S["cc"], S["half"]
    for j, i in enumerate(pick(S["R"])):
        a = ax[r, j]
        img = lambda v: v[i].reshape(K, K).T  # rows = pm2 (minor), columns = pm1 (major)
        if kind.endswith("_marg"):
            for v, (ax_sum, lab, mk) in enumerate(((0, r"$v_{\rm maj}$", "o"), (1, r"$v_{\rm min}$", "s"))):
                D, M = img(S["d"]).sum(ax_sum), img(S["m"]).sum(ax_sum)
                E = np.sqrt((img(S["e"]) ** 2).sum(ax_sum))
                off = 10.0 ** (-2 * v)  # v_min shifted down two decades for legibility
                a.errorbar(cc, D * off, E * off, ls="none", marker=mk, ms=2.5, color="0.2",
                           label=None if j else lab + (r" ($\times 10^{-2}$)" if v else ""))
                a.plot(cc, M * off, color=CM)
            a.set_yscale("log")
            a.set_ylim(1e-7, 1)
            a.set_xlabel(r"$v$ [km s$^{-1}$]")
            if j == 0:
                a.legend(frameon=False, fontsize="x-small", loc="lower center")
        else:
            a.imshow(img(S["d"]), origin="lower", extent=[-half, half, -half, half], cmap="Greys")
            a.contour(cc, cc, img(S["m"]), levels=img(S["m"]).max() * np.array([0.05, 0.2, 0.5, 0.8] if SLIDES else [1e-3, 1e-2, 0.1, 0.5]),
                      colors=CM, linewidths=0.8)
            a.set_title(rf"{LAB[s]}: $R={S['R'][i]:.0f}''$, $N={S['N'][i]}$", fontsize="small")
            a.set_xlabel(r"$v_{\rm maj}$ [km s$^{-1}$]")
    ax[r, 0].set_ylabel("fraction per bin" if kind.endswith("_marg") else r"$v_{\rm min}$ [km s$^{-1}$]")
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print("apertures:", {s: [f"{x:.0f}" for x in np.sort(R)[[0, *(int(q * (len(R) - 1)) for q in QS)]]]
                     for s, R in (("veldist", RM), ("hst_pm", PM["hst_pm"]["R"]), ("gaia_pm", PM["gaia_pm"]["R"]))})
