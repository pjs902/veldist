"""LOS-only fit vs joint fit: radial profiles of the proper-motion moments the LOS-only weights predict.

    python los_only_compare.py PACK_DIR MODEL_HISTS.npz LOS_TAG JOINT_TAG OUT.png [LABEL]

LABEL names the LOS_TAG curve (default 'LOS-only fit'; e.g. 'relative HST' for a relative-PM solve).

MODEL_HISTS.npz from model_hists.py (keys '{set}|data', '{set}|model|{tag}'). Per PM set (HST, Gaia): sigma_T/sigma_R,
mean mu_T (along the rotation), sigma_R and sigma_T per aperture (d2_maps.pm_moments, Sheppard-corrected), median in
log-radius annuli, for the data, the joint fit and the LOS-only fit. MUSE: v_LOS (|v| along the major-axis sign is not
used; the rms of v in the annulus) and sigma_LOS. Prints the annulus values.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).parent / "paper_figs"))
from d2_maps import pm_moments  # noqa: E402

P, Z, LOS, JNT, OUT = Path(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
LAB = sys.argv[6] if len(sys.argv) > 6 else "LOS-only fit"
TAGS = [("data", "data", "k"), ("joint fit", f"model|{JNT}", "C1"), (LAB, f"model|{LOS}", "C2")]


def annuli(R, vals, n=9):
    e = np.geomspace(R.min() * 0.999, R.max() * 1.001, n + 1)
    i = np.digitize(R, e) - 1
    rc = np.array([np.median(R[i == k]) for k in range(n) if (i == k).any()])
    return rc, [np.array([np.median(v[i == k]) for k in range(n) if (i == k).any()]) for v in vals]


def muse(H):
    t = Table.read(P / "veldist_kinematics.ecsv")
    vc = np.asarray(t.meta["vcent"], float)
    h = np.median(np.diff(vc))
    w = H / H.sum(1, keepdims=True)
    m = w @ vc
    s = np.sqrt(np.maximum((w * (vc[None] - m[:, None]) ** 2).sum(1) - h ** 2 / 12, 0))
    return np.hypot(t["xbin"], t["ybin"]), m, s


QS = [("st/sr", r"$\sigma_T/\sigma_R$"), ("mt", r"$\langle\mu_T\rangle$ [km s$^{-1}$]"),
      ("sr", r"$\sigma_R$ [km s$^{-1}$]"), ("st", r"$\sigma_T$ [km s$^{-1}$]")]
fig, ax = plt.subplots(3, 4, figsize=(4 * plt.rcParams["figure.figsize"][0] * 0.7,
                                      3 * plt.rcParams["figure.figsize"][1] * 0.7), constrained_layout=True)
for r, (st, nm) in enumerate([("hst", r"\textit{HST}"), ("gaia", r"\textit{Gaia}")]):
    for lab, key, col in TAGS:
        m = pm_moments(P / f"{st}_veldist_pm_2dhist.npz", Z[f"{st}_pm|{key}"])
        rc, (q, mt, sr, s_t) = annuli(m["R"], [m["st"] / m["sr"], m["mt"], m["sr"], m["st"]])
        for c, v in enumerate([q, mt, sr, s_t]):
            ax[r, c].plot(rc, v, "o-" if lab == "data" else "-", color=col, label=lab)
        print(nm, lab, "R", np.round(rc).astype(int), "sT/sR", np.round(q, 3), "muT", np.round(mt, 2))
    for c, (_, ql) in enumerate(QS):
        ax[r, c].set_xscale("log")
        ax[r, c].set_ylabel(f"{nm} {ql}")
for lab, key, col in TAGS:
    R, v, s = muse(Z[f"veldist|{key}"])
    rc, (vr, sg) = annuli(R, [np.abs(v), s], n=7)
    ax[2, 0].plot(rc, vr, "o-" if lab == "data" else "-", color=col, label=lab)
    ax[2, 1].plot(rc, sg, "o-" if lab == "data" else "-", color=col, label=lab)
    print("MUSE", lab, "R", np.round(rc).astype(int), "|v|", np.round(vr, 2), "sigma", np.round(sg, 2))
ax[2, 0].set_ylabel(r"MUSE median $|v_{\rm LOS}|$ [km s$^{-1}$]")
ax[2, 1].set_ylabel(r"MUSE $\sigma_{\rm LOS}$ [km s$^{-1}$]")
for c in (0, 1):
    ax[2, c].set_xscale("log")
    ax[2, c].xaxis.set_minor_formatter(plt.NullFormatter())
for c in (2, 3):
    ax[2, c].axis("off")
for a in ax[1]:
    a.set_xlabel(r"$R$ [arcsec]")
ax[2, 0].set_xlabel(r"$R$ [arcsec]")
ax[2, 1].set_xlabel(r"$R$ [arcsec]")
ax[0, 0].legend()
fig.savefig(OUT, dpi=150)
print("saved", OUT)
