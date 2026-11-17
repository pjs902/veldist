"""Paper figure d3: radial profiles of the binned kinematics -- dispersions, rotation, anisotropy.

usage: python d3_profiles.py PACK_DIR OUT.pdf
Annuli are log-spaced per dataset (>= 6 bins each); a point is the mean over its bins with the error of
the mean. Rotation: LOS amplitude A from v = A cos(theta - theta_rec) + c in each annulus, theta_rec the
receding major axis (PA_KIN); proper motions: the annulus mean of mu_T (positive along the rotation).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from d2_maps import PA_KIN, load

C = {"MUSE": "#0072B2", "LVM": "#009E73", "HST": "#D55E00", "Gaia": "#E69F00"}


def annuli(R, nper=None, edges=None):
    if edges is None:
        edges = np.unique(np.quantile(R, np.linspace(0, 1, max(2, len(R) // nper + 1))))
    idx = np.clip(np.digitize(R, edges) - 1, 0, len(edges) - 2)
    return [np.flatnonzero(idx == k) for k in range(len(edges) - 1) if (idx == k).sum() >= 6]


def mean_err(R, val, groups):
    r = np.array([np.median(R[g]) for g in groups])
    m = np.array([val[g].mean() for g in groups])
    e = np.array([val[g].std(ddof=1) / np.sqrt(len(g)) for g in groups])
    return r, m, e


def amplitude(x, y, v, groups):
    th_rec = np.radians(90 - PA_KIN)  # x-East frame angle of the receding major axis
    out = []
    for g in groups:
        A = np.c_[np.cos(np.arctan2(y[g], x[g]) - th_rec), np.ones(len(g))]
        coef, res, *_ = np.linalg.lstsq(A, v[g], rcond=None)
        s2 = np.sum((v[g] - A @ coef) ** 2) / (len(g) - 2)
        cov = s2 * np.linalg.inv(A.T @ A)
        out.append((np.median(np.hypot(x[g], y[g])), coef[0], np.sqrt(cov[0, 0])))
    return np.array(out).T


def main():
    hst, gaia, muse, lvm = load(Path(sys.argv[1]))
    for d in (muse, lvm):
        d["R"] = np.hypot(d["x"], d["y"])
    w, h = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(1, 3, figsize=(3 * w, h), constrained_layout=True, sharex=True)
    gm, gl = annuli(muse["R"], 14), annuli(lvm["R"], 15)
    gh, gg = annuli(hst["R"], 70), annuli(gaia["R"], 15)
    prof = {}
    for lab, d, g, key, mk in (("MUSE", muse, gm, "s", "o"), ("HST", hst, gh, "st", "s"), ("Gaia", gaia, gg, "st", "s"),
                               ("HST", hst, gh, "sr", "^"), ("Gaia", gaia, gg, "sr", "^")):
        r, m, e = mean_err(d["R"], d[key], g)
        prof[(lab, key)] = (r, m, e)
        name = {"s": r"$\sigma_{\rm LOS}$", "st": r"$\sigma_T$", "sr": r"$\sigma_R$"}[key]
        ax[0].errorbar(r, m, e, color=C[lab], marker=mk, ls="-" if key != "sr" else "--", ms=4,
                       mfc="w" if key == "sr" else C[lab], label=f"{name}, {lab}" if lab != "Gaia" or key != "s" else None)
    ax[0].set_ylabel(r"dispersion [km s$^{-1}$]")
    ax[0].legend(frameon=False, loc="lower left", ncol=1)
    for lab, d, g in (("MUSE", muse, gm), ("LVM", lvm, gl)):
        r, a, e = amplitude(d["x"], d["y"], d["v"], g)
        prof[(lab, "A")] = (r, a, e)
        ax[1].errorbar(r, a, e, color=C[lab], marker="o", ms=4, label=rf"$A_{{\rm LOS}}$, {lab}")
    for lab, d, g in (("HST", hst, gh), ("Gaia", gaia, gg)):
        r, m, e = mean_err(d["R"], d["mt"], g)
        prof[(lab, "mt")] = (r, m, e)
        ax[1].errorbar(r, m, e, color=C[lab], marker="s", ms=4, label=rf"$\langle\mu_T\rangle$, {lab}")
    ax[1].axhline(0, color="0.6", lw=0.5)
    ax[1].set_ylabel(r"rotation [km s$^{-1}$]")
    ax[1].set_ylim(0, 12)
    ax[1].legend(frameon=False, loc="lower right", ncol=2, bbox_to_anchor=(0.97, 0.0))
    for lab in ("HST", "Gaia"):
        r, st, et = prof[(lab, "st")]
        _, sr, er = prof[(lab, "sr")]
        q = st / sr
        ax[2].errorbar(r, q, q * np.hypot(et / st, er / sr), color=C[lab], marker="s", ms=4,
                       label=rf"$\sigma_T/\sigma_R$, {lab}")
    r, sl, el = prof[("MUSE", "s")]
    rt, st, et = prof[("HST", "st")]
    sti = np.interp(np.log(r), np.log(rt), st)
    q = sl / sti
    ax[2].errorbar(r, q, q * el / sl, color=C["MUSE"], marker="o", ms=4,
                   label=r"$\sigma_{\rm LOS}/\sigma_T$, MUSE/HST")
    print("sLOS/sT", np.round(q, 3), "at", np.round(r))
    ax[2].axhline(1, color="0.6", lw=0.5)
    ax[2].set_ylabel("dispersion ratio")
    ax[2].set_ylim(0.79, 1.08)
    ax[2].legend(frameon=False, loc="lower left")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("projected radius [arcsec]")
        a.axvline(287, color="0.5", ls=":", lw=0.8)
    fig.savefig(sys.argv[2])
    fig.savefig(sys.argv[2].replace(".pdf", ".png"), dpi=200)
    for k, (r, m, e) in prof.items():
        print(k, " ".join(f"{a:.0f}:{b:.2f}+-{c:.2f}" for a, b, c in zip(r, m, e)))


if __name__ == "__main__":
    main()
