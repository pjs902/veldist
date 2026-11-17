"""Paper figure r6: profile of the fit statistic against IMBH mass, diagonal chi2 vs the per-star likelihood.

usage: python r6_imbh_profile_env.py ALL_MODELS.ecsv RESCORE_DIR OLD_PROFILE.npz OUT.pdf [--slides]   (--slides: 16:9)
Each statistic's profile is a smooth lower envelope of the computed models: at each mass a soft minimum of the
models' statistics with Gaussian weights in log M_IMBH (0.08 dex), so it follows the best models near each mass.
Grey points: all rescored models.
  diagonal  = kinchi2 (all datasets, the v4 fit statistic)
  per-star  = kinchi2 - sum(HST diagonal chi2) + sum(HST Laplace term), at each model's v4 weights
The fast-star lower limits on M_IMBH (proper motions + preliminary NIRSpec RVs) are read from the existing
kinematics+fast-star profile (OLD_PROFILE.npz, first mass where the constrained curve exists).
"""
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


def envelope(logm, y, tol=0.04):
    o = np.argsort(logm)
    lm, yy = logm[o], y[o]
    groups, cur = [], [0]
    for i in range(1, len(lm)):
        if lm[i] - lm[cur[0]] <= tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    best = [g[np.argmin(yy[g])] for g in groups]
    return lm[best], yy[best]


def softmin_curve(x, y, xs, width=0.08, frac=0.04):
    """Smooth lower envelope: at each xs, a soft minimum of the models' y with Gaussian weights in x (width in
    dex), temperature T = frac x (80th-percentile spread of y). Follows the best models near each mass, does not
    extrapolate beyond them, and tends to the plain minimum as T -> 0."""
    T = frac * (np.percentile(y, 80) - y.min())
    K = np.exp(-0.5 * ((xs[:, None] - x[None]) / width) ** 2)
    z = (y - y.min()) / T
    return y.min() - T * np.log((K * np.exp(-z)[None]).sum(1) / K.sum(1))


def main():
    t = Table.read(sys.argv[1])
    t = t[np.asarray(t["all_done"], bool)]
    resc = {}
    for f in glob.glob(f"{sys.argv[2]}/rescore_*.npz"):
        z = np.load(f, allow_pickle=True)
        resc[str(z["directory"]).strip("/")] = float(z["kinchi2"]) - z["diag"].sum() + z["laplace"].sum()
    d = np.array([str(x).strip("/") for x in t["directory"]])
    logm = np.log10(np.asarray(t["m-bh"], float))
    have = np.array([k in resc for k in d])
    stats = [(np.array([resc[k] for k in d[have]]), "per-star likelihood", "#0072B2")]
    print(f"{len(t)} finished models, {have.sum()} rescored (both panels use the rescored set)")
    old = np.load(sys.argv[3], allow_pickle=True)
    ok = np.isfinite(old["RV"]) & (old["RV"] - np.nanmin(old["RV"]) < 50)
    lim_rv = 10 ** old["log_m"][ok].min()

    w, h = plt.rcParams["figure.figsize"]
    fig, a0 = plt.subplots(figsize=(1.6 * h * 16 / 9, 1.6 * h) if "--slides" in sys.argv else (w, h), constrained_layout=True)
    ax = [a0]
    lm = logm[have]
    for a, (y, lab, c) in zip(ax, stats):
        ex, ey = envelope(lm, y)
        xs = np.linspace(lm.min(), lm.max(), 300)
        ys = softmin_curve(lm, y, xs)
        y0 = ys.min()
        a.plot(10 ** xs / 1e3, ys - y0, color=c)
        a.scatter(10 ** lm / 1e3, y - y0, s=10, color="0.45", lw=0, zorder=2)
        a.axvspan(1, lim_rv / 1e3, color="0.9", lw=0, zorder=0)
        a.set_ylim(-0.06 * np.percentile(y - y0, 80), np.percentile(y - y0, 80))
        a.set_ylabel(r"$\Delta(-2\ln\mathcal{L})$")
        print(lab, "curve min at %.1fk;" % (10 ** xs[np.argmin(ys)] / 1e3), "best model %.1fk" % (10 ** ex[np.argmin(ey)] / 1e3),
              "envelope:", " ".join(f"{10 ** x / 1e3:.1f}k:{v - y0:.0f}" for x, v in zip(ex, ey)))
    ax[0].set_xscale("log")
    ax[0].set_xlim(4.5, 65)
    ax[0].set_xticks([5, 10, 20, 50], ["5", "10", "20", "50"])
    ax[0].set_xticks([], minor=True)
    ax[0].set_xlabel(r"$M_{\rm IMBH}$ [$10^3\,M_\odot$]")
    ax[0].text(0.02, 0.95, "excluded by\nfast stars", transform=ax[0].transAxes, va="top", fontsize="small",
               color="0.4")
    fig.savefig(sys.argv[4])
    fig.savefig(sys.argv[4].replace(".pdf", ".png"), dpi=200)
    print("fast-star (PM + RV) lower limit: %.0f" % lim_rv)


if __name__ == "__main__":
    main()
