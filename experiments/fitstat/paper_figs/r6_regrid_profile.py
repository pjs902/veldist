"""Paper figure r6: profile of the per-star statistic against M_IMBH and against the enclosed mass M(<5") and M(<20"),
from the full-covariance re-solve of the grid (regrid).

usage: python r6_regrid_profile.py ALL_MODELS.ecsv RESULTS_DIR ENCLOSED_MASS.npz OUT.pdf
Statistic: chi2_kin of regrid_one.py (per-star Laplace on MUSE, HST, Gaia + LVM chi2 at the re-solved weights).
Profile: GP surrogate over the 7 varied parameters (gp_profile.py). Against M_IMBH: min over the models' own values of
the other six of the GP mean. Against M(<r): M(<r) = M_IMBH + M_other(<r; theta), with M_other the stars + dark remnants
+ halo inside r (enclosed_mass.py) of each model's theta; at each M(<r) and each model's theta, M_IMBH = M(<r) -
M_other is put into the GP (kept only inside the sampled M_IMBH range) and the minimum taken. Band: GP sd at the
minimiser. Points: the models. LOO rms of the surrogate printed and shown.
"""
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from gp_profile import _unit, loo_rmse, gp_profile  # noqa: E402

PARS = ("m-bh", "mscale-sbh", "rscale-sbh", "q-stars", "p-stars", "u-stars", "ml")
t = Table.read(sys.argv[1])
d = [str(x).strip("/") for x in t["directory"]]
y = np.full(len(t), np.nan)
for f in glob.glob(f"{sys.argv[2]}/*.npz"):
    z = np.load(f)
    y[d.index(str(z["directory"]).strip("/"))] = float(z["chi2_kin"])
done = np.isfinite(y)
X = np.column_stack([np.log10(np.asarray(t[k], float)) if k == "m-bh" else np.asarray(t[k], float)
                     for k in PARS])[done]
y = y[done]
E = np.load(sys.argv[3])
assert [str(s).strip("/") for s in E["directory"]] == d
y0 = y.min()
rms = loo_rmse(X, y)[0]
print(f"{done.sum()} models; LOO rms {rms:.0f}")

xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 60)
prof, sd, gp = gp_profile(X, y, 0, xs)
lo, span = _unit(X)


def enclosed_profile(r_arcsec, grid):
    """min over models' theta of GP mean at M_IMBH = M(<r) - M_other(<r; theta)."""
    ri = E["r_arcsec"]
    other = sum(E[k] for k in ("M_stars", "M_sbh", "M_halo"))[done]
    m_other = np.array([np.exp(np.interp(np.log(r_arcsec), np.log(ri), np.log(o))) for o in other])
    m_tot = 10 ** X[:, 0] + m_other  # each model's own M(<r) (Plummer IMBH is all inside r >= 1")
    out, s_out = [], []
    for M in grid:
        lm = np.log10(np.clip(M - m_other, 1e-9, None))
        ok = (M > m_other) & (lm >= X[:, 0].min()) & (lm <= X[:, 0].max())
        if not ok.any():
            out.append(np.nan)
            s_out.append(np.nan)
            continue
        V = X[ok].copy()
        V[:, 0] = lm[ok]
        m, s = gp.predict((V - lo) / span, return_std=True)
        i = np.argmin(m)
        out.append(m[i] * y.std() + y.mean())
        s_out.append(s[i] * y.std())
    return m_tot, np.array(out), np.array(s_out)


w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(1, 3, figsize=(3 * w, 1.1 * h), sharey=True, constrained_layout=True)
panels = [(10 ** X[:, 0] / 1e3, 10 ** xs / 1e3, prof, sd, r"$M_{\rm IMBH}$ [$10^3\,M_\odot$]")]
for r in (5.0, 20.0):
    m_tot, _, _ = enclosed_profile(r, [])
    grid = np.geomspace(*np.percentile(m_tot, [3, 97]), 60)  # the extremes have one model each: no surrogate there
    _, p_r, s_r = enclosed_profile(r, grid)
    panels.append((m_tot / 1e3, grid / 1e3, p_r, s_r, rf"$M(<{r:.0f}'')$ [$10^3\,M_\odot$]"))
for a, (xm, xg, p, s, lab) in zip(ax, panels):
    a.scatter(xm, y - y0, s=10, color="0.55", lw=0, zorder=2)
    a.plot(xg, p - y0, color="#0072B2", lw=1.2)
    a.fill_between(xg, p - s - y0, p + s - y0, color="#0072B2", alpha=0.2, lw=0)
    a.set_xscale("log")
    a.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    a.set_xticks([], minor=True)
    a.set_xlabel(lab)
    j = np.nanargmin(p)
    print(lab, "profile minimum at %.1fk" % xg[j],
          "| Delta at grid:", " ".join(f"{xg[k]:.1f}k:{p[k] - y0:+.0f}" for k in range(0, len(xg), 10)))
ax[0].set_ylabel(r"$\Delta(-2\ln\mathcal{L})$")
ax[0].set_ylim(-0.1 * np.percentile(y - y0, 90), np.percentile(y - y0, 90))
ax[0].text(0.03, 0.95, f"{done.sum()} models\nsurrogate LOO rms {rms:.0f}", transform=ax[0].transAxes, va="top",
           fontsize="small")
fig.savefig(sys.argv[4])
fig.savefig(sys.argv[4].replace(".pdf", ".png"), dpi=200)
print("saved", sys.argv[4])
