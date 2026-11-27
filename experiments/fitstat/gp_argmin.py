"""Nuisance values that minimise the r6 GP surrogate at fixed M_IMBH (the profile's argmin), so real libraries can be
built ON the profile rather than on one family's slice (e.g. 007_008 = 3.2e4 at the 1e4 family's nuisance values).

    python gp_argmin.py ALL_MODELS.ecsv RESULTS_DIR M [M ...]

Same inputs and statistic as paper_figs/r6_regrid_profile.py. For each M: the 3 best candidates of
min_i mean(log10 M, theta_i) over the models' own theta_i, with GP sd, the source model and its own Delta.
"""
import glob
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from gp_profile import _unit, gp_profile  # noqa: E402

PARS = ("m-bh", "mscale-sbh", "rscale-sbh", "q-stars", "p-stars", "u-stars", "ml")
t = Table.read(sys.argv[1])
d = [str(x).strip("/") for x in t["directory"]]
y = np.full(len(t), np.nan)
for f in glob.glob(f"{sys.argv[2]}/*.npz"):
    z = np.load(f)
    y[d.index(str(z["directory"]).strip("/"))] = float(z["chi2_kin"])
gp_rows = ~np.isfinite(y) & np.asarray(t["all_done"], bool)
y[gp_rows] = np.asarray(t["kinchi2"], float)[gp_rows]
done = np.isfinite(y)
X = np.column_stack([np.log10(np.asarray(t[k], float)) if k == "m-bh" else np.asarray(t[k], float)
                     for k in PARS])[done]
y, dirs = y[done], np.array(d)[done]
y0 = y.min()
masses = [float(m) for m in sys.argv[3:]]
prof, sd, gp = gp_profile(X, y, 0, np.log10(masses))
lo, span = _unit(X)
ym, ys = y.mean(), y.std()
print(f"{done.sum()} models; best {dirs[np.argmin(y)]} {y0:.1f}")
for M, p, s in zip(masses, prof, sd):
    V = X.copy()
    V[:, 0] = np.log10(M)
    m, sv = gp.predict((V - lo) / span, return_std=True)
    m, sv = m * ys + ym, sv * ys
    assert np.isclose(m.min(), p)
    print(f"M {M:.3g}: profile {p - y0:+.0f} +- {s:.0f}")
    for i in np.argsort(m)[:3]:
        th = " ".join(f"{k} {X[i, j]:.4g}" for j, k in enumerate(PARS) if j)
        print(f"   {m[i] - y0:+7.0f} +- {sv[i]:.0f}  theta of {dirs[i]} (M {10 ** X[i, 0]:.3g}, own {y[i] - y0:+.0f}): {th}")
