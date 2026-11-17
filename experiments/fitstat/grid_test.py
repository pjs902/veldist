"""Grid test: grid bias of the per-star likelihood on the real production Gaia / HST bins.

For each production bin (stars and full error covariances from the production notebook) a 2D Gaussian (m1, m2, s1, s2) is fitted
  (a) to the stars directly, with no grid: star likelihood N(v_i; m, diag(s1^2, s2^2) + C_i);
  (b) as exact cell masses on a K x K grid over the production extent (+-56.46 km/s), by the exact per-star
      likelihood (stat_exact) with the plain design matrix M and with the sharpened M S, over the production extent.
bias = (b) - (a) per axis isolates the grid error: same stars, same family, so (a) is the grid-free answer.
Gaia: K = 17 (production, 6.64 km/s cells), 25, 33.  HST: K = 23 (production, 7.20 km/s), 31, 39.

usage:
  dump (cwd = omegaCen/dynamite_dataprep; data prep only, no fitting): python grid_test.py dump gaia|hst BINS.pkl
  fit:  python grid_test.py fit BINS.pkl OUT.npz [NPROC]
"""
import json
import os
import pickle
import sys
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from veldist.fitstat import sharpen_matrix, stat_exact  # noqa: E402
from veldist.veldist2d import KinematicSolver2D, precompute_design_matrix_2d  # noqa: E402

SETS = {  # notebook, last cell, cut that cell before this text (None: whole cell), PREFIX, half-width, K list
    "gaia": ("gaia_veldist_refit_20261031.ipynb", 38, None, "gaia_v2_noscale", 112.91314746000282 / 2, (17, 25, 33)),
    "hst": ("hst_veldist_refit_20261031.ipynb", 18, "cache_path = Path", "hst", 165.61277588006868 / 2, (23, 31, 39)),
}
def bins_from_notebook(nb, upto, cut, prefix, K0):
    sys.path.insert(0, os.getcwd())
    for k in ("DS_PM_MODE", "DS_GAIA_V2", "DS_GAIA_ESCALE"):
        os.environ.pop(k, None)
    ns = {}
    for i, c in enumerate(json.load(open(nb))["cells"][: upto + 1]):
        if c["cell_type"] == "code":
            src = "\n".join(ln for ln in "".join(c["source"]).splitlines() if not ln.lstrip().startswith(("%", "!")))
            if i == upto and cut:
                src = src[: src.index(cut)]
            exec(compile(src, f"{nb}:cell{i}", "exec"), ns)
    assert ns["PREFIX"] == prefix and ns["grid_kwargs"]["n_bins"] == K0
    return [bd for bd in ns["bin_data_list"] if bd is not None]


def gmass(e, p):
    m1, m2, s1, s2 = p
    h = np.outer(np.diff(ndtr((e - m1) / s1)), np.diff(ndtr((e - m2) / s2))).ravel()
    return h / h.sum()


def fit_one(bd, HALF, KS):
    v, C = np.c_[bd["pm1"], bd["pm2"]], bd["cov"]

    def nll(p):  # grid-free Gaussian, full per-star covariance
        a, d, b = C[:, 0, 0] + p[2] ** 2, C[:, 1, 1] + p[3] ** 2, C[:, 0, 1]
        det = a * d - b * b
        r1, r2 = v[:, 0] - p[0], v[:, 1] - p[1]
        return np.sum(0.5 * (d * r1 ** 2 - 2 * b * r1 * r2 + a * r2 ** 2) / det + 0.5 * np.log(det))

    p0 = [*v.mean(0), *np.sqrt(np.maximum(v.var(0) - C[:, [0, 1], [0, 1]].mean(0), 4.0))]
    ref = minimize(nll, p0, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-8, maxiter=4000)).x
    out = {"ref": ref, "n": len(v), "err": np.sqrt(np.median(C[:, [0, 1], [0, 1]], 0))}
    for K in KS:
        sol = KinematicSolver2D()
        sol.setup_grid(center=(0.0, 0.0), width=(2 * HALF,) * 2, n_bins=K)
        e = np.asarray(sol.grid["edges_x"], float)
        M = np.asarray(precompute_design_matrix_2d(v[:, 0], v[:, 1], C, sol.grid), float)
        for key, MM in (("plain", M), ("sharp", M @ sharpen_matrix(K))):
            r = minimize(lambda p: stat_exact(gmass(e, p), MM, len(MM), 1.0), ref, method="Nelder-Mead",
                         options=dict(xatol=1e-5, fatol=1e-8, maxiter=4000))
            out[f"{key}{K}"] = r.x
    return out


if __name__ == "__main__" and sys.argv[1] == "dump":
    nb, upto, cut, prefix, half, ks = SETS[sys.argv[2]]
    pickle.dump({"bins": bins_from_notebook(nb, upto, cut, prefix, ks[0]), "half": half, "K": ks, "set": sys.argv[2]},
                open(sys.argv[3], "wb"))
elif __name__ == "__main__":
    d = pickle.load(open(sys.argv[2], "rb"))
    HALF, KS, bins = d["half"], d["K"], d["bins"]
    sys.argv = sys.argv[2:]  # -> [BINS, OUT, NPROC]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        res = pool.starmap(fit_one, [(bd, HALF, KS) for bd in bins])
    save = {k: np.array([r[k] for r in res]) for k in res[0]}
    save["K"], save["half"] = np.array(KS), HALF
    np.savez(sys.argv[1], **save)
    ref = save["ref"]
    for K in KS:
        dd = 2 * HALF / K
        q = dd / ref[:, 2:]
        for key in ("plain", "sharp"):
            b = save[f"{key}{K}"][:, 2:] - ref[:, 2:]
            hi = q > 0.5
            print(f"K={K} Delta={dd:.2f} {key}: sigma bias median {np.median(b):+.3f}, "
                  f"Delta/sigma>0.5 ({hi.sum()} axes) {np.median(b[hi]) if hi.any() else np.nan:+.3f}, "
                  f"max |b| {np.abs(b).max():.3f} km/s; Delta/sigma {q.min():.2f}-{q.max():.2f}")
