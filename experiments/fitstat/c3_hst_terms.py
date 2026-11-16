"""C3 data side: per-star HST data for the v4 (addback) pack, in the x-East frame.

Re-runs the cells of omegaCen/dynamite_dataprep/hst_veldist_refit_20261105.ipynb
that build `cat` and the Voronoi bins (cells 2, 4, 8, 10, 13, 14, 16, and the
bin_data_list loop of 18) without fitting anything, checks the star lists
against the cached fit's data fingerprint, then writes for each fitted
aperture (npz order):
  - CSR star rows on the closed grid (stars with < 1e-3 on the grid dropped),
    after the x-West -> x-East flip DYNAMITE's pack applies (pm2 -> -pm2,
    cov12 -> -cov12, i.e. PM_2dhist[:, :, ::-1]);
  - h0 = 0.99 * mean + 0.01 * uniform of the cached 300 cell-mass-corrected
    draws, flipped the same way.

usage (from omegaCen/dynamite_dataprep, DS_PM_MODE unset = addback):
  python c3_hst_terms.py OUT.npz
"""
import ast
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from veldist.veldist import _cell_mass_correction  # noqa: E402
from veldist.veldist2d import precompute_design_matrix_2d  # noqa: E402

NB = "hst_veldist_refit_20261105.ipynb"
CACHE = "hst_veldist_solvers_cache.pkl"

sys.path.insert(0, os.getcwd())  # the notebook imports local modules (centre, ...)
os.environ.pop("DS_PM_MODE", None)  # addback, as v4
cells = json.load(open(NB))["cells"]
ns = {}
def _code(i):
    # drop IPython magics / shell lines
    return "\n".join(ln for ln in "".join(cells[i]["source"]).splitlines()
                     if not ln.lstrip().startswith(("%", "!")))


for i in (2, 4, 8, 10, 13, 14, 16):
    exec(compile(_code(i), f"cell{i}", "exec"), ns)
src18 = _code(18)
tree = ast.parse(src18)
keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_data_fingerprint"]
loop = src18[: src18.index("grid_kwargs = {")]
exec(compile(loop, "cell18-loop", "exec"), ns)
exec(compile(ast.Module(keep, []), "cell18-fp", "exec"), ns)
bdl = ns["bin_data_list"]

cache = pickle.load(open(CACHE, "rb"))
fp = ns["_data_fingerprint"](bdl)
assert fp == cache["key"]["data_fingerprint"], (fp, cache["key"]["data_fingerprint"])
print(f"fingerprint ok: {fp[:12]}  bins {len(bdl)}")

solvers = cache["solvers"]
K = 23
indptr, cells_, masses, apert, h0s, nst = [0], [], [], [], [], []
out_i = 0
for orig, (sol, st) in enumerate(zip(solvers, bdl)):
    if sol is None:
        continue
    cov = st["cov"].copy()
    cov[:, 0, 1] *= -1
    cov[:, 1, 0] *= -1
    M = np.asarray(precompute_design_matrix_2d(st["pm1"], -st["pm2"], cov, sol.grid), dtype=np.float64)
    M = M[M.sum(axis=1) >= 1e-3]
    for row in M:
        nz = np.nonzero(row >= 1e-6 * row.max())[0]
        cells_.append(nz.astype(np.int32))
        masses.append(row[nz].astype(np.float32))
        indptr.append(indptr[-1] + len(nz))
    apert.append(np.full(len(M), out_i, dtype=np.int32))
    nst.append(len(M))
    draws = _cell_mass_correction(np.asarray(sol.samples["intrinsic_pdf"], np.float64), sol.grid["shape"])
    mean = draws.mean(0)
    h0 = (0.99 * mean / mean.sum() + 0.01 / (K * K)).reshape(K, K)[:, ::-1]
    h0s.append(h0 / h0.sum())
    out_i += 1

np.savez(sys.argv[1], PM_star_indptr=np.asarray(indptr, np.int64), PM_star_cell=np.concatenate(cells_),
         PM_star_mass=np.concatenate(masses), PM_star_aperture=np.concatenate(apert),
         PM_lap_h0=np.asarray(h0s), n_star_used=np.asarray(nst))
print(f"wrote {sys.argv[1]}: {out_i} apertures, {indptr[-1]} nonzeros, {sum(nst)} stars")
