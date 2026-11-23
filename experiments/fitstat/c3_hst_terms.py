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

usage:
  dump  (laptop, cwd omegaCen/dynamite_dataprep; notebook cells + cache only, nothing fitted or integrated):
        python c3_hst_terms.py dump OUT.pkl [relative]
  build (node: design matrices, optionally sharpened as veldist's writer does):
        python c3_hst_terms.py build IN.pkl OUT.npz [sharpen] [stretch=S]
stretch=S (docs/ml-reuse-investigation.md): rows for an orbit library reused at s = sqrt(ml/ml_orblib), on the
library's native cells: each star's pm / S, cov / S^2 (= its kernel over the stretched cells S*e); the star set is
the s = 1 set (frozen); h0 regridded to the stretched cells (spline CDF, clipped, 1% uniform). Add 2 N d ln S
(d = 2) to compare the absolute -2lnL across S.
relative: DS_PM_MODE=relative, cache hst_relative_veldist_solvers_cache.pkl (same bins and grid as addback).
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
from veldist.fitstat import sharpen_matrix  # noqa: E402
from veldist.veldist2d import precompute_design_matrix_2d  # noqa: E402

NB = "hst_veldist_refit_20261105.ipynb"
K = 23


def dump(out, relative):
    cache_file = "hst_relative_veldist_solvers_cache.pkl" if relative else "hst_veldist_solvers_cache.pkl"
    sys.path.insert(0, os.getcwd())  # the notebook imports local modules (centre, ...)
    if relative:
        os.environ["DS_PM_MODE"] = "relative"
    else:
        os.environ.pop("DS_PM_MODE", None)  # addback, as v4
    cells = json.load(open(NB))["cells"]
    ns = {}
    def _code(i):  # noqa: E306
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

    cache = pickle.load(open(cache_file, "rb"))
    fp = ns["_data_fingerprint"](bdl)
    assert fp == cache["key"]["data_fingerprint"], (fp, cache["key"]["data_fingerprint"])
    print(f"fingerprint ok: {fp[:12]}  bins {len(bdl)}")

    assert ns["PREFIX"] == ("hst_relative" if relative else "hst")
    solved, h0s = [], []
    for sol, st in zip(cache["solvers"], bdl):
        if sol is None:
            continue
        draws = _cell_mass_correction(np.asarray(sol.samples["intrinsic_pdf"], np.float64), sol.grid["shape"])
        mean = draws.mean(0)
        h0 = (0.99 * mean / mean.sum() + 0.01 / (K * K)).reshape(K, K)[:, ::-1]  # x-East flip of pm2
        h0s.append(h0 / h0.sum())
        solved.append({**st, "grid": sol.grid})
    pickle.dump({"bins": solved, "h0": np.asarray(h0s), "relative": relative}, open(out, "wb"))
    print(f"dumped {len(solved)} apertures to {out}")


def _stretch_h0(h0, grid, s):
    """h0 (n_ap, K, K; pm2 flipped) as masses on the native cells of the data divided by s, i.e. on s * edges."""
    sys.path.insert(0, "/Users/pesmith/research/dynamite-fitstat")
    sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith/veldist-fitstat/dynamite-6aa90a7")
    from dynamite.kinematics import cdf_regrid_matrix
    ex = np.asarray(grid["edges_x"], float)
    ey = -np.asarray(grid["edges_y"], float)[::-1]  # x-East flip of pm2
    Rx, Ry = cdf_regrid_matrix(ex, s * ex), cdf_regrid_matrix(ey, s * ey)
    h = np.clip(np.einsum("ij,ajk,lk->ail", Rx, h0, Ry), 0, None)
    h = 0.99 * h / h.sum(axis=(1, 2), keepdims=True) + 0.01 / (K * K)
    return h / h.sum(axis=(1, 2), keepdims=True)


def build(inp, out, sharpen, stretch=1.0):
    d = pickle.load(open(inp, "rb"))
    assert all(np.allclose(b["grid"]["edges_x"], d["bins"][0]["grid"]["edges_x"]) and
               np.allclose(b["grid"]["edges_y"], d["bins"][0]["grid"]["edges_y"]) for b in d["bins"])  # one grid
    S = sharpen_matrix(K) if sharpen else None
    indptr, cells_, masses, apert, nst = [0], [], [], [], []
    for out_i, st in enumerate(d["bins"]):
        cov = st["cov"].copy()
        cov[:, 0, 1] *= -1
        cov[:, 1, 0] *= -1
        M = np.asarray(precompute_design_matrix_2d(st["pm1"], -st["pm2"], cov, st["grid"]), dtype=np.float64)
        keep = M.sum(axis=1) >= 1e-3  # the s = 1 star set, frozen for every stretch
        if stretch != 1.0:
            M = np.asarray(precompute_design_matrix_2d(st["pm1"] / stretch, -st["pm2"] / stretch,
                                                       cov / stretch ** 2, st["grid"]), dtype=np.float64)
        M = M[keep]
        if S is not None:
            M = M @ S  # S commutes with the pm2 flip (symmetric, persymmetric stencil)
        for row in M:
            a = np.abs(row)
            nz = np.nonzero(a >= 1e-6 * a.max())[0]
            cells_.append(nz.astype(np.int32))
            masses.append(row[nz].astype(np.float32))
            indptr.append(indptr[-1] + len(nz))
        apert.append(np.full(len(M), out_i, dtype=np.int32))
        nst.append(len(M))
    np.savez(out, PM_star_indptr=np.asarray(indptr, np.int64), PM_star_cell=np.concatenate(cells_),
             PM_star_mass=np.concatenate(masses), PM_star_aperture=np.concatenate(apert),
             PM_lap_h0=d["h0"] if stretch == 1.0 else _stretch_h0(d["h0"], d["bins"][0]["grid"], stretch),
             n_star_used=np.asarray(nst), PM_star_sharpened=np.asarray(bool(sharpen)),
             stretch=np.asarray(stretch), jacobian=np.asarray(2 * sum(nst) * 2 * np.log(stretch)))
    print(f"wrote {out}: {len(d['bins'])} apertures, {indptr[-1]} nonzeros, {sum(nst)} stars, sharpened={sharpen}, "
          f"stretch={stretch} (jacobian 2 N d ln s = {2 * sum(nst) * 2 * np.log(stretch):.1f})")


if sys.argv[1] == "dump":
    dump(sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "relative")
else:
    st = [float(a.split("=")[1]) for a in sys.argv[4:] if a.startswith("stretch=")]
    build(sys.argv[2], sys.argv[3], "sharpen" in sys.argv[4:], st[0] if st else 1.0)
