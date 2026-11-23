"""Stretched per-star sidecars for the production Gaia and MUSE sets (docs/ml-reuse-investigation.md).

Same path as export_stars_gaia_muse.py (re-runs the dataprep notebook cells, loads the cached solvers, NOTHING is
fitted, calls the production veldist writers with star_data=..., sharpen=True, then dynamite_frame.convert_datapack),
except that each solved bin's stars are
  1. cut to the stars the writer keeps at s = 1 (>= 1e-3 of the kernel on the grid): the star set is frozen;
  2. divided by s (velocities and errors; 2D: pm / s, cov / s^2).
After writing it checks that the writer kept every star of every bin (the frozen set) and stores stretch and
jacobian = 2 N d ln s. The writer's h0 is the s = 1 one; ml_stretch_pack.py makes the stretched expansion point. At s = 1 the output must equal the production sidecars bit for bit.

usage (cwd = omegaCen/dynamite_dataprep):
  python ml_stretch_export.py OUTDIR --stretch S [--only gaia|muse] --muse-cache PKL
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--stretch", type=float, required=True)
ap.add_argument("--only", choices=["gaia", "muse"])
ap.add_argument("--muse-cache", default=None)
ap.add_argument("--base-stage", default="NGC5139_addback_input")
args = ap.parse_args()
S = args.stretch
OUT = Path(args.outdir).resolve()
STAGE, XEAST = OUT / "xwest", OUT / "xeast"
sys.path.insert(0, os.getcwd())
for k in ("DS_PM_MODE", "DS_GAIA_V2", "DS_GAIA_ESCALE"):
    os.environ.pop(k, None)
if STAGE.exists():
    shutil.rmtree(STAGE)
shutil.copytree(Path(args.base_stage).resolve(), STAGE)
KEPT = {}  # set -> per solved bin, the number of stars kept at s = 1


def stretch_star_data(bdl, solvers, which):
    """bin_data_list for the writer: s = 1 star set, divided by S (see module docstring)."""
    from veldist.veldist import precompute_design_matrix
    from veldist.veldist2d import precompute_design_matrix_2d
    out, kept = [], []
    for sol, st in zip(solvers, bdl):
        if sol is None:
            out.append(st)
            continue
        if which == "gaia":
            M = np.asarray(precompute_design_matrix_2d(st["pm1"], st["pm2"], st["cov"], sol.grid), dtype=np.float64)
            k = M.sum(axis=1) >= 1e-3
            out.append({**st, "pm1": st["pm1"][k] / S, "pm2": st["pm2"][k] / S, "cov": st["cov"][k] / S ** 2})
        else:
            M = np.asarray(precompute_design_matrix(st["vel"], st["err"], sol.grid["centers"],
                                                    bin_width=sol.grid["width"]), dtype=np.float64)
            k = M.sum(axis=1) >= 1e-3
            out.append({**st, "vel": st["vel"][k] / S, "err": st["err"][k] / S})
        kept.append(int(k.sum()))
    KEPT[which] = kept
    return out


def run_nb(nb, upto, patches):
    ns = {"_STRETCH": stretch_star_data}
    for i, c in enumerate(json.load(open(nb))["cells"][: upto + 1]):
        if c["cell_type"] != "code":
            continue
        src = "\n".join(ln for ln in "".join(c["source"]).splitlines() if not ln.lstrip().startswith(("%", "!")))
        for old, new in patches.get(i, []):
            assert old in src, (i, old)
            src = src.replace(old, new)
        exec(compile(src, f"{nb}:cell{i}", "exec"), ns)
    return ns


def finish(npz, prefix, which, dim):
    z = dict(np.load(npz))
    ap_ = z[f"{prefix}_star_aperture"]
    n = np.bincount(ap_, minlength=len(KEPT[which]))
    assert n.tolist() == KEPT[which], f"{which}: the writer dropped stretched stars (frozen set broken)"
    N = int(n.sum())
    z["stretch"], z["jacobian"] = np.asarray(S), np.asarray(2 * N * dim * np.log(S))
    np.savez(npz, **z)
    print(f"{which}: {N} stars (frozen), stretch {S}, jacobian {2 * N * dim * np.log(S):.1f}")


if args.only in (None, "gaia"):
    tmp = OUT / "tmp_gaia"
    ns = run_nb("gaia_veldist_refit_20261031.ipynb", 40, {
        39: [("if solvers is None:",
              "assert solvers is not None, 'cache key mismatch: a refit would be needed'\nif False:")],
        40: [('output_dir=f"NGC5139_input/{PREFIX}_veldist"',
              f"output_dir={str(tmp)!r}, star_data=_STRETCH(bin_data_list, solvers, 'gaia'), sharpen=True")],
    })
    assert ns["PREFIX"] == "gaia_v2_noscale"
    finish(tmp / "gaia_v2_noscale_veldist_pm_2dhist.npz", "PM", "gaia", 2)
    for n in ("aperture.dat", "bins.dat", "pm_2dhist.npz"):
        shutil.copyfile(tmp / f"gaia_v2_noscale_veldist_{n}", STAGE / f"gaia_veldist_{n}")

if args.only in (None, "muse"):
    assert args.muse_cache and Path(args.muse_cache).exists(), "--muse-cache PKL needed"
    tmp = OUT / "tmp_muse"
    ns = run_nb("muse_veldist_refit_20261031.ipynb", 31, {
        15: [("solvers = fit_all_bins(",
              f"solvers = __import__('pickle').load(open({args.muse_cache!r}, 'rb'))['solvers']\n_unused = dict(")],
        31: [('output_dir="."',
              f"output_dir={str(tmp)!r}, star_data=_STRETCH(bin_data_list, solvers, 'muse'), sharpen=True")],
    })
    assert len(ns["solvers"]) == len(ns["bin_data_list"])
    finish(tmp / "veldist_kinematics_stars.npz", "LOS", "muse", 1)
    for n in ("aperture.dat", "bins.dat"):
        shutil.copyfile(tmp / f"veldist_{n}", STAGE / f"veldist_{n}")
    shutil.copyfile(tmp / "veldist_kinematics.ecsv", STAGE / "veldist_kinematics.ecsv")
    shutil.copyfile(tmp / "veldist_kinematics_stars.npz", STAGE / "veldist_kinematics_stars.npz")

import dynamite_frame as f  # noqa: E402

f.convert_datapack(STAGE, XEAST)
print("done:", XEAST)
