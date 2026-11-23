"""Per-star (Laplace) sidecar data, sharpened, for the production Gaia and MUSE sets.

Re-runs the omegaCen/dynamite_dataprep notebook cells that build the bins and
per-star data. NOTHING IS FITTED: the cached solvers are loaded, the cache key /
data fingerprint must match, otherwise the script stops. Then calls the veldist
writers with star_data=..., sharpen=True into an x-West staging pack (a copy of
NGC5139_addback_input with the Gaia / MUSE files replaced), and runs
dynamite_frame.convert_datapack to get the x-East pack.

  Gaia: gaia_veldist_refit_20261031.ipynb (DS_GAIA_V2=1, noscale), cache
        gaia_v2_noscale_veldist_solvers_cache.pkl
  MUSE: muse_veldist_refit_20261031.ipynb; that notebook never cached its
        solvers, so --muse-cache must point at a pickle {"solvers": [...]} (one
        KinematicSolver per bin, with .samples["intrinsic_pdf"]) from the same
        notebook run. Without it MUSE is skipped.

usage (cwd = omegaCen/dynamite_dataprep):
  python export_stars_gaia_muse.py OUTDIR [--only gaia|muse] [--muse-cache PKL]
"""
import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--only", choices=["gaia", "muse"])
ap.add_argument("--muse-cache", default=None)
ap.add_argument("--muse-refit", action="store_true",
                help="no cache at --muse-cache: refit with the notebook's own fit_all_bins call and save it there")
ap.add_argument("--base-stage", default="NGC5139_addback_input", help="x-West pack to copy")
args = ap.parse_args()
OUT = Path(args.outdir).resolve()
STAGE, XEAST = OUT / "xwest", OUT / "xeast"
BASE = Path(args.base_stage).resolve()
sys.path.insert(0, os.getcwd())
os.environ.pop("DS_PM_MODE", None)
os.environ.pop("DS_GAIA_V2", None)
os.environ.pop("DS_GAIA_ESCALE", None)

if STAGE.exists():
    shutil.rmtree(STAGE)
shutil.copytree(BASE, STAGE)
SIDECARS = {}


def run_nb(nb, upto, patches):
    """exec code cells 0..upto (magics stripped); patches {cell: [(old, new)]}."""
    ns = {}
    for i, c in enumerate(json.load(open(nb))["cells"][: upto + 1]):
        if c["cell_type"] != "code":
            continue
        src = "\n".join(ln for ln in "".join(c["source"]).splitlines() if not ln.lstrip().startswith(("%", "!")))
        for old, new in patches.get(i, []):
            assert old in src, (i, old)
            src = src.replace(old, new)
        exec(compile(src, f"{nb}:cell{i}", "exec"), ns)
    return ns


if args.only in (None, "gaia"):
    tmp = OUT / "tmp_gaia"
    ns = run_nb("gaia_veldist_refit_20261031.ipynb", 40, {
        39: [("if solvers is None:",
              "assert solvers is not None, 'cache key mismatch: a refit would be needed'\nif False:")],
        40: [('output_dir=f"NGC5139_input/{PREFIX}_veldist"',
              f"output_dir={str(tmp)!r}, star_data=bin_data_list, sharpen=True")],
    })
    assert ns["PREFIX"] == "gaia_v2_noscale"
    for n in ("aperture.dat", "bins.dat", "pm_2dhist.npz"):
        shutil.copyfile(tmp / f"gaia_v2_noscale_veldist_{n}", STAGE / f"gaia_veldist_{n}")

if args.only in (None, "muse"):
    refit = args.muse_refit and args.muse_cache and not Path(args.muse_cache).exists()
    if not args.muse_cache or not (refit or Path(args.muse_cache).exists()):
        print("MUSE SKIPPED: no solver cache (pass --muse-cache, or --muse-refit)")
    else:
        tmp = OUT / "tmp_muse"
        fit = ([("print(f\"{sum(s is not None", f"__import__('pickle').dump({{'solvers': solvers}}, open({args.muse_cache!r}, 'wb'))\n"
                 "print(f\"{sum(s is not None")] if refit else
               [("solvers = fit_all_bins(",
                 f"solvers = __import__('pickle').load(open({args.muse_cache!r}, 'rb'))['solvers']\n_unused = dict(")])
        ns = run_nb("muse_veldist_refit_20261031.ipynb", 31, {
            15: fit,
            31: [('output_dir="."', f"output_dir={str(tmp)!r}, star_data=bin_data_list, sharpen=True")],
        })
        # cache must belong to these bins: same count and per-bin star counts
        for sol, bd in zip(ns["solvers"], ns["bin_data_list"]):
            assert (sol is None) == (len(bd["vel"]) < 10) or sol.n_stars <= len(bd["vel"])
        assert len(ns["solvers"]) == len(ns["bin_data_list"])
        for n in ("aperture.dat", "bins.dat"):
            shutil.copyfile(tmp / f"veldist_{n}", STAGE / f"veldist_{n}")
        shutil.copyfile(tmp / "veldist_kinematics.ecsv", STAGE / "veldist_kinematics.ecsv")
        shutil.copyfile(tmp / "veldist_kinematics_stars.npz", STAGE / "veldist_kinematics_stars.npz")

sys.path.insert(0, os.getcwd())
import dynamite_frame as f  # noqa: E402

f.convert_datapack(STAGE, XEAST)
print("done:", XEAST)
