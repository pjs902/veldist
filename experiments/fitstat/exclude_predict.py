"""Fit one model without some kinematic sets, then predict them (e.g. LOS only: the extragalactic case,
Kranjcevic / Hogg; or HST held out).

    python exclude_predict.py PRIV_CONFIG MODEL_DIRECTORY PROD.npz OUT.npz [RIDGE_LAMBDA] [EXCLUDE]

EXCLUDE: comma-separated set names left out of the fit (default 'hst_pm,gaia_pm', the LOS-only fit).

RIDGE_LAMBDA overrides the config's ridge_lambda (0.03): with only the LOS rows the ridge dominates many more
directions than in the joint fit, so the prediction has to be checked against it.

1. ``NNLS._solve_laplace`` with ``fit_exclude_sets`` = EXCLUDE: the weights are fit to the other sets plus the
   mass constraints. No files are written in the model.
2. One chi2 read with every set at those weights: the per-set statistic of the excluded sets is the prediction
   (the joint fit's own statistic), compared with PROD's joint-fit ``sets_new``.
Saved (key names from the LOS-only study): w_los = the fitted weights, sets_los = every set at them, the excluded
list, PROD's joint sets, timings.
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402

import ast  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import dynamite as dyn  # noqa: E402
from dynamite import weight_solvers as wsmod  # noqa: E402

cfg, mdir, prod_path, out_path = sys.argv[1:5]
ridge = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "none" else None
exclude = sys.argv[6].split(",") if len(sys.argv) > 6 else ["hst_pm", "gaia_pm"]
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1
model = c.all_models.get_model_from_row(rows[0])
assert "/veldist-fitstat/" in model.directory, model.directory  # private copy only
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset()
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
ws = wsmod.NNLS(config=c, model=model)
assert ws.gram_blockwise and ws.stream_reads
if ridge is not None:
    ws.ridge_lambda = ridge
print(f"ridge_lambda {ws.ridge_lambda}", flush=True)
prod = np.load(prod_path)
sets_joint = ast.literal_eval(str(prod["sets_new"]))
timing = {}

ws.settings["fit_exclude_sets"] = exclude
print(f"excluded from the fit: {exclude}", flush=True)
t0 = time.time()
w, *_ = ws._solve_laplace(orblib)
timing["solve"] = time.time() - t0
print(f"solve without {exclude} {timing['solve'] / 60:.1f} min", flush=True)

ws.settings["fit_exclude_sets"] = []
t0 = time.time()
gp = ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=w)
timing["read"] = time.time() - t0
mass_r2 = (gp.A_mass @ w - gp.b_mass) ** 2
sets, a0 = {}, ws.n_intrinsic
for i, k in enumerate(kin):
    sets[k.name] = float(gp.chi2_by_tag[f"set{i}/"] - mass_r2[a0:a0 + k.n_spatial_bins].sum())
    a0 += k.n_spatial_bins
sets["mass"] = float((gp.row0_vec @ w - gp.b0) ** 2 + mass_r2.sum())
print(f"at w_los: {sets}\njoint fit: {sets_joint}\nd (los - joint): "
      f"{ {k: sets[k] - sets_joint[k] for k in sets_joint if k in sets} }", flush=True)
np.savez(out_path, directory=mdir, w_los=w, w_joint=np.asarray(prod["w_new"], float), m_bh=prod["m_bh"],
         sets_los=str(sets), sets_joint=str(sets_joint), timing=str(timing), ridge_lambda=ws.ridge_lambda,
         excluded=",".join(exclude))
