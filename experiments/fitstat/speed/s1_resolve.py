"""Node acceptance run of the Laplace speedups A + B + D1 on one model (no files written in the model).

    python s1_resolve.py PRIV_CONFIG MODEL_DIRECTORY PROD.npz OUT.npz KAPPA FRAC     (KAPPA / FRAC: 'none' = off)

1. B on real data: one cell-space chi2 read at the production weights (PROD.npz ``w_new``); per set it must
   reproduce PROD's ``sets_new`` (statistic about the data h0). Timed.
2. The Laplace driver itself (``NNLS._solve_laplace``, not ``solve``, which would overwrite the model's
   orbit_weights.ecsv) with ``pm_laplace_rowspace_kappa`` = KAPPA and ``pm_laplace_working_set`` = FRAC.
   Its final read reports the per-set statistic at the new weights, the row-space bound E and the working-set
   rounds (m, KKT gap, violators). Timed; set DYNAMITE_GRAM_PROFILE for per-phase JSON.
Acceptance (docs/laplace-solve-speedup-analysis.md, section 6): per-set chi2 within 0.5 of PROD, E < 0.05, gap <= 0.01.
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

cfg, mdir, prod_path, out_path, kappa, frac = sys.argv[1:7]
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
if kappa != "none":
    ws.settings["pm_laplace_rowspace_kappa"] = float(kappa)
if frac != "none":
    ws.settings["pm_laplace_working_set"] = float(frac)
prod = np.load(prod_path)
sets_prod = ast.literal_eval(str(prod["sets_new"]))
timing = {}

t0 = time.time()
w_prod = np.asarray(prod["w_new"], float)
gp = ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=w_prod)
timing["read_prod"] = time.time() - t0
mass_r2 = (gp.A_mass @ w_prod - gp.b_mass) ** 2
sets_read, a0 = {}, ws.n_intrinsic
for i, k in enumerate(kin):
    sets_read[k.name] = float(gp.chi2_by_tag[f"set{i}/"] - mass_r2[a0:a0 + k.n_spatial_bins].sum())
    a0 += k.n_spatial_bins
E_prod = dict(getattr(ws, "laplace_bound", None) or {})
print(f"read at PROD w_new {timing['read_prod'] / 60:.1f} min: {sets_read}; PROD {sets_prod}; E {E_prod}", flush=True)

t0 = time.time()
w, chi2_tot, chi2_kin, chi2 = ws._solve_laplace(orblib)
timing["solve"] = time.time() - t0
sets_new = {nm: float(v) for nm, v, _ in chi2}
info = ws.laplace_info
E_new = dict(getattr(ws, "laplace_bound", None) or {})
print(f"solve {timing['solve'] / 3600:.2f} h: {sets_new}; d vs PROD "
      f"{ {k: sets_new[k] - sets_prod[k] for k in sets_prod if k in sets_new} }; E {E_new}; "
      f"working set {info.get('working_set')}", flush=True)
np.savez(out_path, directory=mdir, kappa=kappa, frac=frac, w_new=w, w_pass=np.array(info["w_pass"]),
         sets_new=str(sets_new), sets_prod=str(sets_prod), sets_read_prod=str(sets_read), E_prod=str(E_prod),
         E_new=str(E_new), working_set=str(info.get("working_set")), chi2_tot=chi2_tot, chi2_kin=chi2_kin,
         timing=str(timing))
