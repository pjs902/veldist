"""Full per-star (Laplace) re-solve of one v4 model through DYNAMITE's own ``NNLS.solve``.

    python regrid_one.py PRIV_CONFIG MODEL_DIRECTORY OUT.npz

PRIV_CONFIG: $W/regrid/config.yaml (regrid_setup.sh): the production statistic (per-star Laplace on MUSE, HST, Gaia;
LVM chi2) with the accepted speed settings (row-space kappa, working set), pointed at a private copy of the v4 grid
whose 74 models (50 finished + 24 profile-fill of iteration 005) reuse the v4 orbit libraries.
``solve`` writes orbit_weights.ecsv into the private copy; the per-set statistics it does not keep are taken from
``_solve_laplace``'s return. Saved: weights, per-set statistic (Laplace sets offset-free about the data; LVM and
mass chi2), chi2_tot / chi2_kin, the model's parameters, row-space bound E, working-set rounds, timing.
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402

import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import dynamite as dyn  # noqa: E402
from dynamite import weight_solvers as wsmod  # noqa: E402

cfg, mdir, out_path = sys.argv[1:4]
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1, mdir
row = c.all_models.table[rows[0]]
model = c.all_models.get_model_from_row(rows[0])
assert "/veldist-fitstat/regrid/" in model.directory or "/veldist-fitstat/stretch/" in model.directory, \
    model.directory  # private copies only (regrid_setup.sh, stretch_setup.sh)
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset()
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
stat = {k.name: getattr(k, "statistic", "diag") for k in kin}
assert all(stat[n] == "laplace" for n in ("veldist", "hst_pm", "gaia_pm")), stat
ws = wsmod.NNLS(config=c, model=model)
assert ws.gram_blockwise and ws.stream_reads
s = ws.settings
assert s.get("pm_laplace_rowspace_kappa") is not None and s.get("pm_laplace_working_set") is not None, s

kept = {}
_solve_laplace = ws._solve_laplace


def keep(lib):
    kept["r"] = _solve_laplace(lib)
    return kept["r"]


ws._solve_laplace = keep
t0 = time.time()
w, chi2_tot, chi2_kin = ws.solve(orblib, ignore_existing_weights=True)[:3]
t_solve = time.time() - t0
assert "r" in kept, "solve did not run the Laplace driver (or failed: see the log)"
sets = {nm: float(v) for nm, v, _ in kept["r"][3]}
info = ws.laplace_info
print(f"{mdir}: solve {t_solve / 3600:.2f} h; {sets}; E {getattr(ws, 'laplace_bound', None)}; "
      f"working set {info.get('working_set')}", flush=True)
np.savez(out_path, directory=mdir, params=str({k: float(row[k]) for k in row.colnames
                                                 if np.issubdtype(np.asarray(row[k]).dtype, np.number)}),
         m_bh=float(row["m-bh"]), kinchi2_v4=float(row["kinchi2"]), w_new=np.asarray(w, float),
         sets_new=str(sets), chi2_tot=float(chi2_tot), chi2_kin=float(chi2_kin),
         E=str(dict(getattr(ws, "laplace_bound", None) or {})), working_set=str(info.get("working_set")),
         statistic=str(stat), t_solve=t_solve)
