"""Relative-PM pilot: one v4 model re-solved with the HST set in relative mode + per-star (Laplace) statistic.

    python rel_pilot.py PRIV_CONFIG MODEL_DIRECTORY OUT.npz

PRIV_CONFIG is a private copy (c3_private_copy.sh, then rel_config.sh) whose hst_pm entry is relative: True,
statistic: laplace, optionally with rotation_rows, and pm_recentre_mode: vp. The solve is DYNAMITE's own
(NNLS.solve -> _solve_relative -> _solve_relative_vp_laplace) on the private copy, so nothing is written into the
live grid. Saved: weights, chi2 per set as DYNAMITE reports them, the vp history (chi2 per round, centres), the
stop reason, and wall time per phase from the log timestamps. Compare with arm A (e1_solve.py, addback) at the
same model.
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
assert len(rows) == 1
model = c.all_models.get_model_from_row(rows[0])
assert "/veldist-fitstat/" in model.directory, model.directory  # private copy only
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset()
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
hst = [k for k in kin if k.name == "hst_pm"][0]
assert hst.relative and hst.statistic == "laplace"
ws = wsmod.NNLS(config=c, model=model)
assert ws.settings.get("pm_recentre_mode") == "vp"
t0 = time.time()
res = ws.solve(orblib, ignore_existing_weights=True)
weights, chi2_tot, chi2_kin = res[0], float(res[1]), float(res[2])
wall = time.time() - t0
info = {f"pm_{k}": np.asarray(v, dtype=object) for k, v in getattr(ws, "pm_recentre_info", {}).items()}
np.savez(out_path, weights=np.asarray(weights), chi2_tot=chi2_tot, chi2_kin=chi2_kin,
         wall_s=wall, rows=hst.rotation_rows is not None, model=mdir, **info)
print(f"done {mdir}: chi2_kin {chi2_kin:.6g}, wall {wall / 3600:.1f} h, info keys {sorted(info)}")
