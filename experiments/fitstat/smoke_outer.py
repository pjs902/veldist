"""Smoke test: one model, tiny orbit library, production config + the outer relative PM sets (rows), end to end.

    python smoke_outer.py SMOKE_CONFIG V4_ALL_MODELS.ecsv MODEL_DIRECTORY

Parameters are copied from MODEL_DIRECTORY's row of the v4 all_models table; the orbit library is built from scratch
in the smoke output directory with the config's (tiny) orbit numbers, so the outer apertures are projected. Then the
production weight solve. Checks only that everything reads, projects and solves; the numbers are not science.
Prints phase timings, per-set statistic at the solution, and the rotation-row residuals of the outer sets.
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402

import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402

import dynamite as dyn  # noqa: E402

cfg, v4tab, mdir = sys.argv[1:4]
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
print("sets:", [(k.name, type(k).__name__, getattr(k, "statistic", "diag"), getattr(k, "relative", False),
                 k.n_spatial_bins, getattr(k, "rotation_rows", None) is not None) for k in kin], flush=True)
t4 = Table.read(v4tab)
row = t4[[str(d).rstrip("/") == mdir.rstrip("/") for d in t4["directory"]]][0]
parset = c.parspace.get_parset()
for n in parset.colnames:
    if n in row.colnames:
        parset[n] = row[n]
print("parset:", dict(zip(parset.colnames, parset)), flush=True)
model = dyn.model.Model(config=c, parset=parset)
model.setup_directories()
# register the model in all_models, as the parameter-space iterator does (later steps look the parset up there)
T = c.all_models.table
newrow = []
for col in T.colnames:
    if col in parset.colnames:
        newrow.append(parset[col])
    elif col == "time_modified":
        newrow.append(str(np.datetime64("now", "s")))
    elif col == "which_iter":
        newrow.append(0)
    elif col == "directory":
        newrow.append(os.path.relpath(model.directory, c.settings.io_settings["output_directory"] + "models") + "/")
    else:
        newrow.append(T[col].dtype.type(None))
T.add_row(newrow)
print("registered", T["directory"][-1], flush=True)
t0 = time.time()
orblib = model.get_orblib()
t1 = time.time()
print(f"orbit library: {t1 - t0:.0f} s, {orblib.n_orbs if hasattr(orblib, 'n_orbs') else '?'} orbits", flush=True)
ws = model.get_weights(orblib)
t2 = time.time()
w = np.asarray(model.weights, float)
print(f"weight solve: {t2 - t1:.0f} s; kinchi2 {model.kinchi2:.6g}, chi2 {model.chi2:.6g}; "
      f"nonzero weights {np.sum(w > 0)}/{len(w)}", flush=True)
info = getattr(ws, "pm_recentre_info", None)
if info:
    print("relative-PM info:", {k: info[k] for k in info if not hasattr(info[k], "__len__") or len(info[k]) < 10})
print("SMOKE OK", flush=True)
