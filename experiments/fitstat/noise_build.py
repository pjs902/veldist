"""Integrate the orbit library of one model of a noise-replica copy (noise_setup.sh). Node job.

    python noise_build.py CONFIG MODEL_DIRECTORY      (MODEL_DIRECTORY e.g. orblib_007_005/ml02.93)

Builds the library with the copy's orblib_settings (DYNAMITE's own get_orblib, orblib_chunks from the config); the
solve is then regrid_one.py on the same config and model.
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402

import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import dynamite as dyn  # noqa: E402

cfg, mdir = sys.argv[1:3]
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1, mdir
model = c.all_models.get_model_from_row(rows[0])
assert "/veldist-fitstat/noise/" in model.directory, model.directory  # private copies only
assert not any(Path(model.directory_noml, "datfil").glob("*.bz2")), "library exists already"
model.setup_directories()
t0 = time.time()
model.get_orblib()
print(f"{mdir}: library built in {(time.time() - t0) / 3600:.2f} h", flush=True)
