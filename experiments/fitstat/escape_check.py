"""Escape speed of the best models at the fast stars (Haeberle+24 A-G; optional table with vlos for 3D speeds).

    python escape_check.py CONFIG [FAST_STARS.ecsv] [N_BEST]      (on the node, in the config's directory)

v_esc of each model's full potential at the star's projected radius (z = 0, the deepest point on its line of sight)
against its speed (v2D, or 3D where vlos is given): a star with speed - 2 err > v_esc there is unbound in that model
whatever its z. Prints per model kinchi2, M_IMBH, v_esc at each star and the strict-mode chi2 (escape.bound_chi2).
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402

import logging  # noqa: E402

import numpy as np  # noqa: E402

import dynamite as dyn  # noqa: E402
from dynamite_analysis import escape  # noqa: E402
from dynamite_analysis.escape_chi2 import _model_with_physical_masses  # noqa: E402

cfg = sys.argv[1]
stars = escape.load_fast_stars(sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".ecsv") else None)
n_best = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 12
logging.disable(logging.WARNING)
c = dyn.config_reader.Configuration(cfg, reset_logging=False, user_logfile=False, reset_existing_output=False)
t = c.all_models.table
ok = np.flatnonzero(np.asarray(t["all_done"], bool) & np.isfinite(np.asarray(t["kinchi2"], float)))
rows = ok[np.argsort(np.asarray(t["kinchi2"], float)[ok])][:n_best]
v, e = escape.fast_star_speeds(stars)
lab = stars.get("label", [str(i) for i in range(len(v))])
print("stars  ", " ".join(f"{s:>7}" for s in lab))
print("R [\"]  ", " ".join(f"{r:7.3f}" for r in stars["r_arcsec"]))
print("speed  ", " ".join(f"{a:7.1f}" for a in v))
for i in rows:
    m = _model_with_physical_masses(c, int(i))
    ve = escape.v_esc_at(m, np.asarray(stars["r_arcsec"], float))
    chi = escape.bound_chi2(ve, v, e)
    print(f"{float(t['kinchi2'][i]):8.1f} M {float(t['m-bh'][i]):6.0f} v_esc", " ".join(f"{a:7.1f}" for a in ve),
          f"| strict chi2 {chi.sum():7.1f} unbound(>2sig) {int(((v - 2 * e) > ve).sum())}  {t['directory'][i]}")
