"""Exact per-star check of one stretched solve (docs/ml-reuse-investigation.md, node step 4).

    python ml_stretch_exact.py PRIV_CONFIG MODEL_DIRECTORY RESULT.npz OUT.npz

One rhs-only read at the solve's weights (RESULT.npz w_new, from regrid_one.py) captures the model histograms of
every Laplace set (the e1_solve.py pattern). Per set and aperture: the exact -2 log L (fitstat_pm.stat_exact, no
expansion, no h0) and the offset-free Laplace statistic about the data h0, whose sum must reproduce the solve's
reported value. Across s: exact(s) + jac(s) - exact(1) is independent of h0 and of the Laplace approximation.
Writes nothing into the model directory.
"""
import ast
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402

import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import dynamite as dyn  # noqa: E402
from dynamite import fitstat_pm as fs  # noqa: E402
from dynamite import weight_solvers as wsmod  # noqa: E402

cfg, mdir, res_path, out_path = sys.argv[1:5]
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1, mdir
model = c.all_models.get_model_from_row(rows[0])
assert "/veldist-fitstat/stretch/" in model.directory, model.directory
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset()
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
R = np.load(res_path)
assert str(R["directory"]).rstrip("/") == mdir.rstrip("/")
w = np.asarray(R["w_new"], np.float64)
solved = ast.literal_eval(str(R["sets_new"]))

ws = wsmod.NNLS(config=c, model=model)
t0 = time.time()
ws._lap_h0, ws._lap_capture = None, dict(w=w, h={})
ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=w)
H = ws._lap_capture["h"]
ws._lap_capture = None
t_read = time.time() - t0

out, ap0 = dict(directory=mdir, t_read=t_read), 0
for k in kin:
    n = k.n_spatial_bins
    if getattr(k, "statistic", "diag") != "laplace":
        ap0 += n
        continue
    Ma = np.asarray(ws.projected_masses[ap0:ap0 + n], float)
    ap0 += n
    h = H[k.name]
    lap, ex, nonpos, nst = (np.zeros(n) for _ in range(4))
    for a in range(n):
        Lt, g, h0 = k.laplace_terms(a, Ma[a])
        lap[a] = fs.stat_offset_free(h[a], Lt, g, h0)
        M = k.laplace_design(a)
        ex[a] = fs.stat_exact(h[a], M, len(M), Ma[a])
        nonpos[a], nst[a] = np.sum(M @ h[a] <= 0), len(M)
    # the solve's chi2 read is the same statistic at the same weights
    assert np.isclose(lap.sum(), solved[k.name], rtol=1e-6, atol=0.05), (k.name, lap.sum(), solved[k.name])
    print(f"{k.name}: laplace {lap.sum():.2f} (solve {solved[k.name]:.2f}); exact {ex.sum():.2f}; "
          f"stars {int(nst.sum())}; p<=0 {int(nonpos.sum())}", flush=True)
    out.update({f"{k.name}_h": h.astype(np.float32), f"{k.name}_Ma": Ma, f"{k.name}_lap": lap,
                f"{k.name}_exact": ex, f"{k.name}_nonpos": nonpos, f"{k.name}_nstars": nst})
np.savez(out_path, **out)
print(f"done ({t_read / 60:.0f} min read)", flush=True)
