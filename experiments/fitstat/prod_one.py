"""Production-statistic re-solve of one v4 model: per-star Laplace on MUSE, HST and Gaia (LVM GH chi2).

    python prod_one.py PRIV_CONFIG MODEL_DIRECTORY OUT.npz

PRIV_CONFIG is the production config ($W/prod/config.yaml) pointed at a c3_private_copy.sh copy of the model
(the production pack's apertures and bins are byte-identical to v4's, so the v4 orbit library is reused). The solve
is DYNAMITE's own ``NNLS.solve`` as the production grid will run it. One rhs-only read before (at the v4 weights)
and after gives the per-set statistic (Laplace sets: offset-free Laplace about the data; LVM and mass: chi2).
Saved: both weight vectors, per-set statistics, chi2 as the solve reports it, timings.
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402

import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from astropy.io import ascii  # noqa: E402

import dynamite as dyn  # noqa: E402
from dynamite import weight_solvers as wsmod  # noqa: E402

cfg, mdir, out_path = sys.argv[1:4]
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1
row = c.all_models.table[rows[0]]
model = c.all_models.get_model_from_row(rows[0])
assert "/veldist-fitstat/" in model.directory, model.directory  # private copy only
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset()
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
stat = {k.name: getattr(k, "statistic", "diag") for k in kin}
assert all(stat[n] == "laplace" for n in ("veldist", "hst_pm", "gaia_pm")), stat
ws = wsmod.NNLS(config=c, model=model)
assert ws.gram_blockwise
timing = {}


def sets_at(w, tag):
    t0 = time.time()
    gp = ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=w)
    timing[f"capture_{tag}"] = time.time() - t0
    mass_r2 = (gp.A_mass @ w - gp.b_mass) ** 2
    sets, a0 = {}, ws.n_intrinsic
    for i, k in enumerate(kin):
        sets[k.name] = float(gp.chi2_by_tag[f"set{i}/"] - mass_r2[a0:a0 + k.n_spatial_bins].sum())
        a0 += k.n_spatial_bins
    sets["mass"] = float((gp.row0_vec @ w - gp.b0) ** 2 + mass_r2.sum())
    return sets


w_base = np.asarray(ascii.read(model.directory + "orbit_weights.ecsv")["weights"], dtype=np.float64)
sets_b = sets_at(w_base, "base")
print(f"v4 weights: {sets_b}", flush=True)
t0 = time.time()
res = ws.solve(orblib, ignore_existing_weights=True)
timing["solve"] = time.time() - t0
w_new, chi2_tot, chi2_kin = np.asarray(res[0], float), float(res[1]), float(res[2])
sets_n = sets_at(w_new, "new")
print(f"solve {timing['solve'] / 3600:.1f} h; re-solved: {sets_n}; chi2_kin {chi2_kin:.6g}", flush=True)
np.savez(out_path, directory=mdir, m_bh=row["m-bh"], kinchi2_v4=row["kinchi2"], w_base=w_base, w_new=w_new,
         sets_base=str(sets_b), sets_new=str(sets_n), chi2_tot=chi2_tot, chi2_kin=chi2_kin, timing=str(timing),
         statistic=str(stat))
