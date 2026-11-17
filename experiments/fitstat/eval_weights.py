"""Per-set statistics of one model at given orbit weights (one rhs-only read, no solve).

    python eval_weights.py PRIV_CONFIG MODEL_DIRECTORY WEIGHTS.npz KEY OUT.npz [HST_TERMS.npz]

Puts weight vectors from different solves (E1 addback, relative arm B/C, production) on one footing: the
statistics of PRIV_CONFIG. With HST_TERMS the HST set is switched to statistic 'laplace' in memory with those
star terms (the E1 footing: HST per-star, Gaia/MUSE diag); with the production config all three are per-star.
Per set: Laplace sets report the offset-free Laplace statistic about the data, others chi2; 'mass' is the
mass-constraint term.
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402

import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import dynamite as dyn  # noqa: E402
from dynamite import weight_solvers as wsmod  # noqa: E402

cfg, mdir, wfile, key, out_path = sys.argv[1:6]
terms = sys.argv[6] if len(sys.argv) > 6 else None
w = np.asarray(np.load(wfile, allow_pickle=True)[key], dtype=np.float64)
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1
model = c.all_models.get_model_from_row(rows[0])
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset()
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
if terms:
    hst = [k for k in kin if k.name == "hst_pm"][0]
    T = np.load(terms)
    assert bool(T["PM_star_sharpened"])
    for k in hst.LAPLACE_KEYS:
        hst.data[k] = T[k]
    hst.statistic = "laplace"
    hst._init_laplace()
stat = {k.name: getattr(k, "statistic", "diag") for k in kin}
ws = wsmod.NNLS(config=c, model=model)
gp = ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=w)
mass_r2 = (gp.A_mass @ w - gp.b_mass) ** 2
sets, a0 = {}, ws.n_intrinsic
for i, k in enumerate(kin):
    sets[k.name] = float(gp.chi2_by_tag[f"set{i}/"] - mass_r2[a0:a0 + k.n_spatial_bins].sum())
    a0 += k.n_spatial_bins
sets["mass"] = float((gp.row0_vec @ w - gp.b0) ** 2 + mass_r2.sum())
print(f"{mdir} {wfile}[{key}] under {stat}: {sets}; kin sum {sum(v for k, v in sets.items() if k != 'mass'):.1f}",
      flush=True)
np.savez(out_path, directory=mdir, weights_from=f"{wfile}[{key}]", statistic=str(stat), sets=str(sets))
