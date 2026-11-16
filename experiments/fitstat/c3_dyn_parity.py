"""B1 real-data check: DYNAMITE's statistic 'laplace' on a v4 model == C3's rescore.

    python c3_dyn_parity.py PRIV_CONFIG MODEL_DIRECTORY STAR_TERMS.npz RESCORE.npz

Loads the v4 private copy, switches the HST set to statistic 'laplace' in
memory (its data dict gets C3's per-star keys, then ``_init_laplace``, which
also runs the h0 frame guard), and does one rhs-only chi2 read at the model's
stored weights with the dynamite branch's ``construct_gram_and_rhs_blockwise``.
The HST set chi2 must equal C3's sum(laplace - laplace_at_h0) for that model
(veldist.fitstat, float64, rows built from the same terms): checks cell order,
frame, units (M_a scaling) and the float32-row error at 45k orbits.
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

cfg, mdir, terms_path, rescore_path = sys.argv[1:5]
print("dynamite from", dyn.__file__, flush=True)
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1
model = c.all_models.get_model_from_row(rows[0])
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset()
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
si = [k.name for k in kin].index("hst_pm")
hst = kin[si]
T = np.load(terms_path)
for k in hst.LAPLACE_KEYS:
    hst.data[k] = T[k]
hst.statistic = "laplace"
hst._init_laplace()  # includes the PM_lap_h0 vs PM_2dhist frame guard
ws = wsmod.NNLS(config=c, model=model)
w = np.asarray(ascii.read(model.directory + "orbit_weights.ecsv")["weights"], dtype=np.float64)
t0 = time.time()
gp = ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=w)
t_read = time.time() - t0
mass_r2 = (gp.A_mass @ w - gp.b_mass) ** 2
ap0 = ws.n_intrinsic + sum(k.n_spatial_bins for k in kin[:si])
lap_dyn = gp.chi2_by_tag[f"set{si}/"] - mass_r2[ap0:ap0 + hst.n_spatial_bins].sum()
R = np.load(rescore_path)
lap_c3 = float(np.sum(R["laplace"] - R["laplace_at_h0"]))
print(f"read {t_read:.0f} s; HST laplace: dynamite {lap_dyn:.3f}  C3 {lap_c3:.3f}  "
      f"diff {lap_dyn - lap_c3:+.3f} ({(lap_dyn - lap_c3) / max(abs(lap_c3), 1):+.2e} rel)", flush=True)
others = {k.name: gp.chi2_by_tag[f"set{i}/"] for i, k in enumerate(kin) if i != si}
print("other sets (tag chi2, incl. their projected-mass rows):", others, flush=True)
print("done", flush=True)
