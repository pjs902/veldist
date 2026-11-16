"""Laplace full-solve timing (pass 1 + one re-expansion) on a v4 model; derived from c3_dyn_parity.py.

B1 real-data check: DYNAMITE's statistic 'laplace' on a v4 model == C3's rescore.

    python laplace_solve_probe.py PRIV_CONFIG MODEL_DIRECTORY STAR_TERMS.npz RESCORE.npz

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
assert ws.gram_blockwise
ws.settings["pm_laplace_reexpand"] = 1
# time every blockwise build (full Gram builds vs rhs-only reads) and the QP solves
T = []
_build, _qp = wsmod.NNLS.construct_gram_and_rhs_blockwise, wsmod.NNLS._relative_qp


def build(self, *a, **k):
    t0 = time.time(); out = _build(self, *a, **k)
    T.append(("rhs_read" if k.get("rhs_only") else "gram_build", time.time() - t0))
    print(f"{T[-1][0]} {T[-1][1]:.0f} s", flush=True)
    return out


def qp(self, *a, **k):
    t0 = time.time(); out = _qp(self, *a, **k)
    T.append(("qp", time.time() - t0)); print(f"qp {T[-1][1]:.0f} s", flush=True)
    return out


wsmod.NNLS.construct_gram_and_rhs_blockwise, wsmod.NNLS._relative_qp = build, qp
t0 = time.time()
w, chi2_tot, chi2_kin, _ = ws._solve_laplace(orblib)
print(f"laplace solve total {time.time() - t0:.0f} s; chi2_kin {chi2_kin:.1f}; per phase {T}", flush=True)
np.save(model.directory + "laplace_weights_probe.npy", w)
print("done", flush=True)
