"""M4: row-space Laplace rows (pm_laplace_rowspace_kappa) at production weights.

    PYTHONPATH=$W/dynamite-rowspace python m4_rowspace.py PRIV_CONFIG MODEL_DIRECTORY WEIGHTS.npz:KEY KAPPA OUT.npz

1. rhs-only read with the setting on and no chi2: rows reaching the accumulator (acc.rows_used), with and
   without the setting -> the dsyrk row reduction on the real data;
2. chi2 read at the weights with the setting on: per-set statistic (must equal the reference sets_new in
   WEIGHTS.npz to 1e-6) and the reported bound E per Laplace PM set (laplace_bound).
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402

import ast  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import dynamite as dyn  # noqa: E402
from dynamite import weight_solvers as wsmod  # noqa: E402

cfg, mdir, wspec, kappa, out_path = sys.argv[1:6]
wfile, wkey = wspec.split(":")
Z = np.load(wfile, allow_pickle=True)
w = np.asarray(Z[wkey], dtype=np.float64)
ref = ast.literal_eval(str(Z["sets_new"])) if "sets_new" in Z.files else {}
print("dynamite from", dyn.__file__, flush=True)
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1
model = c.all_models.get_model_from_row(rows[0])
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset()
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
ws = wsmod.NNLS(config=c, model=model)
assert ws.gram_blockwise
ws.settings["pm_laplace_rowspace_kappa"] = float(kappa)

used, kept = {}, {}
_add = wsmod.NormalEquationAccumulator.add


def add(self, A, b):
    _add(self, A, b)
    used[self.tag] = used.get(self.tag, 0) + int(np.shape(A)[0])  # rows offered
    kept[self.tag] = kept.get(self.tag, 0) + int(np.count_nonzero(np.asarray(A).any(axis=1)))  # rows reaching dsyrk


wsmod.NormalEquationAccumulator.add = add
t0 = time.time()
gp = ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True)
acc_rows = dict(kept)
t_read = time.time() - t0
print(f"reduced read {t_read:.0f} s; rows offered by tag {used}; rows kept (non-zero) {kept}", flush=True)
used.clear(); kept.clear()

t0 = time.time()
gp = ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=w)
mass_r2 = (gp.A_mass @ w - gp.b_mass) ** 2
sets, a0 = {}, ws.n_intrinsic
for i, k in enumerate(kin):
    sets[k.name] = float(gp.chi2_by_tag[f"set{i}/"] - mass_r2[a0:a0 + k.n_spatial_bins].sum())
    a0 += k.n_spatial_bins
E = dict(getattr(ws, "laplace_bound", {}) or {})
print(f"chi2 read {time.time() - t0:.0f} s; sets {sets}; reference {ref}; bound E {E}", flush=True)
for k, v in ref.items():
    if k in sets:
        print(f"  {k}: rel diff {abs(sets[k] - v) / max(abs(v), 1e-300):.2e}", flush=True)
np.savez(out_path, sets=str(sets), ref=str(ref), E=str(E), rows_offered=str(used), rows_used=acc_rows,
         kappa=float(kappa), t_read=t_read)
print("done", flush=True)
