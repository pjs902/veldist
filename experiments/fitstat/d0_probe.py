"""D0: cost probe for the exact per-star statistic inside DYNAMITE (one v4 model).

    python d0_probe.py PRIV_CONFIG MODEL_DIRECTORY STAR_TERMS.npz OUT.npz [--solve]

Same set-up as rescore_v4.py (private copy, DYNAMITE's own streamed read).
Measures:
  t_read        one rhs-only library read (the per-pass read of a Laplace re-expansion)
  hst_nnz/bytes the HST orbit histograms held in memory as float32 CSR
  t_eval        exact -2logL and its gradient over w, all HST apertures in memory
  check         2(-logL) at the fitted w == rescore_v4's exact sum (self-check)
  t_solve       (--solve) a full baseline solve of this model in the private copy
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402  re-execs under ENV_mkl

import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from astropy.io import ascii  # noqa: E402

import dynamite as dyn  # noqa: E402
from dynamite import weight_solvers as wsmod  # noqa: E402

cfg, mdir, terms_path, out_path = sys.argv[1:5]
do_solve = "--solve" in sys.argv
HST = "hst_pm"
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1, (mdir, rows)
model = c.all_models.get_model_from_row(rows[0])
assert os.path.realpath(model.directory_noml).startswith(os.path.realpath(Path(cfg).parent))
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset(), "parset hash mismatch: library would be rebuilt"
ws = wsmod.NNLS(config=c, model=model)
w = np.asarray(ascii.read(model.directory + "orbit_weights.ecsv")["weights"], dtype=np.float64)

CAP = []
_orig = wsmod.NNLS._prepare_kinematic_block


def _capture(self, kins, orb_veldist, tmp, prj_mass_i, ap0=0):
    obs, err, orb = _orig(self, kins, orb_veldist, tmp, prj_mass_i, ap0=ap0)
    if kins.name == HST:
        CAP.append((ap0, sp.csr_matrix(np.asarray(orb, np.float32).reshape(len(w), -1)),
                    np.asarray(prj_mass_i, np.float64)))
    return obs, err, orb


wsmod.NNLS._prepare_kinematic_block = _capture
t0 = time.time()
ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=w)
t_read = time.time() - t0
wsmod.NNLS._prepare_kinematic_block = _orig
CAP.sort(key=lambda t: t[0])
H = sp.hstack([t[1] for t in CAP], format="csr")  # orbits x (n_ap * K2)
Ma = np.concatenate([t[2] for t in CAP])
del CAP
n_ap = len(Ma)
K2 = H.shape[1] // n_ap
hst_bytes = H.data.nbytes + H.indices.nbytes + H.indptr.nbytes
print(f"read {t_read:.0f} s; H {H.shape} nnz {H.nnz} ({hst_bytes / 1e9:.2f} GB)", flush=True)

T = np.load(terms_path)
ptr, cell, mass, apert = (T[k] for k in ("PM_star_indptr", "PM_star_cell", "PM_star_mass", "PM_star_aperture"))
cols = cell.astype(np.int64) + np.repeat(apert.astype(np.int64), np.diff(ptr)) * K2
Ms = sp.csr_matrix((mass.astype(np.float64), cols, ptr), shape=(len(apert), n_ap * K2))
n_a = np.bincount(apert, minlength=n_ap)
cvec = np.repeat(n_a / Ma, K2)  # extended-Poisson term per model cell
Ht = H.T.tocsr()


def f_and_grad(w):
    h = Ht @ w
    p = np.maximum(Ms @ h, 1e-300)
    f = 2.0 * (-np.log(p).sum() + cvec @ h)
    g = 2.0 * (H @ (cvec - Ms.T @ (1.0 / p)))
    return f, g


f0, _ = f_and_grad(w)
ts = []
for _ in range(5):
    t0 = time.time()
    f_and_grad(w)
    ts.append(time.time() - t0)
t_eval = float(np.median(ts))
print(f"exact 2(-logL) at fitted w {f0:.1f}; f+grad {t_eval:.2f} s (median of 5)", flush=True)

t_solve = np.nan
if do_solve:
    t0 = time.time()
    ws.solve(orblib, ignore_existing_weights=True)
    t_solve = time.time() - t0
    print(f"baseline solve {t_solve:.0f} s", flush=True)

np.savez(out_path, directory=mdir, n_orb=len(w), n_ap=n_ap, n_stars=len(apert), t_read=t_read,
         hst_nnz=H.nnz, hst_bytes=hst_bytes, t_eval=t_eval, exact_at_w=f0, t_solve=t_solve)
print("done", flush=True)
