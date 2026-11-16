"""C3: rescore one finished v4 model with the baseline and the Laplace / exact statistics.

    python rescore_v4.py PRIV_CONFIG MODEL_DIRECTORY STAR_TERMS.npz OUT.npz

Runs DYNAMITE's own streamed read (``construct_gram_and_rhs_blockwise``,
``rhs_only=True``, ``chi2_w=w``) on a PRIVATE copy (``c3_private_copy.sh``), so
the model histograms are exactly the rows the solve used. The kinchi2 it
reproduces is checked against the model's ``all_models`` row (rtol 1e-4).
``NNLS._prepare_kinematic_block`` is wrapped to capture, per HST aperture:
the model histogram h = w @ rows, the data and sigma (both times the MGE
projected mass M_a, as DYNAMITE fits them). Then, per aperture:

  diag    sum(((h - obs) / err)^2)        (baseline; sums to the HST set chi2)
  laplace ||Lt h - target||^2             (veldist.fitstat; constant per aperture)
  exact   2 (-log L), extended Poisson    (veldist.fitstat)
  border  border-ring mass / total model mass
  mratio  model aperture mass / M_a
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402  re-execs under ENV_mkl

import importlib.util  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from astropy.io import ascii  # noqa: E402

import dynamite as dyn  # noqa: E402
from dynamite import weight_solvers as wsmod  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "fitstat", Path(__file__).resolve().parents[2] / "src" / "veldist" / "fitstat.py")
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)

cfg, mdir, terms_path, out_path = sys.argv[1:5]
HST = "hst_pm"

os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False,
                                    reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1, (mdir, rows)
row = c.all_models.table[rows[0]]
model = c.all_models.get_model_from_row(rows[0])
assert os.path.realpath(model.directory_noml).startswith(os.path.realpath(Path(cfg).parent)), model.directory_noml
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset(), "parset hash mismatch: library would be rebuilt"
ws = wsmod.NNLS(config=c, model=model)
assert ws.gram_blockwise and ws.stream_reads
w = np.asarray(ascii.read(model.directory + "orbit_weights.ecsv")["weights"], dtype=np.float64)

CAP = []
_orig = wsmod.NNLS._prepare_kinematic_block


def _capture(self, kins, orb_veldist, tmp, prj_mass_i, ap0=0):
    obs, err, orb = _orig(self, kins, orb_veldist, tmp, prj_mass_i, ap0=ap0)
    if kins.name == HST:
        n_ap = len(prj_mass_i)
        flat = orb.reshape(len(w), -1)
        h = np.zeros(flat.shape[1])
        for o0 in range(0, len(w), 2048):  # float64 accumulation without a float64 copy of the group
            h += w[o0:o0 + 2048] @ np.asarray(flat[o0:o0 + 2048], dtype=np.float64)
        CAP.append((ap0, h.reshape(n_ap, -1), np.reshape(obs, (n_ap, -1)).astype(np.float64),
                    np.reshape(err, (n_ap, -1)).astype(np.float64), np.asarray(prj_mass_i, np.float64)))
    return obs, err, orb


wsmod.NNLS._prepare_kinematic_block = _capture
gp = ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=w)

# self-check: reproduce kinchi2 exactly as prod_chi2.per_dataset does
stars = c.system.get_unique_triaxial_visible_component()
kin = stars.kinematic_data
mass_r2 = (gp.A_mass @ w - gp.b_mass) ** 2
ap0, set_chi2 = ws.n_intrinsic, {}
for si, k in enumerate(kin):
    set_chi2[k.name] = float(gp.chi2_by_tag[f"set{si}/"] - mass_r2[ap0:ap0 + k.n_spatial_bins].sum())
    ap0 += k.n_spatial_bins
kinc = sum(set_chi2.values())
print(f"kinchi2 {kinc:.1f} (all_models {row['kinchi2']:.1f}); per set {set_chi2}", flush=True)
np.testing.assert_allclose(kinc, row["kinchi2"], rtol=1e-4)

CAP.sort(key=lambda t: t[0])
h = np.concatenate([t[1] for t in CAP])
obs = np.concatenate([t[2] for t in CAP])
err = np.concatenate([t[3] for t in CAP])
Ma = np.concatenate([t[4] for t in CAP])
n_ap = len(Ma)
diag = (((h - obs) / err) ** 2).sum(1)
np.testing.assert_allclose(diag.sum(), set_chi2[HST], rtol=1e-4)

T = np.load(terms_path)
ptr, cell, mass, apert, h0n = (T[k] for k in ("PM_star_indptr", "PM_star_cell", "PM_star_mass",
                                               "PM_star_aperture", "PM_lap_h0"))
assert h0n.shape[0] == n_ap
K2 = h.shape[1]
K = int(round(np.sqrt(K2)))
border = np.ones((K, K), bool)
border[1:-1, 1:-1] = False
border = border.ravel()
lap = np.zeros(n_ap)
lap0 = np.zeros(n_ap)
exact = np.zeros(n_ap)
n_neg = np.zeros(n_ap, int)
for a in range(n_ap):
    rws = np.nonzero(apert == a)[0]
    M = np.zeros((len(rws), K2))
    for j, r in enumerate(rws):
        M[j, cell[ptr[r]:ptr[r + 1]]] = mass[ptr[r]:ptr[r + 1]]
    N = len(rws)
    h0 = h0n[a].ravel() * Ma[a]
    F, g = fs.laplace_terms(M, h0, N, Ma[a])
    Lt, tg = fs.laplace_rows(F, g, h0, n_stars=N, mass=Ma[a])
    lap[a] = fs.stat_laplace(h[a], Lt, tg)
    lap0[a] = fs.stat_laplace(h0, Lt, tg)
    exact[a] = fs.stat_exact(h[a], M, N, Ma[a])
    n_neg[a] = int(np.sum(M @ h[a] <= 0))

np.savez(out_path, directory=mdir, m_bh=row["m-bh"], kinchi2=row["kinchi2"], set_chi2=str(set_chi2),
         diag=diag, laplace=lap, laplace_at_h0=lap0, exact=exact, n_nonpos_p=n_neg,
         border=h[:, border].sum(1) / h.sum(1), mratio=h.sum(1) / Ma)
print(f"done {mdir}: diag {diag.sum():.1f} laplace {lap.sum():.1f} exact {exact.sum():.1f} "
      f"border max {np.max(h[:, border].sum(1) / h.sum(1)):.4f} nonpos p {n_neg.sum()}", flush=True)
