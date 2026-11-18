"""M3 + M6 (+ M4 reference): one rhs-only read of a production model at given weights.

    python m3_support.py PRIV_CONFIG MODEL_DIRECTORY WEIGHTS.npz:KEY OUT.npz

Per Laplace PM aperture (``NNLS._laplace_pm_block`` wrapped, rows unchanged):
- M3: fraction of orbits with any non-zero cell in H_a, number of contiguous runs of
  non-zero orbit indices, non-zero cell density of H_a;
- M6: per orbit, the ridge part outside the star row space, delta_a (|H_j|^2 - |H_j V_a|^2)
  (V_a = orthonormal row-space basis of the star rows, singular values > 1e-8 s_max),
  and the full Laplace diagonal |H_j Lt_a'|^2, both summed over the set's apertures;
- r_a = rank of the star rows.
Also the per-set statistic at the weights with the current code (reference for M4).
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402

import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import dynamite as dyn  # noqa: E402
from dynamite import fitstat_pm  # noqa: E402
from dynamite import weight_solvers as wsmod  # noqa: E402

cfg, mdir, wspec, out_path = sys.argv[1:5]
wfile, wkey = wspec.split(":")
w = np.asarray(np.load(wfile, allow_pickle=True)[wkey], dtype=np.float64)
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

stats = {}  # set name -> dict of per-aperture arrays / per-orbit sums
_orig = wsmod.NNLS._laplace_pm_block


def wrapped(self, kins, obs, orb, prj_mass_i, ap0):
    n_orbs, n_ap = orb.shape[:2]
    s = stats.setdefault(kins.name, dict(
        frac=np.full(kins.n_spatial_bins, np.nan), runs=np.zeros(kins.n_spatial_bins, int),
        dens=np.full(kins.n_spatial_bins, np.nan), rank=np.zeros(kins.n_spatial_bins, int),
        ridge_out=np.zeros(n_orbs), diag_full=np.zeros(n_orbs)))
    for a in range(n_ap):
        H = np.asarray(orb[:, a], dtype=np.float64).reshape(n_orbs, -1)
        nz = (H != 0).any(axis=1)
        s["frac"][ap0 + a] = nz.mean()
        s["runs"][ap0 + a] = int(np.count_nonzero(np.diff(nz.astype(np.int8)) == 1) + nz[0])
        s["dens"][ap0 + a] = np.count_nonzero(H) / max(1, nz.sum() * H.shape[1])
        M = kins.laplace_design(ap0 + a)
        _, sv, Vt = np.linalg.svd(M, full_matrices=False)
        V = Vt[sv > 1e-8 * sv[0]].T
        s["rank"][ap0 + a] = V.shape[1]
        mass = float(prj_mass_i[a])
        delta = fitstat_pm.default_delta(len(M), mass)
        HV = H @ V
        s["ridge_out"] += delta * (np.einsum("ij,ij->i", H, H) - np.einsum("ij,ij->i", HV, HV))
    o, d, r = _orig(self, kins, obs, orb, prj_mass_i, ap0)
    rr = np.asarray(r, dtype=np.float64).reshape(n_orbs, n_ap, -1)
    s["diag_full"] += np.einsum("ijk,ijk->i", rr, rr)
    return o, d, r


wsmod.NNLS._laplace_pm_block = wrapped
t0 = time.time()
gp = ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=w)
mass_r2 = (gp.A_mass @ w - gp.b_mass) ** 2
sets, a0 = {}, ws.n_intrinsic
for i, k in enumerate(kin):
    sets[k.name] = float(gp.chi2_by_tag[f"set{i}/"] - mass_r2[a0:a0 + k.n_spatial_bins].sum())
    a0 += k.n_spatial_bins
print(f"read {time.time() - t0:.0f} s; sets at weights: {sets}", flush=True)
out = {f"{n}|{k}": v for n, s in stats.items() for k, v in s.items()}
np.savez(out_path, sets=str(sets), read_s=time.time() - t0, **out)
for n, s in stats.items():
    f = s["frac"]
    share = s["ridge_out"] / np.maximum(s["diag_full"], 1e-300)
    live = s["diag_full"] > 0
    print(f"{n}: orbit support per aperture median {np.nanmedian(f):.3f} [p10 {np.nanpercentile(f, 10):.3f}, "
          f"p90 {np.nanpercentile(f, 90):.3f}]; runs median {int(np.median(s['runs']))}; "
          f"cell density {np.nanmedian(s['dens']):.3f}; rank median {int(np.median(s['rank']))}; "
          f"dropped-ridge share of diag: median {np.median(share[live]):.2e}, p99 {np.percentile(share[live], 99):.2e}, "
          f"max {share[live].max():.2e}; weighted (w) {np.sum(w * s['ridge_out']) / np.sum(w * s['diag_full']):.2e}",
          flush=True)
print("done", flush=True)
