"""Per-aperture and per-star split of the Laplace sets of one solve (the kinematic M_IMBH profile investigation:
where in HST does Delta(3.2e4 - 1e4) come from, is it the IMBH, the statistic, or the grid edge).

    python hst_split.py PRIV_CONFIG MODEL_DIRECTORY RESULT.npz OUT.npz

ml_stretch_exact.py's capture (one rhs-only read at RESULT.npz w_new from regrid_one.py), on any private copy.
Per Laplace set and aperture: offset-free Laplace statistic (sum asserted against the solve), FLOOR-FREE exact
-2 log L (stars with p = (M h)_s <= 0 left out and counted separately), projected mass Ma, captured fraction
sum(h)/Ma, and for the PM sets the fraction of h in the outermost ring of cells and the aperture position.
Per star: aperture index, log p (nan if p <= 0), kernel mass on the grid m_s and, for PM sets, the speed of its kernel centre (km/s units of the
pack's cells). The exact per-star term is log p_s - sum(h)/Ma (the extended-Poisson normalisation split evenly).
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
assert any(f"/veldist-fitstat/{p}/" in model.directory for p in ("regrid", "stretch", "noise")), model.directory
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset()
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
R = np.load(res_path)
assert str(R["directory"]).rstrip("/") == mdir.rstrip("/")
w = np.asarray(R["w_new"], np.float64)
solved = ast.literal_eval(str(R["sets_new"]))
INDIR = Path(c.settings.io_settings["input_directory"])
FILES = {"hst_pm": "hst_veldist_pm_2dhist.npz", "gaia_pm": "gaia_veldist_pm_2dhist.npz"}

ws = wsmod.NNLS(config=c, model=model)
t0 = time.time()
ws._lap_h0, ws._lap_capture = None, dict(w=w, h={})
ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=w)
H = ws._lap_capture["h"]
ws._lap_capture = None
t_read = time.time() - t0

out, ap0 = dict(directory=mdir, t_read=t_read, m_bh=float(c.all_models.table[rows[0]]["m-bh"])), 0
for k in kin:
    n = k.n_spatial_bins
    if getattr(k, "statistic", "diag") != "laplace":
        ap0 += n
        continue
    Ma = np.asarray(ws.projected_masses[ap0:ap0 + n], float)
    ap0 += n
    h = np.asarray(H[k.name], np.float64)
    pm = k.name in FILES
    if pm:
        z = np.load(INDIR / FILES[k.name])
        assert len(z["xbin"]) == n
        K = int(round(np.sqrt(h.shape[1])))
        assert K * K == h.shape[1] == z["PM_2dhist"].shape[1] * z["PM_2dhist"].shape[2]
        cc = np.linspace(-float(z["vxrange"]), float(z["vxrange"]), K + 1)
        cc = 0.5 * (cc[1:] + cc[:-1])
        P1, P2 = [g.ravel() for g in np.meshgrid(cc, cc, indexing="ij")]
        ii, jj = [g.ravel() for g in np.meshgrid(np.arange(K), np.arange(K), indexing="ij")]
        ring = (ii == 0) | (jj == 0) | (ii == K - 1) | (jj == K - 1)
        out[f"{k.name}_x"], out[f"{k.name}_y"] = np.asarray(z["xbin"], float), np.asarray(z["ybin"], float)
        out[f"{k.name}_edgefrac"] = h[:, ring].sum(1) / h.sum(1)
    lap, ex, nonpos, nst = (np.zeros(n) for _ in range(4))
    s_ap, s_logp, s_v, s_m = [], [], [], []
    for a in range(n):
        Lt, g, h0 = k.laplace_terms(a, Ma[a])
        lap[a] = fs.stat_offset_free(h[a], Lt, g, h0)
        M = k.laplace_design(a)
        assert M.shape[1] == h.shape[1]
        p = M @ h[a]
        ok = p > 0
        nonpos[a], nst[a] = np.sum(~ok), len(M)
        ex[a] = 2.0 * (-np.sum(np.log(p[ok])) + len(M) / Ma[a] * h[a].sum())
        s_ap.append(np.full(len(M), a, np.int32))
        s_logp.append(np.where(ok, np.log(np.where(ok, p, 1.0)), np.nan).astype(np.float32))
        s_m.append(M.sum(1).astype(np.float32))  # kernel mass on the grid: p of a uniform h is sum(h) m_s / K2
        if pm:
            m0 = M.sum(1)
            s_v.append(np.hypot(M @ P1, M @ P2) / m0)
    assert np.isclose(lap.sum(), solved[k.name], rtol=1e-6, atol=0.05), (k.name, lap.sum(), solved[k.name])
    print(f"{k.name}: laplace {lap.sum():.2f} (solve {solved[k.name]:.2f}); exact(p>0) {ex.sum():.2f}; "
          f"stars {int(nst.sum())}; p<=0 {int(nonpos.sum())}; captured median {np.median(h.sum(1) / Ma):.4f} "
          f"min {np.min(h.sum(1) / Ma):.4f}", flush=True)
    out.update({f"{k.name}_Ma": Ma, f"{k.name}_capt": h.sum(1) / Ma, f"{k.name}_lap": lap, f"{k.name}_exact": ex,
                f"{k.name}_nonpos": nonpos, f"{k.name}_nstars": nst, f"{k.name}_star_ap": np.concatenate(s_ap),
                f"{k.name}_star_logp": np.concatenate(s_logp), f"{k.name}_star_m": np.concatenate(s_m),
                f"{k.name}_K2": h.shape[1]})
    if pm:
        out[f"{k.name}_star_v"] = np.concatenate(s_v).astype(np.float32)
        out[f"{k.name}_half"] = float(z["vxrange"])
np.savez(out_path, **out)
print(f"done ({t_read / 60:.0f} min read)", flush=True)
