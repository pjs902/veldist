"""E1: full Laplace solve of one v4 model (HST statistic: laplace), vs its v4 baseline solve.

    python e1_solve.py PRIV_CONFIG MODEL_DIRECTORY STAR_TERMS.npz OUT.npz [REEXPAND]

v4's own weights ARE the baseline full solve (diag chi2 on every set). Here
the HST set is switched to statistic 'laplace' in memory (C3's per-star
terms; Gaia/MUSE stay diag until their star data are exported), the weights
are re-solved with DYNAMITE's ``_solve_laplace`` (pass 1 + REEXPAND
re-expansions, default 1), and for both weight vectors one rhs-only read
captures the HST model histograms (float64) and the per-set chi2. Saved:
w_base / w_lap, h_base / h_lap (n_ap, K2: model PM histograms in mass
units), Ma, per-set chi2 (HST as offset-free Laplace about the data h0),
per-aperture Laplace (about the data h0) and exact (-2 log L) for both, the
24-ish stars with p <= 0 per aperture, and timings. Writes nothing into
the model directories.
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
from dynamite import fitstat_pm as fs  # noqa: E402
from dynamite import weight_solvers as wsmod  # noqa: E402

cfg, mdir, terms_path, out_path = sys.argv[1:5]
n_re = int(sys.argv[5]) if len(sys.argv) > 5 else 1
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1
row = c.all_models.table[rows[0]]
model = c.all_models.get_model_from_row(rows[0])
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset()
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
names = [k.name for k in kin]
si = names.index("hst_pm")
hst = kin[si]
T = np.load(terms_path)
for k in hst.LAPLACE_KEYS:
    hst.data[k] = T[k]
hst.statistic = "laplace"
hst._init_laplace()
ws = wsmod.NNLS(config=c, model=model)
assert ws.gram_blockwise
ws.settings["pm_laplace_reexpand"] = n_re
ap0 = sum(k.n_spatial_bins for k in kin[:si])
Ma = None  # MGE projected masses of the HST apertures, set by the first build
timing = {}


def capture(w, tag):
    """One rhs-only read at w: HST model histograms + per-set chi2 (HST: Laplace about data h0)."""
    global Ma
    t0 = time.time()
    ws._lap_h0, ws._lap_capture = None, dict(w=w, h={})
    gp = ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=w)
    h = ws._lap_capture["h"]["hst_pm"]
    ws._lap_capture = None
    timing[f"capture_{tag}"] = time.time() - t0
    if Ma is None:
        Ma = np.asarray(ws.projected_masses[ap0:ap0 + hst.n_spatial_bins], float)
    mass_r2 = (gp.A_mass @ w - gp.b_mass) ** 2
    sets, a0 = {}, ws.n_intrinsic
    for i, k in enumerate(kin):
        sets[k.name] = gp.chi2_by_tag[f"set{i}/"] - mass_r2[a0:a0 + k.n_spatial_bins].sum()
        a0 += k.n_spatial_bins
    sets["mass"] = float((gp.row0_vec @ w - gp.b0) ** 2 + mass_r2.sum())
    lap, ex, nonpos = (np.zeros(len(Ma)) for _ in range(3))
    for a in range(len(Ma)):
        Lt, g, h0 = hst.laplace_terms(a, Ma[a])
        lap[a] = fs.stat_offset_free(h[a], Lt, g, h0)
        M = hst.laplace_design(a)
        ex[a] = fs.stat_exact(h[a], M, len(M), Ma[a]) - fs.stat_exact(h0, M, len(M), Ma[a])
        nonpos[a] = np.sum(M @ h[a] <= 0)
    return h, sets, lap, ex, nonpos


w_base = np.asarray(ascii.read(model.directory + "orbit_weights.ecsv")["weights"], dtype=np.float64)
h_b, sets_b, lap_b, ex_b, np_b = capture(w_base, "base")
print(f"baseline (v4 weights): sets {sets_b}; HST laplace {lap_b.sum():.1f} exact {ex_b.sum():.1f}", flush=True)
t0 = time.time()
w_lap, chi2_tot, chi2_kin, chi2 = ws._solve_laplace(orblib)
timing["laplace_solve"] = time.time() - t0
h_l, sets_l, lap_l, ex_l, np_l = capture(w_lap, "lap")
print(f"laplace solve {timing['laplace_solve']:.0f} s; sets {sets_l}; HST laplace {lap_l.sum():.1f} "
      f"exact {ex_l.sum():.1f}", flush=True)
np.savez(out_path, directory=mdir, m_bh=row["m-bh"], kinchi2_v4=row["kinchi2"], n_reexpand=n_re,
         w_base=w_base, w_lap=w_lap, h_base=h_b.astype(np.float32), h_lap=h_l.astype(np.float32), Ma=Ma,
         sets_base=str(sets_b), sets_lap=str(sets_l), lap_base=lap_b, lap_lap=lap_l,
         exact_base=ex_b, exact_lap=ex_l, nonpos_base=np_b, nonpos_lap=np_l, timing=str(timing),
         w_pass=np.asarray(ws.laplace_info["w_pass"]))
print("done", flush=True)
