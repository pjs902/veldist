"""Enclosed 3D mass M(<r) of every regrid model (Miyazaki, Chiba & Horigome 2026: report M(<r) at resolved radii next
to M_IMBH). Only the mass parameters enter, not the orbit weights, so this runs on all 74 models at once (seconds each).

    python enclosed_mass.py REGRID_CONFIG OUT.npz        (on the node; REGRID_CONFIG = $W/regrid/config.yaml)

Components as paperfig/extract_potential.py (r2): ml x deprojected stellar MGE and the dark-remnant MGE, exact mass
inside a sphere; Plummer IMBH; NFW halo; dark masses PHYSICAL (escape_chi2._model_with_physical_masses).
Saved: directory, m_bh [n_mod], r_arcsec [n_r], M_stars / M_sbh / M_imbh / M_halo / M_total [n_mod, n_r] in Msun.
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402

import logging  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# shortcut: reuses the node-only r2 helpers (veldist-fitstat/paperfig, not in git); move them here if paperfig goes.
sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith/veldist-fitstat/paperfig")
from extract_potential import mge_sphere_mass, nfw_mass  # noqa: E402

import dynamite as dyn  # noqa: E402
from dynamite import physical_system as phs  # noqa: E402
from dynamite_analysis import escape  # noqa: E402
from dynamite_analysis.escape_chi2 import _model_with_physical_masses  # noqa: E402

cfg, out_path = sys.argv[1:3]
os.chdir(Path(cfg).parent)
logging.disable(logging.WARNING)
c = dyn.config_reader.Configuration(cfg, reset_logging=False, user_logfile=False, reset_existing_output=False)
t = c.all_models.table
stars = c.system.get_unique_triaxial_visible_component()
sbh = next(k for k in c.system.cmp_list if isinstance(k, phs.StellarBlackHolesMGE))
bh = next(k for k in c.system.cmp_list if isinstance(k, phs.Plummer))
nfw = next((k for k in c.system.cmp_list if isinstance(k, phs.NFW)), None)
r = np.geomspace(0.05, 300.0, 80)
cols = {k: [] for k in ("M_stars", "M_sbh", "M_imbh", "M_halo")}
for i in range(len(t)):
    model = _model_with_physical_masses(c, i)
    par = model.parset
    cols["M_stars"].append(float(par["ml"]) * mge_sphere_mass(model, stars.mge_pot, r)[0])
    cols["M_sbh"].append(mge_sphere_mass(model, sbh.scaled_mge(par), r)[0])
    mbh, abh = escape._plummer_pars_of(model, bh)
    cols["M_imbh"].append(mbh * r ** 3 / (r ** 2 + abh ** 2) ** 1.5)
    cols["M_halo"].append(nfw_mass(model, nfw, r) if nfw is not None else np.zeros_like(r))
    print(t["directory"][i], f"M_IMBH {mbh:.4g}", flush=True)
cols = {k: np.array(v) for k, v in cols.items()}
tot = sum(cols.values())
np.savez(out_path, directory=np.array([str(d) for d in t["directory"]]), m_bh=np.asarray(t["m-bh"], float),
         m_imbh_phys=cols["M_imbh"][:, -1], r_arcsec=r, M_total=tot, **cols)
for rr in (1, 5, 20):
    j = np.argmin(abs(r - rr))
    print(f"r={r[j]:.2f}'': M_total {tot[:, j].min():.4g}..{tot[:, j].max():.4g} Msun over {len(t)} models")
