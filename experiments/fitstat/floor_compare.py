"""Compare exact-likelihood floor rules for sharpened rows across the v4 models (rescore_v4.py with PLAIN_TERMS).

    python floor_compare.py RESCORE_DIR

Each statistic replaces the HST diagonal chi2 inside kinchi2 (other sets unchanged). Reported per rule: Spearman
against the sharpened Laplace and the plain exact ranking, the best model and its M_BH, and stars floored per model.
"""
import glob
import sys

import numpy as np
from scipy.stats import spearmanr

zs = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{sys.argv[1]}/rescore_f_*.npz"))]
base = np.array([float(z["kinchi2"]) - z["diag"].sum() for z in zs])
mbh = np.array([float(z["m_bh"]) for z in zs])
name = [str(z["directory"]) for z in zs]
floors = zs[0]["floors"]
tot = {"laplace_sharp": base + [z["laplace"].sum() for z in zs],
       "exact_plain": base + [z["exact_plain"].sum() for z in zs],
       "exact_sharp_raw": base + [z["exact"].sum() for z in zs]}
for i, f in enumerate(floors):
    tot[f"floor f={f:g}"] = base + [z["exact_floor"][i].sum() for z in zs]
nfl = {f"floor f={f:g}": np.array([z["n_floored"][i].sum() for z in zs]) for i, f in enumerate(floors)}
print(f"{len(zs)} models; non-positive p_S per model: median {np.median([z['n_nonpos_p'].sum() for z in zs]):.0f}")
print(f"{'statistic':18s} {'rho(lap_s)':>10s} {'rho(ex_pl)':>10s} {'best model':>24s} {'M_BH':>8s} {'floored/model':>14s}")
for k, v in tot.items():
    b = int(np.argmin(v))
    nf = f"{np.median(nfl[k]):.0f}" if k in nfl else ""
    print(f"{k:18s} {spearmanr(v, tot['laplace_sharp'])[0]:10.3f} {spearmanr(v, tot['exact_plain'])[0]:10.3f} "
          f"{name[b]:>24s} {mbh[b]:8.0f} {nf:>14s}")
