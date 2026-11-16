"""C3 gate: compare v4 model rankings under the baseline and the Laplace / exact HST statistics.

usage: python c3_compare.py RESCORE_DIR
Per model: kinchi2 (v4), HST diag (baseline), HST Laplace and exact (as -2 dlogL),
and kinchi2 with HST replaced by Laplace. Deltas are to the best model under each
column. 'sqrt2N' is the v4 grid's unit, sqrt(2 * N_obs) for the whole kinchi2.
"""
import ast
import sys
from pathlib import Path

import numpy as np

rows = []
for f in sorted(Path(sys.argv[1]).glob("rescore_*.npz")):
    z = np.load(f)
    sets = ast.literal_eval(str(z["set_chi2"]))
    lap = float(np.sum(z["laplace"] - z["laplace_at_h0"]))
    rows.append(dict(dir=str(z["directory"]), mbh=float(z["m_bh"]), kin=float(z["kinchi2"]),
                     hst=float(z["diag"].sum()), lap=lap, ex=float(z["exact"].sum()),
                     other=float(z["kinchi2"]) - sets["hst_pm"],
                     border=float(z["border"].max()), nonpos=int(z["n_nonpos_p"].sum()),
                     mratio=float(np.median(z["mratio"]))))
if not rows:
    sys.exit("no rescore_*.npz yet")
n_obs = 1406 * 529 + 149 * 289 + 0  # HST + Gaia cells dominate; LOS rows are small
unit = np.sqrt(2 * n_obs)
cols = {"kinchi2": [r["kin"] for r in rows], "hst_diag": [r["hst"] for r in rows],
        "hst_lap": [r["lap"] for r in rows], "hst_exact": [r["ex"] for r in rows],
        "kin_lap": [r["other"] + r["lap"] for r in rows]}
D = {k: np.array(v) - np.min(v) for k, v in cols.items()}
order = np.argsort([r["mbh"] for r in rows])
print(f"{len(rows)} models; v4 unit sqrt(2N) ~ {unit:.0f}")
print(f"{'directory':24s}{'m_bh':>8s}{'dKin':>9s}{'dKin/u':>8s}{'dHSTdiag':>10s}{'dHSTlap':>9s}"
      f"{'dHSTexact':>11s}{'dKin_lap':>10s}{'border':>8s}{'nonpos':>7s}{'mratio':>7s}")
for i in order:
    r = rows[i]
    print(f"{r['dir']:24s}{r['mbh']:8.0f}{D['kinchi2'][i]:9.1f}{D['kinchi2'][i] / unit:8.2f}"
          f"{D['hst_diag'][i]:10.1f}{D['hst_lap'][i]:9.1f}{D['hst_exact'][i]:11.1f}{D['kin_lap'][i]:10.1f}"
          f"{r['border']:8.4f}{r['nonpos']:7d}{r['mratio']:7.3f}")
if len(rows) >= 3:
    from scipy.stats import spearmanr
    for k in ("hst_lap", "hst_exact", "kin_lap"):
        print(f"Spearman(rank kinchi2, rank {k}) = {spearmanr(cols['kinchi2'], cols[k])[0]:.2f}")
    print("best by each:", {k: rows[int(np.argmin(v))]["dir"] for k, v in cols.items()})
