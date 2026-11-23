"""Per-dataset split of the regrid fit statistic: each model's per-set -2lnL minus the best model's.
Shows which dataset ranks the models (paper, Fit Quality).
    python regrid_sets.py work/regrid/results
"""
import ast
import glob
import sys

import numpy as np

K = ["veldist", "hst_pm", "gaia_pm", "lvm", "mass"]
R = []
for f in glob.glob(f"{sys.argv[1]}/*.npz"):
    z = np.load(f, allow_pickle=True)
    s = z["sets_new"].item()
    s = ast.literal_eval(s) if isinstance(s, str) else s
    assert abs(sum(s[k] for k in K[:4]) - float(z["chi2_kin"])) < 1e-6 * float(z["chi2_kin"])  # kin = 4 sets, no mass
    R.append((float(z["chi2_kin"]), str(z["directory"]), float(z["m_bh"]), s))
R.sort(key=lambda r: r[0])
b = R[0]
print(f"{len(R)} models; best {b[1]} kinchi2 {b[0]:.2f}")
print("  dkin  model                    M_IMBH ", " ".join(f"{k:>8s}" for k in K))
for r in R[:15]:
    print(f"{r[0] - b[0]:6.0f} {r[1][:24]:24s} {r[2]:8.3g}", " ".join(f"{r[3][k] - b[3][k]:+8.0f}" for k in K))
D = np.array([[r[3][k] - b[3][k] for k in K] for r in R[1:]])
print("median difference over the others:", dict(zip(K, np.median(D, 0).round(0))))
print("median share of dkin from hst_pm:", np.median(D[:, 1] / D[:, :4].sum(1)).round(2))
print("models fitting each set better than the best:", dict(zip(K, (D < 0).sum(0))))
print("lvm difference range:", D[:, 3].min().round(0), D[:, 3].max().round(0))
