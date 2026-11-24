"""Compare stretched-data solves of one library across s (docs/ml-reuse-investigation.md).

    python ml_stretch_compare.py BASE REF NAME [NAME ...]

BASE/NAME/result.npz (regrid_one.py), optional BASE/NAME/exact.npz (ml_stretch_exact.py) and BASE/NAME/replace/stretch_meta.npz (ml_stretch_pack.py). REF is the s = 1 run.
Per Laplace set: A(s) - A(1) = Q(s) - Q(1) + E0(s) + jac(s) - E0(1); LVM: chi2(s) - chi2(1). Mass chi2 shown, not summed
(chi2_kin excludes it, as in the regrid ranking).
"""
import ast
import sys
from pathlib import Path

import numpy as np

SETS = {"hst_pm": "hst", "gaia_pm": "gaia", "veldist": "veldist"}


def load(base, name):
    z, m = np.load(base / name / "result.npz"), np.load(base / name / "replace" / "stretch_meta.npz")
    sets = ast.literal_eval(str(z["sets_new"]))
    assert np.isclose(sum(v for k, v in sets.items() if k != "mass"), float(z["chi2_kin"])), name
    return float(m["stretch"]), sets, m


def delta(base, ref, name):
    s, q, m = load(base, name)
    s1, q1, m1 = load(base, ref)
    assert s1 == 1.0
    d = {k: q[k] - q1[k] + float(m[f"e0_{v}"]) + float(m[f"jac_{v}"]) - float(m1[f"e0_{v}"]) for k, v in SETS.items()}
    d["lvm"] = q["lvm"] - q1["lvm"]
    return s, d, q["mass"] - q1["mass"]


if __name__ == "__main__":
    base, ref, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    print(f"{'name':>10} {'s':>7} {'ml':>6} " + " ".join(f"{k:>9}" for k in (*SETS, "lvm")) + f" {'total':>9} {'mass':>8}")
    for n in names:
        s, d, dm = delta(base, ref, n)
        print(f"{n:>10} {s:7.4f} {2.965 * s * s:6.3f} " + " ".join(f"{d[k]:9.1f}" for k in (*SETS, "lvm"))
              + f" {sum(d.values()):9.1f} {dm:8.1f}")
    # exact -2 log L (ml_stretch_exact.py; no h0, no expansion): exact(s) + jac(s) - exact(1).
    # stat_exact floors p at 1e-300, so each star with model p <= 0 adds FLOOR; that count changes between solves
    # and swamps the difference (ml3.065 HST: -37 such stars = -51k). Reported without it, plus the count change.
    FLOOR = -2 * np.log(1e-300)
    if all((base / n / "exact.npz").exists() for n in (ref, *names)):
        X1 = np.load(base / ref / "exact.npz")
        print("exact w/o p<=0 floor:" + "".join(f" {k:>9}" for k in SETS) + f" {'total':>9}   change in # p<=0 stars")
        for n in names:
            X, m = np.load(base / n / "exact.npz"), np.load(base / n / "replace" / "stretch_meta.npz")
            dn = {k: int(X[f"{k}_nonpos"].sum() - X1[f"{k}_nonpos"].sum()) for k in SETS}
            d = {k: X[f"{k}_exact"].sum() + float(m[f"jac_{v}"]) - X1[f"{k}_exact"].sum() - dn[k] * FLOOR
                 for k, v in SETS.items()}
            print(f"{n:>21}" + "".join(f" {d[k]:9.1f}" for k in SETS) + f" {sum(d.values()):9.1f}   {dn}")
