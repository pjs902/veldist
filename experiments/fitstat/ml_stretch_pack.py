"""Assemble the four replacement files of a stretched input pack (docs/ml-reuse-investigation.md).

On the node the pack is a copy of prod/input/NGC5139_perstar_xeast with exactly these files replaced:
  hst_veldist_pm_2dhist.npz      production file, PM_star_* / PM_lap_h0 / n_star_used from c3_hst_terms build stretch=S
  gaia_veldist_pm_2dhist.npz     ml_stretch_export.py output (xeast)
  veldist_kinematics_stars.npz   ml_stretch_export.py output (xeast)
  lvm_kins.ecsv                  production file with v, dv, sigma, dsigma divided by S (h3..h6 and errors unchanged)
Every other array of the replaced files must equal production; the star sets must equal production's; all four
must carry the same stretch. Prints the total Jacobian 2 sum N d ln S, and per set
  E0 = -2 sum_i ln(row_i . h0[aperture_i])
the data-reference term the fork's offset-free Q is measured from (the aperture-mass terms are the same at every S
and cancel), so that A(S) - A(1) = sum Q(S) - sum Q(1) + E0(S) - E0(1) + Jacobian(S) + chi2_LVM(S) - chi2_LVM(1).
HST_ROWS may be the production file itself (no stretch key: S = 1).
    python ml_stretch_pack.py PROD_HST.npz PROD_DIR HST_ROWS.npz EXPORT_XEAST OUTDIR
"""
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

prod_hst, prod_dir, hst_rows, exp_dir, out = map(Path, sys.argv[1:6])
out.mkdir(parents=True, exist_ok=True)
STAR = ("PM_star_indptr", "PM_star_cell", "PM_star_mass", "PM_star_aperture", "PM_lap_h0", "n_star_used")

# HST
P, R = dict(np.load(prod_hst)), dict(np.load(hst_rows))
R.setdefault("stretch", np.asarray(1.0))
R.setdefault("jacobian", np.asarray(0.0))
S = float(R["stretch"])
np.testing.assert_array_equal(R["PM_star_aperture"], P["PM_star_aperture"])
np.testing.assert_array_equal(R["n_star_used"], P["n_star_used"])
assert bool(R["PM_star_sharpened"]) and bool(P["PM_star_sharpened"])
H = {**P, **{k: R[k] for k in STAR}, "stretch": R["stretch"], "jacobian": R["jacobian"]}
np.savez(out / "hst_veldist_pm_2dhist.npz", **H)
jac = {"hst": float(R["jacobian"])}

# Gaia, MUSE: everything but the star arrays must equal production
for f, pre in (("gaia_veldist_pm_2dhist.npz", "PM"), ("veldist_kinematics_stars.npz", "LOS")):
    Z, Q = dict(np.load(exp_dir / f)), dict(np.load(prod_dir / f))
    assert np.isclose(float(Z["stretch"]), S), (f, float(Z["stretch"]), S)
    for k in Q:
        if k.startswith(f"{pre}_star") and k not in (f"{pre}_star_aperture", f"{pre}_star_sharpened"):
            continue
        if k == f"{pre}_lap_h0":
            assert Z[k].shape == Q[k].shape
            continue
        np.testing.assert_array_equal(Z[k], Q[k], err_msg=f"{f}:{k}")
    np.savez(out / f, **Z)
    jac[f.split("_")[0]] = float(Z["jacobian"])

# LVM
t = Table.read(prod_dir / "lvm_kins.ecsv")
for c in ("v", "dv", "sigma", "dsigma"):
    t[c] = t[c] / S
t.meta["stretch"] = S
t.write(out / "lvm_kins.ecsv", overwrite=True)

print(f"stretch {S:.6f} (ml ratio {S * S:.6f}); jacobian per set {jac}; total {sum(jac.values()):.1f}")


def e0(z, pre):
    ptr, cell, mass, ap = (z[f"{pre}_star_{k}"] for k in ("indptr", "cell", "mass", "aperture"))
    h0 = z[f"{pre}_lap_h0"].reshape(len(z[f"{pre}_lap_h0"]), -1)
    seg = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    p = np.bincount(seg, mass.astype(np.float64) * h0[ap[seg], cell], minlength=len(ptr) - 1)
    assert np.all(p > 0), (pre, int((p <= 0).sum()))
    return float(-2 * np.sum(np.log(p)))


E0 = {"hst": e0(H, "PM"), "gaia": e0(dict(np.load(out / "gaia_veldist_pm_2dhist.npz")), "PM"),
      "veldist": e0(dict(np.load(out / "veldist_kinematics_stars.npz")), "LOS")}
print("E0 per set:", {k: round(v, 1) for k, v in E0.items()})
np.savez(out / "stretch_meta.npz", stretch=S, **{f"jac_{k}": v for k, v in jac.items()},
         **{f"e0_{k}": v for k, v in E0.items()})
