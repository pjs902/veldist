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
The expansion point h0 of all three sets is made here, from the production h0, in one way (whatever the builders
wrote is replaced): undo the writer's 1% uniform mix, move the masses to the stretched cells S*e with the fork's
spline-CDF regrid (cdf_regrid_matrix), clip its few negative tail cells, re-apply the 1% mix. Only the expansion point
is regridded, never the model. At ml 3.065 the reference costs E0(s) + jac - E0(1) = +106 for HST with the spline
against +1683 with a linear CDF (which smears by ~h^2/6). At S = 1 this returns production's h0 (to rounding).
    python ml_stretch_pack.py PROD_HST.npz PROD_DIR HST_ROWS.npz EXPORT_XEAST OUTDIR
"""
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

prod_hst, prod_dir, hst_rows, exp_dir, out = map(Path, sys.argv[1:6])
out.mkdir(parents=True, exist_ok=True)
STAR = ("PM_star_indptr", "PM_star_cell", "PM_star_mass", "PM_star_aperture", "PM_lap_h0", "n_star_used")



def stretch_h0(h0, edges, S):
    """h0 (n_ap, *cells), each summing to 1 with the writer's 1% uniform part; edges per cell axis."""
    sys.path.insert(0, "/Users/pesmith/research/dynamite-fitstat")
    from dynamite.kinematics import cdf_regrid_matrix
    n_cell = np.prod(h0.shape[1:])
    h = (h0 - 0.01 / n_cell) / 0.99
    if S != 1.0:
        for ax, e in enumerate(edges, start=1):
            h = np.moveaxis(np.moveaxis(h, ax, -1) @ cdf_regrid_matrix(e, S * e).T, -1, ax)
    h = np.clip(h, 0, None)
    norm = tuple(range(1, h.ndim))
    h = 0.99 * h / h.sum(axis=norm, keepdims=True) + 0.01 / n_cell
    return h / h.sum(axis=norm, keepdims=True)


# HST
P, R = dict(np.load(prod_hst)), dict(np.load(hst_rows))
R.setdefault("stretch", np.asarray(1.0))
R.setdefault("jacobian", np.asarray(0.0))
S = float(R["stretch"])
np.testing.assert_array_equal(R["PM_star_aperture"], P["PM_star_aperture"])
np.testing.assert_array_equal(R["n_star_used"], P["n_star_used"])
assert bool(R["PM_star_sharpened"]) and bool(P["PM_star_sharpened"])
H = {**P, **{k: R[k] for k in STAR}, "stretch": R["stretch"], "jacobian": R["jacobian"]}
eh = np.linspace(-82.80638794003434, 82.80638794003434, 24)  # the HST grid (c3_hst_terms: one grid, all apertures)
H["PM_lap_h0"] = stretch_h0(P["PM_lap_h0"], (eh, eh), S)
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
    if pre == "PM":
        half, K = float(Q["vxrange"]), Q["PM_2dhist"].shape[1]
        assert np.isclose(float(Q["vyrange"]), half)
        e = np.linspace(-half, half, K + 1)
        Z[f"{pre}_lap_h0"] = stretch_h0(Q[f"{pre}_lap_h0"], (e, e), S)
    else:
        meta = Table.read(prod_dir / "veldist_kinematics.ecsv").meta
        K, dv = int(meta["nvbins"]), float(meta["dv"])
        e = (np.arange(K + 1) - K / 2) * dv
        Z[f"{pre}_lap_h0"] = stretch_h0(Q[f"{pre}_lap_h0"], (e,), S)
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
