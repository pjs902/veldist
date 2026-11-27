"""Where does Delta(-2lnL) between two models come from, and how significant is it: compares two hst_split.py files
(LOW, HIGH; the same data, so the same stars in the same order).

    python split_compare.py LOW.npz HIGH.npz [LOW2.npz HIGH2.npz]

Per Laplace set (Delta = HIGH - LOW, -2lnL units):
  - Laplace and floor-free exact Delta; p<=0 stars per model;
  - Laplace Delta by aperture radius (PM sets; MUSE has no positions here) and the median captured fraction / edge-ring
    fraction per radius bin (grid-edge truncation);
  - exact per-star Delta (stars with p > 0 in both) by kernel-centre speed (fraction of the grid half-width) and radius;
  - significance of the exact Delta at FIXED weights: star-level sd 2 sqrt(N var(dl)) (Vuong) and aperture-block sd
    sqrt(n_ap var(D_a)) * sqrt(n_ap/(n_ap-1)); their variance ratio is the empirical correlation inflation (kappa).
With a second pair: correlation of the per-aperture Laplace Delta between the pairs (library-noise reproducibility).
"""
import sys

import numpy as np

A = [np.load(f) for f in sys.argv[1:]]
lo, hi = A[0], A[1]
sets = [k[:-4] for k in lo.files if k.endswith("_lap")]
REDGES = [0, 2, 5, 10, 20, 50, 1e9]
VEDGES = [0, 0.25, 0.5, 0.75, 0.9, 1.01]


def lap_by_r(L, H, s):
    r = np.hypot(L[f"{s}_x"], L[f"{s}_y"])
    d = H[f"{s}_lap"] - L[f"{s}_lap"]
    return r, d


for s in sets:
    dl, de = (hi[f"{s}_{q}"].sum() - lo[f"{s}_{q}"].sum() for q in ("lap", "exact"))
    print(f"\n== {s}: M {float(lo['m_bh']):.3g} -> {float(hi['m_bh']):.3g}; Laplace Delta {dl:+.1f}; exact(p>0) "
          f"Delta {de:+.1f}; p<=0 stars {int(lo[f'{s}_nonpos'].sum())} -> {int(hi[f'{s}_nonpos'].sum())}")
    if f"{s}_x" in lo.files:
        r, d = lap_by_r(lo, hi, s)
        print("   R bin [arcsec]   n_ap   Laplace Delta   capt lo/hi (median)   edge-ring lo/hi (median)  p<=0 lo/hi")
        for a, b in zip(REDGES[:-1], REDGES[1:]):
            m = (r >= a) & (r < b)
            if m.any():
                print(f"   {a:>4g}-{b:<6g}  {m.sum():6d}   {d[m].sum():+12.1f}   "
                      f"{np.median(lo[f'{s}_capt'][m]):.4f}/{np.median(hi[f'{s}_capt'][m]):.4f}         "
                      f"{np.median(lo[f'{s}_edgefrac'][m]):.2e}/{np.median(hi[f'{s}_edgefrac'][m]):.2e}     "
                      f"{int(lo[f'{s}_nonpos'][m].sum())}/{int(hi[f'{s}_nonpos'][m].sum())}")
        o = np.argsort(d)
        print("   largest |Laplace Delta| apertures (R, Delta):",
              " ".join(f"{r[i]:.1f}\":{d[i]:+.0f}" for i in np.r_[o[:4], o[-4:]]))
    # per-star exact term l_s = log p_s - sum(h)/Ma of its aperture
    ap = lo[f"{s}_star_ap"]
    assert np.array_equal(ap, hi[f"{s}_star_ap"])
    ll = [Z[f"{s}_star_logp"].astype(float) - (Z[f"{s}_capt"])[ap] for Z in (lo, hi)]
    ok = np.isfinite(ll[0]) & np.isfinite(ll[1])
    dl_s = np.where(ok, ll[1] - ll[0], 0.0)
    D = -2 * dl_s[ok].sum()
    sd_star = 2 * np.sqrt(ok.sum() * dl_s[ok].var())
    n_ap = len(lo[f"{s}_lap"])
    Da = -2 * np.bincount(ap, weights=dl_s, minlength=n_ap)
    sd_blk = np.sqrt(n_ap * Da.var() * n_ap / (n_ap - 1))
    print(f"   exact per-star (p>0 in both, {ok.sum()} of {len(ok)} stars): Delta {D:+.1f}; sd star {sd_star:.1f}, "
          f"aperture-block {sd_blk:.1f}; kappa_eff (var ratio) {(sd_blk / sd_star) ** 2:.2f}; "
          f"Z_star {D / sd_star:+.1f}, Z_block {D / sd_blk:+.1f}")
    if f"{s}_star_v" in lo.files:
        v = lo[f"{s}_star_v"] / float(lo[f"{s}_half"])
        print("   exact Delta by star speed (v / half-width):",
              "  ".join(f"[{a:g},{b:g}) n {int(((v >= a) & (v < b) & ok).sum())} {-2 * dl_s[(v >= a) & (v < b) & ok].sum():+.1f}"
                        for a, b in zip(VEDGES[:-1], VEDGES[1:])))
        rs = np.hypot(lo[f"{s}_x"], lo[f"{s}_y"])[ap]
        print("   exact Delta by star radius:",
              "  ".join(f"[{a:g},{b:g}) {-2 * dl_s[(rs >= a) & (rs < b) & ok].sum():+.1f}"
                        for a, b in zip(REDGES[:-1], REDGES[1:])))
        fast = v >= 0.75
        print(f"   stars at v >= 0.75 half: {fast.sum()}; p<=0 among them lo/hi "
              f"{int((~np.isfinite(ll[0]) & fast).sum())}/{int((~np.isfinite(ll[1]) & fast).sum())}")

if len(A) == 4:
    print("\n== reproducibility across library replicas (per-aperture Laplace Delta, pair 1 vs pair 2)")
    for s in sets:
        d1 = A[1][f"{s}_lap"] - A[0][f"{s}_lap"]
        d2 = A[3][f"{s}_lap"] - A[2][f"{s}_lap"]
        diff = d1 - d2
        print(f"   {s}: totals {d1.sum():+.1f} / {d2.sum():+.1f}; per-aperture corr {np.corrcoef(d1, d2)[0, 1]:.3f}; "
              f"rms per aperture {d1.std():.2f}, rms of pair difference {diff.std():.2f} "
              f"(library-noise sd of the total ~ {np.sqrt(len(diff) / 2) * diff.std():.1f})")
