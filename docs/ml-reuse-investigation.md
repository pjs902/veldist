# Reusing an orbit library across M/L (Υ): investigation

Started 2026-11-23. User: "it would be so nice to have a version that worked and could let us scan through M/L for
just the price of weight solving". `reuse_orblib_across_ml: False` in every production config since v4 because no
rescaling was found that kept the model non-negative, unbiased and the fit statistic intact.

## Why it matters now

The regrid of the 74 v4 models (per-star statistic) puts the best model at the edge of the explored Υ:
orblib_001_001 (Υ 2.965) is the only model with Υ < 3.05, and the three models that differ from it only in Υ give
Δ(−2lnL) = 0 / +2167 / +5202 at Υ = 2.965 / 3.065 / 3.165. A parabola through them has its minimum near
Υ ≈ 2.77, ~1700 below the current best. Every new Υ currently costs a full orbit library (~20 h on the loaded node).

## The physics is already exact; only the velocity binning is the problem

Multiplying every mass by k = Υ/Υ₀ (stars, IMBH, sBHs, halo) leaves the orbits unchanged and multiplies their
velocities by s = √k. The fork already does the parameter bookkeeping right: `AllModels.get_physical_parameter_table`
scales every dark mass by k = sf² for a reused library. So a reused library is not a Υ scan at fixed dark masses; it
is a *ray* (Υ, M_IMBH, M_sBH, halo) × k, every point of which is a valid model the GP can use through the
physical table. A pure Υ scan still needs libraries; a ray costs one weight solve per point.

What breaks is putting the stretched model onto the data's velocity cells. The fork has two model-side ways
(`kinematics.cdf_regrid_matrix`):

- `kind='spline'` (default): cubic-spline CDF regrid. Conserves mass, reproduces smooth second moments to 1e-4,
  **but gives negative cell masses** (mock: 14–18 negative cells of 6×40 for smooth components, ~370 of 24×40 for
  orbit-like narrow ones, at every s ≠ 1).
- `kind='linear'` (dynamite's former behaviour): non-negative, but assumes flat density inside each source cell, which
  **smears every histogram by ~dv²/6 per axis and lowers the χ² of rescaled models** (docstring: +3% variance at
  dv = 0.5σ for a 4% stretch).

## The proposal: stretch the data, not the model

Keep the library's cell masses as they are, on its native cells e. In the stretched model the same masses sit on
cells s·e (in km/s). Each star's row is the mass of its error kernel in each cell; integrate it over s·e instead of e
(equivalently: divide the star's velocity by s and its covariance by s², and use the native cells). Nothing is
interpolated on the model side, so the model stays exactly non-negative and unsmoothed.

Needed for comparisons across s (the rows carry no 1/(cell width) factor, and the fork reports each per-star set
relative to its expansion point, `stat_offset_free` = Q(h) − Q(h0)):

    A(s) = Q_s(h) + stat_exact(h0_s; rows_s) + 2 · N · d · ln s

with N the stars of the set, d its dimension (1 MUSE, 2 HST/Gaia), h0_s the expansion point for the stretched data
(the s = 1 h0 regridded to s·e; only the expansion point, so its form is second order after the one re-expansion).
At s ≈ 0.97 the Jacobian alone is ~8×10⁴ for ~1.4M star-dimensions: two orders of magnitude above the differences
we need, so getting it right is the test.

Other details:
- **Fixed star set.** The export drops stars with < 1e-3 of their kernel on the grid. Under a stretch that set would
  change with s; it must be frozen at s = 1 (keep the s = 1 stars at every s).
- **Sharpening** (`sharpen_matrix`, the h²/12 correction of the rows) is in cell units: the same S applies at every s.
- **Window.** The stretched model's window is ±s·41 km/s (HST); at s < 1 tail stars lose model support. The
  outlier floor is a density in cell units, so it shifts by ~N_floor · 2d ln s. Count both on the real data.
- **LVM** (χ² on mean velocities): (v_d/s − v_m)/(e/s) = (v_d − s v_m)/e, exact, no correction.
- **Escape constraint**: the potential scales by k, so v_esc by s; already through the physical masses.

## Results

All tests: `pytest experiments/fitstat/test_ml_stretch.py` (26 pass, 2026-11-23).

### Identities and normalization (exact; `test_ml_stretch.py` T1–T2)

- T1: production `precompute_design_matrix` (1D) and `precompute_design_matrix_2d` (2D, correlated and diagonal
  covariances) give rows(v/s, C/s²; e) = rows(v, C; s·e) to 1e-12 (2e-7 for the float32 2D output), s = 0.9–1.12.
- T2: the stretch formula with the Jacobian equals −2lnL computed by independent quadrature (1D quad, 2D dblquad) of
  the stretched flat-cell density; a known-answer Gaussian recovers its scale, and dropping or flipping the Jacobian
  misses it by > 0.02 in s (T2d).

### Mocks

- `ml_stretch_mock.py` (1D, fitted weights by EM, exact log-likelihood): smooth components — every method tracks
  the exact likelihood to ~15 over the Υ scan; orbit-like narrow components on coarse cells — *every* binned method,
  a fresh library included, bends 40–390 from exact (cell discretisation), and the unsharpened stretch bends more
  than the spline regrid (62 vs 36 at h = 5, converged at 4000 EM iterations). The spline regrid puts negative mass
  in 14–400 model cells at every s ≠ 1; the linear regrid bends 2–7× more than a fresh library. With sharpened rows
  the exact-log EM breaks (negative per-star probabilities): production linearises (Laplace), so the sharpened case is
  tested at fixed weights instead.
- `ml_stretch_alias.py` (one HST axis, 610k stars, the dump's errors — median 1.5 km/s against 7.2 km/s cells —
  fixed smooth model, sharpened rows as production): over s = 0.95–1.05 the stretch has detrended rms 10.0,
  peak-to-peak 35, trend +21 per 0.01 in s; a fresh library 5.6 / 19 / −29. Both recover the scale to 0.0025
  (exact 0.9775, truth 0.98); the curvature is ~94 per 0.01 in s, so either trend moves Υ by ~0.2%. Without
  sharpening both carry trends ~10× larger: sharpening is what makes binned Υ comparisons work at all. The stretch's
  noise does not shrink with N (9.6 at 150k stars): a few window-edge stars (±2.5 each) and fast precise tail stars
  (±0.3 each) whose cells move under the stretch. ~0.6% in Υ at HST scale (T3).

### Real data: stretched packs

- `c3_hst_terms.py build … stretch=S` (HST) and `ml_stretch_export.py` (Gaia, MUSE: the production notebook cells and
  writers, nothing refitted) build stretched star rows; `ml_stretch_pack.py` assembles the four replaced files and
  makes all three expansion points; `stretch_setup.sh` builds a private node copy.
- Parity: at s = 1 the Gaia and MUSE exports equal the production regrid input bit for bit (every array; md5 of
  the node files); HST rows equal production to 2e-6 of each row's maximum (laptop vs node float rounding at the
  1e-6 sparsification cut, 4267 of 610,822 stars; the s = 1 run uses the production HST file itself) (T4a, T5).
- Frozen star set: every stretched pack keeps exactly the s = 1 stars per aperture (610,822 HST, 64,539 Gaia,
  24,925 MUSE) (T4b, T6).
- Stars scale by 1/s: on the real Gaia and MUSE stars, row centroids of the stretched packs regress on the s = 1
  ones with slope 1/s to 4 decimals (median residual 0.000 Gaia, 0.002–0.009 km/s MUSE); per star for the Gaia stars
  whose kernel spans ≥ 2 cells (T6).
- Expansion point: the production h0 has a 1% uniform part; regridding it without undoing that costs a second 1%
  (+8.7k HST at s = 1.017). Undo, spline-CDF regrid (+106 HST, against +1683 for a linear CDF), clip, redo. The
  reference then costs the same at every s once the Jacobian is added — E0(s) + 2Nd ln s − E0(1):

  | Υ | s | HST (Jacobian) | Gaia (Jacobian) | MUSE (Jacobian) |
  |---|---|---|---|---|
  | 2.66 | 0.9472 | +780 (−132,611) | +7.5 (−14,012) | −14.5 (−2706) |
  | 2.76 | 0.9648 | +442 (−87,526) | +3.2 (−9248) | −8.5 (−1786) |
  | 2.86 | 0.9821 | +141 (−44,046) | +0.8 (−4654) | −3.2 (−899) |
  | 3.065 | 1.0167 | +106 (+40,524) | +4.7 (+4282) | −0.3 (+827) |
  | 3.165 | 1.0332 | +477 (+79,743) | +17.2 (+8426) | +2.9 (+1627) |

  Residuals are < 0.6% of the Jacobian: its sign and size are right on the real data. The HST residual grows
  ~(s−1)² (spline fidelity on 7.2 km/s cells); it enters only through the Laplace error (T7).
- The fork's h0-vs-data check (`_laplace_setup`: median correlation ≥ 0.9, no flip/transpose gain) passes for every
  pack (correlation ≥ 0.997).

### Node runs (set up 2026-11-24 00:25 UTC: all six `$W/stretch/NAME/` ready; solves wait for load < 176)

Per run, one job: `regrid_one.py CFG orblib_001_001/ml02.96 $W/stretch/NAME/result.npz && ml_stretch_exact.py CFG
orblib_001_001/ml02.96 …/result.npz …/exact.npz` (the exact check: one extra library read). Compare:
`ml_stretch_compare.py $W/stretch s1 ml3.065 ml3.165 ml2.86 ml2.76 ml2.66`.

Each: `stretch_setup.sh W NAME` then `regrid_one.py $W/stretch/NAME/config.yaml orblib_001_001/ml02.96 …/result.npz`
(regrid speed settings; ~1.5–4 h, ≤ ~70 GB, 16 MKL threads). Absolute comparison across s:

    A(s) − A(1) = ΣQ(s) − ΣQ(1) + Σ_sets [E0(s) + jac(s) − E0(1)] + χ²_LVM(s) − χ²_LVM(1)

1. `s1` (production files): must reproduce the regrid's orblib_001_001 per-set values (14,688.96) — identity.
2. `ml3.065`, `ml3.165`: free cross-check against orblib_000_000 (Υ 3.065, Δ = +2167) and orblib_001_000 (Υ 3.165,
   +5202), which share 001_001's other parameters; the stretched models' dark masses are 3.4% / 6.7% higher, so
   agreement to a few hundred is expected, a normalization error would show as tens of thousands.
3. `ml2.86`, `ml2.76`, `ml2.66`: the low-Υ ray (dark masses × 0.965 / 0.931 / 0.897).
4. Exact post-check: from each solve's weights, the model cell masses per aperture and the exact per-star
   −2lnL against each pack's rows (independent of h0 and of the Laplace expansion).
5. Gold standard (stage-3 slot): integrate orblib_001_001's potential with every mass × k (k ≈ 0.93) — the exact
   scaled copy; the stretched solve must match it far below the ~100 library noise.
