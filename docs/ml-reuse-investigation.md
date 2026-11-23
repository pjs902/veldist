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

### (a) 1D mock (`experiments/fitstat/ml_stretch_mock.py`)

RESULTS-A

### (b)–(d) Real data

PLAN-BCD
