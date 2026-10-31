# Cell-mass correction in `clip_uncertainties`

**Status:** on by default since 2026-10-31 (`cell_correction=True` in both
`KinematicSolver.clip_uncertainties` and `KinematicSolver2D.clip_uncertainties`).
`cell_correction=False` restores the previous output exactly.

## The problem

The likelihood treats the intrinsic density as flat inside each cell (the
design matrix integrates each star's error kernel over the cell). Where the
measurement errors blur the data on the cell scale, the data constrain the
fitted density's *variance*, so the fit converges on a flat-step histogram
whose total variance is the true `V`. Part of that variance is the flat steps
themselves (`h²/12`), so the cell masses, read as points at the cell centres,
have `V − h²/12`.

DYNAMITE does not compare densities. It bins its orbit velocities into the same
cells and compares cell masses, and the truth's exact cell masses have
`V + h²/12` (Sheppard). The per-cell values veldist wrote were therefore too
concentrated by `h²/6` in variance per axis, and DYNAMITE would fit a
dispersion 1–5% low on the production grids (more for heavy tails).

Found as the pre-existing slow-gate failure
`test_per_cell_losvd_coverage_2d[anisotropic-hst_faint-gaussian_core]`: one
y-wing cell at 0.28 coverage (p = 0.004 if calibrated), wing cells running
~1.3 half-widths low while the core over-covered (0.90).

## The correction

Each posterior draw is convolved with `[1/12, 5/6, 1/12]` along every grid
axis before the per-cell median and 68% interval are taken. This adds exactly
`h²/6` of point variance per axis, keeps total mass and mean, and — applied per
draw — keeps median and error bar consistent. `compute_summary` and the other
moment functions are unaffected: they work on the raw draws and already add
`h²/12`.

**Regime.** It assumes errors blur the data on the cell scale. When errors are
far below a cell, the fit already approaches the true cell masses and the
kernel over-widens (in the limit by the full `h²/6`). Ziliotto (err/h ≈ 0.17)
shows this: +1.6% on one axis.

## Verification (2026-10-31)

Mocks matched to each production dataset: the production grid, the dataset's
own star count and measured error distribution, per-star *correlated* errors
(|ρ| ≤ 0.45) for 2D, a mean offset of half the rotation span, and three or four
truths including heavy-tailed and two-component ones. 25 mocks per case, real
sampler. "sigma bias" is the dispersion read off the normalised per-cell
medians (what DYNAMITE sees), relative to the same statistic of the exact truth
cell masses, per axis. Raw data: `docs/data/cell-mass-correction-verification.jsonl`.
Script: `verify_cell_correction.py` (session scratchpad, 2026-10-31).

| dataset | truth | K | coverage raw → corr | cells < 0.30 raw → corr | sigma bias raw → corrected |
|---|---|---|---|---|---|
| bellini_f1 | gaussian_tilted | 13 | 0.69 → 0.77 | 0 → 0 | -2.5% → -0.7%; -4.0% → +0.3% |
| bellini_f1 | core_halo | 13 | 0.85 → 0.81 | 0 → 0 | -3.7% → -0.6%; -2.3% → +0.8% |
| bellini_f1 | two_component | 13 | 0.74 → 0.70 | 2 → 0 | -2.6% → -0.2%; -4.2% → -0.1% |
| scalco_f2f3 | gaussian_tilted | 13 | 0.72 → 0.78 | 0 → 0 | -1.9% → +0.1%; -3.4% → +1.1% |
| scalco_f2f3 | core_halo | 13 | 0.86 → 0.86 | 0 → 0 | -3.4% → -0.1%; -4.4% → -0.7% |
| scalco_f2f3 | two_component | 13 | 0.78 → 0.79 | 0 → 0 | -3.0% → -0.1%; -8.3% → -3.8% |
| muse | gaussian | 21 | 0.79 → 0.79 | 0 → 0 | -1.6% → +0.7% |
| muse | skew_normal_h3 | 21 | 0.68 → 0.68 | 0 → 0 | -3.9% → -1.6% |
| muse | student_t_h4 | 21 | 0.63 → 0.62 | 0 → 0 | -7.2% → -4.8% |
| muse | bimodal_counter_rotation | 21 | 0.67 → 0.64 | 1 → 1 | -4.6% → -2.1% |
| gaia | gaussian_tilted | 17 | 0.70 → 0.79 | 0 → 0 | -2.3% → +0.5%; -8.0% → -1.9% |
| gaia | core_halo | 17 | 0.79 → 0.83 | 0 → 0 | -9.5% → -3.4%; -11.1% → -5.2% |
| gaia | two_component | 17 | 0.65 → 0.70 | 11 → 4 | -4.1% → +0.1%; -11.0% → -4.7% |
| ziliotto | gaussian_tilted | 19 | 0.76 → 0.80 | 0 → 0 | -0.6% → +0.4%; -0.8% → +1.6% |
| ziliotto | core_halo | 19 | 0.87 → 0.82 | 0 → 0 | -1.6% → +0.3%; -3.0% → -1.0% |
| ziliotto | two_component | 19 | 0.78 → 0.75 | 6 → 2 | -0.5% → +0.9%; -3.9% → -1.6% |
| hst | gaussian_tilted | 23 | 0.73 → 0.77 | 0 → 0 | -0.8% → +0.4%; -1.8% → +0.8% |
| hst | core_halo | 23 | 0.88 → 0.84 | 0 → 0 | -1.9% → +0.3%; -4.9% → -2.9% |
| hst | two_component | 23 | 0.78 → 0.75 | 3 → 0 | -1.2% → +0.5%; -3.8% → -1.4% |

**Reading.** The correction reduces the σ bias on every axis of every case
except Ziliotto (tiny errors, see Regime), and cuts cells below 0.30 coverage
from 23 to 7. What remains on heavy-tailed truths (MUSE h4 −4.8%, Gaia
core+halo −5.2%) is not the cell effect: it comes from reporting per-cell
*medians* (the DYNAMITE format) and from the prior shrinking the tails, and is
present with or without the correction.

Prior-free maximum-likelihood checks (Gaussian truths, all six production
grids) agree: raw σ bias −1.3% to −4.9%, corrected within ±0.7%.
