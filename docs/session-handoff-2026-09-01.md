# Session Handoff — 2026-09-01, ~05:30

## Status

- **Branch**: `main`, clean, synced with `origin/main` at `3e3a976`
- **Commits this session**: 25 (9 in the tuning phase, `cb42e4b`..`3e3a976`)
- **Uncommitted**: none. One untracked 0-byte file, `fit_all_bins_2d_failures.jsonl`
  (pre-existing, safe to delete)
- **Tests**: 284 passing, 2 xfailed, fast suite. Slow/SBC suite NOT re-run since
  the tuning changes — see Pending.

## What's Done

### The load-bearing fix (earlier this session, before the tuning work)

The resolution-dependent dispersion bias is diagnosed and fixed. Three
components disagreed about what a bin value means:

| component | implied variance |
|---|---|
| likelihood (design matrix integrates the error kernel over cells) | `Sum p(v-mu)^2 + h^2/12` |
| reported moments (point masses at cell centres) | `Sum p(v-mu)^2` |
| "truth" (centres of exact cell masses) | `V + h^2/12` |

Gap = `h^2/6` in variance = `-h^2/(12 sigma)` in sigma. Fixed by adding
`h^2/12` per axis to reported variance and comparing against the CONTINUOUS
truth. Verified flat across a 9x range in `h^2` (K series, 400 fits):

| K | 15 | 19 | 21 | 29 |
|---|---|---|---|---|
| sigma_y bias before | -0.250 | -0.154 | -0.119 | -0.050 |
| after | +0.004 | -0.006 | -0.004 | +0.005 |

A second, independent bug was found by audit and fixed: the 2D Gaussian core
produced density-at-cell-centres rather than cell mass. Real, but worth only
-0.250 -> -0.248; it was NOT the cause.

### Data-driven tuning (this session's goal)

All three datasets now derive their grid from measured quantities.

| dataset | before (hand-picked) | after (measured) |
|---|---|---|
| **Gaia** | sigma_ref 8.0, err 5.0, N=2000, K=17 | sigma 7.03-16.02, err 8.60, N=435, rot 12.8, cps 0.85, **K=25**, 0.70 stars/cell |
| **HST** | sigma_ref 17.0, err 0.24, N=400, K=15 | sigma 11.50-21.54, err 1.51, N=426 (min 174), rot 16.7, cps 0.58, **K=25**, 0.68 stars/cell |
| **MUSE (1D)** | N=150, err 2.5, log_s 0.4, rot 10.0, 37 bins | N=152, err 4.0, log_s 0.62, sigma 10.37-20.01, rot 17.9, **22 bins**, 6.9 stars/bin |

Measured 2026-08-31/09-01 by replaying each production notebook's own binning
call on the real catalogues, read-only. `NGC5139_config_production.yaml`
consumes exactly those three notebooks, so these are the real numbers. The
`core_supplement` / `hybrid_binning` / `sector_*` modules are exploratory and
NOT wired into production.

## What's In Progress

Nothing running. All background jobs completed and their results are folded in.

## What's Pending

Ordered by priority. The first two are gaps in THIS session's deliverable and
should be closed before the tuning is trusted.

1. **MUSE has no simulation verification at all.** Its grid was derived and
   found affordable; no 1D recovery curve was ever run with
   `OMEGACAT_MEASURED`. "Feasible" meant affordable, not validated. Cheapest
   of the outstanding work (1D, 22 bins).

2. **HST at its 174-star minimum is unverified on the adopted grid.** The
   min-occupancy run (`scratchpad/run_hst_min.py`, N=174/250/426) ran at
   cps=0.85 / K=19, which we then rejected. At the adopted cps=0.58 (K=25),
   174 stars gives 0.28 stars/cell, far below the 0.68 floor the sweeps
   measured. Roughly 10% of HST's 1415 bins sit near that floor, and they are
   the sparse outer bins carrying the kinematic signal.

3. **`n_stars` was measured, not optimised.** The goal asked for it. Current
   values were confirmed unbiased at the median; `target_capacity` was never
   swept, so we do not know the margin.

4. **Re-run the slow/SBC suite.** Not run since the tuning changes. Recipe (now
   corrected in CLAUDE.md):
   `XLA_FLAGS="--xla_force_host_platform_device_count=4" pytest tests/ -v --tb=short -m slow`

5. **2D non-Gaussianity** — the skew/kurtosis analogues for HST from the
   original ask. Never started. Recommend standardised cumulants over 2D
   Gauss-Hermite; **write the discretisation test FIRST**, since higher-moment
   Sheppard corrections are shape-dependent and were deliberately left
   unimplemented in `analysis.py`.

6. **Resolution should be per-axis, not per-bin.** `cell_per_sigma` is defined
   against `sigma_lo` (narrowest BIN), but the narrow AXIS of an anisotropic
   ellipsoid is ~0.65x smaller, so cells run ~1.3x that axis's own sigma even
   at the adopted value. This is why `sigma_y` is the metric that breaks first.
   Same correction already made for extent. Design change, not a constant.

7. **`err_cut` mismatch.** The Gaia notebook passes the default
   `PM_QUALITY_CUT_KMS` (7.81 km/s), not the 4x-inflated `err_cut` that
   `GAIA_OUTER` itself declares. Profile and notebook disagree.

8. **HST's minor-axis mean-velocity spread (16.7) exceeds its major-axis one
   (16.0).** Rigid rotation cannot produce that, and per-bin SEM (~0.8 km/s)
   rules out noise. Something roughly isotropic — bulk-motion residual,
   perspective, or edge bins — rides alongside the rotation. Does not affect
   grid sizing (larger axis wins either way), but a rotation-only model would
   misattribute it.

## Key Decisions Made

- **`cell_per_sigma` 0.47 -> regime-dependent.** 0.47 came from a sweep run
  BEFORE the `h^2/12` fix, when refining the grid shrank a bias the estimator
  itself was manufacturing. That sweep measured the bug's resolution-dependence
  and read it as a resolution requirement. Applied to measured profiles it
  demanded Gaia K=43 at 0.24 stars/cell.

- **A single global `cell_per_sigma` is the wrong shape.** HST fails at Gaia's
  0.85 (`rms_z` 1.36) and fails FLAT IN N, so it is not a sparse-bin problem.
  `rms_z` is bias / interval width; HST's `err/sigma_lo` is 0.13 vs Gaia's 1.22,
  so its posterior is sharp and nothing hides a residual discretisation bias.
  Gaia tolerates coarse cells because its large errors inflate the intervals.
  **Precise data needs FINER grids.**

- **The interpolation is empirical and labelled as such.** The natural physical
  model (bias under statistical error, scaling `(1+(err/sigma)^2)^(1/4)`)
  predicts a 1.20x ratio between the regimes against 1.47 measured. Writing the
  physical form would have been a clean equation that disagrees with our own
  data, so `cell_per_sigma_for` is a two-point power law that CLIPS outside its
  anchors rather than extrapolating.

- **HST cps = 0.58, not 0.42.** 0.42 is cleaner (0.0% bias vs 1.0%) but costs
  1.7x the cells for no measured gain across 1415 bins. 0.70 is borderline
  (2.2%, rms_z 1.11).

- **`rotation_span` uses full ptp, not the 5-95 span.** The grid must hold the
  worst bin; a percentile range discards exactly the tail that defines it.

- **The three original hand-picked profiles were kept**, not replaced, because
  the whole calibration campaign ran against them.

## Learnings Captured

- **[Semantics] Four bugs this session shared one shape**: a value numerically
  fine but semantically the wrong quantity — density for mass, discretised truth
  for continuous, `rows[0]` for all metrics, `sigma_ref` for `sigma_lo`. All
  produced plausible numbers and passed every structural check; types were
  identical in each case. What caught the last one was a **round-trip test**:
  feed the calibration its own measured inputs and require it to reproduce the
  measured outputs. A wrong denominator cannot satisfy two anchors at once.

- **[Constants] Every hand-picked scalar checked understated the real spread**
  — rotation_span 1.8x, err_median 1.6x, err_log_sigma 1.55x, HST's error 6x.
  `n_stars` erred the other way but also toward optimism. Five for five. Treat
  as the prior for anything not yet checked.

- **[Test design] Truth choice was load-bearing.** The isotropic truth passes
  at every resolution tested, including cps=1.80 where intervals are half the
  width they should be. The failure only appears on an ANISOTROPIC velocity
  ellipsoid. An isotropic-only validation would have certified a broken grid.

## Files Touched

- `src/veldist/calibration2d.py` — `+282`. Added `sigma_min`/`sigma_max`/
  `rotation_span`/`bins_per_error` fields, `sigma_lo`/`sigma_hi`/`cell_width`/
  `error_floor_width`/`cells_per_sigma_target` properties, `cell_per_sigma_for()`
  and `_CPS_ANCHORS`, `recommend_grid_2d()`, and the `GAIA_OUTER_MEASURED` /
  `HST_MEASURED` profiles. `cell_per_sigma` now defaults to `None` (derived).
- `src/veldist/calibration.py` — `+26`. Added `OMEGACAT_MEASURED`.
- `tests/test_recommend_grid_2d.py` — new, 16 tests pinning the rule's shape.
- `tests/test_calibration2d_diagnostics.py` — reads `cells_per_sigma_target`
  rather than the raw `cell_per_sigma` field, which is now `None` by default.
- `tests/test_calibration2d_profile.py` — registry now has 5 profiles.
- `docs/handoff-2d-tilt-recovery.md` — `+121`. The measured profiles, the
  shared-grid constraint, the regime-dependence finding, the settled rule.
- `CLAUDE.md` — fixed the slow-test parallelism recipe.

## Gotchas for Next Session

- **`recovery_curve_2d` returns one row PER METRIC per `n_stars`**, not one row
  per `n_stars`. Taking `rows[0]` silently reports `mean_x` alone — the metric
  LEAST sensitive to resolution, since the `h^2/12` term biases second moments
  and leaves means untouched. A sweep read that way looks reassuringly flat
  while saying nothing about `sigma_x`/`sigma_y`/`rho`. **This voided one
  400-fit run in this session.**

- **A ratio-valued constant needs its denominator named.** `_CPS_ANCHORS` must
  be expressed against `sigma_lo`, matching how `cell_per_sigma_for` is called.
  An earlier revision anchored on `sigma_ref` and returned 0.463 for HST instead
  of 0.58 — and was right for Gaia only because the value clipped at the top of
  the range.

- **Printing rows as JSON with `if not hasattr(v, "__len__")` drops strings**,
  so `metric` and `truth` vanish from the log while the numbers remain. The
  saved `.json` keeps them. Metric order is `mean_x, mean_y, sigma_x, sigma_y,
  rho`.

- **Backgrounded watcher processes do not survive** the tool call that spawned
  them. A `nohup sh -c 'while ...; do sleep; done'` chain to auto-launch a
  follow-up run was reaped silently. Launch follow-ups explicitly instead.

- **The commit hook rejects long subject lines** even when conventional-commit
  formatted. Keep the subject short.

- **`rtk proxy`** is needed to see unfiltered output from `grep`/`pytest`; the
  plain wrapper truncates and can misreport.

## Resume Command

> Continue on `main` in `/Users/pesmith/research/veldist`. The `h^2/12`
> dispersion bias is fixed and all three datasets (Gaia, HST, MUSE) now derive
> their grids from measured observing profiles — see
> `docs/handoff-2d-tilt-recovery.md`. Two verification gaps remain from that
> work: MUSE has NO simulation verification at all (never ran a 1D recovery
> curve with `OMEGACAT_MEASURED`), and HST's 174-star minimum bins are
> unverified on the adopted cps=0.58 / K=25 grid (0.28 stars/cell, below the
> 0.68 measured floor). Next step: run the MUSE 1D recovery curve first — it is
> the cheapest and has zero coverage — then HST at N=174 on K=25. Read ALL
> metric rows from `recovery_curve_2d`, not `rows[0]`.
