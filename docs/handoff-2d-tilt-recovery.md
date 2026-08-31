# Handoff: 2D tilt (rho) recovery investigation

**Status:** open, but much less alarming than the first version of this doc
claimed. Read the "Correction" section before anything else.

## Where this came from

Real Gaia bins measured at ~435 stars/bin, smaller than the calibration
profile (`GAIA_OUTER`, 2000 stars) that already `xfail`s in
`docs/validation.md`. `rho` (velocity-ellipsoid tilt) was tracked but not
gated in `tests/test_coverage_2d.py`. Landed on `main` (merge commit
`076d76a`, branch `2d-tilt-recovery`, now deleted):

- `rho` moved from `SOFT_METRICS` to `HARD_METRICS` for `gaussian_core` x
  `{hst_bright, hst_faint}` in `tests/test_coverage_2d.py`.
- `RecoveryCurve2D` / `recovery_curve_2d` added to
  `src/veldist/calibration2d.py` — sweeps `n_stars` (not an `ivar` proxy,
  by deliberate scope decision) and measures per-metric coverage / CI-CR
  ratio / bias across `n_real` mock realisations.
- `max_ci_ratio` is advisory-only for `rho` (spec-mandated: the analytic
  CR bound `(1-rho^2)^2/n` only holds under homogeneous per-star errors,
  not Gaia's real heteroscedastic regime) — only coverage gates `rho`'s
  threshold.

## Correction

The first draft of this doc read the coverage numbers against a nominal of
0.99 and concluded "nothing here is calibrated." **That was wrong.**

The nominal is **0.68**. The test criterion is `|median - truth| <= half68`
(`tests/test_coverage_2d.py:147-150`) — a 68% credible-interval hit test.
`coverage_floor(n_real, band=0.99, nominal=0.68)`
(`src/veldist/calibration.py:600`) returns the lower edge of the *99%
binomial band around 0.68*, which is where the 0.44 floor at `n_real=25`
comes from. It is not a coverage target.

## What the real-data sweep found

Ran `recovery_curve_2d` against a freshly re-measured real Gaia profile
(script: `scratchpad/run_gaia_tilt_recovery.py`, not committed —
reproduce by rerunning it, or recreate from `gaia_veldist.ipynb`'s
pipeline + `ObservingProfile2D.from_data` + a `recovery_curve_2d` call),
`truth_name="anisotropic"`, `n_stars = [100, 200, 435, 800, 1600]`,
`n_real=25`:

| metric | 100 | 200 | 435 (real) | 800 | 1600 | read against 0.68 |
|---|---|---|---|---|---|---|
| mean_x | 0.68 | 0.84 | 0.80 | 0.72 | 0.72 | on target |
| mean_y | 0.56 | 0.60 | 0.60 | 0.68 | 0.64 | slightly low, in band |
| rho | 0.68 | 0.72 | 0.52 | 0.64 | 0.56 | noisy, all in band |
| sigma_x | 0.44 | 0.60 | 0.60 | 0.60 | 0.64 | low but in band |
| sigma_y | 0.80 | 0.80 | 0.60 | 0.76 | 0.60 | in band |

Every number is inside the 99% binomial band for `n_real=25` at p=0.68.
At that sample size each coverage estimate carries roughly +/-0.19 at 1
sigma, so the sweep cannot distinguish "calibrated" from "10 points low."
**The sweep is underpowered, not damning.**

`sigma_y`'s "threshold not reached" is likewise *not* a coverage failure.
`RecoveryCurve2D.threshold` applies a second gate,
`ci_width <= 1.5 * cr_bound`, to every metric except `rho`. `sigma_y`'s
ratio is 1.77-1.81 across the whole sweep, so it trips the efficiency
check while passing coverage. CI/CR > 1 everywhere (1.1-1.8) is expected
and fine; only `sigma_y` exceeds the 1.5 gate.

So there is exactly one real signal in this run: **`sigma_y` posterior
intervals are ~1.8x the Cramer-Rao bound.**

## Ruled out

The `gaussian_core` 2D prior does not enforce axis-alignment or isotropy
(which would have explained systematically poor `rho` recovery
structurally):

- Null space (`_null_space_basis_2d`, `veldist2d.py:399-442`) spans
  `{1, x, y, x^2, xy, y^2}` — the full space of bivariate-Gaussian
  log-densities at *any* rotation and aspect ratio. The `xy` term is what
  lets the ellipse tilt; it is unpenalized like every other null-space
  direction.
- The GMRF roughness penalty (`build_gmrf_precision`, `veldist2d.py:296-396`)
  uses `edge_weight=1.0`, `diag_weight=1/sqrt(2)` on all 8 lattice
  neighbours — exact 4-fold (90 deg) symmetry, no axis bias in the
  smoothness metric either.

The model is already rotation-invariant.

## Plan

### Phase 1 — power

Rerun `recovery_curve_2d` at `n_real=100`, only `n_stars` in {435, 1600},
`truth=anisotropic`. 200 fits, roughly the cost of the 125 already run. At
`n_real=100` the 99% band around 0.68 narrows to about [0.56, 0.79] —
tight enough to actually call. This is the only experiment that can
distinguish the live hypotheses; run it before anything else.

### Phase 2 — correlated measurement errors

`_draw_stars` (`calibration2d.py:221-240`) builds a **diagonal** per-star
covariance: `err_x` and `err_y` drawn independently, zero off-diagonal.
Real Gaia has `pmxy_cov` — the reproduction script computes it (line 70)
and then discards it, and `ObservingProfile2D.from_data` has no parameter
for it. Correlated *measurement* errors are exactly what leaks into a
fitted *intrinsic* `rho`, so this is the most likely remaining source of
tilt-specific trouble.

1. Verify `KinematicSolver2D.add_data` honours off-diagonal `cov`. It
   accepts a full 2x2, so it probably does — check, do not assume.
2. Extend `ObservingProfile2D` to carry the measured error-correlation
   distribution, and `_draw_stars` to draw from it.
3. Rerun Phase 1. If `rho` coverage moves, that is the answer, and it is a
   real science result about what Gaia can say about tilt.

### Phase 3 — data-driven calibration

Only once Phases 1-2 give a trustworthy `rho`. Turn the recovery curve
into a per-bin decision rule: for each real Voronoi bin, look up its star
count and error profile and emit a per-metric trust flag, rather than one
global threshold. Natural home is a `RecoveryCurve2D.trustworthy(metric,
n_stars)` interpolator over the swept curve.

### Phase 4 — HST skew/kurtosis analogs

No 2D higher-moment machinery exists yet (`analysis.py`'s
`gauss_hermite_fit` is 1D-only). Needs its own brainstorm. First scoping
decision: standardised third/fourth cumulants of the 2D pdf (cheap, 4+5
numbers, computable straight from `intrinsic_pdf` the same way
`_moments_from_pdf_samples_2d` does it) versus a genuine 2D
Gauss-Hermite expansion (literature-comparable, far more machinery).
Recommend starting with cumulants: they slot into the existing coverage
harness with no new fitting code, and answer "is there detectable
non-Gaussianity" before we invest in "what is its h3/h4."

### Also open

Why are `sigma_y` intervals ~1.8x the CR bound while `sigma_x` sits at
~1.3x? The truth is anisotropic (`sx=1.18*sigma`, `sy=0.76*sigma`), so the
narrower axis is the one with the inefficient posterior. Worth checking
whether this is a grid-resolution effect — `cell_per_sigma` is defined
against `sigma_ref`, so the `sy` axis is sampled more coarsely in units of
its own width.

## Reproducing

```bash
/opt/miniconda3/envs/main/bin/python <path>/run_gaia_tilt_recovery.py
```
(needs the `main` conda env — has `kinutils`/`dynamite`/editable `veldist`
pointing at this repo; loads the real Gaia catalogue from
`/Users/pesmith/research/omegaCen/dynamite_dataprep/gaia_data/NGC_5139_oCen.txt`)

Full raw output (with per-fit NUTS progress) is not preserved — only the
final `curve.report()` summary above was captured. Recreate the script
from the pipeline in `gaia_veldist.ipynb` if it's no longer in scratchpad.
