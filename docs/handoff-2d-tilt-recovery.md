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

## Phase 1 result (n_real=100)

Rerun at `n_real=100`, `n_stars` in {435, 1600}, same measured profile and
`anisotropic` truth. Floor 0.56; each estimate now carries +/-0.047, tight
enough to call. Script: `scratchpad/run_phase1_power.py`.

| metric | 435 | 1600 | CI/CR | bias | verdict |
|---|---|---|---|---|---|
| mean_x | 0.74 | 0.63 | 1.14 | +0.05 | passes |
| mean_y | 0.70 | 0.63 | 1.32 | +0.06 | passes |
| rho | 0.60 | 0.52 | 1.65 | +0.03 | **fails at 1600** |
| sigma_x | 0.67 | 0.63 | 1.28 | +0.09 | passes |
| sigma_y | 0.58 | 0.47 | 1.78 | -0.25 | **fails at 1600** |

Three findings:

**Coverage falls as star count rises, on every metric.** 435 -> 1600 loses
6-11 points across the board. More data making calibration worse is the
signature of a fixed bias with a shrinking interval around it: at 435 the
intervals are wide enough to cover the offset, at 1600 they are not. A pure
identifiability limit would flatten out rather than degrade. This is the
most important thing in the run and it was invisible at `n_real=25`, where
+/-0.19 per estimate hid the trend (and happened to scatter upward at small
n, making `rho` look centred on nominal).

**`sigma_y` carries a real, persistent bias of -0.25**, essentially
unchanged between 435 and 1600, while every other metric's bias is +0.02 to
+0.10 and shrinking. `sigma_y` is the narrow truth axis (`sy = 0.76*sigma`)
and the bias is negative -- the fit under-disperses the axis that is most
coarsely resolved relative to its own width.

**`rho`'s failure is an interval-width problem, not a centring one.** Its
bias (+0.03, shrinking) is among the smallest in the table, but CI/CR =
1.65 is the second widest. `rho` and `sigma_y` are the only two metrics
above the 1.5 efficiency gate, and `rho = cov_xy / (sigma_x * sigma_y)` is
mechanically coupled to `sigma_y`. These are plausibly one defect, not two.

This demotes Phase 2 as originally written. Correlated measurement errors
would show up as a `rho` *bias*, and `rho`'s bias is the smallest thing in
the table. The grid-resolution question, filed as a P2 sideline, is now the
leading hypothesis for both failing metrics.

## Phase 2 (revised): grid resolution

`cell_per_sigma = 0.47` is defined against `sigma_ref`, but the anisotropic
truth's narrow axis is `sy = 0.76 * sigma_ref` -- so the narrow axis
actually gets 0.47/0.76 = 0.62 cells per its own sigma, well coarser than
the value chosen by measurement. Test at fixed `n_stars=1600` (clearest
failure) over `cell_per_sigma` in {0.47 (K=15, Phase 1 baseline), 0.37
(K=19), 0.33 (K=21)}. Script: `scratchpad/run_phase2_grid.py`.

**Caveat that could sink this hypothesis:** Sheppard's correction says
binning *inflates* a variance by ~h^2/12, so a too-coarse grid should
over-disperse `sigma_y`, and the observed bias is negative. This is not
fatal -- `_discretised_truth_moments` computes the truth on the same grid,
so the leading-order inflation cancels and the residual can go either way
-- but if the bias does not move with resolution, the hypothesis is dead
and Phase 2-as-originally-written (correlated errors via `pmxy_cov`) comes
back off the shelf.

Note also that `ObservingProfile2D`'s docstring records K=19 "breaks on
anisotropic truths" at N=400 (1.1 stars/cell). This test runs at N=1600,
so K=19 sees 4.4 stars/cell and K=21 sees 3.6 -- the regime where finer
cells should be affordable.

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

### Phase 1 — power — DONE

See "Phase 1 result" above.

### Phase 2 — grid resolution — RUNNING

See "Phase 2 (revised)" above. Supersedes the correlated-error work below
as the next thing to run, on the Phase 1 evidence.

### Phase 2b — correlated measurement errors (deferred)

Run this only if the grid test comes back null.

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

Every metric loses coverage from 435 to 1600 stars, not just the two that
fail outright. If Phase 2 explains `sigma_y` and `rho`, check whether the
same fix lifts `mean_x`/`mean_y`/`sigma_x` at 1600 too — their small
positive biases (+0.05 to +0.09) may share the mechanism.

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
