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

## Phase 2 result: grid resolution

At `n_stars=1600`, `n_real=100`, floor 0.56. Script:
`scratchpad/run_phase2_grid.py`.

| metric | K=15 | K=19 | K=21 |
|---|---|---|---|
| mean_x | 0.63 | 0.64 | 0.65 |
| mean_y | 0.63 | 0.63 | 0.62 |
| rho | 0.52 FAIL | 0.62 | 0.64 |
| sigma_x | 0.63 | 0.63 | 0.61 |
| sigma_y | 0.47 FAIL | 0.62 | 0.58 |

| bias | K=15 | K=19 | K=21 |
|---|---|---|---|
| sigma_y | -0.250 | -0.154 | -0.119 |
| sigma_x | +0.099 | +0.121 | +0.145 |
| rho | +0.045 | +0.019 | +0.019 |

`sigma_y`'s bias falls monotonically with resolution and both failing
metrics clear the floor at K=19. That is the dependence a discretisation
bias should show and one that neither a measurement-error nor an
identifiability explanation could produce. **Grid resolution is confirmed
as the cause of the two Phase 1 failures.**

The Sheppard sign objection resolves as a prior-resolution interaction
rather than a Sheppard term: since `_discretised_truth_moments` bins the
truth identically, exact cell-mass recovery would cancel the inflation to
zero. A negative residual that shrinks with finer cells means the model
*under-recovers* mass in the narrow axis's outer cells -- the softmax-GMRF
field cannot express enough structure across a profile spanning ~1.6 cells
per sigma, so it flattens toward the grid.

### The two runs are paired

`recovery_curve_2d` resets `rng = np.random.default_rng(seed)` *inside* the
`n_stars` loop (`calibration2d.py:480`), so every K setting at a given
`n_stars` sees identical mock datasets. Comparisons across K are paired.

But only aggregates are retained -- per-realisation hits are discarded --
so McNemar cannot be computed from saved output. **K=19 vs K=21 on
`sigma_y` (0.62 vs 0.58) is 4 net flips in 100 shared realisations and is
not significant**; unpaired it is ~0.6 sigma. Read the bias column there,
not the coverage. Retaining per-realisation hit vectors is a cheap tooling
fix and a prerequisite for any future K comparison.

### Why x and y respond oppositely

The grid is square (`±3.5*sigma_ref`, square cells) but the truth is not
(`sx = 1.18*sigma_ref`, `sy = 0.76*sigma_ref`). Each axis therefore gets a
different resolution *and* a different extent, in opposite directions:

| axis | cells per own sigma (K=15 -> K=21) | half-extent in own sigma |
|---|---|---|
| x (wide) | 0.40 -> 0.28 | **2.97** |
| y (narrow) | **0.62** -> 0.43 | 4.63 |

x is well-resolved but tightly truncated; y is roomy but under-resolved.
Refining cells fixes y's problem and leaves x's boundary where it was --
and `sigma_x`'s bias *grows* (+0.099 -> +0.145) because finer cells let the
field pile mass more sharply against an unmoved boundary.

**Prediction (untested):** raising `n_sigma_grid` should move `sigma_x` and
leave `sigma_y` alone. One run settles it.

**The underlying defect:** `ObservingProfile2D` parameterises the grid
against a single scalar `sigma_ref`, which is only meaningful for an
isotropic truth. For an anisotropic velocity ellipsoid no single square-grid
setting can put both axes at their calibrated resolution *and* extent, since
`cell_per_sigma` and `n_sigma_grid` are shared while the requirement is
per-axis. That is why tuning K trades one axis against the other.

### What the grid fix did not touch

- `rho`'s CI/CR stays 1.63-1.66 at every K. `rho`'s remaining failure mode
  is interval *width*, and grid resolution does not address it. This is
  what re-opens the correlated-error question (see Phase 2b).
- The 435 -> 1600 coverage decline. At K=19/1600 every metric sits at
  0.62-0.64, still below the 435-star K=15 numbers (0.67-0.74).
- `cell_per_sigma=0.37` is a *candidate* default only. It was chosen at one
  star count on one truth and is not validated at 435 stars or on the HST
  profiles.

## Mechanism: two effects, cleanly separated

All at `n_stars=1600`, `n_real=100`, anisotropic truth (`sx=1.18*sigma_ref`,
`sy=0.76*sigma_ref`). The K series holds extent at 3.5*sigma_ref on BOTH
axes and refines cells only:

| grid | cell/sigma_ref | sigma_y bias | sigma_x bias |
|---|---|---|---|
| K=15 | 0.467 | -0.250 | +0.099 |
| K=19 | 0.368 | -0.154 | +0.121 |
| K=21 | 0.333 | -0.119 | +0.145 |
| K=29 | 0.241 | -0.050 | +0.198 |

Refining cells at fixed extent monotonically **fixes** `sigma_y` and
monotonically **worsens** `sigma_x`. Two mechanisms:

- **Resolution** causes under-dispersion (negative bias). More cells fix it.
- **Truncation** causes over-dispersion (positive bias), and finer cells make
  it WORSE -- they let the fitted field pile mass more sharply against a near
  boundary.

**Root cause:** extent is set as 3.5*`sigma_ref` on both axes, but what
matters is 3.5 of each axis's OWN sigma. The wide axis gets only 2.97 `sx`
(starved); the narrow axis gets 4.60 `sy` (wasteful).

Two independent corroborations:

- The isotropic control gives both axes exactly 3.5 of their own sigma, and
  shows only the negative resolution component on both (-0.170/-0.222 at
  K=15, shrinking to -0.072/-0.126 at K=21). No positive component anywhere.
- Grids A/B (below) trimmed the narrow axis's extent to DYNAMITE's minimum
  and its bias promptly flipped positive, i.e. the truncation hypothesis
  correctly predicted a new axis's behaviour before it was measured.

### Grids A/B: a confounded experiment, kept as evidence

| grid | cell/sigma_ref | y half-extent | sigma_x bias | sigma_y bias | rms_z x/y |
|---|---|---|---|---|---|
| A 29x17 | 0.250 | 2.13 sigma_ref | +0.186 | +0.124 | 1.13 / 1.17 |
| B 37x21 | 0.190 | 2.00 sigma_ref | +0.267 | +0.238 | 1.28 / 1.43 |

These changed TWO variables at once against the K series -- cells finer AND
y extent trimmed from 3.5 to ~2.0 sigma_ref -- because they were sized to
DYNAMITE's minimum `width/sigma >= 5`. Resolution and extent are therefore
confounded here and neither effect can be attributed from A/B alone. Kept
because the direction corroborates the truncation mechanism.

**Lesson worth keeping:** sizing a grid minimally against a spec is the right
way to build one you intend to ship and the wrong way to build one you intend
to learn from. Minimising extent to buy cells silently moved a second
variable. The K series was a clean experiment only by accident of
construction.

### DYNAMITE's extent minimum is too tight for this inference

`sanity_check`'s `width/sigma >= 5` (half-extent 2.5 sigma) is a threshold
for DYNAMITE's chi-square against a known histogram. We are *deconvolving*
measurement error, and the fitted field piles mass against a near boundary.
Grids A/B sat at 2.63-2.80 `sy` -- above DYNAMITE's minimum -- and were badly
over-dispersed. Use 3.5 of each axis's own sigma, not 2.5.

### What rms_z bought immediately

At grid B, `mean_x`/`mean_y` come back at rms_z 1.01-1.03 (essentially
perfect interval calibration) while `sigma_x`/`sigma_y` reach 1.28/1.43
(badly overconfident), with `mean_z` +0.79/+0.96 -- nearly a full
interval-width of standardised bias. On coverage alone all five metrics
looked like variations on "somewhat below floor". The means are demonstrably
fine and the problem is entirely in the second moments.

At grid C, **`rho` has bias +0.014 and rms_z 0.98** -- essentially perfect.
The metric this whole investigation started from, which read 0.52 coverage on
the original grid, is correctly calibrated purely from grid geometry: no
change to the prior, no `pmxy_cov` modelling. The tilt was always
recoverable; the grid was hiding it.

## Grid D: the fix, validated at 1600 stars

Grid D holds grid C's cell size (0.241 `sigma_ref`, square cells) and only
reallocates the extent per-axis, at 3.5 of each axis's OWN sigma:
**35 x 23 = 805 cells**, half-extent 3.57 `sx` / 3.65 `sy`.

| metric | C bias | D bias | C rms_z | D rms_z |
|---|---|---|---|---|
| mean_x | +0.065 | +0.016 | 1.02 | 1.01 |
| mean_y | +0.068 | +0.053 | 1.03 | 1.01 |
| rho | +0.014 | +0.010 | 0.98 | 0.94 |
| sigma_x | +0.198 | **+0.043** | 1.15 | 1.01 |
| sigma_y | -0.050 | -0.045 | 1.06 | 1.06 |

`sigma_x`'s bias fell 4.6x at identical resolution, purely from giving the
wide axis 3.57 `sx` of room instead of 2.97. `sigma_y` held as predicted.
Every `rms_z` is within 0.06 of 1.0 and every bias is under 0.06: all five
posteriors have honest interval widths. **No model change was involved --
only how the grid is sized.**

### The tuning rule (validated at n_stars=1600 ONLY)

- half-extent >= 3.5 x each axis's own sigma
- square cells at <= 0.241 `sigma_ref`
- odd bin count per axis (DYNAMITE requirement)

### rms_z vs the Cramer-Rao gate

`sigma_y` still reports "threshold not reached" at grid D. That verdict comes
from `threshold()`'s `ci_width > 1.5 * cr_bound` check (ratio 1.75) -- the
analytic CR bound the class docstring already flags as unreliable under
heteroscedastic errors. `rms_z = 1.06` measures the same intervals against
their own residuals and says they are fine. The CR bound is wrong here, not
the posterior. Without `rms_z` those two cases are indistinguishable, which
is precisely what made the original table unreadable. **`threshold()`'s CI/CR
gate should probably be replaced by an `rms_z` band for every metric, not
just exempted for `rho`.**

### The open gap that matters

Everything above is at `n_stars=1600`, where 805 cells is ~2.0 stars/cell.
Gaia's real bins have **435** stars: 0.54 stars/cell on the same geometry,
well past where `ObservingProfile2D`'s docstring records the method breaking
down (~1.8 stars/cell effective limit; K=19 "breaks on anisotropic truths" at
1.1). So the 1600-star results establish the SHAPE of the bias surface but
not where the affordable point on it lies for real data. At real volumes the
resolution term and the stars-per-cell term must collide, and the tuning rule
has to arbitrate between them. Sweeping `n_stars` on grid D's geometry is the
next run.

## SOLVED: the resolution-dependent dispersion bias

The `sigma` under-dispersion that scales with `h/sigma` is a three-way
bookkeeping disagreement about what a cell value means. It is not a modelling
error, not the prior, and not the Gaussian-core measure bug (that was real,
was fixed, and moved this bias by 0.002 -- see below).

### The mechanism

The design matrix is `M[i,m] = int_cell_m N(v; obs_i, Sigma_i) dv`, and the
likelihood is `M @ intrinsic_pdf`. Deriving the forward model that implies:

    P(obs) = int p(v) N(obs; v, err) dv
           = sum_m int_cell_m p(v) N(obs; v, err) dv
           ~= sum_m (p_m / h) int_cell_m N(obs; v, err) dv        [p ~ const]

Pulling `p(v) ~= p_m/h` out of the integral is a **piecewise-constant**
assumption: the likelihood spreads each cell's mass uniformly across the cell.
So the density it actually implies has

    Var(q) = sum_m p_m (v_m - mu)^2 + h^2/12

with `h^2/12` the exact variance of a uniform across one cell.

Three places then disagree:

| | treats `p_m` as | variance |
|---|---|---|
| likelihood (`M @ p`) | piecewise-constant over the cell | `sum p (v-mu)^2 + h^2/12` |
| reported moments | point masses at cell centres | `sum p (v-mu)^2` |
| `_discretised_truth_moments` | centres of exact cell masses | `V + h^2/12` |

The data drive the first to `V`, so `sum p (v-mu)^2 = V - h^2/12` while the
comparison target is `V + h^2/12`. Gap `h^2/6` in variance, i.e.

    bias(sigma) ~= -h^2 / (12 * sigma)

### Confirmation

Against the isotropic control at 1600 stars (no truncation confound, both
axes identical by construction), predicted vs measured:

| K | predicted | measured (mean of x, y) | ratio |
|---|---|---|---|
| 15 | -0.202 | -0.196 | 0.97 |
| 19 | -0.126 | -0.129 | 1.03 |
| 21 | -0.103 | -0.099 | 0.96 |

Within 4% at every resolution. On anisotropic `sigma_y` the ratios are 0.94 /
0.93 / 0.88 / 0.70 (K=15/19/21/29) -- good where the bias dominates, drifting
at K=29 where it is small and the truncation term competes.

### The fix

Make the estimator target the CONTINUOUS quantity and compare against the
CONTINUOUS truth:

- reported moments: `var_x += h_x^2/12`, `var_y += h_y^2/12`. The covariance
  is unchanged (cells are axis-aligned and the within-cell distribution is a
  uniform product, so x and y are independent within a cell); `rho` is
  recomputed from the corrected variances.
- truth: the analytic continuous moments, NOT the centre-discretised ones.

This is exact, not approximate. `Var(q) = sum p (v-mu)^2 + h^2/12` holds
exactly for a piecewise-constant reconstruction, so the corrected report is
`V` and the target is `V`.

Note `_discretised_truth_moments`' docstring previously argued the opposite --
that centre-discretised truth is "the fair comparison" because a continuous
truth "charges the model for grid discretisation". That was correct for the
UNCORRECTED estimator. With the correction it would double-count.

### Confirmed by measurement

K=15, anisotropic, 1600 stars, `n_real=25` (the configuration with the
largest bias, so the signal is unambiguous):

| metric | original | after core fix | after within-cell fix | rms_z now |
|---|---|---|---|---|
| sigma_y | -0.250 | -0.248 | **+0.077** | 1.02 (was 1.40) |
| sigma_x | +0.099 | +0.061 | +0.162 | 0.93 |
| rho | +0.045 | +0.027 | **-0.003** | 0.88 |
| mean_x | -- | +0.049 | -0.024 | 0.82 |
| mean_y | -- | +0.058 | -0.017 | 0.95 |

`sigma_y` moved by +0.325 against a predicted `h^2/(12*sigma)` = +0.265. At
`n_real=25` the SE on bias is ~0.06, so the residual +0.077 sits ~1.3 sigma
from zero -- consistent with the bias being removed, not merely reduced.
`rms_z` fell 1.40 -> 1.02, so the intervals are honest too, which the core
fix never achieved.

The residual `sigma_x` +0.162 is NOT this bug: it is the truncation term
(the wide axis gets only 2.97 `sx` of extent at this grid), diagnosed and
fixed separately by per-axis extent sizing, which took it to +0.043 in grid
D. The two fixes are orthogonal and compose.

### Still to do

- Re-run the K series (15/19/21/29) at `n_real=100` to confirm the fix
  removes the h-dependence across resolutions, not just at K=15.
- `analysis.py` has the SAME deficit and is NOT fixed. `compute_moments`,
  `compute_summary` and their `_maps` variants all use
  `sum_m p_m (v_m - mean)^2` at bin centres with no `h^2/12` term, so every
  reported dispersion is biased low by about `h^2/(12*sigma)`. This is the
  user-facing science API, so correcting it changes published numbers
  (including `docs/validation.md` and anything written to DYNAMITE) --
  a deliberate scope decision, not a patch. Note the 1D coverage tests call
  `compute_summary` directly, so the 1D exposure IS this one; there is no
  separate 1D calibration helper to fix.
- `calibration.py::true_moments` is already correct (integrates the
  continuous truth on a 400001-point grid).

### Why this hid for so long

It biases only SECOND moments (means are untouched), it scales away as
`h -> 0` so it reads as "we just need a finer grid", and it is invisible to
every normalisation check since all three quantities sum to 1. It is the same
family as the Gaussian-core measure bug -- a disagreement about whether a cell
value is a density or a mass -- but in a different place, and it is the larger
of the two by roughly a factor of two.

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

### Phase 2 — grid resolution — DONE

Confirmed. See "Phase 2 result" above.

### Phase 2b — correlated measurement errors — REINSTATED

Demoted after Phase 1 on the grounds that `rho`'s *bias* was small. That
argument only ever covered centring; `rho`'s actual failure mode is
interval width (CI/CR ~1.65), which the grid fix did not move at any K.
Correlated measurement errors are a live candidate for that residual, so
this comes back on the list rather than staying shelved.

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

---

## Data-driven tuning (2026-09-01)

The resolution bias is fixed (see above), so the open question became: what
grid should each real dataset actually use? Every tuning constant in
`calibration.py`/`calibration2d.py` was hand-picked. This section records
what the production notebooks say those constants should be.

### The profiles were measured, not guessed

Measured by replaying each production notebook's own binning call on the
real catalogue (`omegaCen/dynamite_dataprep/{gaia,hst,muse}_veldist.ipynb`),
read-only, bins with <10 stars excluded. `NGC5139_config_production.yaml`
consumes exactly these three notebooks' outputs, so these are the numbers
the pipeline really runs on. The `core_supplement`/`hybrid_binning`/
`sector_*` modules are exploratory and are NOT wired into production.

| field | hand-picked | measured | error |
|---|---|---|---|
| `OMEGACAT.rotation_span` | 10.0 | 17.9 | 1.8x too small |
| `OMEGACAT.err_median` | 2.5 | 4.0 | 1.6x too small |
| `OMEGACAT.err_log_sigma` | 0.4 | 0.62 | 1.55x too small |
| `HST_BRIGHT.err_median` | 0.24 | 1.51 | 6x too small |
| `GAIA_OUTER.n_stars` | 2000 | 435 | 4.6x too large |
| `rotation_span` (2D) | absent | 12.8-16.7 | field did not exist |

**Every hand-picked scalar that differs from measurement understates the
real spread**, with `n_stars` the sole exception (and it errs toward
optimism too: fewer real stars than assumed). Treat this as the prior for
any constant not yet checked.

Per-bin star counts, which set what occupancy the calibration must cover:

| | median | min | bins |
|---|---|---|---|
| Gaia | 435 | 320 | 148 |
| HST | 426 | **174** | 1415 |
| MUSE | 152 | 69 | 163 |

HST's minimum is the case never tested -- the whole campaign ran at N~400.

### The shared-grid constraint is what binds

DYNAMITE takes one scalar `vxrange`/`vyrange` per map, so ONE grid serves
every spatial bin. Extent must hold `sigma_max` plus `rotation_span`;
resolution must serve `sigma_min`. Since sigma spans ~2x across bins within
a single axis (Gaia pm1 8.4-16.0, HST pm1 12.0-21.5), those two pull apart,
and in 2D the cost squares. `ObservingProfile2D` previously had only a
scalar `sigma_ref` and so could not express this at all.

Applying the rule with `cell_per_sigma=0.47` gives Gaia K=43 at 0.24
stars/cell and HST K=31 at 0.44 -- not affordable. But 0.47 was measured
BEFORE the h^2/12 fix, when refining the grid shrank a bias the estimator
itself was manufacturing; that sweep was measuring the bug's
resolution-dependence and calling it a resolution requirement. Re-measuring
it is in progress.

**1D MUSE is already feasible** on measured numbers: 22 bins at 6.9
stars/bin, against the hand-picked profile's 37 at 4.1. The wider measured
`rotation_span` is more than offset by the measured dispersion range being
narrower than assumed (10.4-20.0 vs 7.0-22.0).

### Not rotation

For HST the *minor*-axis mean-velocity spread (16.7 km/s) slightly exceeds
the *major*-axis one (16.0). Rigid rotation should put nearly all of that
on the major axis with the minor flat near zero. Per-bin SEM is ~0.8 km/s
against a 16 km/s spread, so it is not sampling noise on the means.
Something roughly isotropic -- bulk-motion residual, perspective effects,
or edge bins -- rides alongside the rotation. It does not change grid sizing
(the larger axis wins either way), but a rotation-only model would
misattribute it.

### Trap for the next person

`recovery_curve_2d` returns one row PER METRIC per `n_stars`, not one row
per `n_stars`. Taking `rows[0]` silently reports `mean_x` alone -- which is
the metric LEAST sensitive to resolution, since the h^2/12 term biases
second moments and leaves means untouched. A sweep read that way looks
reassuringly flat while saying nothing about `sigma_x`/`sigma_y`/`rho`.
This voided one 400-fit run here.

### cell_per_sigma cannot be one global constant (2026-09-01)

Re-measured at Gaia's real numbers post-h^2/12-fix, `cell_per_sigma=0.85`
is right for Gaia (sigma_y bias 2.8% of the narrow axis, both rms_z near 1).
**It fails for HST**, and the failure is not about star count:

| N | stars/cell | sigma_y bias | sigma_y rms_z |
|---|---|---|---|
| 174 | 0.48 | +0.438 | 1.34 |
| 250 | 0.69 | +0.414 | 1.29 |
| 426 | 1.18 | +0.359 | 1.36 |

Flat in N. HST's sigma_y intervals are ~30% too narrow at the MEDIAN bin,
so the 174-star minimum is not the problem -- the whole dataset is
mis-calibrated on sigma_y at this resolution.

**Why**: `rms_z` is bias over interval width, and HST's intervals are tiny.
Its `err/sigma` is 0.094 against Gaia's 1.05 -- the measurement errors are
~1% of the signal, so the posterior is sharp and there is nothing to hide a
residual discretisation bias behind. Gaia tolerates 0.85 not because its
grid is better but because its large errors inflate the intervals enough to
swallow the same absolute error.

**Consequence for the tuning rule**: the resolution requirement scales with
how PRECISE the data is, not just how wide the LOSVD is. Discretisation
error must stay below the statistical floor, and that floor falls as
`err/sqrt(N)`. Precise datasets need FINER grids -- the opposite of the
intuition that good data is easier to fit. A single `cell_per_sigma` shared
across regimes is the wrong shape; it should be a function of
`err_median/sigma` (and weakly of `n_stars`).

This is why `HST_BRIGHT`'s original hand-picked `err_median=0.24` mattered
so much: it is 6x below the measured 1.51, and it is exactly the parameter
that sets how fine HST's grid has to be.

Sweep in progress at cps {0.42, 0.58, 0.70} on HST's measured profile to
locate where it calibrates. Gaia's 0.85 stands.

---

## 2026-09-01 (afternoon): observing-profile audit

Read-only replay of both production notebooks' own selections and PowerBin
calls, to close the three open profile questions. Scripts:
`scratchpad/phase_a_gaia.py`, `scratchpad/phase_a_hst.py`.

### The mock error distributions were wrong in BOTH measured profiles

`draw_errors` back-derived its log-normal width from `err_cut`, assuming the
quality cut sits at p95. That assumption failed in opposite directions.

| profile | real median | real p95 | mock p95 (before) | after |
|---|---|---|---|---|
| `gaia_outer_measured` | 8.33 | 27.54 | **7.81** (65% pinned at the cut) | 28.55 |
| `hst_measured` | 1.535 | 3.14 | **7.78** | 3.08 |

**Gaia** declared `err_cut = PM_QUALITY_CUT_KMS` = 7.81 km/s, *below* its own
`err_median` of 8.60. `sigma_log` clamped to its 0.25 floor, then the clip
collapsed the rest: 65% of every mock's per-star errors came out at exactly
7.81. Gaia's notebook applies no meaningful error cut at all -- its filter is
`pmrae/pmdece < 10 mas/yr` = 260 km/s, and `PM_QUALITY_CUT_KMS` appears there
only as a reference line on a plot. Measured over the 64537 stars in
[300, 1500) arcsec: median 8.33, p95 27.54, p99 40.35 -> `err_log_sigma` 0.727.

**HST**'s `err_cut` is genuinely right (the 0.3 mas/yr cut is applied; the
largest error in 610846 selected stars is 7.741 against the 7.81 cut) -- but
it sits at ~p100, not p95, so the back-derivation returned 0.999 against a
measured 0.434.

**The directions are not equivalent.** `rms_z` is bias over interval width.
Gaia's errors were too SMALL, which understates interval width and makes its
anchor conservative. HST's were too LARGE, which inflates the intervals and
made coarse cells look more acceptable than they are -- so **HST's 0.58 may
be too coarse**. Both `_CPS_ANCHORS` entries were measured with these errors
and are unearned until re-measured; only HST's can be wrong in a harmful way.

Fixes: `err_log_sigma` is now an explicit field (as 1D has always had), a
construction-time guard rejects `err_cut <= err_median`, and `from_data`
measures the spread instead of assuming. The estimator is p95/median, not
`std(log)`: the real distributions are heavier-tailed than log-normal and
`std(log)` fits the body while understating the tail (HST 0.287 vs 0.434,
giving a mock p95 of 2.42 against 3.14). The tail is what sets interval
inflation, so it is the part that must be reproduced.

### HST's minor-axis mean-velocity excess is the re-added rotation (CLOSED)

Linear gradient fit to the 1415 bin means, in the PA-aligned frame:

```
m1 (major axis):  dx=-0.0072  dy=+0.0296 km/s/arcsec   resid sd=1.26
m2 (minor axis):  dx=-0.0302  dy=-0.0080 km/s/arcsec   resid sd=1.23
```

Equal magnitude, orthogonal directions: a pure curl at ~0.030 km/s/arcsec,
with the divergence term ~10x smaller. The notebook re-adds rotation as
`d_pmra`/`d_pmdec` from the 1D **tangential** rotation curve, and a tangential
field has equal RMS in any two orthogonal Cartesian components by
construction. So `ptp2 ~= ptp1` (16.72 vs 16.05) is required, not anomalous.

The earlier framing -- "rigid rotation cannot produce that" -- was wrong. It
cannot for rotation about an in-plane axis; the re-added field is tangential
about the line of sight, which can and must. Alternatives ruled out: not edge
bins (inner 90% by radius: 15.95 / 16.72, essentially unchanged), not a bulk
residual (both axis means within 0.15 km/s of zero, and mean-subtraction
leaves the spread identical). `rotation_span = 16.7` stands. No action.

### Per-axis resolution (#6) is much smaller than projected

|  | axis ratio (median) | narrowest axis | declared `sigma_min` | ratio |
|---|---|---|---|---|
| HST | 0.894 (min 0.713) | 11.87 (p1) | 11.50 | **1.03** |
| Gaia | 0.767 (p16 0.586) | 5.67 (p5) | 7.03 | **0.81** |

`sigma_min` was already measured per-component across both axes, so it
already sits at or below the narrowest principal axis. **HST needs no
correction at all**; Gaia needs ~0.81, i.e. ~1.23x finer cells, not the 1.54x
projected from the assumed 0.65.

That 0.65 came from `truths_for`'s synthetic anisotropic truth (sy/sx = 0.65),
not from omega Cen -- real bins are much rounder. The test truth is *harder*
than the data, which is the right way round, and means the per-axis
correction is largely already paid for by validating against something more
anisotropic than reality.

**Estimator caveat**: Gaia's raw minimum minor axis is 0.91 km/s (ratio 0.13),
which would demand an absurd grid. It is noise -- Gaia's `err/sigma` is ~0.9,
so `cyy - <e_y^2>` is a difference of comparable numbers and a few bins
deconvolve to near zero. Use p5 (5.67). The same reasoning that makes full
`ptp` correct for `rotation_span` makes the raw min wrong here: that tail is
physical, this one is the estimator's.
