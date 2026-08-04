# Validation

This page summarises the statistical validation in the test suite for the 1D
(`KinematicSolver`) and 2D (`KinematicSolver2D`) solvers, and the 2D solver's
performance gate. It reports what these tests found, including a known
unresolved bias. See `PLAN.md` §1.2, §1.3, §3.3, and §3.4 for the full
methodology.

## What SBC validates

Simulation-Based Calibration (SBC; Talts et al. 2018) checks the sampler
against the model itself: draw a parameter from the model's own prior,
simulate synthetic data from it, run inference, and check that the true
parameter's rank among the posterior draws is uniformly distributed. This
is a test of implementation correctness — a wrong random-walk prior, an
off-by-half-bin design matrix, or a `numpyro.factor` term invisible to
`Predictive` all show up as non-uniform rank histograms. It does **not**
check whether the model describes real data well.

## What coverage validates

Frequentist coverage testing validates the *model against reality*, not
against itself. For a handful of fixed, physically motivated truths (not
drawn from the model's prior), we generate many independent mock datasets,
fit each, and check whether the true value falls inside the reported 68%
credible interval the stated fraction of the time. This test matters for a
downstream consumer such as Dynamite's NNLS $\chi^2$, which takes the
reported uncertainties literally. Coverage can fail even when SBC passes
cleanly, because SBC's "truth" is always self-consistent with the model's
prior. A smoothness prior that shrinks away genuine sharp features in real
data is invisible to SBC but shows up directly in coverage.

## 1D solver results

**SBC** (`tests/test_calibration.py`, `n_bins=15`, `n_stars=200`,
500 warmup + 1200 samples, `n_sims=30`): 6/6 test quantities pass the
Bonferroni-corrected KS uniformity test for **both** the RW1 and
Gaussian-core priors, with p-values in the range 0.52–0.97. The SBC harness
is now parametrised over ``SBC_PRIORS = ["rw1", "gaussian_core"]``
(see `tests/test_calibration.py`). This SBC run caught an earlier bug: the
original random-walk prior used `numpyro.factor` on an unconditioned base
measure. Because `numpyro.factor` is invisible to `Predictive`, the SBC
"truth" was silently drawn from the wrong distribution. The prior was
rewritten generatively; SBC then passed cleanly.

**Prior-predictive null-space test** (`tests/test_prior_predictive.py`):
the Gaussian-core prior's prior-predictive median velocity dispersion is
~45 km/s on a 400 km/s wide grid (vs. ~115 for uniform), confirming the
null space is quadratic, not flat. This is resolution-invariant to within
3% relative spread across n_bins = 20/40/80.

**Bias tests** (`tests/test_moment_bias.py`): for a Gaussian truth
(σ = 40 km/s, N = 150 stars, n_bins ∈ {20, 80}), the Gaussian-core prior
shows |kurtosis bias| < 0.35 and |σ bias| < 3%, and the σ bias does not grow
with bin count. The RW1 negative control still reproduces the known
+1.1 kurtosis and +4% σ bias at n_bins = 80. Note that the pre-fix version of
this result was uninformative for the non-Gaussian deviation: with the
deviation term inert (marginal SD ~0.0036), a posterior collapsed onto a
Gaussian trivially satisfied the Gaussian-truth bias thresholds.

**Coverage** (`tests/test_coverage.py`, `n_real=25` per truth, `n_stars=150`,
`n_bins=20`, four truths: Gaussian, Student-$t$ ($\nu=6$), skew-normal,
counter-rotating bimodal): parametrised over both priors. For the
**Gaussian-core prior** (without truncation) this test currently **fails**, and
also failed before the RW3 scaling fix — it is a standing known-red result, not
a regression, and is marked `xfail(strict=False)` so that a genuine improvement
shows up as an XPASS.

The Gaussian truth shows over-coverage in kurtosis (1.000, all 25/25 intervals
contain the truth) and skewness (0.960); the error bars are conservative but
valid. This row is not evidence about the deviation term either way — kurtosis
coverage was also 1.000 *before* the fix, because a posterior collapsed onto a
Gaussian covers a Gaussian truth perfectly.

The non-Gaussian truths still show under-coverage in kurtosis and tail_weight.
Measured pre-fix → post-fix: bimodal kurtosis 0.000 → 0.320 and bimodal
tail_weight 0.000 → 1.000 (both out of catastrophic failure); Student-$t$
kurtosis 0.000 → 0.040 (still below the 0.30 floor); skew-normal skewness and
kurtosis 0.000 → 0.000 (unchanged). The earlier attribution in this document to
"an inherent finite-data limitation, not the flat-null-space bug" is withdrawn:
that diagnosis was made with an inert deviation term and cannot be supported.
The cause of the residual under-coverage — the skew-normal case above all — is
an open question. Full table in
`docs/superpowers/plans/2026-08-03-rw3-measurements.md`. For the **RW1
prior**, `n_sigma_truncate=3.0` is applied (see `analysis.truncate_pdf_samples`);
the test remains marked `xfail` pending a better tail-handling approach for
heavy-tailed truths. See `PLAN.md` §1.3 for numbers.

**Non-Gaussian deviation prior** (`sigma3` in `generate_gaussian_core_curve`):
the deviation scale is standardised via the Sørbye–Rue generalised-variance
constant (Sørbye & Rue 2014, *Spatial Statistics* 8, 39–51), so `sigma3`
directly means the typical log-density departure from a Gaussian LOSVD,
independent of grid resolution. The prior on `sigma3` is a penalised-complexity
(PC) prior (Simpson et al. 2017, *Statistical Science* 32, 1): an Exponential
whose base model, `sigma3 = 0`, is an exactly Gaussian LOSVD. A prior-
predictive check confirms the PC prior makes non-Gaussian LOSVDs with
|excess kurtosis| p90 ≈ 1.36 reachable a priori.

**Default prior and regularisation**: the adopted configuration (2026-08-03,
from the validation campaign decision) is `SIGMA3_RATE=0.35` (Exp(0.35)) at
`rw_order=3`. At the science target (σ=22, n_real=100) this gives 41/45
coverage entries in the nominal band, 1 catastrophic, with v_mean efficiency
1.13× and sigma efficiency 1.35×. The decision record and full comparison
table are in `docs/superpowers/specs/2026-08-03-regularisation-decision.md`.

`KinematicSolver.run()` defaults to `prior="gaussian_core"`. Pass `prior="rw1"`
for the previous behaviour. The penalty order defaults to 3 and is available
as `rw_order=`.

## 2D solver results

**SBC** (`tests/test_calibration_2d.py`, `K=10` (100 cells), `n_stars=250`,
500 warmup + 1200 samples, `n_sims=30`): 6/6 test quantities (`mean_x`,
`mean_y`, `sigma_x`, `sigma_y`, `rho`, `smoothness_sigma`) pass; 0/30
simulations failed. The 2D model's GMRF prior was already implemented
generatively (`z ~ N(0, I)` plus a deterministic Cholesky-whitening
transform, never a bare `numpyro.factor` penalty) from the start, following
the lesson learned from the 1D SBC bug above. This was verified via
`test_prior_predictive_is_smooth_2d`, which checks that `Predictive` draws of
`intrinsic_pdf` are diffuse GMRF-like fields and not near-one-hot spikes.

**Recovery** (`tests/test_veldist2d.py`, slow tests): the acceptance
criterion for "2D minimally working" per the plan is recovering the full
covariance — not just the marginals — of a tilted bivariate Gaussian, since
marginal recovery alone would be equally well passed by a separable
(non-GMRF) prior. `test_recover_tilted_gaussian` fits mock data drawn from a
bivariate Gaussian with covariance $\mathrm{Var}(x)=\sigma_x^2$,
$\mathrm{Var}(y)=\sigma_y^2$, $\mathrm{Cov}(x,y)=\rho\sigma_x\sigma_y$,
using $\sigma_x=8$, $\sigma_y=6$, $\rho=0.6$, and recovers all three
independent covariance components within the posterior half-68CI
(5$\sigma$ tolerance). `test_recover_isotropic_gaussian` is the $\rho=0$
control. `test_2d_marginal_matches_1d` cross-checks the 2D solver's
$\hat v_1$ marginal against a direct 1D fit of the same data — both pass.
One finding: these tests needed `n_stars=2000` rather than the plan-suggested
200–400, because at lower star counts the GMRF prior induces a measurable
finite-sample shrinkage bias in the recovered variance (the same underlying
mechanism as the 1D kurtosis bias above — many free grid cells relative to
star count). This was fixed by raising $N$, a genuine mitigation, rather than
by loosening the tolerance.

**Performance gate** (`PLAN.md` §3.4): the plan defines an explicit,
measurable gate before considering any SVI/Pathfinder escalation. Run
`K=20` (400 cells), `N=5000` mock stars, 500 warmup + 1000 samples on CPU
with 4 chains, and proceed with plain NUTS if wall time < 10 min, minimum
ESS/`n_samples` > 0.1, and maximum $\hat R$ < 1.01. Measured:

| Criterion | Threshold | Measured | Verdict |
|---|---|---|---|
| Wall time | < 600 s | 87.9 s | PASS |
| min(ESS)/n_samples | > 0.1 | 3.11 | PASS |
| max($\hat R$) | < 1.01 | 1.0023 | PASS |

All three criteria pass with wide margin (ESS and $\hat R$ were checked
across `smoothness_sigma`, the latent `z` vector, and `intrinsic_pdf`, not
just the cheapest scalar; `intrinsic_pdf` was the binding constraint on both
ESS and $\hat R$). All three criteria pass. Per the plan, **no escalation is
warranted**; the SVI/Pathfinder ladder (K reduction, `dense_mass=True`,
GPU, Pathfinder-for-init, full SVI) was not built, as the plan instructs. Full numbers and the
reproduction procedure are recorded in `PLAN.md` §3.4 "Gate result
(measured)".

## How to reproduce

The slow tests below run actual NUTS sampling and take from tens of seconds
to several minutes each; they are excluded from the default fast test run
(`pytest tests/ -v --tb=short -m "not slow"`).

```bash
# 1D SBC
pytest tests/test_calibration.py -m slow -v

# 1D coverage (currently xfail on kurtosis, see above)
pytest tests/test_coverage.py -m slow -v

# 2D SBC
pytest tests/test_calibration_2d.py -m slow -v

# 2D recovery tests (tilted/isotropic Gaussian, 1D/2D marginal consistency)
pytest tests/test_veldist2d.py -m slow -v
```

The §3.4 performance gate is a one-off measurement, not a pytest test (it
calls `numpyro.infer.MCMC`/`NUTS` directly on `model_2d` with `num_chains=4`,
which `KinematicSolver2D.run()` does not currently expose). See `PLAN.md`
§3.4 for the exact procedure.
