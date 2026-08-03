# Validation

This page summarises the statistical validation currently in the test suite
for both the 1D (`KinematicSolver`) and 2D (`KinematicSolver2D`) solvers, and
the 2D solver's performance gate. It reports what these tests found — including
a known, currently-unresolved bias — rather than only the results that look
good. See `PLAN.md` §1.2, §1.3, §3.3, and §3.4 for the full methodology and
the reasoning behind each design choice.

## What SBC validates

Simulation-Based Calibration (SBC; Talts et al. 2018) checks the sampler
against the model itself: draw a parameter from the model's own prior,
simulate synthetic data from it, run inference, and check that the true
parameter's rank among the posterior draws is uniformly distributed. This
is a test of implementation correctness — a wrong random-walk prior, an
off-by-half-bin design matrix, or a `numpyro.factor` term invisible to
`Predictive` all show up as non-uniform rank histograms. It does **not**
check whether the model is a good description of real data.

## What coverage validates

Frequentist coverage testing validates the *model against reality*, not
against itself. For a handful of fixed, physically-motivated truths (not
drawn from the model's own prior), generate many independent mock datasets,
fit each, and check whether the true value falls inside the reported 68%
credible interval the stated fraction of the time. This is the test that
matters for a downstream consumer such as Dynamite's NNLS $\chi^2$, which
takes the reported uncertainties literally. Coverage can fail even when SBC
passes cleanly, because SBC's "truth" is always self-consistent with the
model's prior — a smoothness prior that shrinks away genuine sharp features
in real data is invisible to SBC but shows up directly in coverage.

## 1D solver results

**SBC** (`tests/test_calibration.py`, `n_bins=15`, `n_stars=200`,
500 warmup + 1200 samples, `n_sims=30`): 6/6 test quantities
(`v_mean`, `sigma`, `skewness`, `kurtosis`, `tail_weight`, `smoothness_sigma`)
pass the Bonferroni-corrected KS uniformity test, with reported p-values in
the range 0.52–0.97. This SBC run is what caught and led to fixing a real
bug: the original random-walk prior was implemented via `numpyro.factor` on
an unconditioned base measure, which is invisible to `Predictive` — the SBC
"truth" was silently drawn from the wrong distribution. The prior was
rewritten generatively (explicit `steps` sample sites, cumsum + mean-centre),
after which SBC passed cleanly.

**Coverage** (`tests/test_coverage.py`, `n_real=25` per truth, `n_stars=150`,
`n_bins=20`, four truths: Gaussian, Student-$t$ ($\nu=6$), skew-normal,
counter-rotating bimodal): coverage is currently within the acceptable band
for `v_mean`, `sigma`, and `skewness` across all four truths. **`kurtosis`
has a known, quantified bias, partially mitigated** — the test is marked
`@pytest.mark.xfail(strict=False, ...)` in the repository as of this
writing (not fully resolved). Root cause: tail leakage from the RW1
smoothness prior — the fitted posterior carries excess mass in the
outermost grid bins ($\sim 5\sigma$ out), and kurtosis's fourth-power
weighting amplifies that small residual by a factor of $\sim 5^4 = 625$,
producing roughly +1.6 to +2.5 excess-kurtosis bias even for a plain
Gaussian truth (whose true excess kurtosis is 0). `compute_summary()` now
exposes an opt-in `n_sigma_truncate` parameter (see
`veldist.analysis.truncate_pdf_samples`) — a per-draw analogue of
`KinematicSolver.truncate_losvd()`, but applied to the raw posterior samples
`compute_summary` consumes rather than only the Dynamite export path.
Empirically, `n_sigma_truncate=3.0` fixes the bias for the Gaussian and
mildly-skewed truths (kurtosis coverage 0/25 → 21/25 and 0/25 → 20/25
respectively), but does **not** fix it for heavy-tailed or bimodal truths
(Student-$t$, counter-rotating bimodal remain near 0/25) — a fixed-$\sigma$
truncation cut necessarily trades away genuine tail mass along with leaked
mass for those cases, so it is not a general solution and is not applied by
default. This is tracked as an open item in `TASKS.md`. If a future fix
resolves the remaining cases, the test will show as an unexpected pass
(XPASS), which is the signal to remove the marker.

## 2D solver results

**SBC** (`tests/test_calibration_2d.py`, `K=10` (100 cells), `n_stars=250`,
500 warmup + 1200 samples, `n_sims=30`): 6/6 test quantities (`mean_x`,
`mean_y`, `sigma_x`, `sigma_y`, `rho`, `smoothness_sigma`) pass, 0/30
simulations failed. The 2D model's GMRF prior was already implemented
generatively (`z ~ N(0, I)` plus a deterministic Cholesky-whitening
transform, never a bare `numpyro.factor` penalty) from the start, following
the lesson learned from the 1D SBC bug above — this was verified rather than
assumed, via `test_prior_predictive_is_smooth_2d`, which checks that
`Predictive` draws of `intrinsic_pdf` are diffuse GMRF-like fields and not
near-one-hot spikes.

**Recovery** (`tests/test_veldist2d.py`, slow tests): the acceptance
criterion for "2D minimally working" per the plan is recovering the full
covariance — not just the marginals — of a tilted bivariate Gaussian, since
marginal recovery alone would be equally well passed by a separable
(non-GMRF) prior. `test_recover_tilted_gaussian` fits mock data drawn from a
bivariate Gaussian with covariance $\mathrm{Var}(x)=\sigma_x^2$,
$\mathrm{Var}(y)=\sigma_y^2$, $\mathrm{Cov}(x,y)=\rho\sigma_x\sigma_y$,
using $\sigma_x=8$, $\sigma_y=6$, $\rho=0.6$, and recovers all three
independent covariance
components within the posterior half-68CI (5$\sigma$ tolerance).
`test_recover_isotropic_gaussian` is the $\rho=0$ control, and
`test_2d_marginal_matches_1d` cross-checks the 2D solver's $\hat v_1$
marginal against a direct 1D fit of the same data — both pass. One finding
along the way: these tests needed `n_stars=2000` rather than the
plan-suggested 200–400, because at lower star counts the GMRF prior induces
a measurable finite-sample shrinkage bias in the recovered variance (the
same underlying mechanism as the 1D kurtosis bias above — many free grid
cells relative to star count). This was fixed by raising $N$, a genuine
mitigation, rather than by loosening the tolerance.

**Performance gate** (`PLAN.md` §3.4): the plan defines an explicit,
measurable gate before considering any SVI/Pathfinder escalation — run
`K=20` (400 cells), `N=5000` mock stars, 500 warmup + 1000 samples on CPU
with 4 chains, and proceed with plain NUTS if wall time < 10 min, minimum
ESS/`n_samples` > 0.1, and maximum $\hat R$ < 1.01. Measured directly this
session:

| Criterion | Threshold | Measured | Verdict |
|---|---|---|---|
| Wall time | < 600 s | 87.9 s | PASS |
| min(ESS)/n_samples | > 0.1 | 3.11 | PASS |
| max($\hat R$) | < 1.01 | 1.0023 | PASS |

All three criteria pass with wide margin (ESS and $\hat R$ were checked
across `smoothness_sigma`, the latent `z` vector, and `intrinsic_pdf`, not
just the cheapest scalar; `intrinsic_pdf` was the binding constraint on both
ESS and $\hat R$). Per the plan, the gate passing means **no escalation is
warranted** — the SVI/Pathfinder ladder (K reduction, `dense_mass=True`,
GPU, Pathfinder-for-init, full SVI) was not built, consistent with the
plan's instruction not to build it speculatively. Full numbers and the
reproduction procedure are recorded in `PLAN.md` §3.4, "Gate result
(measured)".

## How to reproduce

All of the slow tests below run actual NUTS sampling and take from tens of
seconds to several minutes each; they are excluded from the default fast
test run (`pytest tests/ -v --tb=short -m "not slow"`).

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
which `KinematicSolver2D.run()` does not currently expose) — the procedure
and exact configuration to reproduce it are recorded in `PLAN.md` §3.4.
