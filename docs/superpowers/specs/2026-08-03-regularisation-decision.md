# Regularisation Configuration Decision

> **SUPERSEDED 2026-08-04.** `Exp(0.35)` was adopted on coverage evidence
> alone and fails simulation-based calibration (5/30 against a 2% budget).
> The adopted rate is now **`Exp(5.0)`** — see "2026-08-04 revision" at the
> end of this document. The campaign results below remain valid as measured;
> it is the decision drawn from them that changed.

**Date:** 2026-08-03
**Branch:** `fix/rw3-deviation-scaling`

## Candidates tested

Two candidates from the regularisation sweep, both `rw_order=3` (the penalty
order hypothesis was not supported — see the null-space measurements doc):

1. **Exp(0.35)** — `SIGMA3_RATE = 0.35`, 6.6× looser than current default
2. **Exp(0.20)** — `SIGMA3_RATE = 0.20`, 11.5× looser than current default

## Campaign results (n_real=100)

### σ=22 (the science target — 37 bins, 5.0 km/s, err/σ=0.11)

| Metric | Exp(0.35) | Exp(0.20) |
|---|---|---|
| in-band /45 | **41/45** | **41/45** |
| catastrophic | 1 | 1 |
| v_mean efficiency | 1.13× | **1.04×** |
| sigma efficiency | 1.35× | **1.32×** |
| h3 coverage (skew_normal) | 0.400 | **0.430** |
| h4 coverage (skew_normal) | 0.560 | **0.580** |

### σ=7 (narrow dispersion — 37 bins, 30% informative)

| Metric | Exp(0.35) | Exp(0.20) |
|---|---|---|
| in-band /45 | 23/45 | 24/45 |
| catastrophic | 13 | 14 |
| h3 coverage (skew_normal) | 0.200 | 0.210 |

Coverage is essentially identical at σ=7, confirming the problem is structural
(empty grid), not a tuning issue.

**Ignore the wall-clock runtimes in the campaign log.** The machine slept
mid-run, so the 32590 s recorded for Exp(0.20) at σ=7 is elapsed time, not
sampler time, and none of the four timings are comparable. If sampler cost ever
becomes a tiebreaker between priors, measure it with num_steps / divergences per
chain rather than wall clock.

Raw campaign output: `2026-08-03-regularisation-campaign-log.txt` (this
directory). Per-realisation JSON was written to `/tmp/campaign_results/` and is
not preserved; regenerate with the campaign driver in the validation-campaign
plan (n_real=100, same seeds) if the per-truth detail is needed again.

## Decision

**Adopt `SIGMA3_RATE = 0.35` (`Exp(0.35)`).**

Rationale:
- Both candidates are effectively tied at σ=22 (41/45 in-band, 1 catastrophic).
  Exp(0.20) has marginally better h3 coverage (0.430 vs 0.400) and efficiency
  (1.04 vs 1.13), but these are within binomial noise at n_real=100.
- σ=7 does not separate them either (23/45 vs 24/45, within binomial noise).
  With nothing measured favouring the looser prior, the tiebreaker is the
  default principle: prefer the prior that shrinks harder toward the base
  model. Exp(0.20) has not been shown to buy anything.
- Exp(0.35) is close to the sweeps' leading candidate and was measured at
  n_real=100 rather than the sweep's n_real=25, so the 41/45 number is more
  reliable.

## What was given up

1. **h3/h4 at σ=7 remain unrecoverable.** The coverage is still catastrophic
   (13/45 entries, h3 at 0.200). The cause is the empty-grid problem diagnosed
   in Task 7: only 30% of bins carry mass. **Per-bin fitting on a matched grid
   is the remedy**, and would be its own implementation plan.
2. **h3/h4 at σ=22 are improved but not at nominal coverage.** At 0.400/0.560
   for skew_normal_h3's skewness/kurtosis, h3 is still below the [0.44, 0.92]
   band. The finite-sample achievable skewness at N=150 is 0.43 against a true
   0.45, so this is partly a fundamental data limit, not just model shrinkage.
3. **More degenerate prior draws.** The looser prior raises the fraction of
   prior-predictive draws that put all their mass in one bin, so
   `test_gaussian_core_prior_is_not_uniform` had its ceiling moved from 0.25 to
   0.40. Prior draws are not posterior draws, so this is acceptable, but it is
   the direct cost of the loosening and the number to watch if the rate is
   loosened further.
4. **The penalty order lever does not work as hypothesised.** The softmax
   nonlinearity decouples the log-density polynomial null space from the PDF
   moments, so raising rw_order to 4 or 5 does not free h3/h4. The next option
   is a mode-order split scale (two sigma3 parameters).

## Current default changed

- `SIGMA3_RATE` set to `0.35` in `src/veldist/veldist.py`
- `rw_order` remains 3 (unchanged)


---

## 2026-08-04 revision: adopt `Exp(5.0)`

**What changed:** the acceptance criterion was made explicit, and SBC was
finally run over the rate (campaign Task 6, never executed at the time).

**Acceptance criterion (Peter):** minimally correct = recover `v_mean` and
`sigma` well, with properly calibrated uncertainties. Recovering h3/h4 well is
optional. That resolves the trade this document agonised over: calibration
wins, h3/h4 coverage is expendable.

**SBC over the rate** (n_sims=30, 2% budget, JAX x64 to match `conftest.py`):

| rate | failed | verdict |
|---|---|---|
| 0.35 | 5/30 (16.7%) | ✗ |
| 1.0 | 1/30 (3.3%) | ✗ |
| 2.303 | 2/30 (6.7%) | ✗ |
| **5.0** | **0/30** | ✓ |
| 10.0 | 0/30 | ✓ |

Every failure is low ESS on the `sigma3` site (ESS 3.4–18.5 against a threshold
of 20) — a funnel: a loose prior lets `sigma3` approach zero and the sampler
cannot traverse the neck. No NaN posteriors, no sampler exceptions.

**Tightening is free for `v_mean` and `sigma`** (n_real=50, oMEGACat profile,
N=150, σ=22). Across rates 0.35 → 1.0 → 5.0, coverage, efficiency, and bias are
flat, on the Gaussian truth *and* on non-Gaussian truths (`skew_normal_h3`,
`flat_top_tangential`, `student_t_h4`). Representative rows, rate 0.35 → 5.0:

| truth | metric | coverage | efficiency (std) |
|---|---|---|---|
| gaussian | v_mean | 0.70 → 0.70 | 1.09 → 1.08 |
| gaussian | sigma | 0.78 → 0.66 | 0.99 → 0.98 |
| skew_normal_h3 | sigma | 0.64 → 0.60 | 1.07 → 1.07 |
| flat_top_tangential | sigma | 0.78 → 0.82 | 0.68 → 0.68 |
| student_t_h4 | sigma | 0.72 → 0.72 | 1.03 → 1.03 |

Note the Gaussian truth alone cannot answer this — it lies exactly in the
prior's null space, so tightening cannot hurt it by construction. The
non-Gaussian truths are what make the result meaningful.

**The old "tight priors cost 2.7–4.3× σ efficiency" claim does not replicate.**
It was the std of 25 medians on a different grid; with n_real=50 and a robust
scatter estimator there is no penalty at any rate. Do not carry it forward.

**What is given up:** h3/h4, now formally out of scope. Also note
`test_gaussian_core_prior_spans_nongaussian_shapes` is a bracket, not a pin —
it passes for every rate from 0.35 to 50+, so it cannot select a rate. SBC does.
