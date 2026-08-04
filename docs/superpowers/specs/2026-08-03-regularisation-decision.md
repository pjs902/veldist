# Regularisation Configuration Decision

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

Both are essentially identical at σ=7, confirming the problem is structural
(empty grid), not a tuning issue.

## Decision

**Adopt `SIGMA3_RATE = 0.35` (`Exp(0.35)`).**

Rationale:
- Both candidates are effectively tied at σ=22 (41/45 in-band, 1 catastrophic).
  Exp(0.20) has marginally better h3 coverage (0.430 vs 0.400) and efficiency
  (1.04 vs 1.13), but these are within binomial noise at n_real=100.
- Exp(0.20) at σ=7 may be worse than Exp(0.35) since a looser prior in an
  already grid-starved regime gives NUTS more dimensions to wander in — the
  matched-grid diagnosis (Task 7) showed the σ=7 problem is structural, not a
  tuning problem, so the more conservative prior is safer.
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
3. **The penalty order lever does not work as hypothesised.** The softmax
   nonlinearity decouples the log-density polynomial null space from the PDF
   moments, so raising rw_order to 4 or 5 does not free h3/h4. The next option
   is a mode-order split scale (two sigma3 parameters).

## Current default changed

- `SIGMA3_RATE` set to `0.35` in `src/veldist/veldist.py`
- `rw_order` remains 3 (unchanged)
