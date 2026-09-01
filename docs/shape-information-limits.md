# What sets our ability to recover h3 and h4

Written 2026-09-01 after three successive misreadings of the same sweep. The
first two blamed the estimator (one was a real bug, fixed) and the third
blamed "the prior" without establishing that the shrinkage was irreducible.
The question is analytic and did not need any of those sweeps.

## The result

Write the intrinsic LOSVD as a Gauss-Hermite series about a Gaussian of
dispersion `s`:

```
f(v) = N(v; 0, s) [ 1 + h3 He_3(v/s) + h4 He_4(v/s) + ... ]
```

Convolution with a Gaussian measurement error of width `e` is **exact** in
this basis. By Rodrigues' formula `N(0,s) He_n(v/s) = (-s)^n d^n/dv^n
N(v;0,s)`, and convolution commutes with differentiation, so with
`S^2 = s^2 + e^2`:

```
[N(0,s) He_n(./s)] * N(0,e) = (s/S)^n N(0,S) He_n(./S)
```

The n-th Hermite coefficient is attenuated by `A_n = (1+r^2)^(-n/2)` with
`r = e/s`. Verified numerically to five decimals in
`scratchpad/gh_information.py`.

The Fisher information about `h_n` from `N` stars follows from the
orthogonality `int N(0,S) He_n(w)^2 dv = n!`:

```
I_nn = N A_n^2 n!
```

The Gauss-Hermite basis is **Fisher-orthogonal** -- the information matrix is
diagonal in `(h3, h4, ...)` -- so estimating the other coefficients costs
nothing at leading order. This is the property van der Marel & Franx (1993)
designed the basis for. Converting to the standardised cumulants
`compute_summary` reports (`gamma1 = 6 h3`, `gamma2 = 24 h4`, from
`w^3 = He_3 + 3He_1` and `w^4 = He_4 + 6He_2 + 3`):

```
sd(skewness)        = sqrt( 6/N) * (1 + r^2)^(3/2)
sd(excess kurtosis) = sqrt(24/N) * (1 + r^2)^2
```

At `r = 0` these reduce to the textbook `sqrt(6/N)` and `sqrt(24/N)`.

**The deconvolution exponent is the moment order.** That is the whole story:
measurement errors cost the dispersion nothing at first order, skewness the
3/2 power of `(1+r^2)`, and kurtosis the square. It is also why h4 is always
about twice as expensive as h3 (`sqrt(24)` vs `sqrt(6)`) -- a theorem, not a
property of our prior.

## What it says about our datasets

Smallest |h3| and |h4| detectable at 3 sigma, at each profile's actual
occupancy:

| dataset | err/sigma | N | min h3 | min h4 |
|---|---|---|---|---|
| HST | 0.13 | 426 | **0.36** | 0.74 |
| MUSE (sigma_max) | 0.20 | 152 | 0.64 | 1.31 |
| MUSE (sigma_min) | 0.386 | 152 | **0.73** | 1.63 |
| Gaia (sigma_min) | 1.22 | 435 | **1.39** | 4.39 |

Stars needed for a 3-sigma detection of the test truths:

| dataset | err/sigma | N for h3=0.454 | N for h4=1.0 | N for h4=0.305 |
|---|---|---|---|---|
| HST | 0.130 | 276 | 231 | 2483 |
| MUSE (sigma_max) | 0.200 | 295 | 253 | 2717 |
| MUSE (sigma_min) | 0.386 | 398 | 377 | 4047 |
| Gaia | 1.220 | 4037 | 8282 | 89030 |

**The ordering is HST > MUSE > Gaia, and it is dataset-dependent:**

* **HST** (r=0.13, N=426) is the only dataset that can do shape work.
  Threshold h3 ~ 0.36. Never tested for h3/h4 -- every shape conclusion in
  this repo so far comes from MUSE.
* **MUSE** (r=0.386, N=152) cannot measure h3 below 0.73. Its binding
  constraint is **occupancy, not errors**: at 152 stars/bin it is 2.6x short
  of what its own test truth needs. Larger spatial bins would buy it.
* **Gaia** (r=1.22) needs ~4000 stars/bin for h3 and ~89000 for a realistic
  h4. Not achievable at any occupancy.

## Why the earlier sweeps misled

MUSE's test truth has h3 = 0.454 against a 3-sigma threshold of 0.73. It is a
**1.85 sigma signal**. Everything reported as a "failure" is a
correctly-behaving Bayesian estimator shrinking a sub-2-sigma signal toward
its prior's base model.

The flat-in-`ivar` result that misled two separate readings was measured over
`ivar` 0.8 -> 1.6 at sigma_min. That is a factor 2 in N and therefore only
**sqrt(2) in signal-to-noise**, moving h3 from 1.5 to 2.1 sigma. Both points
sit deep inside the regime where any sensible prior dominates. The curve was
flat because the sweep never left that regime, not because the shrinkage is
irreducible.

Two corroborating checks:

* The measured posterior interval on h3 at the operating point (0.209) is
  NARROWER than the Cramer-Rao bound (0.254). That is the expected signature
  of prior shrinkage, and `RecoveryCurve.efficiency`'s own docstring warns
  about it: "<1 is not better than optimal: it means the prior is shrinking
  estimates."
* h4 was shrunk harder than h3 in every measured row. That follows from the
  formulas above rather than from anything about `gaussian_core`.

This also vindicates the `gaussian_core` design rather than indicting it. Its
PC prior has a Gaussian base model, and shrinking a 1.85-sigma signal to ~26%
of its value is close to what a correctly calibrated Bayesian estimator
should do.

## Caveats on quoting these numbers

1. These are thresholds for the **strong** test truths (h3 = 0.454). A
   realistic omega Cen h3 of ~0.1 needs ~20x more stars, so the honest
   headline may be that nobody measures h3 per-bin at these occupancies.
2. This is the parametric Cramer-Rao bound for a **2-parameter** Gauss-Hermite
   model. Our nonparametric fit has 22-64 free bins and can only do worse, so
   these are **optimistic floors**, not predictions of our pipeline.
3. `r` is taken at `err_median`; the real error distributions are
   heavy-tailed (`err_log_sigma` 0.43-0.73), and `(1+r^2)^2` is convex, so
   averaging over the real distribution makes h4 worse than the median-`r`
   figure suggests.

Literature scale for comparison: Amorisco & Evans (2012), "Line Profiles from
Discrete Kinematic Data" (arXiv:1204.5181), work at N ~ 800 per bin for dwarf
spheroidals and report their maximum-likelihood method beating a Gauss-Hermite
expansion by a factor ~2 in variance at that sample size.

## The test plan this implies

The analytics make a **falsifiable quantitative prediction**, which turns the
remaining sweeps from fishing into confirmation.

**Test A -- information scan (primary).** MUSE, sigma_min, `ivar` in
{1.145, 2.3, 4.6, 9.2, 18.4}, i.e. S/N on h3 from 1.85 to 7.4. Truths
`skew_normal_h3`, `student_t_h4`, `rotating_tangential`.
*Prediction:* recovery fraction tracks the S/N and reaches ~0.9 between
`ivar` 4.6 and 9.2. If instead it stays flat out to 16x the operating
information, the prior IS stiff and `SIGMA3_RATE` needs revisiting.

**Test B -- regularisation strength (conditional on A).** `SIGMA3_RATE` is a
hand-set module constant (0.35). Re-run the operating point and one
high-`ivar` point at rate x0.1 / x1 / x10. Only worth running if A shows the
shrinkage exceeding what the S/N justifies.

**Test C -- HST shape recovery.** The analytics say HST is the one dataset
that can do this, and it has never been tested. Highest value per fit.

Run order: **C, then A, then B if needed.** C answers the policy question
("which datasets get shape measurements reported?"), A validates the model
against the theory, B is contingent.
