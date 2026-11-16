# Fit statistic for DYNAMITE: design sketch (2026-11-16, in progress)

How DYNAMITE should score orbit-model PM histograms against veldist output.
**Baseline = addback mode + DYNAMITE's diagonal χ² on veldist's floored
medians/σ.** Every option is judged against it on accuracy, bias, calibration
and solve time.

Scripts: `experiments/fitstat/` (`fitstat.py` runs on astro-node-12 in
`veldist-fitstat/`). Synthetic aperture = 430 stars, per-axis err 1.5 km/s,
production HST grid (K=23, ±82.8 km/s).

## Facts established (code and data reads, 2026-11-16)

- **DYNAMITE** (`origin/master`; the node may differ): rows and data are
  divided by σ before the Gram build; `normalise()` scales `PM_2dhist` and σ
  together; the data are multiplied by the MGE projected mass per aperture.
- **Production PM solve is a QP** (`nnls_solver: admm`, `gram_blockwise`):
  takes a Gram `P` and a linear `q`, so linear terms are allowed.
- **Orbit PM histograms pile out-of-range velocities into the border cells**
  (Fortran `histogram_velbin`); Python zeroes edges for 1D only. veldist's
  design matrix has closed border cells and drops off-grid stars → a
  data/model mismatch in the border ring.
- **ml rescaling** regrids with a not-a-knot spline of the CDF; mass outside
  the shared window is dropped, and an overflow spike can give negative cells.
- **Floor**: 62.5% of HST and 71.9% of Gaia cells sit at 1% of the
  aperture's maximum half-width (production pack, verified).
- **Kernels are narrow**: median HST per-axis error 1.5 km/s vs 7.2 km/s
  cells, so each star's kernel covers 1–2 cells and the per-star likelihood is
  close to a multinomial on cell counts.
- **Fast stars are cut upstream**: of 24 stars within 3″ faster than 60 km/s,
  2 pass `selection_hq_astrometry_and_membership`; no HQ star within 30″ is off
  the grid. The HST histograms constrain an IMBH through the dispersion
  profile, not through the fast stars.
- **HST is fitted in relative mode** in the newer configs (`relative: True`
  plus oMEGACat II rotation rows), and addback is the control; the user's
  current default is addback.

## Options

| | statistic | DYNAMITE change | veldist change |
|---|---|---|---|
| A | diag χ² on median/floored σ (**baseline**) | none | none |
| B | full-covariance χ² | whiten each aperture block by L⁻¹ | keep draws, write Ledoit-Wolf C (529² per aperture) |
| C | Laplace: quadratic expansion of the per-star −2 log L | P += Aᵀ F A, q += Aᵀ(F h₀ − g) per aperture | write F, g, h₀ per aperture (analytic from M) |
| D | IRLS: C repeated, re-expanded about the current model | C plus an outer loop, with a Gram rebuild per round | design matrices |
| E | exact per-star (EM / mirror descent) | new solver; A resident in RAM | design matrices |

Addback and relative are a separate axis: relative = projection with infinite
variance on the per-aperture shift, addback = zero variance, and addback with
a covariance = Woodbury with a finite Σ_m. Any of B–E works in either mode;
in relative mode the projection is applied in the F- (or C-) whitened space.

## Results so far (synthetic; laptop pilot)

σ fit, 8 seeds, mean known (addback setting):

| statistic | bias | scatter | claimed σ | scatter/claimed |
|---|---|---|---|---|
| A diag floored | −0.25 | 0.33 | 0.066 | **5.1** |
| A diag, no floor | −0.38 | 0.43 | 0.031 | 13.8 |
| B full-cov | −0.24 | 0.30 | 0.36 | 0.8 |
| E exact | −0.39 | 0.33 | 0.42 | 0.8 |
| C Laplace | −0.52 | 0.29 | 0.43 | 0.7 |

(The 8 realisations are themselves −0.18 low, with a standard error of about 0.12.)

Penalty per aperture for 1% of mass in a 60 km/s ring that the data don't
have: A 700–1300, B 77–174, E 2–3.5, C 5.5–7.3. The posterior covariance is
mostly prior: in occupied cells the likelihood gives about 2% of the
posterior precision, and the posterior σ is about 0.3× Poisson.

Tail **recovery** (1 seed): with 1% truth at 90 km/s, beyond the grid, only
the exact per-star statistic with open border cells recovers f (0.009 ±
0.005). C expanded about the posterior mean gives f = 0: tail stars have
p₀ ≈ 0, and one quadratic step can at most double p. A defensive expansion
point (0.99·mean + 0.01·uniform) gives 0.005. For an in-grid tail at 60 km/s,
C matches E.

## Current lean

- **A is miscalibrated by about 5× in σ** and overweights tails by orders of
  magnitude. B fixes the calibration but encodes the prior (tails about 40×).
- **C is the cheapest calibrated option**: one Gram build, the same number of
  rows as now (F is 529×529 per aperture and nearly banded), plus a linear
  term the admm QP already accepts. Its weak spot is the tails beyond the grid.
- **D** (2–5 rounds) fixes C's tails at 2–5× the solve cost, for central
  apertures only if it can be restricted to them.
- **Any per-star option needs open border cells** to match DYNAMITE's
  histograms.

## Implementation sketch for C

veldist (`write_dynamite_kinematics_2d`, optional `laplace=True`):

1. Build M with **open border cells** (integrate the kernel to ±∞ in the
   border columns) and keep the off-grid stars.
2. h₀ = 0.99·(posterior mean) + 0.01·uniform; p₀ = M h₀.
3. Per aperture: g = N·1 − Mᵀ(1/p₀) (the multinomial normalisation folded
   into the gradient), F = Mᵀ diag(1/p₀²) M; write F (or its Cholesky factor),
   g and h₀ to the npz (about 1.1 MB per aperture in float32; 1.6 GB for HST).
4. Unit test: for a Gaussian truth, the quadratic reproduces the exact
   −2ΔlogL to within a few per cent over ±10% in σ.

DYNAMITE (`ProperMotions`, new `statistic: laplace`):

1. `normalise()` scales h₀ by `hist_scale` and F, g by the inverse
   (likelihood invariance).
2. `_prepare_kinematic_block`: instead of `/ econ`, whiten the block by
   Lᵀ (F = L Lᵀ) and set the data to Lᵀ(h₀ − F⁻¹g) where F is regular. Empty
   cells (F ≈ 0) go into `q` as the linear penalty `Aᵀ g`, not as rows.
3. Relative mode: `_relative_pm_block` projects in the L-whitened space (G̃ = Lᵀ G).
4. χ² bookkeeping, plots and `kinchi2` report −2ΔlogL for this set.
5. Test: with F = diag(1/σ²) and g = 0 it must reproduce the baseline bit for bit.

## Open

- Full-scale phase-1 results (node): calibration across N, the addback mean
  error, tails, outliers, edges, `laplace_mix`.
- Phase 2 (decision gate): rescore finished v4 models with A and C along the
  IMBH mass profile.
- Time one production solve (Gram build vs admm iterations).
- Where to pin the expansion point h₀ (posterior mean vs pass-0 model).
