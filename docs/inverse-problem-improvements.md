# What the inverse-problem survey actually buys us

Written 2026-09-14, after the literature survey in
`context/tomography_inverse_problems.md` and a careful read of the current model.
This is the "so what" document: what to build, in what order, what not to build.

Organised as **three independent tracks**. They share one piece of machinery
(the averaging kernel, Track 1 item 1) and otherwise do not block each other.

| track | scope | headline |
|---|---|---|
| **1** | the method as it stands, 1D and 2D | one possible live systematic; several cheap diagnostics |
| **2** | spatial regularisation across Voronoi bins | motivation is **occupancy**, not accuracy; couple the null space only |
| **3** | joint LOS + PM 3D retrieval | the only place the tomography framing is literally correct |

Two framing corrections carried in from the survey, because they change what is
worth doing:

- The default prior is `gaussian_core`, not RW1. Gaussians are already exactly
  unpenalised, and it is already Merritt's (1997) Silverman penalty in scaled
  generative GMRF form. Anything justified by "veldist relaxes toward a flat
  curve" is void.
- The companion paper's spatial-prior win (`2026A&A...710A.135J`) is **0.1–0.3%**
  in recovery error, not the "significant" improvement its abstract implies. The
  case for Track 2 has to be made on other grounds — and it can be.

---

# Track 1 — improvements to the method as it stands

No new dimensions, no spatial coupling. This track is where the only
bug-risk item lives, so it goes first regardless of interest in the others.

## 1.1 Uniform-resolution audit (P1, bug-risk)

**Fessler & Rogers 1996** (*IEEE TIP* 5(9), 1346–1358) prove that a penalised
estimator with a *spatially invariant* penalty has a *spatially varying* local
impulse response, because the Fisher weighting varies with the local data. The
exact form, from Fessler's own textbook (`c-srp.pdf` §22.4.3.1, eq. 22.4.16),
exact for a linear model with quadratic penalty:

```
l̄⁽ʲ⁾ = [F + R̈]⁻¹ F e_j ,    F = MᵀWM
```

and the cleanest statement of the mechanism (§22.4.3.2): for white noise and a
quadratic penalty, `l̄⁽ʲ⁾ = [MᵀM + σ²βR]⁻¹MᵀM e_j`, so **"the regularization
parameter β effectively is scaled by the noise variance σ²."**

Our Voronoi bins span an order of magnitude in star count. If effective velocity
resolution varies with occupancy, then **any measured radial trend is partly a
resolution gradient rather than physics** — which would affect results already
shipped.

**Measured 2026-09-14 (main session, `scratchpad/track1_resolution.py`).**
Sign established empirically, and it is the *opposite* of Stayman & Fessler's
emission-tomography direction: **low-occupancy bins over-smooth** (prior
dominates), high-occupancy bins resolve. Spike truth concentration:
0.933 (N=10) → 0.993 → 0.999 → 1.000 (N=500). Gaussian σ=15 truth:
23.7 (N=10, +58%) → 17.2 (+15%) → 15.5 (+3%) → 15.1 (+0.5%). Mechanism:
the Gaussian null space leaves s0 free with a broad LogNormal(span/8)
prior, so sparse bins inflate toward a broad Gaussian. Hierarchical sigma3
adapts (1.9→1.8 over the range) but does not equalise resolution.
**Impact: negligible at shipped occupancies** (MUSE 152, HST 426, Gaia 435:
+3% to nil) — no live systematic in shipped results. **Severe at the
`min_stars=10` floor** (+58%): floor bins are prior-dominated and should be
flagged as such, and this is Track 2's strongest motivation. No penalty
reweighting needed for shipped data; audit closed unless the floor is used
for science.

**The sign must be computed, not imported.** Stayman & Fessler 2000 §V measured
their conventional space-invariant penalty "blurring more in high-count regions
than in low-count regions," but that is for an emission-tomography system matrix
where `A = diag{c_i} G diag{s_j}` and `W` tracks photon counts. Our `F = MᵀWM`
has different structure — per-star error widths and per-bin counts, not ray
geometry. Compute it here.

If confirmed, the fix is **Stayman & Fessler 2000** (*IEEE TMI* 19(6), 601–615)
certainty weighting: `R(f) = Σ κ_j κ_k ψ(f_j − f_k)` with
`κ_j ∝ sqrt(Σ_i M_ij² W_ii)`. Their own caveat: this gets mean FWHM right but
"the responses are still quite asymmetric."

## 1.2 Averaging kernel / `d_s` per bin (P2)

Three fields built the same matrix: Fessler's local impulse response, Menke's
model resolution matrix `(GᵀG+λL)⁻¹GᵀG`, and Rodgers' averaging kernel
`A = Λ_post⁻¹Λ_data` with `d_s = tr(A)`. Under a Laplace approximation they are
one object. Computable as **post-processing on existing NUTS draws**: Gaussian-fit
the draws of the latent, linearise softmax (`J = diag(p) − ppᵀ`), use
`Λ_data = Λ_post − Λ_prior`.

Three caveats that must be stated, not hidden:

- **softmax is nonlinear**, so this is a linearisation, least accurate exactly
  where the posterior is skewed — edge and low-count bins.
- **`Λ_prior` is singular.** The `gaussian_core` null space is `{1, u, u²}` in 1D
  and `{1,x,y,x²,xy,y²}` in 2D, so the prior contributes zero precision in 3 (or
  6) directions. Use a pseudo-inverse or restrict to the penalised subspace;
  naively inverting gives garbage. Upside: `d_s` then counts ≈3 (or 6) fully
  data-driven degrees of freedom before any shape information, which is the right
  accounting.
- **The theory is derived at fixed penalty weight.** We infer `sigma3`. So `d_s`
  is a conditional, `sigma3`-fixed slice through the hierarchical posterior; it
  omits the `Var_β[E(x|y,β)]` term and cannot see a perturbation in one bin
  shifting the inferred global smoothing. Report as an approximation.

Note this **complements rather than replaces** `RecoveryCurve.efficiency`, which
already compares credible-interval width to the Cramér–Rao bound with `<1`
documented as "the prior is shrinking estimates." That covers the summary
statistics; `d_s` covers the whole curve per bin.

**Measured 2026-09-14: not reproduced by grid coarseness alone.**
Anisotropic mock (sx=15, sy=11.5, ratio 0.77, homoscedastic errors): y CI/CR =
1.10 (K=9) → 1.07 (K=15) → 1.02 (rectangular (15,21)); x stays 1.00–1.03.
Nowhere near the observed 1.8×, so coarse cells are at most a 10% contributor.
Remaining suspects: heavy-tailed real errors (untested here) or real-data
specifics. Downgraded to P3: confirm on real HST data with its measured error
distribution before spending more. The averaging kernel (§1.2) is deferred
with it — Exp A/B already answer the practical questions empirically, so
d_s is documentation-grade, not diagnosis-grade. Do not build now.

## 1.3 Posterior-mean-sums-to-1 test (P2, trivial)

Means sum to 1 exactly by linearity of expectation, regardless of dependence
structure. Medians have no such additivity, and for right-skewed non-negative
marginals each sits below its mean — which is where the observed 0.85–0.95 comes
from. So that figure is a derived consequence, not a tolerance. Two lines of
test, plus a wording fix in `CLAUDE.md`, which currently reads as though the
0.85–0.95 applies to marginals generally.

## 1.4 Prior correlation length vs claimed peak separation (P2)

Bayes-LOSVD's bimodality failure was diagnosed precisely (Falcón-Barroso &
Martig 2021 §5.2): *"the level of correlation between velocity bins imposed by
this prior is too strong and smooths the solution too much"* — and an order-1
prior at 30 km/s sampling recovered both peaks where order-2 at 60 km/s destroyed
them. That is the same coupling between grid resolution and effective
regularisation as `cell_per_sigma`. Verify our correlation length is short
relative to any peak separation we claim. Their failure case was an
autoregressive prior on the density; ours is an order-3 penalty on a
null-space-projected deviation, so the lesson is suggestive rather than a direct
read-across.

## 1.5 Delta / checkerboard injection tests (P2)

The recovered posterior mean from a spike truth **is** a row of the empirical
resolution matrix — the exact, non-linearised version of 1.2, and the geophysics
checkerboard test specialised to 1D. Reuses the SBC harness; needs new truth
generators only. Full NUTS runs, so budget a handful of representative velocities
rather than all bins.

## 1.6 Huber / GGMRF on the RW3 increments (P3)

Weakened after reading the code. The CT argument (Rudin–Osher–Fatemi mechanism:
a quadratic penalty charges growing marginal cost per unit jump and so disperses
real discontinuities) assumes relaxation toward a *flat* field. `gaussian_core`
relaxes toward a Gaussian, so Merritt's half of the argument is already answered.
What remains: bimodality lives in the penalised deviation, and that deviation is
still quadratically penalised, so a Huber form on the RW3 increments could pass
one large departure more cheaply than many small ones.

**Check the cheap thing first.** The `_rw_deviation_scale` docstring records that
h3 retention is 0.13–0.16 across `rw_order` 3–5 — i.e. the *order* knob buys
nothing, because softmax decouples the log-density null space from the PDF
moments. Confirm the penalty-*shape* knob behaves differently before investing.

## 1.7 Already done — do not re-open

The `sigma3` funnel is found and fixed: non-centred `d3`,
`target_accept_prob=0.95` (measured: 17% failures with p5 ESS 50 at 0.8, versus
1% and p5 ESS 217 at 0.95), `dense_mass=True`. This is worth stating in a paper,
because the companion 2026 paper hit the identical pathology on `σ_CAR` and
**abandoned hierarchical inference of it**, hand-fixing the scale at 0.001/0.03.

---

# Track 2 — spatial regularisation

## 2.1 The motivation is occupancy, not accuracy

The accuracy case is weak: 0.1–0.3% (§ above). The real case is in our own docs.
`docs/shape-information-limits.md` states MUSE's binding constraint is
**"occupancy, not errors"** — 152 stars/bin against the 398 its own test truth
needs, a factor 2.6 short. Larger spatial bins would buy that, and larger bins
were considered and declined.

**Spatial coupling is the only remaining way to buy effective occupancy without
enlarging the bins.** That is a well-posed motivation, it targets precisely the
regime where Fan's logarithmic rate says a bin cannot stand alone, and it applies
with more force at the `min_stars=10` floor.

## 2.2 Couple the null-space coefficients, not every velocity channel

This is the main new idea here, and it comes from the null-space structure rather
than from the literature.

The companion paper put a CAR prior on **every velocity channel** at each
spaxel, hit divergences, hand-fixed `σ_CAR`, and listed velocity-dependent
`σ_CAR` as future work — because smoothing effectiveness depends on whether the
local density exceeds `σ_CAR` ("velocity channels with density > σ_CAR are most
effectively smoothed"), so a single global scale smooths the dominant component
better than the sub-dominant one.

`gaussian_core` has already split the curve into exactly the right two pieces:

- **Null-space coefficients** — `v0, s0` in 1D; the 6 bivariate-Gaussian
  components in 2D. These are mean velocity, dispersion and (in 2D) the velocity
  ellipsoid: **the quantities that genuinely do vary smoothly across a cluster**,
  and the quantities DYNAMITE consumes.
- **Deviation coefficients** (`d3`) — higher-order shape. No strong physical
  reason to be spatially smooth, and the noisiest part of the fit.

Put the CAR prior on the null-space coefficients only. That gives:

- **~2–3 coupled numbers per bin in 1D, ~6 in 2D**, instead of `K` or `K²`
  channels. Far smaller, far better conditioned.
- **The velocity-channel smoothing problem does not arise**, because the coupled
  part has no velocity channels. Their open problem is sidestepped, not solved.
- **Much lower funnel risk.** The pathology scales with how many weakly-identified
  parameters share one scale hyperparameter — and we know from 1.7 that this
  codebase has already been bitten by it once.
- It borrows strength on exactly the quantities the acceptance criterion is
  stated in (`v_mean`, `sigma`, calibrated uncertainties), rather than on shape
  detail the information limits say is unrecoverable at these occupancies anyway.

## 2.3 The machinery already exists

`build_gmrf_precision` **is** an intrinsic CAR: `Q = D − W`, a graph Laplacian,
singular with a constant null space, ridge-regularised relative to
`mean(diag(Q))`. It is currently pointed at a velocity-cell lattice. Point it at
a **Voronoi adjacency graph** and it is the spatial prior, unchanged.

Two notes so nobody over-engineers this:

- The Sørbye–Rue standardisation already in the codebase matters *more* on an
  irregular graph than on a lattice, because Voronoi bins have varying neighbour
  counts, and it is what makes one hyperparameter mean the same thing across
  them.
- The DCT/analytic-eigenvalue route from Track 3 does **not** apply here — a
  Voronoi adjacency graph has no lattice structure. It does not need to: the
  matrix is `n_bins × n_bins` (hundreds), so a dense factorisation is fine.

Worth reading first: Mitzi Morris's Stan ICAR case study
(`mc-stan.org/users/documentation/case-studies/icar_stan.html`) for the
soft-sum-to-zero and scaling-factor reparameterisations that make ICAR sample
efficiently under NUTS. No maintained NumPyro-native port was found.

## 2.4 Validation

**NEGATIVE at the confirmation round — do not promote as-is.**
`scratchpad/track2_spatial_proto.py`, B=30 bins × 2 seeds (pooled n=60,
se≈0.06 on coverage): v0-coupled CAR (α=0.9, non-centred) gives v_mean RMSE
**4.26 vs 4.22 independent — zero accuracy gain**, coverage **0.38 vs 0.60**
(nominal 0.68). The prototype's −26% RMSE win did not survive: at B=15 it
was seed luck. Worse, coupling actively degrades coverage (Δ≈0.2, ~3σ), and
the inferred spatial scale is unstable (sig_sp 2.97 vs 1.25 across seeds —
the weakly-identified scale showing exactly the funnel signature §1.7
warns about). **Root cause worth recording**: the truth here is a
*deterministic linear gradient* — real v0 does vary smoothly, but under it
the coupled model's extra structure buys nothing and the shared scale
hyperparameter degrades every bin's intervals. Verdict: **spatial coupling
on the null-space coefficients is a NO-GO at this design.** Do not build
src/veldist/spatial.py. If revisited, the failure condition in this file
(better point estimates with worse coverage) fired exactly as written.

Do this in 1D first, where the existing `test_sbc_calibration` and
`test_per_bin_losvd_coverage` gates can validate it directly. A spatial prior
that improves point estimates while degrading coverage is a failure, and coverage
is the acceptance criterion — this is exactly the undercoverage-from-
regularisation-bias failure Kuusela & Panaretos (2015, arXiv:1505.04768) argue
the unfolding literature routinely misses.

---

# Track 3 — joint LOS + PM 3D retrieval

Current state: `veldist2d` is the **PM plane only** (`pm1`, `pm2`, per-star 2×2
covariance); `veldist.py` is LOS only. Nothing models the joint distribution, and
the two products are reconciled downstream — which is what `TASKS.md`'s "fix
NNLS with multiple kinematic datasets: figure out correct stacking" is about.

## 3.1 Why this one is different

Everywhere else the honest word is *deconvolution*. **Here the tomography
content is real**, because observation is partial:

| sample | observes | its row of M is |
|---|---|---|
| MUSE | `v_los` | a blurred **plane integral** through the velocity cube |
| HST / Gaia | `pm1, pm2` | a blurred **line integral** along the LOS axis |
| overlap | all three | a blurred **point** |

Two orthogonal projection directions plus a full-rank subset — few-view
tomography, which the survey identified as the correct precedent (not
missing-wedge, which assumes near-continuous coverage with a contiguous gap).
The **overlap stars are the only thing constraining the off-diagonal terms**
`⟨v_los v_x⟩`, `⟨v_los v_y⟩`, and those off-diagonals *are* the anisotropy. So
this is the mass–anisotropy degeneracy as an information-deficit problem, a
framing the survey established nobody has published.

## 3.2 The architecture already does the hardest part right

Extending the null space to 3D gives

```
{1, x, y, z, x², y², z², xy, xz, yz}          — 10 dimensions
```

which is exactly the **trivariate Gaussian log-densities**: 3 means + 6
independent covariance components + normalisation. The 3D null space is
therefore **the full velocity ellipsoid, left completely unpenalised.**

The smoothness prior would otherwise shrink anisotropy toward isotropy — a
systematic pointing exactly the direction that makes a spurious result look
plausible. In this architecture it cannot, by construction. The 2D docstring
already states the principle for the bivariate case: the null space is "what the
prior must leave free so that the velocity ellipsoid is not shrunk."

**This is the argument for building inside `gaussian_core` rather than bolting
`v_los` onto the 2D solver.**

## 3.3 Partial observation falls out of the design matrix

`precompute_design_matrix_2d` Path 1 already factorises a diagonal covariance
into an outer product of two 1D matrices. Extend that and:

- Block-diagonal covariance (2×2 PM block ⊕ independent RV variance) factorises
  as `M_i = M_pm(i) ⊗ m_los(i)`.
- **A missing axis contributes a row of ones.** Marginalising over an axis means
  summing that axis's cell masses, which is 1 by construction. So an RV-only star
  is `ones(K_x) ⊗ ones(K_y) ⊗ m_los(i)`; a PM-only star is `M_pm(i) ⊗ ones(K_v)`.

That is the whole mechanism. It is also what Extreme Deconvolution does for
missing dimensions (Bovy, Hogg & Roweis 2011), so it is a precedent to cite, not
a novelty to defend. Keeping it factorised is the separable-footprint trick
(Long, Fessler & Balter 2010) and is what PNKR itself does — they report "the
large scale nature of the problem renders assembly of the full matrix infeasible"
and use a Kronecker product `M = Ψ ⊗ Φ`.

## 3.4 Blockers — the GMRF plumbing, not the design matrix

Sizing first: at `K = 25` per axis, `n_cells = 15 625`; a 500-star bin gives `M`
at 31 MB float32. **The design matrix is not the problem.**

| blocker | where | why it breaks | fix |
|---|---|---|---|
| dense precision | `build_gmrf_precision` allocates `np.zeros((n_cells, n_cells))` | 15 625² × 8 B ≈ 2 GB, then O(n³) dense Cholesky | Kronecker sum `L_x⊕L_y⊕L_z`; never form it |
| dense pseudo-inverse | `_gmrf_deviation_scale_2d` calls `np.linalg.pinv`, documented O(k⁶) | O(k⁹) in 3D | analytic lattice eigenvalues + DCT — see 3.5 |
| Cholesky passed to model | `model_2d(matrix, n_cells, L)` | as above | matrix-free `cumsum` + null-space projection, as `generate_gaussian_core_curve` already does in 1D |

## 3.5 Use face-only connectivity in 3D — and the eigenvalue route then works

**Correction to an earlier draft of this document**, which claimed
`_gmrf_deviation_scale_2d`'s documented "−12% drift from k=9 to k=21" meant a
tuned `SIGMA3_RATE_2D` does not transfer across grids. That reading was
backwards. The returned constant is *supposed* to vary with `k`: the raw
projected field's variance is resolution-dependent and the constant varies
inversely to cancel it. The −12% is the size of the correction being applied, not
residual error. The docstring says plainly that the standardisation is what makes
`SIGMA3_RATE_2D` transfer. **There is no drift bug.**

What is true, measured here:

| k | connectivity | max off-diagonal after DCT |
|---|---|---|
| 9 | edge-only (`diag_weight=0`) | 6.3e-15 ✓ |
| 9 | **default 8-conn** (`1/√2`) | **1.04** ✗ |
| 13 | edge-only | 8.1e-15 ✓ |
| 13 | **default 8-conn** | **0.80** ✗ |

The DCT basis diagonalises `Q` exactly for edge-only connectivity and not at all
for the default. The reason is the boundary: the diagonal-neighbour term gives
`D_x⊗D_y − A_x⊗A_y`, which is not a Kronecker sum because the path degree vector
is not a multiple of the identity — interior cells have 8 neighbours, edge cells
fewer.

For **face-only (6-)connectivity in 3D** the route is exact (verified at k=7):
eigenvalues match the analytic `2(1−cos πj/k)` sums, and `diag(pinv)` from the
DCT matches the direct computation to 1e-10 — with no `(n×n)` factorisation
anywhere.

**And there is an independent reason to want 6-connectivity in this cube.**
`build_gmrf_precision`'s own docstring flags that `diag_weight = 1/√2` loses its
geometric justification when cells are non-square. In a joint cube that stops
being an open question: two axes are proper motions, the third is a line-of-sight
velocity, and a "corner-touching neighbour" spans a diagonal in a space with **no
common metric across axes**. `1/√2` is meaningless there. So face-only
connectivity is the defensible choice on physical grounds, and it makes the
analytic route exact as a side effect.

## 3.6 Do the information calculation first (P1)

`docs/shape-information-limits.md` opens by noting three successive sweeps
misread the same result and that "the question is analytic and did not need any
of those sweeps." Same discipline applies, and this is the highest-value task in
Track 3.

Extend the Gauss–Hermite attenuation result to the joint case. Cross terms are
constrained **only by the `N_both` overlap stars**, with attenuation
`(1+r²)^(−1)` per component (moment order 2, from `A_n = (1+r²)^(−n/2)`):

```
sd(⟨v_los v_x⟩ / s_los s_x)  ~  (1 + r_los²)^(1/2) (1 + r_pm²)^(1/2) / sqrt(N_both)
```

**If `N_both` is small, the anisotropy is unconstrained no matter what we
build.** Gate resolved 2026-09-14, measured — CONDITIONAL GO. The formula above is exact at ρ≈0 (Var(ĉ)=(XY+C²)/N; verified by simulation to <3% homoscedastic, heteroscedastic log-normal, and ρ=0.3). Cross-match counted directly in `omegaCen/dynamite_dataprep/merged.parquet`: fiducial MUSE sample is 24,928 stars, **100% with HST PMs** (the join requires the counterpart), median **N_both=101/bin** (min 57, 246 bins), vlos_err median 2.34 km/s, PM err median 0.054 mas/yr (≈1.41 km/s). At operating point (r_los≈0.16, r_pm≈0.12) sd(ρ̂)≈1.02/√N_both ≈ 0.10/bin; 3σ per-bin threshold ρ≈0.30. N_both needed: ρ=0.1→~940, ρ=0.2→~235, ρ=0.3→~104. Physical tilt/anisotropy signals are ρ~0.05–0.15 → **per-bin 3D without spatial pooling is NO-GO; with pooling of the ellipsoid components (Track 2 generalised) it is GO.** Outer field (Gaia r_pm~1.2, sparse RVs) is NO-GO regardless. Order flips: spatial coupling first, joint 3D second and coupled.

## 3.7 Where `d_s` earns its keep

In 1D the averaging kernel is a nice-to-have. Here it answers the question that
matters: **how many of the 10 ellipsoid degrees of freedom did the data actually
constrain, per bin?** That is the mass–anisotropy degeneracy made quantitative
and per-bin, and it is what tells a DYNAMITE consumer whether a fitted anisotropy
is measurement or prior. `Λ_prior` is singular by 10 here, so pseudo-inverse or
penalised subspace.

Track 2's idea also generalises: in 3D, the spatially-coupled quantities would be
the 10 ellipsoid components — which is a physically motivated spatial prior on
the velocity ellipsoid itself.

---

# What not to build

- **Kaczmarz / ART / PNKR-style iterative solvers.** Point estimate plus a
  stopping rule; the posterior is the product here. Randomised Kaczmarz is only
  interesting as a warm-start if NUTS ever dominates runtime, and it operates on
  unconstrained logits rather than the simplex.
- **Missing-wedge or Fourier-slice machinery.** Even Track 3 is few-view, not
  limited-angle-with-contiguous-gap. Wrong precedent.
- **A non-Bayesian fast mode** (CIL/ODL-style operator–regulariser–solver
  frameworks). Interesting architecture, no need.
- **RooUnfold cross-check.** Needs `M` exported into ROOT, checks only the mean
  shape, does not feed the DYNAMITE deliverable.
- **Raising `rw_order`.** Measured: 0.13–0.16 h3 retention across orders 3–5.
  Already settled.

# Suggested order

1. **1.1** uniform-resolution audit — tests for a live systematic in shipped
   results.
2. **1.2** averaging kernel, run first against the open 2D `sigma_y`-vs-`sigma_x`
   question, which is a real test case.
3. **1.3** the two-line test. **3.6** the information calculation — cheap, and it
   gates all of Track 3.
4. **2.2** null-space spatial coupling in 1D, validated against the existing SBC
   and coverage gates.
5. **Track 3**, if and only if 3.6 says the overlap sample supports it.
