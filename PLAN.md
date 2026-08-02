# Implementation Plan

Companion to `TASKS.md`. Goals: (A) a *robust*, statistically validated 1D
solver; (B) a first working 2D (proper-motion) solver.

### How to read this document

Sections marked **⚠ Gotchas** list the specific ways the step fails in
practice, with the symptom you will actually observe. **Fallback** blocks give
an ordered retreat path — if the primary approach resists after a couple of
hours, take the next rung rather than pushing harder. Every fallback is a
legitimate stopping point, not a failure; the only wrong move is to silently
weaken an assertion until it passes.

Two rules that override everything else below:

1. **Never loosen a failing statistical test to make it green.** If a
   calibration or coverage test fails, either the model is wrong or the test
   is wrong. Work out which. Writing `atol=0.5` to get a pass destroys the
   entire point of Part 1.
2. **Commit after each numbered step in the final table.** Several steps
   change the model's posterior; when a later test fails you need to be able
   to bisect which change did it.

The organising principle for both goals is the same: **we do not currently
have any test that would detect a wrong posterior.** Every existing inference test
(`tests/test_inference.py`) checks a single mock realisation against a loose
tolerance — those catch a broken pipeline, not a miscalibrated one. Everything
below is ordered so that the calibration machinery lands *before* the 2D solver,
because the 2D solver needs it to be trustworthy at all.

---

## Part 0 — Findings from the code sweep (new work, not in TASKS.md)

These came out of reading `veldist.py` critically. They are prerequisites: no
point calibrating a model with known defects.

### 0.1 `total_flux` is a broken likelihood term (P0, correctness)

`model()` at `src/veldist/veldist.py:160-197`:

```python
total_flux = numpyro.sample("total_flux",
    dist.TruncatedNormal(loc=N_stars, scale=sqrt(N_stars), low=0.0))
...
log_prob = jnp.sum(jnp.log(per_star_prob)) + N_stars * jnp.log(total_flux)
```

The comment calls this a Poisson log-likelihood. The extended (marked-point-
process) Poisson likelihood for `N` events with intensity `λ(v) = Φ · p(v)` is

```
log L = −Φ + Σᵢ log(Φ · p(vᵢ))  =  −Φ + N log Φ + Σᵢ log p(vᵢ)
```

The `−Φ` term is **missing**. Consequences:

- The posterior for `total_flux` is `TruncatedNormal(N, √N) × Φ^N`. Maximising
  gives `Φ² − NΦ − N² = 0` → `Φ_MAP = N(1+√5)/2 ≈ 1.618 N`. The parameter is
  biased ~62% high and its "uncertainty" is meaningless.
- `intrinsic_pdf` is `softmax(...)`, so it is normalised independent of `Φ`.
  `total_flux` therefore has **zero** influence on the science output. It is a
  free nuisance dimension that only costs NUTS geometry.

**Decision to make, then implement:**

- *Option A (recommended, lazy):* delete `total_flux` entirely. The model is a
  normalised-shape model; total counts are not inferred and not used
  downstream. `n_stars` is already tracked on the solver for `bin_flux`.
- *Option B:* add the `−total_flux` term to make it a correct extended
  likelihood. Only worth it if we later want the un-normalised intensity (e.g.
  for the "multiple tracers" someday-task, where relative populations matter).

Go with A now; leave a one-line comment pointing at B. This is a strictly
simplifying change and will show up as a small speedup in the slow tests.

**Test:** `test_model_has_no_dead_parameters` — run 50 warmup / 50 samples on
tiny mock data, assert `set(samples) == {"steps", "smoothness_sigma",
"intrinsic_pdf"}`.

### 0.2 The random-walk prior is not translation-invariant (P0, correctness)

`generate_smooth_curve()` samples `n_bins − 1` i.i.d. steps and takes a cumsum
with `curve[0] = 0`. Under this prior, `Var(curve[k]) = k · σ²` — bin 0 is
pinned and bin `K−1` is the loosest. The prior is therefore **asymmetric in bin
index**: the LOSVD is more strongly regularised at the low-velocity edge of the
grid than at the high-velocity edge.

The intended object is an *intrinsic* RW1 GMRF, which is translation-invariant
and has no special bin. Its (improper) log-density is

```
log p(x | σ) = −(1 / 2σ²) · Σₖ (x[k+1] − x[k])²      + const
```

with a sum-to-zero gauge fixed by the softmax. Implement as:

```python
x = numpyro.sample("x", dist.Normal(0, 10).expand([n_bins]).to_event(1))  # vague base
numpyro.factor("rw1", -0.5 * jnp.sum(jnp.diff(x)**2) / smoothness_sigma**2
                     - (n_bins - 1) * jnp.log(smoothness_sigma))
```

Note the `−(K−1) log σ` normalisation term: the current implementation gets this
for free from `dist.Normal(0, σ)`, and it *must* be carried over or
`smoothness_sigma` will collapse to zero. This is the single most likely place
to introduce a subtle bug — SBC (§1.2) is what will catch it.

The vague base `Normal(0, 10)` supplies the missing rank-1 direction (the
overall level), which the softmax then ignores. Alternative: keep the cumsum
parameterisation but centre it, and accept the asymmetry as a known
approximation. **Recommend implementing the GMRF form**, because §3 (the 2D
solver) needs exactly this machinery generalised to 2D — doing it in 1D first
means the 2D version is a change of precision matrix, not a change of approach.

Delete the now-redundant `centered_curve = latent_curve − mean(latent_curve)`
line (softmax is shift-invariant; that line is a no-op).

**⚠ Gotchas**

- **Forgetting `−(K−1)·log σ`.** This is the failure mode. `numpyro.factor`
  adds a raw log-density; unlike `dist.Normal(0, σ)` it does *not* supply the
  normalising constant. Omit it and the likelihood becomes monotone increasing
  as `σ → 0`: the sampler drives `smoothness_sigma` to ~1e-8 and the LOSVD
  comes out as a perfectly flat line. **Symptom:** posterior median
  `smoothness_sigma` orders of magnitude below its prior scale, and a
  featureless `intrinsic_pdf`. If you see that, this is the cause.
- **`dist.Normal(0, 10)` base is not neutral.** It weakly pulls the latent
  curve toward zero. Because softmax is shift-invariant this only affects the
  overall level (which is unidentified anyway), so it is harmless — but do not
  "improve" it to `dist.ImproperUniform`, which removes the sole thing keeping
  the sampler's mass-matrix adaptation numerically stable in that direction.
- **`.to_event(1)` is required.** Without it NumPyro treats the `K` components
  as a batch dimension and the plate semantics differ. It will still run and
  give subtly wrong results. Check `mcmc.print_summary()` shows one `x`
  parameter of shape `(K,)`, not `K` scalars.
- **Divergences after this change are expected initially.** The
  `factor`-based prior has different geometry than the cumsum version. Try
  `NUTS(model, target_accept_prob=0.9)` before concluding anything is wrong.

**Fallback ladder**

1. Non-centred reparameterisation: sample `z ~ N(0,1)^K`, build `x` from `z`
   by explicit cumsum-and-centre. Keeps the translation invariance, avoids
   `numpyro.factor` entirely, and sidesteps the `log σ` trap because
   `dist.Normal` supplies the constant. **This is the safer first attempt if
   `factor` gives you trouble** — it is mathematically the same prior:
   `x = cumsum(σ·z)`, then `x −= x.mean()`.
2. If the symmetric prior proves genuinely hard, keep the existing cumsum form
   but **document the asymmetry in `theory.md` and add a test that quantifies
   it** (fit the same mock forwards and reversed in velocity; report the
   difference in recovered `skewness`). A known, measured bias is acceptable;
   an unknown one is not.
3. Only if both fail: leave the prior alone and note it as a limitation. Do
   not block Parts 1–3 on this.

### 0.3 `smoothness_sigma` prior is grid-resolution dependent (P1)

`HalfNormal(0.1)` is a prior on the *per-bin step* in log-density. Halving the
bin width doubles the number of steps needed to traverse the same velocity
range, so the same physical smoothness requires a different `σ`. The existing
`test_grid_resolution_consistency` is loose enough not to catch this.

Fix: reparameterise in physical units. For an RW1 approximating a continuous
process, the step scale should go as `σ_step = σ_phys · √(dv)` (Brownian) or the
precision as `1/dv` (see the discretised-derivative form). Sample
`σ_phys ~ HalfNormal(s)` and set `σ_step = σ_phys * sqrt(grid["width"])`,
passing `dv` into `model()`.

**Test:** `test_smoothness_prior_is_resolution_invariant` — fit the same mock
data on `n_bins ∈ {40, 80}` grids over the same velocity range; assert the
posterior-median `v_mean` and `sigma` agree to within the tighter of the two
posteriors' half-68CIs. Mark `slow`.

### 0.4 `gpu=True` default in `run()` (P2)

`KinematicSolver.run()` defaults `gpu=True` and calls the *global*
`numpyro.set_platform("gpu")`, which raises on machines without a GPU and can
never be undone within the session. Change the default to `gpu=None` meaning
"leave the platform alone"; only call `set_platform` when explicitly `True`.
This also removes a latent CI hazard.

### 0.5 Bare `assert` in library code (P2)

`clip_uncertainties` (`veldist.py:478`) and `write_dynamite_kinematics`
(`veldist.py:852`) use `assert` for user-facing validation. These vanish under
`python -O`, and the second one is explicitly guarding against silent Dynamite
corruption. Convert both to `raise ValueError`. Folds naturally into the ruff
EM101/EM102 cleanup.

---

## Part 1 — Statistical validation framework (P0)

This is the centrepiece. Three layers, cheapest first.

### 1.1 Analytic unit tests for `analysis.py` (fast, no MCMC)

`analysis.py` has zero tests but is almost entirely deterministic array maths
with **known closed-form answers**. No sampling needed — construct exact PMFs
on a grid and check against analytic values. New file `tests/test_analysis.py`.

Helper: `analytic_pmf(dist, centers, width)` = `dist.cdf(edges).diff()`,
renormalised. Feed it as a single-row `(1, n_bins)` "sample set".

| Test | Input | Assertion |
|---|---|---|
| `test_moments_gaussian` | `N(μ=12, σ=8)` on a wide fine grid | `v_mean ≈ 12`, `sigma ≈ 8`, `skewness ≈ 0`, `kurtosis ≈ 0` to 1e-2 |
| `test_moments_skewnormal` | `scipy.stats.skewnorm(a=4)` | `skewness` matches `dist.stats(moments='s')` to 2% |
| `test_kurtosis_student_t` | `t(ν=6)`, excess kurtosis `= 6/(ν−4) = 3` | within 5% (grid truncation biases this low — assert the *sign* and order of magnitude, and document the bias) |
| `test_tail_weight_gaussian` | Gaussian | `≈ 1 − erf(1/√2) = 0.31731` to 1e-3 |
| `test_tail_weight_ordering` | Gaussian vs. `t(6)` vs. uniform | `tw(uniform) < tw(gauss) < tw(t)` |
| `test_sigma_iqr_gaussian` | Gaussian | `sigma_iqr ≈ sigma` to 1% |
| `test_sigma_iqr_heavy_tails` | `t(6)` | `sigma_iqr < sigma` (the documented anisotropy diagnostic) |
| `test_cdf_percentile_recovers_quantiles` | Gaussian | vs. `dist.ppf([0.25,0.5,0.75])` to within half a bin width |
| `test_v_asymmetry_sign` | right-skewed mixture | `v_asymmetry > 0`; sign flips under `v → −v` |
| `test_bimodality_score_counts` | 1, 2, 3-component mixtures with well-separated peaks | score `== 1, 2, 3` |
| `test_bimodality_ignores_tail_noise` | unimodal Gaussian + 0.5%-amplitude ripple in the tail | score `== 1` (exercises the 1%-of-max threshold) |
| `test_half_68ci_gaussian` | 200k draws from `N(0,1)` | `≈ 1.0` to 2% |
| `test_summary_uncertainty_shrinks_with_n` | same PMF replicated with narrow vs. wide posterior spread | half-68CI ordering correct |
| `test_compute_summary_keys` | any | exactly the 10 documented keys; all `(float, float)` except `bimodality_score: int` |
| `test_compute_summary_maps_nan_masking` | `[solver, None, solver]` (stub objects, no MCMC) | index 1 is NaN in every metric; `bimodality_score["uncertainty"]` is all-NaN |
| `test_compute_summary_maps_all_none` | `[None, None]` | raises `ValueError` |

Use lightweight stub objects (`SimpleNamespace(samples={"intrinsic_pdf": …},
grid={"centers": …})`) for the `_maps` tests — do not run MCMC.

Also worth adding here: **`test_gh_conversion_consistency`**. The docstring
claims `h3 ≈ −skewness/√6`, `h4 ≈ kurtosis/√24`. Fit a Gauss–Hermite series to
the same analytic PMF with `scipy.optimize` and check the conversion holds to
the stated `|h| ≲ 0.2` validity range. This directly validates a claim we make
to users and lets us cross-check against literature GH maps.

**Cost:** ~1 s total. All fast-marked. This alone closes the P0 coverage gap on
`analysis.py`.

### 1.2 Simulation-Based Calibration (the correctness test)

This is the one test that can prove the posterior is right rather than merely
plausible. SBC (Talts et al. 2018, arXiv:1804.06788) exploits the identity

> if `θ̃ ~ p(θ)` and `ỹ ~ p(y|θ̃)` and `θ⁽¹⁾…θ⁽ᴸ⁾ ~ p(θ|ỹ)`,
> then the **rank** of `θ̃` among the `L` posterior draws is Uniform{0…L}.

Any deviation from uniformity is a bug or a miscalibration, and the *shape* of
the deviation is diagnostic:

- ∪-shaped histogram → posterior **too narrow** (overconfident)
- ∩-shaped → posterior **too wide**
- monotone ramp → posterior **biased** in that direction

New file `tests/test_calibration.py`.

**Procedure (`run_sbc(n_sims, n_bins, n_stars, thin)`):**

1. Draw `smoothness_sigma` and the latent curve **from the model's own prior**
   — use `numpyro.infer.Predictive(model, num_samples=1)` with no observations,
   so the prior draw is guaranteed identical to what inference assumes. Do not
   hand-roll the prior draw; that is the classic way to get a vacuously passing
   SBC.
2. From the resulting `intrinsic_pdf`, sample `n_stars` true velocities
   (multinomial over bins, then uniform within bin), add per-star Gaussian
   noise with errors drawn from a realistic distribution.
3. Run the solver. Thin the chain to `L = 99` roughly-independent draws
   (thin by `n_samples // 100`; SBC requires *independent* draws, and
   autocorrelated ones manufacture ∪-shapes that look like real bugs).
4. Rank the true value among the `L` draws, for each **test quantity**.

**Test quantities** — rank the derived scalars, not the 400 raw bins:
`v_mean`, `sigma`, `skewness`, `kurtosis`, `tail_weight`, `smoothness_sigma`,
and `log p(θ̃)`. The last is the standard catch-all: it is sensitive to
misfit in directions no individual marginal exposes.

**Assertion:** per-quantity, a uniformity test on the ranks. Use the ECDF-based
test rather than a χ² on binned ranks — it is bin-free and has better power at
small `n_sims`. `scipy.stats.kstest(ranks / L, "uniform")`, assert
`p > 0.005`, Bonferroni-adjusted over the ~7 quantities.

**Cost and how to make it affordable.** This is the real design problem: `n_sims
= 100` × a full NUTS run is minutes-to-hours. Mitigations, in order:

- Use a small configuration: `n_bins = 20`, `n_stars = 300`,
  `num_warmup = 400`, `num_samples = 1000`. SBC detects *implementation* bugs,
  which do not care about problem size.
- `jax.vmap` the whole SBC loop. The design matrix shape is fixed across
  simulations, so all `n_sims` posteriors can be sampled in one vectorised
  `MCMC.run` over a batched `matrix` argument via `numpyro`'s
  `chain_method="vectorized"`. This is the single biggest win — expect
  1–2 orders of magnitude over a Python loop.
- Mark `slow`, run in CI only on the ubuntu/3.13 job (matching the existing
  slow-test policy in `.github/workflows/tests.yml`).
- Cache the rank array to `tests/data/sbc_ranks.npz` and add a fast test that
  re-runs only the uniformity assertion on the cached ranks. Regenerate
  deliberately, not automatically.

**⚠ Gotchas** — SBC is easy to write in a way that passes vacuously. In rough
order of how often each one bites:

- **Hand-rolling the prior draw.** If step 1 samples `θ̃` from anything other
  than the model's own prior, the test is meaningless — and it usually
  *passes*, which is worse than failing. Use `Predictive(model, num_samples=1)`
  with no conditioning. If you find yourself writing `sigma = rng.halfnormal(...)`
  to mirror what the model does, stop: that mirror will drift out of sync the
  first time the model changes.
- **`numpyro.factor`-based priors are invisible to `Predictive`.** This is
  not hypothetical — it is exactly what SBC caught in this codebase during
  §0.2's implementation. The original §0.2 fix expressed the RW1 penalty as
  `numpyro.factor("rw1_prior", -0.5*sum(diff(x)**2)/sigma**2 - ...)` applied
  to an unconditioned `x ~ Normal(0, 10)^K` base. This is *correct* for NUTS
  (the potential function integrates factor terms along with sample-site log
  densities, so the posterior is right), but `Predictive` with no data
  conditioning does pure ancestral sampling through `sample` sites only — it
  has no mechanism to account for a `factor` term, since a factor is not a
  distribution you can forward-sample from. The result: `Predictive` drew `x`
  as plain i.i.d. `Normal(0, 10)` noise (RW1 penalty completely ignored),
  producing an unregularized, near-one-hot `intrinsic_pdf` as the SBC "truth"
  — a draw from a *different* distribution than the one the posterior was
  actually conditioning against. Symptom: `smoothness_sigma`'s rank was ~0 in
  every single simulation (KS p≈0.0000), while quantities less sensitive to
  the spike (`v_mean`) still passed — a partial failure pattern that is easy
  to misdiagnose as "some other bug in just this one quantity."

  **The fix, and the general rule: build priors generatively, not via
  `factor`.** Reparameterise so every random quantity is an actual
  `numpyro.sample` site and the returned value is a deterministic function of
  those draws — e.g. sample the RW1 *increments* directly
  (`steps ~ Normal(0, sigma_step)`, `K-1` of them), then `x = cumsum(steps)
  - mean(cumsum(steps))` (the mean-centering is what restores translation
  invariance relative to the naive pinned-at-zero cumsum — verified
  numerically that this produces a bin-index-symmetric variance profile).
  This is simultaneously simpler than the factor version (no manual
  normalising-constant term — `dist.Normal` supplies it), correct under both
  NUTS and `Predictive`, and requires no vague base-measure hack for an
  unconstrained null-space direction (there isn't one; the parameterisation
  is already full-rank in `steps`). **This is the same lesson the 2D solver's
  §3.2 design already follows** (`z ~ N(0, I_K²)` sample site,
  `x = σ·L⁻ᵀz` as a deterministic transform) — do not "simplify" it into a
  `factor`-based GMRF penalty on an unconditioned base measure, even though
  that looks more directly like the textbook GMRF density. If a prior *must*
  be expressed as a `factor` for some reason, it cannot be validated by
  `Predictive`-based SBC at all — fall back to coverage testing (§1.3) for
  that component instead.
- **Autocorrelated posterior draws.** Un-thinned NUTS output produces
  ∪-shaped rank histograms that look exactly like an overconfident posterior.
  **Always check `ESS > L` before ranking**, and thin to at most `ESS/2` draws.
  If a rank histogram is ∪-shaped, verify the ESS *before* concluding the
  model is broken — this is the single most common false alarm in SBC.
- **`L` not matching the histogram binning.** Ranks run `0…L` inclusive, i.e.
  `L+1` possible values. Off-by-one here manufactures spurious structure at
  the histogram edges. Using the ECDF/KS test as specified avoids the issue
  entirely, which is part of why it is preferred.
- **Ranking a quantity that is constant under the prior.** `bimodality_score`
  is an integer with heavy ties; ranks with ties are not uniform. **Exclude it
  from SBC.** Same caution for any metric that saturates.
- **Silent NaNs.** A prior draw with an extreme `smoothness_sigma` can produce
  a degenerate `intrinsic_pdf` and a failed fit. Do not drop those simulations
  — dropping failures is selection bias and biases the test toward passing.
  Count them, and if more than ~2% of simulations fail, fix the failure mode
  before interpreting any ranks.
- **Reading too much into a small run.** With `n_sims = 100` the test only
  detects fairly gross miscalibration. A green SBC at `n_sims = 100` is
  meaningful; a *marginal* p-value at `n_sims = 100` is noise. Do not tune
  against it.

**Fallback ladder** (in cost order — do not skip to the bottom)

1. **`vmap`/vectorised sampling is fiddly.** If batching the `matrix` argument
   through `chain_method="vectorized"` resists, fall back to a plain Python
   loop with `n_sims = 30` and run it once locally rather than in CI. Thirty
   simulations still detects the ∪/∩/ramp signatures that matter; it just has
   less power. Ship that, and revisit the vectorisation later.
2. **Still too slow.** Shrink the problem, not the number of simulations:
   `n_bins = 10`, `n_stars = 100`, `num_samples = 500`. Implementation bugs are
   scale-free. `n_sims` is the axis that buys statistical power, so it is the
   *last* thing to cut.
3. **Cannot get SBC working at all.** Do not skip validation entirely — drop
   to §1.3 coverage testing, which is strictly weaker but far simpler (no
   ranks, no prior draws, just "is the truth inside the interval the right
   fraction of the time"). Implement §1.3 first in that case and treat SBC as
   a stretch goal. **A repo with coverage tests and no SBC is in far better
   shape than one with neither.**

**Sequencing note:** write and run SBC against the *current* model first, before
the Part 0 fixes. If §0.2's prior asymmetry is a real problem, SBC on the
per-bin quantities should show it, and we get a demonstration that the test has
teeth. Then fix, and confirm SBC goes green. A calibration test that has never
failed is a calibration test you do not yet trust.

### 1.3 Frequentist coverage over mock realisations

SBC validates the sampler against the model. Coverage validates the *model
against reality* — specifically, whether our stated ±half-68CI error bars mean
what we tell Dynamite they mean. This matters because the downstream consumer
(NNLS χ²) takes those uncertainties literally.

New: `tests/test_coverage.py`, `slow`.

For each of a handful of fixed, physically-motivated truths — Gaussian; a
`h4>0` radially-anisotropic analogue (Student-t); a `h3≠0` rotating-side
analogue (skew-normal); a counter-rotation bimodal — generate `n_real = 50`
independent mock datasets, fit each, and record whether the truth falls inside
the 68% and 95% credible interval of each summary metric.

Assert empirical coverage is within the binomial confidence band:
for `n=50` at nominal 0.68, the 99% band is roughly `[0.50, 0.83]` — wide, but
it catches gross miscalibration (e.g. 0.30, which is what a too-narrow posterior
produces). Report the actual numbers in the failure message; the numbers are as
valuable as the pass/fail.

**Expected honest outcome:** coverage will be *under*-nominal for `kurtosis` and
`tail_weight` on the non-Gaussian truths, because the RW1 prior shrinks toward
smoothness and a smoothness prior is a genuine bias when the truth has sharp
features. This is not a bug — it is the bias-variance trade the prior buys us.
The value of the test is quantifying it so we can state it in the paper, and
so we notice if it ever gets *worse*.

### 1.4 Design-matrix correctness (fast)

`precompute_design_matrix` is the one piece where a silent off-by-half-a-bin
error would corrupt everything downstream and pass every existing test.

- `test_design_matrix_vs_quadrature`: compare each `M[i,j]` against
  `scipy.integrate.quad` of the Gaussian over `[edge_j, edge_{j+1}]`, to 1e-10.
- `test_design_matrix_rows_sum_to_one_for_interior_stars`: a star well inside
  the grid with `err << grid width` must have `M[i].sum() ≈ 1`.
- `test_design_matrix_small_error_limit`: `err → 0.01 · bin_width` puts ≥99% of
  the row mass in the single containing bin (this is the aliasing case the box
  integration exists to prevent — assert it directly rather than trusting the
  comment).
- `test_design_matrix_offgrid_star_has_negligible_mass`: documents the grid-
  truncation behaviour so the "does the grid cover the support" assumption is
  explicit.

---

## Part 2 — Pipeline / export round trip (P0, from TASKS.md)

`fit_all_bins` and `write_dynamite_kinematics` are ~250 lines of index
arithmetic with zero tests, and the index arithmetic is the dangerous part: a
bug in `orig_to_new` silently mislabels which sky position has which LOSVD, and
nothing downstream would flag it.

New file `tests/test_pipeline.py`. **Do not run MCMC** for most of these —
construct fake solved solvers.

```python
def fake_solver(centers, pdf_mass, n_stars=100, n_draws=200):
    """A KinematicSolver with a synthetic posterior, no MCMC."""
    s = KinematicSolver()
    s.setup_grid(...)
    s.n_stars = n_stars
    # Dirichlet jitter around pdf_mass -> valid simplex rows
    s.samples = {"intrinsic_pdf": rng.dirichlet(pdf_mass * 500, size=n_draws)}
    return s
```

That fixture is worth building carefully — it unlocks fast tests for
`clip_uncertainties`, `truncate_losvd`, `compute_summary_maps`, and the whole
writer.

**`clip_uncertainties`:**
- floors are applied: with one artificially zero-variance bin, output
  uncertainty `== floor_fraction * max(raw_half_width)`, and `> 0` everywhere.
- the documented 0.85–0.95 marginal-median sum holds (`0.8 < sum < 1.0`) —
  encodes the convention so a future refactor that "fixes" it by renormalising
  trips the test and forces the conversation.
- raw `self.samples` is unmodified (it is documented as non-destructive).
- raises `ValueError` before `run()`.

**`truncate_losvd`:**
- bins beyond `n_sigma` are zeroed and their uncertainty set to `abs_floor`.
- auto-calls `clip_uncertainties` when not already run.
- idempotent: applying twice equals applying once.

**`fit_all_bins`** (one genuinely slow test, tiny settings):
- bins below `min_stars` → `None` at the right index and a `UserWarning`.
- per-bin seed derivation: two bins with *identical* data produce *different*
  posterior draws (proves `base_seed + i` is actually threaded through).
- all-bins-below-threshold → all `None`, no crash.

**`write_dynamite_kinematics`** — the round trip:
- write with a mixed `[solver, None, solver, solver]` list, read back with
  `astropy.table.Table.read`, assert:
  - row count `== 3`; `binID_dynamite == [1,2,3]`.
  - `xbin`/`ybin` match metadata **for the surviving bins only** — this is the
    test that catches an `orig_to_new` bug.
  - `bins.dat` contains no reference to the skipped bin's ID, and every pixel
    that pointed at it is now `0`.
  - `losvd_j`/`dlosvd_j` column count `== 2 * nvbins` and interleaved order.
  - `meta["dv"]`, `meta["vcent"]`, `meta["nbins"]`, `meta["nvbins"]` correct.
- `v` and `sigma` columns match `compute_summary` on the same solver to within
  a bin width — cross-validates the writer's inline moment code against
  `analysis.py`. (These are two independent implementations of the same
  quantity in the codebase right now; either make the writer call
  `analysis.compute_summary`, or keep the test that pins them together.
  Preference: have the writer call `analysis.py`.)
- grid-mismatch between solvers → `ValueError`.
- all-`None` → `ValueError`.
- each `bin_flux_mode`; `'nstars'` with `n_stars=None` → `ValueError`.
- `pixel_bin_ids` accepted as both `(nx*ny,)` and `(ny, nx)`, giving identical
  output.

**⚠ Gotchas**

- **The `fake_solver` fixture must produce a *valid simplex*.** Rows of
  `intrinsic_pdf` must sum to exactly 1 or `clip_uncertainties`'s internal
  assertion (`mean_sum ≈ 1.0`, `rtol=1e-3`) fires. `rng.dirichlet` guarantees
  this; hand-built arrays with added noise do not. Do not "fix" a failure here
  by loosening that assertion — it is load-bearing.
- **The Dirichlet concentration matters.** `pdf_mass * 500` gives a narrow,
  realistic posterior. Use a much smaller multiplier and the draws are so
  diffuse that the 0.85–0.95 marginal-median-sum test fails for reasons that
  have nothing to do with the code under test.
- **1-indexed vs 0-indexed bin IDs.** `binID_BayesLOSVD` is 0-indexed,
  `binID_dynamite` is 1-indexed, `pixel_bin_ids` is 1-indexed with 0 meaning
  masked, and `orig_to_new` is a lookup array whose element 0 is deliberately
  unused. Four conventions in one function. **Write the mixed-`None` test
  first and let it drive the reading of the code**, rather than reading the
  code and then writing a test that agrees with it — the latter just encodes
  whatever bug is already there.
- **`Table.read` round-trips metadata lossily.** `meta["vcent"]` comes back as
  a list, not an ndarray; `meta["PSF"]` comes back as a dict with the key
  ordering not guaranteed. Compare with `np.allclose(np.array(...))` and
  compare dict *contents*, not repr.
- **`output_dir` must be a `tmp_path` fixture**, never a repo-relative path.
  The writer calls `mkdir(parents=True, exist_ok=True)` and `overwrite=True`,
  so a wrong path silently clobbers real files.

**Fallback:** if constructing `fake_solver` proves awkward (e.g. because
`KinematicSolver` grows validation that rejects hand-set `samples`), use a
`SimpleNamespace` duck-type instead — the writer only touches `.grid`,
`.samples`, `.clipped_samples`, and `.n_stars`. Slightly less faithful, far
less friction. If even that fights you, fall back to one genuinely slow
end-to-end test with `n_bins=10, num_samples=100`; it is worth having a
correct slow test rather than no test.

**Golden-file check:** commit one small reference `.ecsv` under `tests/data/`
and diff against it. Cheap insurance against accidental format drift in a file
whose consumer (Dynamite) is not in this repo and cannot be tested against here.

---

## Part 3 — 2D solver: `KinematicSolver2D` (P1)

Only start once Parts 0–2 are green. The 2D solver is the 1D solver with a
different design matrix and a different precision matrix; if the 1D one is
validated, the 2D one is a contained change, and the SBC harness from §1.2
transfers directly.

New module: `src/veldist/veldist2d.py`. Keep it separate from `veldist.py`
until the shared structure is obvious — premature unification of the two will
cost more than the duplication.

### 3.1 Grid and design matrix

`setup_grid_2d(center=(cx, cy), width=(wx, wy), n_bins=K)` → `K×K` grid,
flattened row-major to length `K²`. Store `centers_2d` `(K², 2)`, `area`.

Design matrix `M` shape `(N, K²)`:

```
M[i, m] = ∫∫_{cell m} N(μ = pm_i, Σ = C_i) dμx dμy
```

Two approximations, in order of implementation:

1. **Start: centre-point × area.** `M[i,m] = pdf_bivariate(center_m; pm_i, C_i) * dA`.
   Simple, differentiable, correct in the limit `err >> bin size`.
2. **Then: exact box integration for the diagonal-Σ case.** When `C_i` is
   diagonal (no PM correlation), the 2D integral factorises into a product of
   two 1D `erf` differences — i.e. exactly `precompute_design_matrix` applied
   per axis, outer-producted. This is exact, cheap, and removes the aliasing
   failure mode. **Do this early**, not as a deferred item: Gaia PM errors are
   often comparable to a sensible bin size, which is precisely where centre-
   point evaluation breaks — the same reason the 1D code uses box integration.
   Keep centre-point only as the fallback for correlated `C_i`.

For correlated `C_i` (Gaia gives `pmra_pmdec_corr`, and it is not negligible),
the factorised trick fails. Options: rotate to the eigenbasis of `C_i` per star
and integrate over the rotated (now non-axis-aligned) cell — messy; or accept
centre-point for correlated stars; or a 2×2 Gauss–Legendre sub-cell rule, which
is 4× the cost and kills most of the error. **Recommend the 2×2 sub-cell rule
as the general path** — it is four lines, handles arbitrary `Σ`, and is a
strict improvement over centre-point.

**Memory:** `(N, K²)` at `N = 10⁵`, `K = 20` is 4×10⁷ floats = 320 MB in
float64. Use float32 for `M` (it is a likelihood weight, not an accumulator)
and this halves. At `K = 30` it is 720 MB — this is the practical ceiling and
should be documented. If we need larger, the design matrix is very sparse
(a star constrains only cells within a few σ) and a `BCOO` sparse matrix is the
escape hatch — but do not build that until `K = 20` is working and measured.

**⚠ Gotchas**

- **The `(N, K²)` matrix will OOM before you expect it to.** JAX allocates
  intermediates during construction, so peak memory is 2–3× the final array.
  Build `M` in chunks over stars (`for chunk in range(0, N, 5000)`) and
  concatenate. Do this from the start; retrofitting it after an OOM at
  `N = 10⁵` wastes an afternoon.
- **`bin area` is `dx*dy`, not `dx`.** Easy slip when porting from the 1D code.
  **Check:** for a star with error much larger than the grid, `M[i].sum()`
  should be ≈ the fraction of that star's error ellipse covered by the grid —
  and for a star well inside with small error, ≈1. If your rows sum to ≈`K` or
  ≈`1/dA`, you have a missing or doubled area factor.
- **Gaia's `pmra_pmdec_corr` is a correlation, not a covariance.** Build
  `C_i = [[σx², ρσxσy], [ρσxσy, σy²]]`. Feeding `ρ` where the covariance is
  expected produces a matrix that is not positive-definite for most stars and
  silently NaNs.
- **Perspective rotation is already applied** to the PM data upstream (see the
  Completed list in `TASKS.md`). Do not re-apply it.

**Fallback:** if the 2×2 sub-cell rule complicates the code more than it is
worth on the first pass, ship centre-point evaluation with an explicit
**guard**: assert at `add_data` time that `min(σ_pm) > 0.7 · bin_size`, and
raise a clear error telling the user to coarsen the grid otherwise. That turns
a silent accuracy failure into a loud, actionable one, which is an acceptable
first version.

### 3.2 GMRF prior

Precision matrix `Q` of shape `(K², K²)` for a 2D first-order intrinsic GMRF
with 8-connectivity:

```
Q = D − W        # W = adjacency weights, D = diag(row sums of W)
```

with weight 1 for the 4 edge-neighbours and `1/√2` for the 4 diagonal
neighbours (the `1/√2` is the natural distance weighting; equal weights are
also defensible — pick one, document it, and make it a keyword so the choice
is testable rather than baked in).

`Q` is singular by construction (constant vector in the null space), which is
correct — the softmax removes that direction anyway. For Cholesky, add a ridge:

```python
Q_reg = Q + eps * jnp.eye(K**2)      # eps ~ 1e-6 * mean(diag(Q))
L = jnp.linalg.cholesky(Q_reg)
```

`eps` scaled relative to `diag(Q)` rather than absolute, so it does not change
meaning when the connectivity weights change.

**Latent parameterisation** (non-centred, essential for NUTS):

```python
z = numpyro.sample("z", dist.Normal(0, 1).expand([K**2]).to_event(1))
x = smoothness_sigma * jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
intrinsic_pdf = jax.nn.softmax(x)
```

`L` is computed once outside the model and closed over — it does not depend on
any sampled parameter, so `Q`'s Cholesky is a one-time `O(K⁶)` = 6.4×10⁷ flops
at `K=20`. Fine. Never put a Cholesky inside the model body.

Prior normalisation: as in §0.2, the `−(rank Q)·log σ` term must be present or
`smoothness_sigma` collapses. Rank is `K² − 1`.

**⚠ Gotchas** — this section has the highest bug density in the whole plan.

- **Row-major flattening is the number-one hazard.** Cell `(ix, iy)` maps to
  flat index `m = ix*K + iy`. Every adjacency computation must use the same
  convention, and it is very easy to build `Q` with `x` and `y` transposed.
  The result *still looks plausible* — it is a valid GMRF, just with the axes
  swapped, so it only shows up as a wrong `ρ` sign in the tilted-Gaussian test.
  **Mitigation:** build the adjacency from an explicit
  `np.ravel_multi_index` call, never from hand-written arithmetic, and make
  `test_gmrf_prior_is_isotropic` (§3.3) the first test you write.
- **Boundary cells.** Corner cells have 3 neighbours, edge cells 5, interior 8.
  If you build `W` by shifting arrays, wrap-around (periodic boundaries) is the
  default failure and it is silent. **Check:** `Q.sum(axis=1)` must be ≈0 for
  *every* row, and `np.diag(Q)` must take exactly three distinct values for a
  square grid with `K > 2`. If `diag(Q)` is constant, you have periodic
  boundaries.
- **`solve_triangular` direction.** We want `x = σ·L⁻ᵀz`, so the call is
  `solve_triangular(L.T, z, lower=False)`. Using `L` with `lower=True` gives
  `L⁻¹z`, which is a *different* (and wrong) covariance. Both run without
  error. **Check:** draw 5000 prior samples and verify
  `cov(x) ≈ σ²·pinv(Q)` on a small `K=4` grid.
- **Cholesky of a singular matrix.** `Q` is singular by construction. Without
  the ridge, `jnp.linalg.cholesky` returns **NaNs silently** rather than
  raising (unlike scipy). Assert `jnp.all(jnp.isfinite(L))` immediately after
  factorising — a NaN `L` propagates into an all-NaN posterior that takes a
  long time to trace back.
- **`eps` too large.** A large ridge quietly turns the intrinsic GMRF into a
  proper one with a finite correlation length, over-smoothing the result.
  Scale it as `1e-6 * mean(diag(Q))` and add a test that the recovered LOSVD
  is insensitive to `eps` over `[1e-8, 1e-5]` relative.
- **Float32.** If `M` is float32 for memory (§3.1), keep `Q`, `L`, and the
  latent maths in **float64**. Cholesky of a near-singular matrix in float32 is
  where precision loss actually hurts. Set `jax.config.update("jax_enable_x64",
  True)` and cast `M` down explicitly rather than relying on defaults.

**Fallback ladder**

1. **8-connectivity giving trouble → use 4-connectivity** (edge neighbours
   only, weight 1). Simpler, standard, and the resulting prior is mildly
   anisotropic along the diagonals — acceptable for a first working version,
   and `test_gmrf_prior_is_isotropic` becomes an xfail with a documented
   reason rather than a blocker.
2. **`Q`/Cholesky machinery too fragile → use a separable prior temporarily**
   (independent RW1 along rows and columns). This was ruled out for the final
   design because it cannot represent genuine `μx`–`μy` covariance in the
   *prior*. But note carefully: the **likelihood can still recover correlation**
   in the posterior, because the data constrain the joint distribution
   directly. A separable *prior* is a weaker regulariser, not a wrong model.
   So this is a legitimate way to get the rest of the 2D pipeline working end
   to end, then swap the prior in afterwards. **If you are stuck, take this
   rung** — it decouples the two hard problems.
3. **Everything 2D is fighting you → build the design matrix and the
   likelihood only, and fit with a fixed, user-specified `smoothness_sigma`**
   (no hyperprior). Removes the hierarchical layer, which is where most of the
   sampling difficulty lives. Ship that as "2D minimally working", and add the
   hyperprior as a follow-up.

### 3.3 Tests for the 2D solver

Mirror the 1D structure. The mock truth should be a **tilted** bivariate
Gaussian — correlation is the whole reason we are not doing two independent 1D
fits, so it must be in the primary test.

- `test_gmrf_Q_properties` (fast): `Q` symmetric; row sums `≈ 0`;
  `Q @ ones ≈ 0`; second-smallest eigenvalue `> 0` (connected graph);
  `Q_reg` positive-definite (Cholesky succeeds).
- `test_gmrf_prior_is_isotropic` (fast): draw from the prior with `z ~ N(0,I)`;
  assert the empirical spatial autocorrelation of `x` is the same along
  x, y, and the two diagonals to within Monte Carlo error. This is the test
  that the 8-connectivity weighting is doing what we claim, and it is the one
  that would catch a row-major/column-major transposition in the adjacency
  construction — the single most likely 2D bug.
- `test_design_matrix_2d_factorises` (fast): for diagonal `C_i`, the box-
  integrated 2D matrix equals the outer product of two 1D calls, to 1e-12.
- `test_design_matrix_2d_vs_quadrature` (fast): vs. `scipy.integrate.dblquad`
  on a few cells, including a correlated `C_i`.
- `test_recover_tilted_gaussian` (slow): mock from `Σ = [[σx², ρσxσy],
  [ρσxσy, σy²]]` with `ρ = 0.6`. Assert the recovered **covariance matrix**
  (all three independent components) matches truth within the posterior
  half-68CI. Recovering `ρ` is the acceptance criterion for "2D minimally
  working" — recovering the marginals only would be passed by a separable
  model too, so this is the test that justifies the whole GMRF design decision.
- `test_recover_isotropic_gaussian` (slow): the `ρ = 0` control.
- `test_2d_marginal_matches_1d` (slow): fit 2D, marginalise over `μy`, compare
  to a 1D fit of the `μx` data alone. They should agree — a strong consistency
  check that costs one extra assertion.
- **SBC for the 2D model** (slow, CI-only): reuse the §1.2 harness with the 2D
  model and test quantities `{mean_x, mean_y, σx, σy, ρ, smoothness_sigma,
  log p(θ̃)}`. Use `K = 10` (100 cells) for the SBC configuration — small
  enough to be affordable, large enough to exercise the same code paths.

### 3.4 Performance gate and the SVI question

`K² = 400` latent dimensions with a Gaussian prior and a smooth likelihood is
well within NUTS's comfort zone — the geometry is close to Gaussian, which is
exactly what the non-centred parameterisation buys. **Try NUTS first.**

Define the gate explicitly so it is a measurement and not a vibe:

> Run `K = 20`, `N = 5000` stars, 500 warmup + 1000 samples on CPU.
> **Proceed** if wall time < 10 min *and* `min(ESS)/n_samples > 0.1` *and*
> `max(r_hat) < 1.01` over 4 chains.
> Escalate only if it fails.

Escalation ladder, in order (each step is cheaper than the next):

1. Reduce `K` to 15 and check whether the science tolerates it.
2. `dense_mass=True` in NUTS (400×400 mass matrix is affordable and can
   dramatically help correlated posteriors).
3. GPU — the design matrix multiply is the bottleneck and it is pure BLAS.
4. `numpyro.infer.Pathfinder` for initialisation only, still sampling with NUTS.
5. Full SVI with a low-rank multivariate-normal guide.

Step 4 is the sweet spot if we get there: it keeps the exact posterior and only
replaces the expensive warmup. **Do not build 5 speculatively** — and note that
if we do, SBC (§1.2) is not optional for it, because variational posteriors are
systematically too narrow and SBC's ∪-shaped histogram is the standard way to
quantify by how much.

### 3.5 Explicitly deferred (out of scope for "minimally working")

- Marginalisation over a missing PM axis (stars with only one component).
  Straightforward once the machinery works — sum `M` over the missing axis.
- 3D joint LOS + 2D PM. The design matrix becomes `(N, K³)`; `K = 15` is
  3375 cells, still tractable, but it is a separate project.
- Dynamite 2D output format. Needs a format spec first; `context/` has the 1D
  BayesLOSVD spec only.
- Spatial coherence across Voronoi bins (the "someday" item). Note that the
  GMRF machinery built here is *exactly* what that needs, one dimension up —
  worth keeping the adjacency-matrix construction generic (take a neighbour
  list, not a grid shape) so it is reusable. That is the one piece of
  generality worth paying for now.

---

## Part 4 — Housekeeping (P1–P2)

### 4.1 Ruff (P1, 15 min)

`ruff check --fix src/` clears the 8 EM101/EM102 violations. Do it in the same
commit as §0.5 (the `assert` → `raise ValueError` conversions), since both are
edits to the same `raise` statements. Then add `ruff check` to
`.github/workflows/tests.yml` so it stays clean.

### 4.2 In-code TODOs (P1)

- **`veldist.py:161` — "test this vs simple Dirichlet prior on the weights."**
  Superseded by §0.1: `total_flux` is being deleted. A Dirichlet prior directly
  on the weights is a *different* proposal — it would replace the softmax-of-
  RW1 construction entirely, and it cannot express smoothness (Dirichlet is
  exchangeable over bins; neighbouring velocity bins would be a priori
  independent, which is the opposite of what we want). **Resolve as: rejected,
  with reason.** Write the two sentences into `docs/theory.md` and delete the
  comment.
- **`veldist.py:358` — "decide if we want to work in density space internally."**
  Resolve as: **stay in mass space.** Mass is what the likelihood needs (it
  multiplies the design matrix, which is also a mass), what the simplex
  constraint applies to, and what the Dynamite ECSV format expects. Density is
  a presentation concern and belongs only in `plot_result`. Delete the comment
  and add one line to the `intrinsic_pdf` docstring making the convention
  explicit. Add `test_intrinsic_pdf_is_mass_not_density` asserting every
  posterior row sums to 1 regardless of bin width.

### 4.3 Docs (P2 — smaller than TASKS.md claims)

The sweep contradicts the ticket: `docs/theory.md` (8.5 KB) and
`docs/examples.md` (8.2 KB) are real, substantive prose, not placeholder text.
The actual docs debt is narrower:

- `docs/generate_all_example_images.py` and `docs/generate_example_images.py`
  are near-duplicates (24 KB and 12 KB) and **both** import the legacy
  `compute_moments`. Delete the smaller one, port the survivor to
  `compute_summary`.
- Add the resolved-TODO reasoning from §4.2 to `theory.md`.
- Once §1.3 has run, add a short "Validation" page reporting the SBC and
  coverage results. This is the section reviewers will actually want, and it is
  nearly free once the tests exist — the tests emit the numbers.
- Once §0.1–0.3 land, `theory.md`'s model description needs updating to the
  GMRF form.

Deprecating `compute_moments`: emit a `DeprecationWarning` now, remove after
the docs scripts are ported. It is a 20-line function with a superset
replacement.

### 4.4 Staleness sweep findings (P2, mechanical)

From a pass over `CLAUDE.md`, `README.md`, `docs/`, and config:

**Two parallel, orphaned documentation-image systems.** The docs pages
reference exactly four images — `fig_prior`, `fig_design_matrix`,
`fig_bimodal`, `fig_kin_maps` — produced by the matching `docs/fig_*.py`
scripts. Everything else is dead weight:

- `generate_example_images.py` (12 KB) and `generate_all_example_images.py`
  (24 KB) generate a *different* set of 12 images (`ex1_*`, `example_*`) that
  **no `.md` page references**. Both import legacy `compute_moments`.
- `docs/README_IMAGES.md` and `docs/images/README.md` document that dead
  system, and contradict each other about which script to run.
- 15 orphaned PNGs in `docs/images/` (of 19 tracked; 2.2 MB total).
- `fig_deconvolution.py` and `fig_summary_metrics.py` produce
  `fig_deconvolution.png` / `fig_summary_metrics.png`, which are also
  unreferenced — but these are *current*-API scripts and the figures look like
  ones the docs should probably be using. **Decide: reference them from
  `examples.md`, or delete the scripts.** Do not leave them in limbo.

Recommended: delete both `generate_*_images.py`, both image READMEs, and the
15 orphaned PNGs; replace with a single `docs/make_figures.py` that runs the
`fig_*.py` scripts. This also resolves the "port to `compute_summary`" item
above by deletion rather than by porting.

**Doc/code contradictions:**

- `theory.md` states the likelihood as `ln L = Σ ln([Mw]_i)` — with **no
  `total_flux` term**. The docs already describe the correct model; the code
  has the extra broken term (§0.1). Good corroboration, no docs change needed
  once §0.1 lands.
- `theory.md` writes `u₀ ~ N(0, σ_smooth)` but the code hardcodes
  `curve[0] = 0.0`. Harmless (softmax is shift-invariant, so the `u₀` draw is
  an unidentified shift) but it should be stated correctly, and §0.2 rewrites
  this passage anyway.
- `theory.md`: "Users can run on GPU by passing `gpu=True`" — implies opt-in,
  but `gpu=True` is the *default* (§0.4). Fix together with the code change.
- `CLAUDE.md` says `run()` "infers `intrinsic_pdf` and `smoothness_sigma`",
  omitting `steps` and `total_flux`. Will be accurate once §0.1 removes
  `total_flux`; add `steps`/`x` then.
- `README.md`, `index.md`, and `examples.md` all say Dynamite
  "`histLOSVD` / BayesLOSVD" format. `context/dynamite_format_spec.md` only
  ever says `BayesLOSVD`. **Verify against Dynamite's source which class name
  is right** and use one name consistently — this is the kind of thing that
  costs a user an afternoon.

**Config drift:**

- `docs/conf.py` hardcoded `version = release = '0.1.0'` while
  `pyproject.toml` says `0.2.0`. *Fixed* — now read from installed metadata.
  Note the local editable install is itself stale at 0.1.0; re-run
  `pip install -e ".[dev]"`.
- `ruff` is used but is **not** in `[project.optional-dependencies].dev` and
  is **not** run in CI. Add both (folds into §4.1).
- `.readthedocs.yaml` has a redundant `post_install: pip install .` on top of
  the `python.install` block below it. Harmless, but it doubles build time.
- `.gitignore` still had a `# Beads / Dolt files` block. *Fixed* — removed.
- `CLAUDE.md`'s Build & Test section documents pytest but not `ruff check`.
- **Memory directory is empty** — nothing stale there.

---

## Suggested order

| # | Work | Gate |
|---|---|---|
| 1 | §4.1 ruff + §0.5 asserts | trivially green |
| 2 | §1.1 `test_analysis.py` + §1.4 design matrix | fast tests, closes the P0 coverage gap |
| 3 | §2 pipeline/export round trip | fast tests via the `fake_solver` fixture |
| 4 | §1.2 SBC harness, **run against the current model** | expect to see §0.2's asymmetry |
| 5 | §0.1–0.3 model fixes | SBC goes green — this is the payoff |
| 6 | §1.3 coverage + §4.2 TODOs | 1D declared robust |
| 7 | §3.1–3.3 2D solver | tilted-Gaussian recovery |
| 8 | §3.4 performance gate | measure before escalating |
| 9 | §4.3 docs incl. validation page | |

Steps 1–3 are a day and are pure risk reduction. Step 4 is the one with real
uncertainty in it — budget for the `vmap` work being fiddly. Steps 5–6 are
where the 1D solver actually becomes defensible. The 2D solver does not start
until step 7, and that is deliberate: it inherits every piece of validation
machinery built above, which is far cheaper than building it twice.

---

## Part 5 — Symptom → cause triage

When something breaks, check here before debugging from scratch.

| Symptom | Most likely cause |
|---|---|
| `smoothness_sigma` posterior collapses to ~0; LOSVD is flat | Missing `−(rank Q)·log σ` normalisation in a `numpyro.factor` prior (§0.2, §3.2) |
| All-NaN posterior in the 2D solver | Cholesky of singular `Q` returned NaN silently — ridge missing or too small (§3.2) |
| 2D recovers correct `σx`, `σy` but wrong-sign `ρ` | Row-major/column-major transposition in the adjacency build (§3.2) |
| `diag(Q)` is constant | Periodic boundary conditions leaked in via array shifts (§3.2) |
| SBC ranks are ∪-shaped | Check ESS **first** — autocorrelated draws mimic overconfidence. Only if ESS is fine is the posterior genuinely too narrow (§1.2) |
| SBC ranks show a monotone ramp | Genuine bias. Suspect the prior asymmetry (§0.2) or a design-matrix half-bin offset (§1.4) |
| SBC passes suspiciously easily on first write | Prior draw is probably hand-rolled and out of sync with the model (§1.2) |
| SBC rank ~0 (or ~L) in nearly every simulation for one specific quantity, others pass | A `numpyro.factor`-based prior term — `Predictive` cannot forward-sample through it, so the SBC "truth" is drawn from the wrong distribution. Rewrite the prior generatively (real `sample` sites + deterministic transform), not as a `factor` penalty. This actually happened during §0.2's implementation — see the §1.2 gotcha for the full story (§0.2, §1.2, §3.2) |
| Single SBC simulation fails with "inadequate ESS" | Normal NUTS variability at small `n_sims`/`num_samples`, not necessarily a bug — but don't just drop it silently. First increase `num_warmup`/`num_samples` a bit and see if it goes away before suspecting the model (§1.2) |
| Coverage under-nominal for `kurtosis`/`tail_weight` only | Expected — smoothness-prior shrinkage. Quantify and document, do not "fix" (§1.3) |
| `clip_uncertainties` assertion `mean_sum ≈ 1.0` fires | Test fixture is not producing valid simplex rows (§2) |
| Marginal medians sum to ~0.9 | **Correct and expected.** Independent marginals of a joint simplex posterior. Do not renormalise (§2) |
| Dynamite reports `econ` zeros | An uncertainty floor was bypassed — check `clip_uncertainties` ran on every bin |
| Wrong LOSVD attached to a sky position | `orig_to_new` remapping in `write_dynamite_kinematics` (§2) |
| 2D design matrix rows sum to ≈`1/dA` | Missing bin-area factor (§3.1) |
| OOM building the 2D design matrix | Chunk over stars; peak is 2–3× the final array (§3.1) |
| NUTS divergences right after the §0.2 prior change | Expected geometry change — try `target_accept_prob=0.9` before investigating |

**General rule:** a *statistical* test failing (SBC, coverage) means either the
model or the test is wrong, and you must determine which before touching
tolerances. A *deterministic* test failing (design matrix, index arithmetic,
`Q` properties) is always a code bug — those have exact answers.

---

## Sources

- [Talts, Betancourt, Simpson, Vehtari, Gelman (2018), *Validating Bayesian
  Inference Algorithms with Simulation-Based Calibration*](https://arxiv.org/pdf/1804.06788)
- [Stan User's Guide — Simulation-Based Calibration Checking](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html)
- [`sbi` — Simulation-based calibration diagnostics (ECDF-based rank tests)](https://sbi-dev.github.io/sbi/v0.23.1/tutorials/11_diagnostics_simulation_based_calibration/)
