# Tasks

Detailed implementation plans for everything below: **`PLAN.md`**.
Suggested execution order is the table at the end of that file.

## Acceptance criterion

**Minimally correct = recover `v_mean` and `sigma` well, with properly
calibrated uncertainties.** (Peter, 2026-08-04.) If `h3`/`h4` cannot be
recovered well but the uncertainties are calibrated, that is an acceptable
outcome.

This settles the trade that has driven the last several rounds of work.
Loosening the `sigma3` prior buys h3/h4 coverage and breaks SBC; **calibration
wins.** Work aimed only at freeing h3/h4 is out of scope, which retires two
items that were previously P0/P1 (penalty order, mode-order split scale — both
measured, both ruled out). What remains is: pick the prior that passes SBC,
confirm `v_mean`/`sigma` coverage and efficiency survive it, and report the
h3/h4 limitation honestly rather than engineering around it.

## Now

- **[P0] The adopted prior fails SBC, worse than the one it replaced.**
  Measured 2026-08-04 at the adopted default (`SIGMA3_RATE=0.35`,
  `prior="gaussian_core"`, n_sims=30): **6/30 (20%) simulations failed** — NaN
  posterior, sampler exception, or inadequate ESS — against the 2% budget.
  The old `Exp(2.303)` default was 2/30 (6.7%). Loosening made it **3× worse**.
  This is the predicted mechanism, not a surprise: the failure was already
  diagnosed as the `Exponential` PC prior's upper tail (`sigma3` ~3–4
  saturating the softmax into near-delta LOSVDs), and loosening the rate makes
  that tail fatter. `SIGMA3_RATE` was selected on coverage alone; coverage is
  blind to this.
  **Consequence: `Exp(0.35)` is not science-ready and the regularisation
  decision is not closed.** Coverage improved (26/45 → 41/45 in-band at σ=22)
  and calibration regressed; those were traded without the trade being
  measured. Options, none yet tested: a prior with a lighter upper tail at the
  same body (the six one-parameter families already swept all shared the
  problem, so this likely means a bounded or truncated scale), or accepting a
  tighter rate and recovering coverage elsewhere.
  Reproduce: `pytest "tests/test_calibration.py::test_sbc_calibration[gaussian_core]"`.
  **Note the runner exits 0 on failure — check the report text, not `$?`.**

- **[P1] Mode-order split scale.** With `rw_order` ruled out, this is the
  remaining candidate for freeing h3/h4 without breaking SBC. Two `sigma3`
  parameters — a loose one for the lowest few deviation modes (carrying h3/h4)
  and a tight one for the rest — attacks the problem the plan identified:
  one scalar cannot serve components needing opposite treatment.
  Sequence this *after* the SBC item above: a split scale is harder to
  calibrate than one scalar, and starting it from a base that fails SBC means
  debugging two unknowns at once.

- **[P1] Matched-grid per-bin fitting** — the diagnosed remedy for the σ=7
  collapse (see below). Fit each spatial bin on a grid matched to its local σ,
  aggregate posterior samples onto the shared DYNAMITE output grid. Exact
  provided the fitted grid is at least as fine as the output grid and edges
  align — a precondition to enforce, not assume.

## Open, previously misfiled as resolved

These sat under "Resolved by the validation campaign" but were never resolved:
the SBC regression (now re-measured and worse, promoted to Now above), the
heavy-tailed kurtosis coverage item, the efficiency check, and the 2D §3.5
items. Campaign **Task 6 ("Put SBC in the loop") was never executed** — which
is why the adopted default shipped without its gate being re-measured.

## Resolved by the validation campaign

**Validation campaign plan:** `docs/superpowers/plans/2026-08-03-validation-campaign.md`
(8 tasks: penalty order RW3/4/5, per-bin LOSVD calibration, map-level bias,
SBC in the loop, sigma=7 diagnosis, decision at n_real=100).

- **[P0] Regularisation adopted:** `Exp(0.35)` (`SIGMA3_RATE=0.35`), `rw_order=3`.
  41/45 in-band at σ=22 (n_real=100). Set as default.
  `docs/superpowers/specs/2026-08-03-regularisation-decision.md`.
  ⚠️ **Adopted on coverage evidence only — it fails SBC at 20%. See the P0
  item under Now; treat this default as provisional.**

- **Penalty order hypothesis ruled out.** Orders 4 and 5 do not free h3/h4
  moments — the softmax nonlinearity decouples the log-density polynomial null
  space from PDF moments. Documented in the measurements doc and the xfail
  markers on `test_penalty_order_controls_which_moments_are_free`.

- **Map-level bias measured.** skew_normal_h3 skewness shows 21.4σ bias over
  200 bins at the current prior. New test `test_map_level_bias_is_small_against_the_map_uncertainty`
  marked `xfail(21.4σ)`.

- **σ=7 collapsed diagnosed.** The empty grid is the dominant cause, not err/σ.
  Matched-grid fitting is the recommended remedy.
  `docs/superpowers/plans/2026-08-03-narrow-dispersion-diagnosis.md`.

- **[P0] The sigma=7 bins fail at every prior strength**
  Corrected-profile sweep, n_real=25, four Exponential rates from 2.303 to
  0.20 (a 12x span in prior scale):

  | sigma | in-band /45 | catastrophic | h3/h4 in-band /6 |
  |---|---|---|---|
  | 22 | 33 → 40 | 7 → 1 | 1 → 4 |
  | **7** | **22 → 23** | **16 → 11** | **0 → 0** |

  At sigma=22 loosening helps steadily. At sigma=7 nothing moves: every h3/h4
  entry stays 0.00–0.36, and in-band goes 22→23 out of 45. Efficiency is ~1
  throughout (0.91–0.93 on the mean, 1.08–1.23 on sigma), so this is not an
  information-loss problem — the posterior is confidently wrong, not noisy.
  Two candidate causes with different remedies: only 30% of the shared grid
  carries mass at sigma=7, and err/sigma is 0.36 there versus 0.11 at
  sigma=22 (the dwarf-spheroidal regime, per Amorisco & Evans 2012).
  **Task 7 of the campaign separates them before any regularisation is
  adopted** — if it is the grid, the remedy is per-bin fitting with
  aggregation onto the shared output grid; if it is err/sigma, there is no
  remedy and the map must be reported with the limitation stated.

- **[P0] SBC regression from the RW3 scaling fix** — **STILL OPEN, re-measured
  and worse (6/30). Promoted to the Now section above; kept here for the
  diagnosis history.** (`tests/test_calibration.py`)
  `test_sbc_calibration[gaussian_core]` passes on `main` and **fails on
  `fix/rw3-deviation-scaling`**: 2/30 simulations fail (NaN posterior, sampler
  exception, or inadequate ESS) against a 2% budget. The implementation plan's
  test matrix did not include SBC, so this was not caught.
  The `main` pass was vacuous — with the deviation inert, `sigma3` did not
  affect the simulated data, so its posterior reproduced its prior and ranks
  were uniform by construction. This is the first informative run of the test.
  Likely cause is the `Exponential` PC prior's upper tail: prior-predictive
  draws reach `|skewness|` p99 = 12.2 with `sigma3` ~3–4, saturating the
  softmax into near-delta functions. `SIGMA3_RATE` was pinned on
  `|excess kurtosis|` p90 alone, which is blind to that tail.
  **Blocks calling this prior science-ready.** Numbers in
  `docs/superpowers/plans/2026-08-03-rw3-measurements.md`.

- **[P2] Efficiency check against the statistical optimum** — **OPEN, never done.**
  Amorisco & Evans (2012) §3.3 show unbinned ML attains StD(μ) = 1/√N and
  StD(σ) = 1/√(2N), while *binned* Gauss-Hermite does not reach the latter —
  binning costs precision even in the second moment, before any higher-moment
  question. Their unbinned gain over binned GH is ≈2 on h4 at N=800 and larger
  on h3, quoted as a lower bound (measured with zero uncertainty and uniform
  membership). veldist is unbinned in velocity with exact per-star errors, so
  it should inherit this. Measuring our StD(σ) against 1/√(2N) is a cheap,
  sharp test of whether the implementation extracts what the data contain.

- **[P0] External baseline on N — another group's results, not a hard limit**
  Amorisco & Evans (2012, MNRAS 424, 1899): "extremely difficult to measure
  reliably the shape of any velocity distribution with a sample size
  significantly smaller than N = 200". Sanders & Evans (2020, MNRAS 499,
  5806): negative excess kurtosis needs ~200 stars, positive needs ≳2000.
  On the error floor, an earlier note here was too pessimistic: SE20 find
  sign determination generally needs ≲2 km/s, but **negative excess kurtosis
  remains detectable to ~5 km/s**. Scaled to ω Cen's σ ≈ 9–21 (their tests are
  for σ ≈ 11), oMEGACat's 2–3 km/s is comfortably inside the achievable regime
  for our dominant signal. Full detail and corrections in
  `docs/superpowers/specs/2026-08-03-discrete-losvd-literature.md`.
  **Treat these as calibration of what has been achieved, not as proven
  bounds.** Neither paper derives a Cramér-Rao bound; both report what their
  own per-bin maximum-likelihood estimators delivered, and both report
  *significant detection* thresholds, which is a stricter question than
  posterior calibration. Neither pools information across spatial bins.
  They do explain our coverage table as expected rather than anomalous:
  student_t (h4>0) 0.000 at N=150 is 13x short of their required sample.
  **The sharp version:** Sanders & Evans note the uncertainty on h3/h4 for
  small samples is ≈ (2N)^(−1/2). Against our targets h3 ≈ 0.066, h4 ≈ 0.051
  that gives, per bin: N=150 → 1.1σ/0.9σ, N=200 → 1.3σ/1.0σ, N=250 →
  1.5σ/1.1σ, N=2000 → 4.2σ/3.2σ. So a per-bin h3 is a **~1σ measurement at
  any affordable N**, and 150 → 200 buys 15% in significance, not a
  qualitative change. A correctly calibrated posterior should be returning
  h3 ≈ 0.07 ± 0.06 per bin; ours returns far tighter, centred near zero.
  **Near-term stance: target N=200 per bin.** Matches the published baseline,
  and is cheap — 30k spectra still gives ~150 Voronoi bins across r_h, far
  more than the 27 apertures of van de Ven et al. A calibrated 1σ-per-bin
  result is a respectable product because ~150–200 bins constrain the *map*
  far better than any bin constrains itself; see the hierarchical item below.
  See `docs/superpowers/specs/2026-08-03-path-forward.md`.

- **[P0] Regularisation strength for our use case** (`generate_gaussian_core_curve`)
  The output stays the nonparametric histogram LOSVD written as DYNAMITE
  `BayesLOSVD` — no parametric model is being adopted for inference. The whole
  problem is therefore setting the deviation prior's strength so the posterior
  is *calibrated* at N=150–200 with err/σ ≈ 0.15.
  Measured diagnosis: a single scalar `sigma3` controls all 40 latent
  dimensions at once, and the low-order modes (which carry h3/h4) and the
  high-order modes (wiggle) need opposite treatment. Loosen enough to free
  h3/h4 and the softmax saturates, breaking SBC; tighten enough to control the
  wiggle and h3/h4 are crushed. Six prior families (Exponential, HalfNormal,
  Gamma ×2, LogNormal ×2) all hit this, so it is not a matter of picking a
  better one-parameter prior.
  Two candidate fixes, both staying nonparametric:
  - **Split the scale by mode order.** Separate penalties for the low-order
    deviation modes and the high-order ones, instead of one `sigma3`. Directly
    targets the measured cause.
  - **Reduce `n_bins`.** At N=150 over 40 bins there are ~3.75 stars per bin;
    20 bins gives 7.5 and halves the dimensionality mismatch at no modelling
    cost. Cheapest thing to try and not yet tested against *coverage*.
  **Sweep done** (12 configs, n_bins × Exponential rate, scored on coverage +
  efficiency; full table in the measurements doc). Established:
  - The current prior is worst or near-worst on every axis at every n_bins:
    26/45 in-band, 11–13 catastrophic entries, ~zero h3/h4 coverage, and the
    worst σ efficiency in its row (2.69× the statistical optimum at n_bins=20,
    4.27× at 40). Nothing recommends keeping it.
  - **Coverage is monotone in looseness and never turns over**, so it cannot
    select an optimum alone. Efficiency and SBC are the binding constraints.
  - **Empty grid dimensions are expensive**: n_bins=40 sits at 4.2–4.3× σ
    efficiency regardless of prior strength — the prior cannot rescue it.
  - Leading candidate `n_bins=20, Exp(0.35)`: 33/45, 5 catastrophic, 1.35× σ
    efficiency. Direction, not a value to adopt — see the caveat below.
  **Two caveats that gate adopting any of it.** (a) The n_bins axis is
  confounded with the empty-bin fraction: the harness grid is a fixed 320 km/s
  against σ=15–42 truths, so at n_bins=20 only ~6 of 20 bins carry mass, and
  "20 beats 40" may be reading "fewer empty dimensions". (b) The efficiency
  numbers are the std of 25 medians and so are outlier-sensitive; the n_bins=30
  row is non-monotonic (3.11 → 1.20 → 2.99 → 3.27) in a way that looks like bad
  realisations, not physics. Re-measure with a robust scatter and keep the
  per-realisation medians.

- **[P0] Size the velocity grid to the data**
  The current grid is simultaneously too wide and too coarse. At 320 km/s with
  20 bins the resolution is 16 km/s — **6.4× coarser than the 2–3 km/s
  measurement errors** — while only ~29% of bins carry any mass. Both are
  fixed together by sizing to the data.
  DYNAMITE requires one shared velocity grid for all spatial bins, so it must
  hold the σ≈20 km/s central dispersion plus the ~10 km/s rotation span in the
  mean velocity: **160–200 km/s wide, 5 km/s bins (2× the measurement error;
  finer is unrecoverable since errors convolve it away) → 32–40 bins.** That is
  3× finer velocity resolution *and* a higher informative fraction than the
  current setup. Outer bins at σ≈9 will still only use ~45% of the grid — with
  a factor-2.2 dispersion range some emptiness is structural.
  This likely also explains `n_sigma_truncate`: it was added to suppress
  residual mass in far-edge bins, which is the symptom of a grid this wide.
  Fixing the grid may retire the band-aid.
  **Possible further win:** the DYNAMITE constraint is on the *output*, not the
  inference. veldist could fit each spatial bin on a grid matched to its local
  σ, then aggregate each posterior LOSVD sample onto the shared output grid
  before taking medians and intervals. If the fitted grid is at least as fine as
  the output grid and their edges align, that aggregation is exact — just
  summing mass per sample — so uncertainties propagate correctly. Not yet
  investigated.
  Also check whether the harness truths need rescaling: `gaussian` at σ=30 and
  `bimodal_counter_rotation` at σ=42 are both well outside ω Cen's 9–21 range.

- **[P2] Heavy-tailed kurtosis coverage still open** — **OPEN.** (`PLAN.md` §1.3)
  The RW3 deviation scaling fix improved coverage broadly but did not close it.
  `test_coverage_over_mock_realisations[gaussian_core]` still fails, as it did
  on `main` before the fix — this is not a regression. Both columns below are
  measured; full table in `docs/superpowers/plans/2026-08-03-rw3-measurements.md`.
  - Skew-normal skewness and kurtosis: 0.000 → 0.000, still 0/25. The
    credible intervals never contain the truth, and the finite-sample
    achievable skewness at N=150 is 0.811 vs a true 0.851, so this is not an
    estimator artefact. **This is the sharpest open question** — it is the one
    result the scaling fix was expected to move and did not.
  - Student-t kurtosis: 0.000 → 0.040, still below the 0.30 catastrophic
    floor; tail_weight 0.120 → 0.160. True excess kurtosis 2.82 vs achievable
    1.18 at N=150, so about half this gap is the estimator, not the model.
  - Bimodal kurtosis: 0.000 → 0.320, now above the floor but below the nominal
    band. Bimodal tail_weight went 0.000 → 1.000.
  The earlier claim that this was an "irreducible finite-data limitation" has
  been withdrawn: that diagnosis was made with an inert deviation term
  (SD ~0.0036) that could not respond to any non-Gaussian truth. What replaces
  it is not a new diagnosis — the cause is genuinely not yet known. Note that
  the Gaussian truth's kurtosis coverage was 1.000 both before and after, so
  that row is evidence for nothing in either direction.

- **[P1] 2D solver: §3.5 deferred items** — **OPEN, untouched.** (`PLAN.md` §3.5)
  `KinematicSolver2D` is now feature-complete for "minimally working" per
  the plan's own acceptance criterion (recovering ρ from a tilted bivariate
  Gaussian — done, passing), has its own SBC harness (passing, 6/6
  quantities), and has passed the §3.4 performance gate (see Completed).
  What's left: the explicitly-deferred §3.5 items (PM-axis marginalisation,
  3D, Dynamite 2D output format).

## Considered and closed

- **Parametric near-Gaussian model as the inference model — not adopted.**
  The Sanders & Evans (2020) uniform/Laplace kernel families were proposed as a
  replacement for the nonparametric LOSVD. Rejected: the output path is
  DYNAMITE `BayesLOSVD` (histogram), which the existing writer already
  produces, and converting a parametric posterior into it would either need a
  GH conversion (no closed form — their eq. 5 is only valid to 10% for
  |h4| < 0.01) or project a 4-parameter posterior onto ~40 strongly correlated
  bins that DYNAMITE's χ² would treat as independent.
  **They remain valuable as mock-data generators**, and are already used that
  way: `flat_top_tangential` is their uniform kernel (excess kurtosis −1.00)
  and `rotating_tangential` is the skewed two-piece variant. Their families
  span −0.188 < h4 < 0.145, so they can generate the full expected range.

- **Within-bin rotation gradient faking tangential anisotropy — not a concern.**
  Amorisco & Evans (2012) §2.3 note a velocity gradient across an aperture
  convolves the LOSVD with an arcsine kernel, producing spurious flat-topping
  that reads as negative h4, and that h4 is affected once V_app/σ ≳ 0.7. That
  threshold is far away at our bin sizes. Taking ω Cen's inner rotation
  gradient as ~1 km/s per arcmin (V ≈ 8 km/s at 8′, van de Ven et al.) against
  σ ≈ 15 km/s, and 30k stars over r_h ≈ 4.65′:

  | N/bin | bins | bin radius | ΔV across bin | V_app/σ | margin to 0.7 |
  |---|---|---|---|---|---|
  | 150 | 200 | 20″ | 0.66 km/s | 0.046 | 15× |
  | 250 | 120 | 25″ | 0.85 km/s | 0.059 | 12× |
  | 2000 | 15 | 72″ | 2.40 km/s | 0.166 | 4× |

  Safe by 15× at the operating point, and still 4× at the coarse binning that
  positive-h4 detection would need. Perspective rotation contributes ~0.03 km/s
  across a bin (0.8 km/s over ~20′) and is negligible as a *shape* systematic —
  it remains a correction to the mean velocity field, as van de Ven et al.
  applied, but does not distort the within-bin LOSVD.
  Recorded so this is not re-raised; revisit only if bins ever get much larger.

## Someday (long-term, not active)

- Multiple kinematically distinct tracers: separate design matrices per
  stellar population
- PyPI packaging
- Spatially coherent velocity distribution inference (joint across bins,
  shared smoothness scale-length) — feasibility unclear, revisit once 2D
  solver and export pipeline are both tested and stable

## Completed

- RW3 deviation scaling fix (P0, `docs/superpowers/plans/2026-08-03-rw3-deviation-scaling.md`):
  Replaced the ad-hoc `(bin_width/span)**2.5` scaling with Sørbye–Rue
  generalised-variance standardisation and replaced `HalfNormal(1.0)` with an
  `Exponential` PC prior. The deviation's marginal SD went from ~0.0036 to
  ~1.0, making non-Gaussian LOSVDs visible to the likelihood. The skewed-
  shoulder recovery test (`test_skewed_shoulder_distribution_recovery`) now
  passes (kurtosis z-score < 3, was 199.82). Coverage improved on nine metrics
  and regressed on none, but remains below nominal for the heavy-tailed and
  skewed truths — see the open item above.

- Gaussian-null-space (RW3) prior (P0, `docs/superpowers/plans/2026-08-03-gaussian-null-space-prior.md`):
  Added `generate_gaussian_core_curve` (free Gaussian core + triple-integrated
  RW projected orthogonal to {1, u, u²}), `model_gaussian_core`, and
  `prior=` selector on `KinematicSolver.run()`. Prior-predictive null-space
  tests, sigma/kurtosis bias tests (all pass), SBC for both priors (pass),
  and coverage parametrised over both priors. Default flipped to
  `prior="gaussian_core"` in commit 4b3bca2. `n_sigma_truncate` is now a
  legacy option needed only with `prior="rw1"`.

- 2D solver: §3.4 performance gate (P1, `PLAN.md` §3.4) — measured directly:
  `K=20`, `N=5000` stars, 500 warmup + 1000 samples, 4 chains, CPU. Wall
  time 87.9 s (< 10 min threshold); min(ESS)/n_samples 3.11 across
  `smoothness_sigma`/`z`/`intrinsic_pdf` (> 0.1 threshold); max(r_hat)
  1.0023 (< 1.01 threshold). **PASS on all three criteria**, no escalation
  (K reduction, `dense_mass`, GPU, Pathfinder, SVI) needed. Full numbers and
  reproduction procedure recorded in `PLAN.md` §3.4 "Gate result (measured)".
- Validation page in docs (P2): `docs/validation.md` — reports 1D/2D SBC,
  1D coverage (incl. the still-open kurtosis xfail), 2D recovery, and the
  §3.4 performance gate result; wired into the docs toctree.
- 2D solver core (P1, `PLAN.md` Part 3): `src/veldist/veldist2d.py` —
  grid, design matrix (box integration + 2×2 Gauss-Legendre for correlated
  covariance), GMRF prior (generative `z ~ N(0,I)` + Cholesky whitening
  transform — confirmed Predictive-compatible on first attempt, no factor-
  based bug like 1D's), `KinematicSolver2D`. Tests: `tests/test_veldist2d.py`
  (fast structural tests + slow tilted/isotropic covariance recovery +
  1D/2D marginal consistency, all passing) and `tests/test_calibration_2d.py`
  (SBC, 6/6 quantities pass, 0/30 failed sims). One finding along the way:
  recovery tests needed `n_stars=2000` rather than the plan's suggested
  200–400 — at smaller N the GMRF prior induces a measurable finite-sample
  shrinkage bias in recovered variance (same underlying mechanism as the
  kurtosis bug above: many free grid cells relative to star count). Fixed by
  raising N (a real mitigation), not by loosening tolerances.
- Model correctness fixes (P0): removed the broken `total_flux` term
  (incomplete extended-Poisson likelihood, zero influence on `intrinsic_pdf`);
  replaced the pinned-cumsum random-walk prior with a translation-invariant,
  generatively-sampled intrinsic RW1 prior (increments + mean-centering, not
  a `numpyro.factor` penalty — factors are invisible to `Predictive`, see
  SBC note below); made `smoothness_sigma` grid-resolution-invariant;
  `run(gpu=...)` now defaults to leaving the platform untouched instead of
  crashing on GPU-less machines
- Test coverage for `analysis.py` and the batch/export pipeline (P0):
  `tests/test_analysis.py`, `tests/test_design_matrix.py`,
  `tests/test_pipeline.py`, `tests/test_model.py`
- Fix ruff lint violations (P1): `src/` is ruff-clean; `ruff check src/`
  now runs in CI
- Frequentist coverage tests (P0, `PLAN.md` §1.3): `tests/test_coverage.py`.
  Found the kurtosis tail-leakage bias above — new P0 item added for the fix
- Simulation-based calibration (P0, `PLAN.md` §1.2): `tests/test_calibration.py`.
  Caught and led to fixing a real bug: the RW1 prior's `numpyro.factor`
  implementation was invisible to `Predictive`, so SBC's prior draws didn't
  match what NUTS actually conditioned against. Fixed by rewriting the prior
  generatively (see Model correctness fixes below); now passes cleanly
  (6/6 quantities, p=0.52–0.97, n_sims=30)
- Resolve in-code TODOs (P1): both resolved by the model correctness fixes
  above (density-space TODO resolved as "stay in mass space", documented in
  `plot_result`'s docstring)
- Docs cleanup (P2): removed the orphaned `generate_*_images.py` script pair
  and 15 unreferenced PNGs; referenced the two live-API figure scripts
  (`fig_deconvolution.py`, `fig_summary_metrics.py`) from `examples.md`;
  fixed stale `histLOSVD`/`BayesLOSVD` naming, `docs/conf.py` version
  hardcoding, `.readthedocs.yaml` redundant install step, `.gitignore` beads
  leftovers
- Fix NNLS with multiple kinematic datasets: figure out correct stacking (P1)
- Apply perspective rotation correction to all Gaia PM data (P1)
- SDSS-V data integration and prep pipeline (P1)
- Diagnose and fix cvxopt weight solving broken with h3/h4 (P1, bug)
- Fix Voronoi binning artefacts in low-density outer regions (P1, bug) ×2
- Build PM data prep pipeline (cleaning, membership, coordinate alignment) (P1)
- Run JAM models with PM data only (P2)
- Round-trip integration test: veldist → Dynamite → results (P2)
- Run JAM models with full combined dataset (LOS + PM) (P2)
- Benchmark legacy NNLS vs scipy for speed and memory (P2)
- ω Cen: include outer archival data (Holger's dataset) (P2)
- Orbit integration parallelisation in Dynamite (P3)
- JAM: experiment with dark components as point masses (BH analogues) (P3)
- JAM: test radial M/L ratio variation (P3)
- Bayesian optimisation for Dynamite parameter space exploration (P3, feature)
- Science: DM halo shapes in dwarf galaxies (core/cusp, triaxiality) (P4)
- Science: simultaneous sBH + IMBH modelling, revisit literature detections (P4)
