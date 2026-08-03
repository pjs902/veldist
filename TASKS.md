# Tasks

Detailed implementation plans for everything below: **`PLAN.md`**.
Suggested execution order is the table at the end of that file.

## Now

- **[P0] SBC regression from the RW3 scaling fix** (`tests/test_calibration.py`)
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

- **[P1] Skew-normal LOSVDs are recovered as symmetric at N~150**
  Measured: posterior median skewness +0.038 against a true +0.851, with a
  half-68CI of 0.078 — the posterior is *confidently* symmetric, missing by
  ten interval half-widths. Not a resolution artefact (n_bins=55 is no
  better). The model is consistent — posterior `sigma3` goes 0.277 → 0.943 →
  3.396 and recovered skewness +0.001 → +0.111 → +0.760 as n_stars goes
  150 → 400 → 2000 — so this is prior shrinkage dominating a weak likelihood,
  not an inability to express skew (15.5% of prior draws exceed |skew| 0.85).
  Science impact: at 150 stars per Voronoi bin, which is realistic, genuine
  h3 signal is erased and reported with small error bars. Fixing this and the
  SBC regression is likely one design change, not two: both point at the
  shape of the `sigma3` prior.

- **[P2] Heavy-tailed kurtosis coverage still open** (`PLAN.md` §1.3)
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

- **[P1] 2D solver: §3.5 deferred items** (`PLAN.md` §3.5)
  `KinematicSolver2D` is now feature-complete for "minimally working" per
  the plan's own acceptance criterion (recovering ρ from a tilted bivariate
  Gaussian — done, passing), has its own SBC harness (passing, 6/6
  quantities), and has passed the §3.4 performance gate (see Completed).
  What's left: the explicitly-deferred §3.5 items (PM-axis marginalisation,
  3D, Dynamite 2D output format).

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
