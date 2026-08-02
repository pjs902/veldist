# Tasks

Detailed implementation plans for everything below: **`PLAN.md`**.
Suggested execution order is the table at the end of that file.

## Now

- **[P0] Fix kurtosis tail-leakage bias in `compute_summary`** (`PLAN.md` §1.3
  "ACTUAL outcome")
  Found by the coverage tests: `compute_summary`'s kurtosis is biased
  +1.6 to +2.5 even for a plain Gaussian truth (should be ~0). Root cause:
  the RW1 prior leaks a small amount of posterior mass into far-edge grid
  bins; kurtosis's 4th-power weighting amplifies it ~625×. `truncate_losvd()`
  already exists to suppress exactly this leakage but only patches
  `clipped_samples` (Dynamite export), never the raw samples
  `compute_summary` consumes — so this is currently live and unmitigated for
  any `analysis.py` user. `tests/test_coverage.py` is marked `xfail` pending
  this fix. Directions in the plan: extend tail suppression to the raw-sample
  path, and/or widen grid margin guidance.

- **[P1] 2D solver: continue `KinematicSolver2D`** (`PLAN.md` §3.3–3.5)
  Grid, design matrix (box integration + 2×2 Gauss-Legendre for correlated
  covariance), and GMRF prior (§3.1/§3.2) are done and fast-tested
  (`src/veldist/veldist2d.py`, `tests/test_veldist2d.py`). Smoke-tested
  end-to-end (real MCMC recovers a tilted-covariance mean correctly). Still
  open: the slow recovery tests (tilted/isotropic Gaussian covariance
  recovery, 1D-marginal consistency), SBC for the 2D model, and the §3.4
  performance gate. Try NUTS first — K²=225 (K=15) already ran fine in the
  smoke test; only reach for SVI/Pathfinder if the performance gate fails.

- **[P2] SVI / Pathfinder as NUTS fallback**
  Conditional on the 2D solver above actually needing it — don't build
  speculatively.

- **[P2] Validation page in docs**
  Once §1.2 SBC and §1.3 coverage results exist, add a short "Validation"
  page to the docs reporting them — cheap once the tests exist, since the
  tests emit the numbers.

## Someday (long-term, not active)

- Multiple kinematically distinct tracers: separate design matrices per
  stellar population
- PyPI packaging
- Spatially coherent velocity distribution inference (joint across bins,
  shared smoothness scale-length) — feasibility unclear, revisit once 2D
  solver and export pipeline are both tested and stable

## Completed

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
