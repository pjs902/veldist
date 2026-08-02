# Tasks

Detailed implementation plans for everything below: **`PLAN.md`**.
Suggested execution order is the table at the end of that file.

## Now

- **[P0] Statistical validation framework** (`PLAN.md` §1)
  Analytic unit tests for `analysis.py` and design-matrix correctness: done
  (`tests/test_analysis.py`, `tests/test_design_matrix.py`). SBC harness
  (`tests/test_calibration.py`) in progress against the now-corrected model.
  Still open: §1.3 frequentist coverage over repeated mocks.

- **[P1] 2D solver: `KinematicSolver2D` for bivariate PM distributions**
  Design is settled: square K×K grid, (N, K²) design matrix using centre-point
  bivariate Gaussian PDF × bin area, 2D GMRF prior with 8-connectivity
  (separable 1D random walks ruled out — can't represent genuine covariance).
  Precision matrix Q shape (K², K²); add ridge ε·I before Cholesky. Latent:
  `z ~ N(0, I_K²)`, `x = σ_smooth · L⁻ᵀ z`. Start with K=20. Test on mock
  tilted bivariate Gaussian. Defer: 4-corner box integration, marginalisation
  over missing PM axis, Dynamite 2D output.
  Try NUTS first — K²=400 with a sparse GMRF prior should be fine. Only
  reach for SVI/Pathfinder (below) if that measurably stalls.

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
  replaced the pinned-cumsum random-walk prior with a translation-invariant
  intrinsic RW1 GMRF; made `smoothness_sigma` grid-resolution-invariant;
  `run(gpu=...)` now defaults to leaving the platform untouched instead of
  crashing on GPU-less machines
- Test coverage for `analysis.py` and the batch/export pipeline (P0):
  `tests/test_analysis.py`, `tests/test_design_matrix.py`,
  `tests/test_pipeline.py`, `tests/test_model.py`
- Fix ruff lint violations (P1): `src/` is ruff-clean; `ruff check src/`
  now runs in CI
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
