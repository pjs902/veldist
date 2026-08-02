# Tasks

## Now

- **[P0] Test coverage for `analysis.py` and the batch/export pipeline**
  `compute_summary`, `compute_summary_maps`, `truncate_losvd`,
  `clip_uncertainties`, `fit_all_bins`, and `write_dynamite_kinematics` have
  zero tests — only the core single-bin NUTS solver path is covered. Add a
  round-trip test: mock bins → `fit_all_bins` → `write_dynamite_kinematics`
  → read back ECSV.

- **[P1] Fix ruff lint violations**
  8 EM101/EM102 violations (raw string/f-string literals in `raise`) in
  `veldist.py` and `analysis.py`. `ruff check --fix`.

- **[P1] Resolve in-code TODOs**
  `veldist.py:161` (Dirichlet vs. current prior on weights — undecided) and
  `veldist.py:358` (whether to work in density space internally). Make the
  call and delete the comments, or write the tradeoff down here properly.

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

- **[P2] Rewrite veldist docs**
  Currently LLM-placeholder text. Also confirm no lingering
  `compute_moments` references in examples (prefer `compute_summary`).

## Someday (long-term, not active)

- Multiple kinematically distinct tracers: separate design matrices per
  stellar population
- PyPI packaging
- Spatially coherent velocity distribution inference (joint across bins,
  shared smoothness scale-length) — feasibility unclear, revisit once 2D
  solver and export pipeline are both tested and stable

## Completed

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
