# Tasks

Migrated from beads (`bd`) on 2026-08-02. Open items below; a completed-work
log follows for history.

## Open

- **[P2, blocked] 2D solver: `KinematicSolver2D` for bivariate PM distributions**
  Design is settled: square K×K grid, (N, K²) design matrix using centre-point
  bivariate Gaussian PDF × bin area, 2D GMRF prior with 8-connectivity
  (separable 1D random walks ruled out — can't represent genuine covariance).
  Precision matrix Q shape (K², K²); add ridge ε·I before Cholesky. Latent:
  `z ~ N(0, I_K²)`, `x = σ_smooth · L⁻ᵀ z`. Start with K=20. Test on mock
  tilted bivariate Gaussian. Defer: 4-corner box integration, marginalisation
  over missing PM axis, Dynamite 2D output.
  Blocked on: SVI/Pathfinder fallback below (only if NUTS proves too slow at
  K²=400 — worth trying NUTS first before committing to that dependency).

- **[P3] SVI / Pathfinder as NUTS fallback for large parameter spaces**
  If NUTS proves too slow for the K² parameter space in `KinematicSolver2D`,
  implement NumPyro SVI with mean-field or structured guide, or the
  Pathfinder algorithm, as a faster alternative.

- **[P3] Rewrite veldist docs (currently LLM-generated placeholder text)**
  Needs a proper rewrite once the API stabilises. Also update end-to-end
  examples to use `compute_summary` instead of the deprecated
  `compute_moments`.

- **[P3] Multiple kinematically distinct tracers**
  Support for multiple stellar populations with distinct kinematics.
  Requires separate design matrices per tracer population. Useful for all
  resolved stellar systems with known multiple components.

- **[P4] PyPI packaging**
  README lists PyPI install as "coming soon". Not blocking science, but
  needed for broader adoption.

- **[P4] Spatially coherent velocity distribution inference (joint across bins)**
  Joint inference across spatial bins with a shared smoothness scale-length.
  Uncertain feasibility — data may be either too well-constrained per bin
  (making joint inference redundant) or not well-sampled spatially. Deferred
  until single-bin inference is mature.

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
