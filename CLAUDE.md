# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Task Tracking

Open work items live in `TASKS.md`. Update it directly when tasks are added
or completed.

## Session Completion

**When ending a work session**, work is NOT complete until `git push` succeeds.

1. Update `TASKS.md` for anything completed or newly needed.
2. Run quality gates (if code changed) - tests, linters, builds.
3. `git pull --rebase && git push`, then `git status` to confirm up to date
   with origin.

## Build & Test

```bash
pip install -e ".[dev]"

# Fast tests only (excludes full MCMC runs)
pytest tests/ -v --tb=short -m "not slow"

# Run a single test
pytest tests/test_basic.py::test_setup_grid -v

# Full test suite including slow MCMC inference tests
pytest tests/ -v --tb=short
```

Slow tests (`@pytest.mark.slow`) run actual NUTS sampling. **CI does not run them** — `KinematicSolver.run` defaults to `num_chains=4`, which a GitHub runner serialises, making the suite hours rather than minutes. They are `workflow_dispatch`-only (Actions tab → Run workflow).

**Run them locally before merging anything that touches the model, the prior, or the sampler:**

```bash
# Make the 4 chains parallel; without this they serialise and it is ~4x slower.
python -c "import veldist; veldist.set_host_devices(4)"
pytest tests/ -v --tb=short -m slow
```

The gates that matter most there are `test_sbc_calibration` (both priors) and `test_per_bin_losvd_coverage`. Note the `rtk` pytest wrapper can report exit 0 on a failing run and has misreported collection, so read the summary line rather than trusting `$?`; `rtk proxy pytest ...` gives unfiltered output.

## Architecture

`veldist` infers the intrinsic Line-of-Sight Velocity Distribution (LOSVD) from discrete stellar velocities with per-star measurement errors. The core idea: pre-compute a static design matrix M (N_stars × N_bins) that encodes the convolution of each star's Gaussian measurement error with each velocity bin, then infer the bin weights via NUTS MCMC.

**Source layout:** `src/veldist/`
- `veldist.py` — all inference logic: `precompute_design_matrix`, `model` (NumPyro probabilistic model), `KinematicSolver` class, `fit_all_bins` (batch pipeline), `write_dynamite_kinematics` (Dynamite ECSV/aperture/bins output)
- `analysis.py` — post-inference statistics. Three families, deliberately not
  interchangeable: `compute_summary` (ordinary moments, primary API, 10 scalars
  per bin), `compute_percentile_summary` (robust Bowley/Moors analogues), and
  `gauss_hermite_fit` (literature-comparable h3/h4). Each has a `_maps` batch
  variant. Helpers: `cdf_percentile`, `tail_weight`, `bimodality_score`,
  `half_68ci`. `compute_moments` is legacy.

**Key data flow:**
1. `KinematicSolver.setup_grid(center, width, n_bins)` — defines the velocity histogram grid
2. `KinematicSolver.add_data(vel, err)` → `precompute_design_matrix` — bakes observations into M once (avoids recomputing per MCMC step)
3. `KinematicSolver.run()` → NUTS on `model(matrix, n_bins, bin_width)` — infers the latent curve `x`, `smoothness_sigma` (physical, resolution-independent hyperparameter on the intrinsic RW1 random-walk prior — see `docs/theory.md`), and the `intrinsic_pdf` deterministic (probability mass per bin, each sample sums to 1, `= softmax(x)`)
4. `KinematicSolver.clip_uncertainties()` — post-processes samples into per-bin median + half-68CI, with uncertainty floors (prevents Dynamite NNLS failures from zero-uncertainty bins)
5. `write_dynamite_kinematics(solvers, output_dir, voronoi_bin_metadata)` — writes three Dynamite input files: ECSV kinematics, `aperture.dat`, `bins.dat`

   **The writers do no frame handling and no frame validation** — `angle_deg` and the sky orientation of the positions/PM components are written verbatim and are entirely the caller's responsibility. Two ways to get this wrong, both of which silently invert fitted rotation while leaving surface brightness and sigma looking correct:
   - `angle_deg` is `-theta_maj` with `theta_maj` the *receding* major axis measured CCW from +x **in the caller's own frame**. Dynamite's docs phrase it as `90 - position_angle`, but that is `pafit`'s angle on the same x/y/v — *not* a sky PA. The two agree only mod 180.
   - Dynamite's projection is right-handed with the LOS along `x' x y'`, so `(x, y, v_los)` must be right-handed; with y=North and v receding-positive that means **x-East**. `pm2` (minor-axis component) flips sign under an East/West mirror, `pm1` does not.

   The consumer that gets this right, with the full derivation and sources, is `omegaCen/dynamite_dataprep/dynamite_frame.py`.

**Batch pipeline:** `fit_all_bins(bin_data_list, grid_kwargs, run_kwargs)` runs steps 1–4 for every Voronoi bin, skipping bins below `min_stars=10` (returns `None` at that index). Uses `base_seed + bin_index` to decorrelate chains.

**Important conventions:**
- `intrinsic_pdf` samples are **probability mass** (dimensionless, each row sums to 1), not density. Convert to density by dividing by `grid["width"]` for plotting.

  **This is a hard invariant, not a labelling choice, and it is easy to violate silently.** The design matrix M is built by *integrating* each star's error kernel over each cell (2x2 Gauss-Legendre in 2D). So the likelihood `M @ intrinsic_pdf` is dimensionally correct only if `intrinsic_pdf` is mass. Any code that produces a cell value by *evaluating a density at the cell centre* and then normalising has silently substituted density-at-a-point for mass, and the two differ at second order:

  ```
  mass = int_cell f  ~=  h * f(centre) + (h^3/24) * f''(centre)
  ```

  Because `f''` is positive in the tails and negative near the peak, centre-sampling under-weights the tails and **under-estimates the dispersion by O(h^2)**. The failure mode is nasty: it is invisible on fine grids, scales away as you refine (so it looks like a resolution problem you can buy your way out of), and biases *second* moments while leaving means untouched.

  This exact bug lived in `generate_gaussian_core_field_2d`, where `softmax(-quad/2)` produced the Gaussian density at cell centres rather than its cell mass -- so the free, unpenalised Gaussian core was systematically narrower than the Gaussian it was meant to represent. It cost roughly half of a measured `sigma_y` bias that ran from -0.250 (K=15) to -0.050 (K=29). See `docs/handoff-2d-tilt-recovery.md`.

  **When writing or reviewing any code that builds a per-cell quantity, state which one it is.** If it is mass and comes from a smooth density, it needs cell integration (exact, quadrature, or the second-order `+ (h^2/24) f''/f` term), not point evaluation. The same question applies to anything comparing model output against an analytic truth: `_discretised_truth_moments` integrates the truth over cells for precisely this reason.
- Marginal per-bin medians from `clip_uncertainties` intentionally sum to 0.85–0.95, not 1 — this is expected because marginals are taken independently from a joint simplex posterior.
- `truncate_losvd()` is an optional repair step for diagnosed tail contamination; it is not called by `fit_all_bins` and should be avoided unless there is a clear reason.
- `compute_moments` is a legacy function; prefer `compute_summary` for new code.

**Dynamite output format** is documented in `context/dynamite_format_spec.md`.
