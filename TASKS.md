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

### 2026-09-01 profile audit and verification campaign

**Shape (h3/h4) policy is now settled per dataset** --
`docs/shape-policy-decisions.md`, resting on the Fisher analysis in
`docs/shape-information-limits.md`:

- **MUSE: h3/h4 not measured. ACCEPTED, not deferred** (Peter, 2026-09-01):
  bigger spatial bins are declined, spatial resolution is worth more. Do not
  open further tuning items against MUSE shape recovery -- no grid, prior or
  `SIGMA3_RATE` change closes a ~10x information deficit.
- **Gaia: `cell_per_sigma = 0.85` CONFIRMED** by re-anchor after the
  `err_cut` fix; flat across cps 0.70-1.15 because errors dominate
  (`err/sigma = 1.22`). `cps = 1.00` is an equally valid, 1.4x cheaper option.
  Shape work out of reach by ~2 orders of magnitude.
- **HST: the one dataset that can do shape work** (`err/sigma = 0.131`).
  Occupancy scan in progress, `scratchpad/run_hst_shape.py`.

**Convention trap, recorded because it already caused one error:** the field
quotes h3/h4 (van der Marel & Franx), `compute_summary` reports standardised
cumulants, and `gamma1 = 6.93*h3`, `gamma2 = 19.6*h4`. Always say which.


Status of the six items carried out of the data-driven tuning session. See
`docs/handoff-2d-tilt-recovery.md` (2026-09-01 afternoon section) for the
measurements behind each.

- **[RESOLVED 2026-09-01] MUSE 1D recovery curve.** PASSES on `v_mean` and
  `sigma` at every one of 18 sweep cells, both dispersion ends, CI/CR 0.98 and
  1.12. Shape metrics fail, and the reason is now understood analytically
  rather than asserted: at `err/sigma = 0.386` and 152 stars/bin the smallest
  3-sigma-detectable skewness is 0.73, so MUSE's own test truth (0.454) is a
  **1.85 sigma signal**. See `docs/shape-information-limits.md`. The earlier
  "flat in ivar = prior shrinkage" reading in this entry was wrong twice over:
  part of the kurtosis bias was an estimator bug (fixed, `b734de6`), and the
  ivar sweep spanned only sqrt(2) in signal-to-noise, entirely inside the
  regime where any sensible prior dominates. Original scope: `OMEGACAT_MEASURED` had its grid
  derived (22 bins, 152 stars/bin) but no simulation verification at all.
  Sweeping ivar around the operating point at both dispersion ends
  (0.355 at `sigma_max`, 1.145 at `sigma_min`), 3 truths, 30 realisations:
  `scratchpad/run_muse_recovery.py`.

- **[RESOLVED 2026-09-01] HST cps x n_stars sweep** (`scratchpad/run_cps_sweep.py`).
  Crosses two items onto one set of fits:
  - is HST calibrated at its 174-star minimum bin on the ADOPTED grid? The
    earlier min-occupancy run used cps=0.85 / K=19, since rejected.
  - re-anchor `_CPS_ANCHORS`' HST entry, which was measured with mock errors
    2.5x too broad (see below). Over-broad errors inflate the intervals
    `rms_z` divides by, so 0.58 may be too coarse -- the unsafe direction.

  Note the adopted HST grid is **K=25** (0.68 stars/cell at 426), not the
  K=27 quoted in the 2026-09-01 morning handoff.

- **[QUEUED] Gaia re-anchor.** Same script, same reason: its entry (0.85) was
  measured with errors collapsed onto a spike. Its error points the safe way
  (real errors are larger, inflating intervals), so 0.85 is more likely
  conservative than optimistic, but it is not yet earned.

- **[RESOLVED 2026-09-01] `err_cut` mismatch.** Was filed as a
  notebook-vs-profile disagreement; it was worse. `draw_errors` back-derived
  its log-normal width from `err_cut` assuming the cut sits at p95, and that
  failed in both measured profiles and in opposite directions -- Gaia's cut
  (7.81) sat BELOW its own median (8.60) and collapsed 65% of every mock's
  errors onto the cut; HST's cut is real but sits at ~p100, over-broadening
  the tail 2.5x. `err_log_sigma` is now an explicit field, guarded at
  construction, and measured by `from_data` (anchored on p95/median, not
  `std(log)`, which fits the body and understates the tail).

- **[RESOLVED 2026-09-01, no action] HST minor-axis mean-velocity excess.**
  Not an anomaly. A linear-gradient fit to the 1415 bin means gives
  `dm1/dy = +0.0296` and `dm2/dx = -0.0302` km/s/arcsec -- equal magnitude,
  orthogonal, i.e. a pure curl with the divergence 10x smaller. The notebook
  re-adds rotation from the 1D TANGENTIAL rotation curve, and a tangential
  field has equal RMS in any two orthogonal Cartesian components by
  construction, so `ptp2 ~= ptp1` is required. Not edge bins (inner 90%
  unchanged), not a bulk residual (means within 0.15 km/s of zero).
  `rotation_span = 16.7` stands.

- **[DOWNGRADED] Per-axis rather than per-bin resolution.** Real axis ratios
  are 0.894 (HST) and 0.767 (Gaia) median, not the 0.65 assumed -- that came
  from the synthetic test truth, which is more anisotropic than the cluster.
  `sigma_min`, already measured per-component across both axes, sits at or
  below the narrowest principal axis: HST needs NO correction (ratio 1.03),
  Gaia ~0.81 (1.23x finer cells, not 1.54x). Fold into the 2D
  non-Gaussianity work rather than paying for its own campaign; re-express
  `_CPS_ANCHORS` in the same sweep that re-measures them.

- **[NOT STARTED] 2D non-Gaussianity** (skew/kurtosis analogues for HST).
  Recommend standardised cumulants over 2D Gauss-Hermite. **Write the
  discretisation test FIRST** -- higher-moment Sheppard corrections are
  shape-dependent and are deliberately unimplemented in `analysis.py`.

- **[NOT STARTED] Re-run the slow/SBC suite.** Not run since the tuning
  changes. `XLA_FLAGS="--xla_force_host_platform_device_count=4" pytest
  tests/ -v --tb=short -m slow`.

- **[NOT STARTED] `n_stars` was measured, not optimised.** `target_capacity`
  was never swept, so the margin is unknown. The HST sweep above gives the
  occupancy floor, which is the number that decides whether to raise the
  PowerBin target or accept a few prior-dominated bins.

- **[RESOLVED 2026-08-06] 1D `num_samples` raised 1000 -> 3000; 2D sampler
  defaults fixed (`dense_mass=False`, `num_samples=3000`).** On real HST
  proper-motion data (`omegaCen/dynamite_dataprep/hst_veldist.ipynb`), the
  2D solver's inherited-from-1D `dense_mass=True` default (added the same
  day, then reverted) was measured to be actively pathological: NUTS hit
  `max_tree_depth` (1023 steps/sample) on nearly every sample regardless of
  `target_accept_prob`, gave *lower* min ESS (~200-290) than
  `dense_mass=False` (~260-470 at the same `num_samples=1000`) despite
  ~30x more leapfrog steps, and cost ~100s/bin in JIT compile alone on a
  run with ~1400 distinct star-count shapes (~43h projected for a full
  ~1400-bin run). `KinematicSolver2D.run` now defaults to
  `dense_mass=False`, `target_accept_prob=0.95` (0 divergences across 5
  test bins vs. several at 0.8), `num_samples=3000` (default raised from
  1000: roughly triples min ESS for near-identical per-bin wall time,
  since compile time dominates over sampling here). A `max_tree_depth`
  parameter was also added (not present before) so this pathology can be
  capped/diagnosed in the future without re-deriving it from scratch.
  `fit_all_bins_2d` also gained an `n_jobs` parameter (spawned
  `ProcessPoolExecutor`, parallelises across bins since they're
  independent, not across chains) — measured 1.8x wall-time speedup on 12
  real bins with `n_jobs=4` (contended by another job at measurement time,
  likely an underestimate), byte-identical posteriors to sequential at
  matching seeds.

  **Correction + memory fix (2026-09-04).** The "near-identical per-bin
  wall time, since compile time dominates over sampling" claim above does
  not hold at production grid size: measured at K=23, N=426, 500+3000, a
  bin costs 27.5 s cold vs 26.0 s JIT-warm, so compile is ~5% and sampling
  ~95%. `num_samples` is a near-linear knob on run time; 3000 is kept for
  the ESS but is not free. Shape-bucketing star counts to reuse compiles
  is correspondingly not worth it, and neither is caching the per-bin
  `setup_grid` (GMRF build + Cholesky is 1.7 ms against 26 s).

  What *was* costing real time was memory. `fit_all_bins_2d` returned
  solvers holding every draw: at K=23 x 3000 draws that is ~26 MB/bin, so
  the ~1400-bin HST set held ~36 GB in the returned list — the whole
  machine — which is why that run was pinned to `n_jobs=2` and took 4.4 h.
  `_fit_one_bin_2d` now shrinks each solver inside the worker, before it
  crosses the process boundary: drops `matrix`/`Q`/`Q_reg`/`L` and
  `samples["x"]` (`intrinsic_pdf` is a deterministic softmax of it), and
  thins `intrinsic_pdf` 10x to float32 (~0.63 MB/bin, ~0.9 GB total).
  `clip_uncertainties()` still runs on the FULL draws, so DYNAMITE output
  is unchanged; at ~31 leapfrog steps/sample the kept draws are
  near-independent. New `thin` kwarg on `fit_all_bins_2d` (`thin=1` keeps
  every draw, `thin=0` leaves `samples` untouched); guarded by
  `test_fit_all_bins_2d_thins_returned_samples`.

  Separately, 1D's `KinematicSolver.run` default `num_samples` was raised
  1000 -> 3000 on the same "more ESS for ~free wall time" reasoning.
  Unlike the 2D change, this touches an already-SBC/coverage-validated
  regime (`docs/validation.md`), so `test_sbc_calibration` (both priors)
  and `test_per_bin_losvd_coverage` (all three truth shapes) were
  re-run at `num_samples=3000` before adopting it: **5 passed, 3 xfailed
  (pre-existing, unrelated), 0 failures** — the calibration holds at the
  new sample count. `tests/test_calibration.py`'s `NUM_SAMPLES` constant
  was updated to match (that file's own comment requires it track
  `KinematicSolver.run`'s default, or SBC validates a configuration nobody
  runs).

  **Still open:** the 2D solver has no equivalent SBC campaign to 1D's
  (`test_sbc_calibration_2d`/`test_coverage_2d.py` exist but are a
  different, smaller check) — the `dense_mass=False`/`tap=0.95`/
  `num_samples=3000` combination is measured-good on real data (ESS,
  divergences, wall time) but not formally SBC-validated the way 1D's
  regularisation choice was. Treat it as strong evidence, not the same
  tier of validation as 1D's.

- **[RESOLVED 2026-08-04] The prior was never the problem — the sampler was.**
  Every SBC failure was low ESS on `sigma3`: a funnel, where the deviation
  scale approaching zero narrows the posterior into a neck that a step size
  adapted on the funnel's mouth cannot traverse. Tightening the prior made SBC
  pass by *deleting* that geometry — and the geometry is where the shape
  information lives, so every tightening step cost h3/h4 and per-bin coverage.

  Raising `target_accept_prob` from NumPyro's 0.8 to 0.95 keeps the geometry
  and samples it properly. At `SIGMA3_RATE=0.35`, n_sims=100: 0.8 → 17%
  failures (p5 ESS 50), 0.95 → 1/100 (p5 ESS 217). Extra warmup does not
  substitute — 1500 warmup at 0.8 still failed, so the step size is the
  binding constraint. Cost is ~2x wall time, no warmup change.

  **Adopted: `SIGMA3_RATE=0.35`, `target_accept_prob=0.95` (new default in
  `KinematicSolver.run`), warmup unchanged at 500.** This dominates every
  alternative considered:

  | config | SBC /100 | per-bin (skew) | h3/h4 mean | cost |
  |---|---|---|---|---|
  | Exp(0.35), tap 0.8 (old) | 17 ✗ | 0.692 | 0.603 | 1x |
  | Exp(5.0), tap 0.8 | 1 ✓ | 0.609 | 0.393 | 1x |
  | Exp(1.0), tap 0.8 | 2 ✓ | 0.680 | 0.570 | 1x |
  | **Exp(0.35), tap 0.95** | **1 ✓** | **0.710** | **0.603** | ~2x |

  Per-bin coverage re-measured at the shipped settings: 0.724 / 0.710 / 0.709
  against nominal 0.68, no informative bin below the 0.30 floor.
  **Do not lower `target_accept_prob` to 0.8**; it reintroduces the failures.

  Two further sampler defaults changed with it, both measured. `dense_mass`
  is now True: the `d3` components are correlated through the cumulative sum
  and the null-space projection, and switching from diagonal takes min ESS on
  `intrinsic_pdf` from 119 to 1188 and max r_hat from **1.0161 to 1.0015**, in
  *less* wall time. That r_hat is above the usual 1.01 threshold, so the
  configuration this branch was about to ship was marginally non-converged.
  `num_chains` is now 4, because with one chain there is no r_hat and nothing
  could have reported that. All three settings are exported constants that the
  SBC harness imports, after it was found testing `target_accept_prob=0.8`
  while the solver shipped 0.95.

- **[P1] Moments are lossy — check per-bin coverage.** Tightening `SIGMA3_RATE`
  1.0 → 5.0 left *every* moment metric flat (coverage, efficiency, bias on
  v_mean and sigma) while per-bin LOSVD coverage fell 0.680 → 0.609. Per-bin is
  the artifact DYNAMITE chi-squares, so it is the one that matters when they
  disagree. Guarded by `test_per_bin_losvd_coverage`; interpretation guide in
  `docs/validation.md`.

- **[P1] The test suite cannot see float32 bugs.** `tests/conftest.py` enables
  JAX x64 for the whole suite, but `KinematicSolver` runs in JAX's default
  float32. Any numerical fragility in the production dtype is therefore
  invisible to every test by default — which is exactly how the design-matrix
  cancellation bug (fixed 2026-08-04, see Completed) survived. The x64 default
  is there for good reasons (chained cumsum/QR precision), so the fix is not
  to remove it but to run the numerically sensitive tests in both regimes.
  `test_design_matrix_is_strictly_positive_in_float32` shows the pattern:
  wrap the call in `jax.enable_x64(False)`.

- **[P2] Real ω Cen data.** Once the above is settled, move to real data — it
  will inform tuning far better than the mock harness can. The harness truths
  at σ=30 and σ=42 are outside ω Cen's 9–21 range, so further tuning against
  it has limited value.

- **[P1] 2D solver: the harness was never calibrated to the PM observing
  regime, and its failures were mostly that.** Added the 2D analogues of the
  1D correctness gates (`tests/test_coverage_2d.py`: moment coverage +
  per-cell LOSVD coverage, alongside the pre-existing
  `test_sbc_calibration_2d`).

  **One real bug fixed:** `model_2d`'s `smoothness_sigma ~ HalfNormal(0.1)`
  was drastically too tight at any star count — the posterior wanted ≈3.3 but
  the prior forced it to ≈0.84, giving a near-uniform fit (isotropic truth
  σ=6 recovered as ~9.0, against 11.5 for a perfectly uniform grid). Widened
  to `HalfNormal(3.0)`.

  **Retracted root cause.** An earlier entry here blamed the absence of a
  Sørbye-Rue rescale in `build_gmrf_precision`. Measured directly on the
  6-dimensional quadratic projection that the rescale actually uses, the
  constant drifts about −12% from K=9 to K=21 (2.311 → 2.028). That still
  cannot explain a σ bias that *tripled* over the same range, so the
  retraction stands — real but not the driver.

  Note the first figure recorded here was ~6% (SR scale 1.777 → 1.670 over
  K=10→20). That was computed by projecting out only the *constant* vector, a
  1-dimensional null space, and understated the drift by about half. The
  conclusion was unaffected, but do not re-cite the 6% number.

  **Actual driver: `N_STARS`.** The harness used 150 stars/bin — the *LOS*
  `OMEGACAT` number — copied into 2D without checking. PM catalogues go ~6 mag
  deeper than the spectroscopy (faint limit 24 vs 18), so the science target
  is 250–500 stars/bin in the inner region and up to ~2000 in the outer
  (Gaia) region. Sweeping N at K=10, isotropic σ=6:

  | N_stars | stars/cell | σ bias | tail mass vs truth |
  |---|---|---|---|
  | 150 | 1.5 | +0.369 | 4.3× |
  | 500 | 5.0 | +0.046 | 2.8× |
  | 1500 | 15.0 | −0.033 | 2.0× |
  | 5000 | 50.0 | −0.094 | 1.5× |

  The dispersion bias is negligible by N=500, i.e. across the whole science
  range. Prior scale ∈ {3,5,10} and `target_accept_prob` ∈ {0.8,0.95} were
  also swept and had **no effect** — do not re-tune those.

  Far-field leak into the grid corners is real and persists (tail mass 1.5–2×
  truth even at N=5000) but is second-order once the data pins the bulk.
  Truncating at 3σ is not the fix: it biases σ *low* by −0.3 at every N,
  because it removes genuine mass along with leaked mass — the same trade 1D
  found with `n_sigma_truncate`.

  **DYNAMITE 2D output format: it exists.** Previously listed here as
  deferred/undecided, so `veldist2d` risked producing an output with no
  consumer. It has one. DYNAMITE PR #442 "2d histograms" (merged 2026-06-03,
  commit `9ccc416`) adds `ProperMotions` and `Histogram2D` to
  `dynamite/kinematics.py`. **On `main`, NOT in any tagged release** —
  `v5.0.0` (2026-01-14) predates it, so pin to a commit. An earlier attempt
  (PR #307) was closed unmerged as incomplete.

  Format is a NumPy `.npz` (not ECSV/FITS), keys: `PM_2dhist` and
  `PM_2dhist_sigma`, both `(n_apertures, n_bins[0], n_bins[1])`, plus
  `binID_dynamite`, `nstarbin`, `vxrange`, `vyrange`, `xbin`, `ybin`. The
  conventional `aperture.dat`/`bins.dat` pair is unchanged — only the velocity
  data moved to `.npz`. Three conventions line up with what we already emit:

  - values are **probability mass normalised to 1 per spatial bin** — exactly
    `intrinsic_pdf`'s convention, no density conversion needed;
  - the grid is **uniform and symmetric about zero**,
    `linspace(-vxrange, vxrange, n_bins+1)` — matches `setup_grid_2d(center=(0,0))`;
  - the chi² **flattens K×K to K² and treats each cell as an independent
    Gaussian** via the same NNLS path as 1D `BayesLOSVD`
    (`weight_solvers.py::construct_nnls_matrix_and_rhs`). So per-cell coverage
    is the gating artifact for 2D, for the same reason per-bin coverage is in
    1D.

  Constraints and hazards:
  - **Bin counts must be ODD** (`set_default_hist_bins` raises `ValueError`
    otherwise). Any even-K measurement is unshippable — the first pass of the
    star-count sweep used K ∈ {8,10,12} and had to be discarded.
  - DYNAMITE allows **rectangular** grids (its dev config uses 15×11);
    `setup_grid_2d` takes a scalar `n_bins` and only builds square K×K.
  - Passing `hist_width`/`hist_center`/`hist_bins` in the config **raises** —
    "2d histogram metadata is always determined by the data!" — yet the
    checked-in `dev_tests/user_test_config_ml_with_pm.yaml` still supplies
    them, so upstream fixtures are stale and this feature is not yet
    test-covered. Expect the interface to move before release.
  - `chi2_kinmap` is hardcoded to `GaussHermite` and returns NaN for
    `ProperMotions`; no tutorial notebook covers PM (all 8 in
    `docs/tutorial_notebooks/` are 1D).

- **[P1] 2D harness bug: sigma was scored against the wrong truth.** The
  star-count sweep compared posterior sigma — computed from **cell centres** of
  a K×K histogram — against the **continuous** true sigma. Binning inflates the
  second moment by ~h²/12 (Sheppard), which at K=9 over a 119 km/s grid is a
  fixed +0.5 km/s offset on sigma=17. At `hst_bright` (err/sigma=0.014) the
  posterior interval is only ~sigma/sqrt(2N) ≈ 0.76 km/s, so that offset alone
  collapses coverage to 0.04–0.24 while `mean_x`/`mean_y` stay fine at
  0.68–0.80 — the signature of a discretisation offset, not a model defect.
  Fair comparison is against the **discretised** truth (true probability mass
  per cell, moments at cell centres), which is also exactly what DYNAMITE
  chi-squares. Sweep now scores both and reports them separately. If the
  continuous sigma is wanted for science, it needs an explicit Sheppard
  correction in post-processing — that is a real, separate deliverable.

  **Resolution (2026-08-05):** The profiling/calibration gap that was "what is
  actually open" is resolved. `calibration2d.py` provides three calibrated PM
  observing profiles (HST_BRIGHT, HST_FAINT, GAIA_OUTER) with grid width and
  bin count derived from the oMEGaCat measurement regime. A cell_per_sigma
  sweep (K ∈ {9,11,13,15,17,19}, n=25, both HST regimes, both truths, under
  the gaussian_core prior) established cell_per_sigma=0.47 (K=15, 225 cells,
  1.8 stars/cell) as the effective limit — finer grids break on anisotropic
  truths at N=400. The default was set at that value. The GMRF-era discovery
  that 2D is the easier per-bin problem (larger N, comparable err/sigma) was
  confirmed.

  **Mean_y offset — checked (2026-08-05), not real.** The cell_per_sigma sweep
  showed a +0.14 to +0.22 km/s mean_y bias at hst_bright with mean_x staying
  small, and it looked reproducible: a from-scratch re-run at a second seed
  epoch (20260901) landed on the same sign and magnitude (+0.22 across
  K=9/15/19). That re-run's data generator turned out to have a real bug —
  `rng.normal(0, err[:, None])` with no `size` argument broadcasts to shape
  `(n, 1)`, silently adding the *same* noise draw to both PM components of a
  star instead of independent per-axis noise (the shipped test suite,
  `tests/test_coverage_2d.py`, does not have this bug — it draws `err_x`/
  `err_y` separately). A controlled A/B at a third, independent seed (90000)
  — buggy correlated-noise data vs. correct independent-axis noise, same
  solver — gave the *same* small-and-insignificant result under both
  conditions: mean_y bias +0.05 to +0.07 km/s, mean_x bias -0.21 to -0.22 km/s,
  both well inside the ~0.17-0.23 km/s per-batch noise floor set by sampling
  only n_real=20-25 realisations of 400 stars each (SE ~ sigma/sqrt(N)/
  sqrt(n_real)). Landing on the same axis with similar sign in the first two
  batches was coincidence, not a reproducible model defect — the bug does not
  explain it either, since the A/B comparison shows it makes no difference.
  No further action needed.

  **Still open:**
  - Sheppard correction if reporting continuous σ (not needed for DYNAMITE
    output, which chi-squares per-cell mass, not cell-centre moments).

- **[P1] 2D solver: `target_accept_prob`/`dense_mass` ported from 1D without
  re-validating on 2D.** (2026-08-06.) `KinematicSolver2D.run` previously
  hardcoded NumPyro's defaults (`target_accept_prob=0.8`, diagonal mass
  matrix) with no way to override them — unlike 1D, which found `tap=0.8`
  gave 17% SBC failures at `SIGMA3_RATE=0.35` from a funnel in the
  smoothness-deviation scale (`docs/validation.md`). Added
  `target_accept_prob=0.95`, `dense_mass=True` as new parameters/defaults
  on `KinematicSolver2D.run`, on the expectation that the 2D
  `gaussian_core` prior has the same funnel-prone structure, **not on 2D-
  specific measurement** — no SBC/coverage re-run at this setting has been
  done yet. `test_sbc_calibration_2d`/`test_coverage_2d` now exercise this
  default; re-run them and confirm they still pass (or improve) before
  treating this as settled the way the 1D case is.

- **[RESOLVED 2026-08-11] Branch `validation-and-binning`: Gaussian MLE baseline,
  GH fitting, data-driven observing profiles, recovery-curve sweep and an
  information-content floor.** What shipped: `veldist.baseline.gaussian_mle`
  (the two-parameter exact-optimum comparison point, see `docs/validation.md`
  "Comparison against a Gaussian MLE baseline"); Gauss-Hermite `h3`/`h4`
  fitting in `analysis.py` (`gauss_hermite_fit`); `ObservingProfile.from_data`
  in `calibration.py`, fitting a profile's grid and error distribution
  directly from a real catalogue instead of by hand; the `recovery_curve`
  sweep and `RecoveryCurve` in `calibration.py`; the measured
  `PROXY_TO_GH` percentile-to-Gauss-Hermite mapping; information-content
  binning helpers in `binning.py` (`make_ivar_sn_func`, matching the
  `sum_i 1/(sigma^2+err_i^2)` definition used everywhere else); and a
  `min_ivar` floor parameter on `fit_all_bins` so low-information bins can be
  skipped the same way `min_stars` already does.

  Merged to main in 21 commits. See `docs/validation.md` for the full results:
  veldist matches the exact Gaussian MLE optimum to 0.999±0.003 on v_mean and
  1.016±0.005 on sigma; ties on all nine truths for first two moments (18/18
  cells, max |t|=1.41); wins decisively on shape (TV distance 0.0712 vs 0.2168,
  t=31.8 on bimodal LOSVD). Recovery campaign: both v_mean and sigma calibrate
  at ivar down to 0.1; real oMEGACat bins at ivar 0.39 are well within
  calibrated range.

- **[RESOLVED 2026-08-11] Recovery-curve campaign complete.** 1920 NUTS fits
  across 6 ivar values (0.1–3.2), 2 dispersions (19.04, 13.37 km/s), 4 truth
  shapes, 40 realisations each. Both `v_mean` and `sigma` calibrate at every
  information content down to ivar 0.1 (pinned bottom; the true threshold may
  be lower). Real oMEGACat bins at ivar 0.39 are comfortably within the
  calibrated range. See `docs/validation.md` "Recovery-curve results".
  The coverage-gate defect that made earlier runs report sigma as
  uncalibrated (fixed `min_coverage=0.60` → binomial band at `n_real`) was
  corrected mid-campaign without re-running MCMC, since the 192 rows were
  already on disk.

- **[RESOLVED 2026-08-11] Real `ObservingProfile` measured from oMEGACat.** Fitted via
  `ObservingProfile.from_data` on the real oMEGACat line-of-sight catalogue
  (24,925 stars after quality cuts, out of 717,934 rows). Fixture committed
  at `tests/data/omegacat_profile.json`. See `docs/validation.md` "The
  measured observing profile" for the full comparison against the hand-typed
  `OMEGACAT` constant.

  **Outstanding decision:** whether to update the `OMEGACAT` constant in
  `src/veldist/calibration.py` to these measured values. Not done as part of
  this task, since every result currently in `docs/validation.md` was
  computed under the hand-typed values and changing the constant would
  silently invalidate them and alter the velocity grid used by the existing
  calibration tests. Left for the project owner to decide.

## Ruled out by measurement — do not re-raise

- **Mode-order split scale — measured, does not work.** (2026-08-04, prior
  predictive, n=20k draws, script
  `docs/superpowers/specs/2026-08-04-split-scale-measurement.py`.) The idea was
  two `sigma3` parameters: loose on the low-order deviation modes that carry
  h3/h4, tight on the high-order modes that are just wiggle. **There is no such
  separation to exploit.** Above a 2-mode split, tightening every higher mode
  to zero changes h3/h4 retention by <2% (1.007 → 1.025) and the degenerate
  draw fraction not at all (0.148 vs 0.150) — all the h3/h4 spread already
  lives in the two lowest modes. But so does the roughness: killing all modes
  above the first four cuts non-degenerate PDF roughness only 3–7% (0.0762 →
  0.0737), while switching the deviation off entirely cuts it 4× (→ 0.0177).
  After softmax, mode order does not partition "shape we want" from "wiggle we
  don't"; both come from the same few smoothest modes. The only knob that does
  anything is the low-mode rate, which is the same knob as `SIGMA3_RATE`
  (Exp(0.15) on the low modes inflates h3 spread 16× and drives degeneracy
  0.15 → 0.28 — the identical loose-prior/degeneracy trade already known).
  The prototype was validated first: it reproduced the published baseline
  (h3 retention 0.125, h4 0.028) before measuring anything new.
  **This is the second hypothesis killed by the same fact** — softmax mixes
  what the log-density basis separates (the first was `rw_order`). Treat the
  whole "reparameterise the prior to free h3/h4" direction as exhausted.

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

- Docs pass (2026-08-06): rewrote `theory.md`'s smoothing-prior section for
  `gaussian_core` (previously only documented `rw1`), with a from-scratch
  derivation of the `QQ^T` projection (`docs/fig_projection.py`) rather than
  naming the operation unexplained. Added 2D coverage to `examples.md`
  (`KinematicSolver2D`, `calibration2d`, `fit_all_bins_2d`/`n_jobs`,
  `dynamite2d`), previously undocumented despite being merged. Fixed the
  flagship `fig_deconvolution.py` demo (was unimodal-blending its "shoulder"
  feature into invisibility) and `fig_2d_recovery.py` (had a transposed
  x/y reshape that made a correct recovery look wrong, and used an invented
  error scale ~30x outside any calibrated profile — now uses
  `HST_FAINT.draw_errors()`). Every `docs/fig_*.py` now calls
  `matplotlib.rcdefaults()` so figures don't inherit a developer's personal
  `matplotlibrc`. `fig_kin_maps.py` cut from >20 min to under a minute by
  fixing `n_stars_per_bin` (was forcing a JIT recompile per distinct star
  count across ~40 shapes); redesigned to plot naive-vs-`veldist` sigma bias
  against known ground truth instead of unreliable skew/kurtosis maps.
  `fig_sigma3_rate.py` added to `validation.md` (previously table-only).
  Added explicit naive-estimator comparisons to Examples 1, 2b, and 3, with
  honest reporting where results were mixed rather than a clean win (2D
  single-draw `cov_xx` recovery was worse than the naive sample covariance
  on one draw, within posterior uncertainty — noted in text, not hidden).

- Design-matrix cancellation bug (P0, commit 5cda2cc): `precompute_design_matrix`
  could return *negative* probabilities (to -1.2e-7) for bins ~5-7 sigma from a
  star, where both edge CDFs are ~1e-7 and their float32 difference cancels.
  The negative propagated into `matrix @ intrinsic_pdf`, `log` of it is NaN,
  and NUTS died at initialisation with numpyro's opaque "Unit distribution got
  invalid log_factor parameter". The old `+ 1e-30` guarded exact zeros, a
  different hazard, and was 22 orders of magnitude too small for this one;
  `jnp.clip` covers both. Reachable in production (float32) and invisible to
  the suite (x64) — see the P1 item on the precision gap.


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
- 2D tilt/rho recovery — see `docs/handoff-2d-tilt-recovery.md` for full context:
  - Phase 1: rerun `recovery_curve_2d` at n_real=100, n_stars={435,1600} to get
    a coverage estimate tight enough to call (P1)
  - Phase 2: model correlated measurement errors — `_draw_stars` builds a
    diagonal per-star covariance, but real Gaia has `pmxy_cov` (P1)
  - Phase 3: per-bin trust rule, `RecoveryCurve2D.trustworthy(metric, n_stars)` (P2)
  - Phase 4: 2D skew/kurtosis analogs for HST data (start with standardised
    third/fourth cumulants, not full 2D Gauss-Hermite) (P2)
  - Investigate why sigma_y intervals run ~1.8x the CR bound while sigma_x sits
    at ~1.3x; suspect a `cell_per_sigma` grid-resolution effect on the narrower
    truth axis (P2, bug)
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
