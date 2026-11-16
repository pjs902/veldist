# E2: end-to-end mock for the omega Cen IMBH fit statistic (plan, 2026-11-16)

Read-only survey, written by a research agent. Nothing was run. Items marked UNVERIFIED are guesses.
Local trees: `OC` = `/Users/pesmith/research/omegaCen/dynamite_dataprep`, `DF` = `/Users/pesmith/research/dynamite-fitstat`
(branch `feature/pm-fit-statistic`), `VD` = `/Users/pesmith/research/veldist` (branch `feature/fit-statistic`).
The v4 model dirs, orbit libraries and `orbit_weights.ecsv` are on the node only; nothing local.

## (a) Inventory of reusable pieces

Histogram-level closed loop (the nearest precedent; it never draws stars):
- `OC/relative_pm/v2_closed_loop.py` (`gen` at L198-400). Reads a real library through DYNAMITE's own path
  (`orblib.read_vel_histograms`, `NNLS._prepare_kinematic_block`), forms truth `X_a = w_true @ rows`, writes mock packs
  with Poisson noise at the REAL `nstarbin`. Also does the LVM GH zero-point Newton solve (L284-315: model LOSVD ->
  v, sigma at h1=h2=0, then real `dv` noise) and the MUSE LOSVD mock (L375-387). REUSE the LVM block and the
  "model prediction in solver units" idiom. Limit: 720-orbit, 15x15 library, no star level, old centre/pack,
  x-East pack taken as given.
- `OC/relative_pm/mock_operator.py` (toy 40-bin mock, numpy NNLS) and `fine_real_solves.py` (4320-orbit library,
  solve/moments/report subcommands): useful for the relative/addback config wiring only.
- `OC/relative_pm/NODE_START.md`, `handoff_2026-10-19.md`, `handoff_2026-10-20.md`,
  `node_run_2026-10/{STATUS_2026-10-20.md,STATUS_2026-10-21_landed.md,run_relative_prod.py}`: node procedure, timings, caveats.

Truth histograms from a finished node model (all in `VD/experiments/fitstat/`, run with `ENV_mkl` via `grid_env`):
- `c3_private_copy.sh`: read-safe private copy of a finished model dir (symlinks the big .bz2/.npz). Use it.
- `rescore_v4.py` (+ `c3_dyn_parity.py`, `e1_solve.py`): wrap `NNLS._prepare_kinematic_block`, stream the read with
  `construct_gram_and_rhs_blockwise(rhs_only=True, chi2_w=w)` and capture per HST aperture `h = w @ rows`, data,
  sigma and `M_a` in DYNAMITE's fitted units (mass x probability; ml scaling and the border handling already applied).
  This is exactly "the truth model's per-aperture histogram". It is HST-only (`HST = "hst_pm"`); the same wrapper works
  for `gaia_pm`, `veldist`, `lvm` by changing `kins.name`. `e1_solve.py` already saves `h_base` (n_ap, K2), `Ma`, timings.
- `c3_compare.py` / `results/c3_compare.txt`: the 10 v4 rescored models (m_bh 5.5k-34k, ml ~3.05-3.16, border mass <=4e-4,
  mratio 1.004, 24 non-positive stars per model).

Star-level pipeline pieces (veldist):
- Writers `VD/src/veldist/dynamite2d.py::write_dynamite_kinematics_2d(star_data=, sharpen=)` and
  `veldist.py::write_dynamite_kinematics(star_data=, sharpen=)` (1D writes `<stem>_stars.npz`).
- `VD/src/veldist/fitstat.py`, `VD/experiments/fitstat/c1.py` (2D `simulate`/`gmass`: Gaussian truth, per-star error,
  `KinematicSolver2D` fit, six statistics) and `c1_1d.py` (MUSE-like, 9 s/seed, K=22, N=152). Templates for the draw/fit loop.
- DYNAMITE fork: `DF/dev_tests/test_laplace_pm.py` (the npz keys the Laplace set needs: `PM_star_indptr/cell/mass/aperture`,
  `PM_lap_h0`, `PM_star_sharpened`), `DF/dynamite/fitstat_pm.py`, branch `feature/pm-fit-statistic` (`_solve_laplace`, `statistic: laplace`).

Production data-prep (what the mock must imitate), all `OC/`:
- HST: `hst_veldist_refit_20261105.ipynb` (addback), `hst_relative_veldist_refit_20261105.ipynb`. Catalogue
  `merged.parquet` + `../HST_rotation/rotation_corrected_proper_motions.fits`; per-star errors `pmra_corrected_err`,
  `pmdec_corrected_err` (diagonal in the catalogue frame, no correlation column), rotated into the pm1/pm2 frame by
  `R C R^T` (cell [16]); `PM_QUALITY_CUT_KMS` error cut; PowerBin `target_capacity=400` on `Grid(cell_width=5)`;
  pinned grid K=23, width 165.61 km/s (cell 7.2 km/s); `min_stars=10`; `n_jobs=4`; NUTS 500+3000.
- Gaia: `gaia_veldist_refit_20261031.ipynb`, `gaia_v2_noscale_veldist_run.ipynb` (csv in `gaia_data/`, `pmx_err/pmy_err`
  or `pmrae/pmdece`, no error rescale, no correlation use confirmed; grid K=17; PowerBin on non-empty cells, 149 bins).
- MUSE: `muse_veldist_refit_20261031.ipynb` (stars `vel`, `e_RVel` from `merged.parquet`, `bin_data_list={vel,err}`,
  `fit_all_bins(..., num_chains=4, min_stars=10)`, `write_dynamite_kinematics`). LVM: `lvm_kins.ecsv` (v, dv per bin; sigma placeholder).
- Packing and frames: `assemble_datapack.py`, `dynamite_frame.py` (`convert_datapack`, `check_datapack`),
  production packs `NGC5139_production_input_xeast/`, `NGC5139_relative_input_xeast/` (geometry, mge, apertures, bins).
- Paper Sec. 2.4 (`/Users/pesmith/research/oCen_Dynamite_Paper_I/sample701.tex` L207-208): 163 MUSE bins (median 152 stars),
  1405 HST bins (median ~430), 148 Gaia bins (production pack 149), PA 104.3, d = 5494 pc; orbit matrix 45,000 orbits.

NOT found locally: any star-level mock, any code that draws stars from orbit-model histograms, and the v4 config/weights.
`dynamite_analysis/` and `dynamite_fitting/` hold plotting and grid driver code only (nothing needed for E2 beyond the report plots).

## (b) E2 pipeline

Place the new scripts in `OC/e2_mock/` (needs catalogues and packs); shared helpers can go to `VD/experiments/e2/`.

1. `e2_truth.py` (NODE, one heavy job, ~ one library read). Input: private copy of the chosen v4 model
   (`c3_private_copy.sh`), its `orbit_weights.ecsv`. Hook `_prepare_kinematic_block` as in `rescore_v4.py`, for all four sets.
   Output `truth.npz`: per set `h[a]` (HST (1405,23,23), Gaia (149,17,17), MUSE (163,K_los), all as probability mass
   `h/h.sum()`, with the border-cell mass recorded), `M_a`, aperture `xbin/ybin/nstarbin`, velocity edges, `w_true`,
   `ml`, m_bh and the other parameters. LVM: do the GH Newton of `v2_closed_loop.py` L284-315 to get truth (v, sigma) per bin.
   Also store truth moments (below) and, for relative mode, `m_true` per aperture.
   - Grid/ml: the PM and LOS histogram grids are baked into `infil/` (apertures + `vxrange`, `hist_width/bins`); the mock
     pack must use the SAME velocity grids (HST K=23 +/-82.8, Gaia K=17, MUSE the pinned LOSVD grid, LVM width 154.3, 79 bins).
     ml enters only through DYNAMITE's CDF-spline regrid of the ml=1 library (design doc facts); mass beyond the window is dropped and
     an overflow spike can go negative, so clip negatives to 0 and renormalise, and record how much was clipped. Capturing AFTER
     `_prepare_kinematic_block` bakes all of this in; do not re-derive from raw `.bz2` histograms.
   - Border cells: Fortran piles out-of-range velocities into the border ring. Border mass is <=4e-4 on v4 (c3_compare), so
     zero the ring and renormalise (matches veldist's closed grid), but assert < 1e-3 per aperture.
2. `e2_draw.py` (laptop). For each set, keep the REAL stars' structure and replace only their velocities:
   - Keep per bin: the real star list (bin membership, positions, per-star errors/covariances, quality cut already applied),
     therefore real `N_a` and the real heteroscedastic error distribution. Draw `v_true ~ p_a` (below), then
     `v_obs = v_true + N(0, cov_star)`. HST: use the rotated `cov11/cov22/cov12` from cell [16]; Gaia: `pmx_err/pmy_err`;
     MUSE: `e_RVel`; LVM: real `dv`.
   - Do NOT draw N from `nstarbin` and errors from a pooled distribution; reusing the real stars preserves the
     radial and magnitude-dependent error mix and the exact bin sizes, and costs nothing.
   - Within-cell sampling: the cells are probability MASS, and flat-in-cell sampling adds h^2/12 per axis
     (HST: 4.3 km/s^2 on ~300, sigma +0.7%; Gaia cells are larger, so more). That is the same O(h^2) bias class as the
     `generate_gaussian_core_field_2d` bug in `VD/CLAUDE.md`, and at the same size as the effects to be measured. RECOMMENDED:
     build the 2D CDF `F` on the cell edges (exact cumulative sums of the mass), interpolate with a bicubic / not-a-knot spline (the same
     operator DYNAMITE uses for the ml regrid, `ProperMotions.shift_histograms`) to 5x finer edges, take differences,
     clip negatives and rescale so every parent cell's mass is preserved exactly, sample a sub-cell by mass, then draw flat
     within the sub-cell (residual (h/5)^2/12, 25x smaller). 1D is the same with a 1D spline. Gate: the refined density's cell masses equal
     the input to 1e-12, and 1e6 draws reproduce its sigma_x, sigma_y, sigma_xy to 0.1%.
   - Truth moments (what the fits are judged against) are those of the refined density (and for comparison the cell-mass moments minus
     the Sheppard h^2/12); compute them by the same integration (cf. `_discretised_truth_moments`), not from cell centres.
   - Frame: draw directly in the DYNAMITE x-East frame (see (d)), pm1/pm2 on the aperture angle, writing `angle_deg = PA - 90 = 14.3`.
   - Seeds: `default_rng([seed, set])`, 3-5 realisations (the fit is independent of m_bh, so noise realisations are cheap relative to libraries).
   - Output: `mock_s{k}/{hst,gaia,muse}_stars.parquet` (per star: bin id, v, cov) + `lvm_kins.ecsv`.
3. `e2_fit.py` (laptop or node, no DYNAMITE). Headless version of the production notebooks:
   `fit_all_bins_2d(bin_data_list, grid_kwargs=PINNED, run_kwargs=PRODUCTION, min_stars=10, n_jobs=4)` for HST and Gaia,
   `fit_all_bins(..., num_chains=4)` for MUSE. Pin the grids exactly as the notebooks (K=23/165.61, K=17, MUSE pinned) so the mock packs are
   comparable with the real ones. Then
   `write_dynamite_kinematics_2d(solvers, ..., star_data=bin_data_list, sharpen=True)` and the 1D equivalent. The same files serve both
   statistics: baseline reads `PM_2dhist/sigma`, Laplace reads the star arrays (confirm `statistic: diag` ignores them).
   Skipped bins (<10 stars) must produce the same renumbering as production; match the mock apertures to production by `xbin,ybin,nstarbin`.
4. `e2_pack.py`. Copy geometry (`*_aperture.dat`, `*_bins.dat`, `mge.ecsv`, rotation rows/weights for relative mode) from the real xeast pack;
   write the mock kinematics; `check_datapack` (adapted); two configs differing only in `statistic: diag|laplace` on `hst_pm`, `gaia_pm`, `veldist`.
   Mass rows: keep the real MGE (v4 mratio 1.004, so the truth model is consistent with it, unlike the 720-orbit library).
5. `e2_orblibs.sh` (NODE). Truth model parameters (q,p,u, sBH component, ml) fixed; vary m_bh only: 7-9 values bracketing the truth
   (e.g. truth x {0.25, 0.5, 0.7, 1, 1.4, 2, 4}). One orbit library per m_bh at the SAME apertures/grids as the truth library; `ml` is
   solved on the existing library via `reuse_orblib_across_ml` (add 2-3 ml values per library if the profile should include ml).
   The truth library itself is the m_bh = truth point (no rebuild needed, but its `models/` copy must be the private one).
6. `e2_solve.py` (NODE, serial; one heavy job at a time, 56 GB peak on pass 0). For each (m_bh, statistic, noise seed): weight solve with the
   mock pack (reuse `OC/.../run_relative_prod.py` for the relative driver, `VD/experiments/fitstat/e1_solve.py` for the Laplace solve
   plus `_solve_laplace` re-expansion). Output per run: kinchi2 per set, per-set Delta chi2 (HST/Gaia/MUSE/LVM/mass), weights, timings.
   Use addback mode first (user default; it avoids the rotation-rows and recentring complexity); relative as a second axis.
7. `e2_report.py` (laptop). Per statistic: profile of total (and per-set) chi2 vs m_bh, minimum, Delta chi2 = 1257 unit (sqrt(2N), the
   production unit) vs a Delta-chi2 calibrated by the per-aperture C1 numbers; recovery of truth m_bh and of the truth sigma_R/sigma_T/beta
   profiles (from `truth.npz` moments); bias and coverage over seeds. Also report the chi2 at the truth m_bh (must be near its minimum for the correct statistic).
   Sanity gate before any library is built: refit the TRUTH weights with the mock pack at m_bh = truth (E2 closure): sigma/beta
   recovered to <1%; if not, the failure is in step 2-4, not the statistic.

Addback realism (optional, step 2): draw ABSOLUTE stars (truth mean kept), then optionally add the real addback error
(`addback - relative` per-bin shift, `fine_real_solves.py pack`) to emulate the Gaia-tied 1D curve mis-centring; without it, addback is idealised.

## (c) Cost estimate

- veldist fits (independent of m_bh; once per noise seed): HST 1405 bins K=23 NUTS 3000 draws. The notebook records 4.4 h at `n_jobs=2`
  (pre-audit, ~22 s wall per bin per worker), so ~2.2 h at `n_jobs=4` on the 36 GB laptop (8 workers swapped; do not raise). Gaia 149 bins K=17
  (smaller grid, UNVERIFIED ~15-30 min). MUSE 163 bins 1D, 4 chains (c1_1d: ~9 s/bin) ~25 min. Stars draw/export: minutes. Total ~3 h per seed on the laptop,
  likely <1 h with the node's cores (node load must be checked first). 3 seeds ~ 9 h laptop, run sequentially, one heavy job.
- Orbit libraries (node): the 90k-orbit library on the node took about 1.5 h end to end (launched 13:57, expected 15:30; STATUS 2026-10-21), plus
  about 35 min of serial bzip2 compression, now minutes with `lbzip2` (perf/read-path). The paper's v4 library is 45k orbits, so
  ~0.7-1.5 h each (UNVERIFIED: check a v4 model's build time in its log). 8 m_bh values -> ~6-12 node-hours. Libraries are ~12 GB each (datfil); plan disk.
- Weight solves (node): addback production solve: HST Gram read 9.2 min per read before the speedups (bunzip2 dominated), ADMM 291 s,
  total ~1-1.5 h before; the 3c074c6 hold-across-passes branch cuts reads by 2-4x. Relative: 5.3 h before, ~2.6 h after (pass 0 alone 64 min, 56 GB).
  The Laplace solve (pass 1 + 1 re-expansion) is untimed locally: `laplace_solve_probe.py` and `e1_solve.py` record the timings on the node; read
  those before costing (expect ~2x the baseline Gram work). Rough total: 8 m_bh x 2 statistics x ~1 h addback ~ 16 node-hours per seed, relative x2-3.
  MUSE/LVM/Gaia sets add little. The node is shared (check `uptime`/free memory before each job; one heavy job at a time).
- Cheapest first step: E2-lite at the 10 existing v4 libraries (`c3_compare.txt`): they differ in m_bh and in other parameters, so not a clean line, but a
  mock pack scored with `rescore_v4.py` (rhs-only reads, no solves, ~minutes each) already tests ranking flips with no new library.

## (d) Open questions and risks

1. FRAME BUG TO FIX FIRST: `OC/dynamite_frame.py::_convert_pm` mirrors only `PM_2dhist`, `PM_2dhist_sigma` and `xbin`. It does `dict(np.load(src))`,
   so `PM_star_cell` (cell indices), `PM_lap_h0` and the star arrays pass through UNCHANGED, i.e. with the vy axis un-mirrored relative to the histograms;
   the 1D `<stem>_stars.npz` is not touched or copied either. A real-data Laplace run through `convert_datapack` would silently mirror pm2 for the
   histogram only. E2 avoids this by drawing and writing in x-East directly (as the e2 pipeline above), but the production path needs `convert_datapack`
   to reverse the vy index of the per-star cells (`ix*K + iy -> ix*K + (K-1-iy)`) and `PM_lap_h0`, plus a check. A cheap E2 test: write the same mock in
   x-West, convert, and compare Laplace scores with the direct x-East write.
2. Relative vs addback: relative needs rotation rows/weights (build from the truth `m_true` as in `v2_closed_loop.py` L336-398), pass 0/pass 1
   and recentring (start dependence documented in `start_test`). Addback with the true mean is idealised; use the injected-error variant for a fair baseline.
3. Truth model choice: a model near the real best fit has weights with ~13k non-zeros and, per `STATUS_2026-10-21`, its own over-rotation and PA offsets. That is fine for
   a mock, but the truth must be fixed (q,p,u, sBH, ml) along the line, otherwise the "truth" changes with m_bh.
4. Fast stars / escape: the mock has none beyond the grid (border <=4e-4), whereas the real set was cut upstream (HQ + membership; no HQ star within 30 arcsec off-grid).
   So E2 cannot test tail contamination or the escape-velocity cut; C4 (injection) covers it. If wanted, inject a per-bin tail fraction from the model's
   own escape velocity and apply the same `PM_QUALITY_CUT_KMS`.
5. Selection effects: the mock keeps real positions/errors but draws velocities independent of magnitude/error; real kinematics vary with mass/magnitude
   (equipartition, outer-field luminosity functions are still an open item). Crowding-dependent errors correlate with radius, which the real star list preserves.
6. Stars outside the apertures, the PSF/aperture weights and the per-bin Voronoi extent: the model histogram is the bin-integrated distribution, stars are
   placed at real positions but not re-weighted; negligible for 400-star bins, but give the position-dependent mean inside a bin a thought (relative-mode shifts handle it, addback does not).
7. Pinning: grids, `min_stars`, bin counts and MUSE's 1.47 core-supplement scaling (if still used) must be pinned from the real run, otherwise differences from E2 are not attributable to the statistic.
8. Laplace/Gaia/MUSE statistics: `e1_solve.py` switched only HST to Laplace ("Gaia/MUSE stay diag until their star data are exported"); confirm the DYNAMITE branch handles
   `statistic: laplace` on `gaia_pm` and `veldist` (the 1D `LaplaceStatistic`, commit 941272b) in one solve before the line is launched.
9. kinchi2 is not comparable between baseline and Laplace; compare each statistic's own profile (position and width of the minimum), not absolute values.
