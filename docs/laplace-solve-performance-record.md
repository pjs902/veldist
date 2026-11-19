# Laplace production solve: everything known about performance (2026-11-18)

Companion to `docs/superpowers/plans/2026-11-18-laplace-solve-speedup.md` (first-pass plan).
Code: DYNAMITE fork `~/research/dynamite-fitstat`, branch `feature/pm-fit-statistic`
(laptop HEAD a464c01; the node copy at `$W/dynamite-fitstat` is at 2cd498b, so node line
numbers differ from the laptop's). W = `/nexus/posix0/MIA-astro-env/nneum/pesmith/veldist-fitstat`.
Statistic maths: `docs/fit-statistic-design.md`; Laplace helpers `dynamite/fitstat_pm.py`.

## 1. The problem

DYNAMITE fits orbit weights w (n_orbs = 45 000, NNLS) by a QP in normal-equation form.
`NNLS.construct_gram_and_rhs_blockwise` streams the orbit library, forms constraint rows
block by block and folds each into `G = A'A` (45k x 45k float64, 15.1 GiB, upper triangle)
with one `scipy.linalg.blas.dsyrk` per block (`NormalEquationAccumulator.add`), plus
`v = A'b`. A never exists (would be 124 GiB at float64). Then `finalize` mirrors and
column-normalises G into P in place, and ADMM (`nnls_solver: admm`, `admm_free_p`) solves.

Kinematic sets in production (`$W/prodone/config.yaml`):

| set | type | statistic | apertures | cells/aperture | rows |
|---|---|---|---|---|---|
| veldist (MUSE) | BayesLOSVD 1D | laplace | 166 | 21 | 3.5k |
| hst_pm | ProperMotions 2D | laplace | 1406 | 23x23 = 529 | 744k |
| lvm | GaussHermite | diag | small | | small |
| gaia_pm | ProperMotions 2D | laplace | 149 | 17x17 = 289 | 43k |

HST dominates: one Gram build is about rows x n_orbs^2 = 7.4e5 x 2.0e9 = 1.5e15 flop.

Laplace rows (`NNLS._laplace_pm_block`, per aperture a, MGE projected mass M_a):
- star rows: dense `M` (n_stars x K2) of each star's error-kernel cell masses (sparse:
  13.6M non-zeros over 610 822 HST stars, ~22 cells per star), built by a Python loop in
  `ProperMotions.laplace_design`;
- `p0 = M h0`, `F = M' diag(1/p0^2) M`, `g = n/M_a - M'(1/p0)` (`fitstat_pm.laplace_terms`);
- `Lt` = upper Cholesky of `F + delta I`, `delta = 1e-3 n / M_a^2`
  (`default_delta`; veldist C1b: a 100x smaller ridge biased sigma by ~1 km/s, so the
  ridge is part of the statistic, not only a numerical guard);
- model rows `H_a Lt'` (H_a = orbit histograms of aperture a, n_orbs x K2, float32 in
  storage, cast to float64), data `Lt h0`, errors 1, plus linear term `2 g' H_a w` via
  `acc.add_linear`. Statistic = `|Lt (h - h0)|^2 + 2 g'(h - h0)`.
- Stars per HST aperture: min 176, median 432, max 703 (< 529 cells for most), total
  610 822. Gaia: 324 / 428 / 571 per aperture, 64 539 total, 289 cells.
  So rank(F_a) <= n_stars_a, but the ridge makes F + delta I full rank.

The diagonal (baseline) statistic has the same rows (K2 per aperture, divided by sigma),
so one Gram build costs the same with either statistic.

`_solve_laplace` (re-expansion driver; `pm_laplace_reexpand: 1`):
1. build Gram about the data h0 -> ADMM -> w1
2. capture read: rhs-only pass that records `h_a = w1 @ H_a` (but still builds the full
   Laplace rows and `H @ Lt'` it never uses)
3. new expansion point `((1-f) h_a/sum h_a + f/K2) M_a`, f = 1e-3; full Gram rebuild
   (F, Lt, g all recomputed) -> ADMM (warm) -> w2
4. final chi2 read about the data h0 at w2 (rhs-only with chi2_w)
veldist C1c: one re-expansion lands within ~0.03 km/s of the exact per-star likelihood.

Solver settings: `nnls_dtype: float32` (orbit-library storage / rows), Gram float64,
`stream_orblib_reads: True`, `gram_chunk_rows: 8192`, `gram_aperture_group: 128`,
`admm_alpha: 1.9`, `admm_tol: 1e-10`, `admm_free_p: True`, `ridge_lambda: 0.03`.

## 2. Hardware and run conditions

astro-node-12: AMD EPYC 9474F, 2 sockets x 48 cores, SMT 2 (192 threads), 2 NUMA nodes,
1.4 TB RAM; shared with other users at load ~100-180. Our cap: 16 cores (soft), keep
300 GB free. Production jobs ran with `MKL_NUM_THREADS=4 OMP_NUM_THREADS=4`, under
`nice -n 19`, six re-solves concurrently (prod1-prod6) plus E1/relative jobs, so absolute
times are contaminated by contention; shares within a run are not. Peak RSS per solve
~44 GB (G 16 GB + blocks + library chunks). Orbit library reads: bzip2 archives,
decompressed via subprocess (`orblib._decompress`), streamed.

## 3. Measured wall time

prod1 (orblib_001_001, from log timestamps):

| phase | wall |
|---|---|
| config + projected masses (MGE) | 24 min |
| pass-1 Gram build | ~4.5 h |
| ADMM 1 | 363 it, 491 s |
| capture read + pass-2 Gram build | ~8.4 h |
| ADMM 2 | 302 it, 435 s |
| final chi2 + post | 1.1 h |
| total `solve` | 51 600 s = 14.3 h |

All six: solve 49 800-56 400 s (13.8-15.7 h); capture_base 1 000-2 600 s, capture_new
1 100-1 800 s (separate model-histogram dumps by the driver `prod_one.py`).

py-spy dumps every 2 min (`$W/prof/orblib_002_000.log`, 413 samples = 13.8 h;
`orblib_004_005.log`, 480 samples). Stage = line of `_solve_laplace` on the stack
(node numbering), top frame:

orblib_002_000:

| stage | samples (x2 min) | top frames |
|---|---|---|
| pass-1 build (L2493) | 149 (5.0 h) | dsyrk 132, read/getitem/decompress 6, Laplace block 11 |
| ADMM 1 (L2494) | 5 | |
| capture read (L2498) | 33 (1.1 h) | `H = float64(orb[:, a])` 8, `H @ Lt'`-line/capture 11, reads ~12 |
| pass-2 build (L2508) | ~196 (6.5 h) | dsyrk 163, Laplace block 15, reads ~16 |
| ADMM 2 (L2509) | 3 | |
| final chi2 (L2512) | 14 (0.5 h) | Laplace block, reads |
| other | 16 | |

Overall: dsyrk 295/413 = 71%; the Laplace block (float64 copy of H, `H @ Lt'`,
`laplace_terms`) ~13%; reads/decompression ~6%; ADMM ~2%.
orblib_004_005: dsyrk 328/480 = 68%, same shape (pass-1 build 228 samples, pass-2 ~165).
`laplace_terms` itself (dense M, F, Cholesky) is rarely sampled: cheap.

Historic (other runs, other conditions; `docs/e2-mock-plan.md`): diagonal addback
production solve ~1-1.5 h total (HST Gram read 9.2 min per read before read speedups,
ADMM 291 s); relative-PM mode 5.3 h before, ~2.6 h after hold-across-passes (pass 0
alone 64 min, 56 GB). Thread counts of those runs are not recorded here (a config
comment mentions a 192-MKL-thread overlap). So the "Laplace is 10x slower" comparison is
not like-for-like: per build the cost is the same; the Laplace driver does two builds,
a capture read and a chi2 read.

Relative + Laplace (vp mode, relB6/relC6, running since 2026-11-17 17:32 / 18:42 UTC):
each pass ~5-8 h, ADMM 310-700 s per pass, 352 -> 218 iterations as passes converge;
relB6 at 5 passes after 26 h and still running.

## 4. Existing instrumentation

- `DYNAMITE_GRAM_PROFILE=<path>`: opt-in per-phase wall/RSS JSON (`GramProfiler`):
  phases read_vel_histograms, downcast, prepare_kinematic_block, kin_econ_divide,
  `<tag>dsyrk`, `<tag>cross_matvec`, finalize. Not yet used on a Laplace run.
- `$W/launch.sh` jobs, py-spy at `$W/../ENV_mkl/bin/py-spy`.

## 5. Constraints on any change

- Bitwise agreement across the five read paths and `dev_tests/test_gram_blockwise.py`
  (econ divide before accumulation; row 0 as equality) must hold.
- Column-norm spread spans many decades (accumulator docstring): float32 Gram is risky.
- The ridge delta is part of the statistic (see above).
- Memory: G is 16 GB; anything adding another n_orbs^2 array doubles the peak.
- Accuracy target: per-set chi2 within ~1 of current at prod1 weights; the IMBH profile
  differences are hundreds to thousands, so tolerances can be explicit.
- Local data for experiments: `experiments/fitstat/work/speed/{hst,gaia}_veldist_pm_2dhist.npz`
  (star rows PM_star_indptr/cell/mass/aperture, PM_lap_h0), and
  `experiments/fitstat/work/model_hists_prod1.npz` (per-aperture data, err and model
  histograms at v4, E1 and prod1 weights; keys like `hst_pm|model|prod_orblib_001_001_ml02.96:w_new`,
  shape (1406, 529); check its normalisation against M_a before using).
  The orbit library itself is only on the node (45k orbits, ~12 GB bzip2 per model).

## 6. Node checks of the speedup plan (2026-11-18 evening)

Plan and analysis: `docs/laplace-solve-speedup-analysis.md`. Scripts: `experiments/fitstat/speed/`.

- L1 (support overlap, E1 outputs, 9 models): pass-1 support 13.8k-17.6k of 45k orbits; ~1% of the
  pass-2 support is new and carries 0.25-0.53% of the pass-2 weight.
- M3/M6 (`m3_support.py`, one rhs-only read of prod1 = orblib_001_001 at `w_new`, 33 min at load ~185):
  orbits with any non-zero cell per aperture, median (p10-p90): MUSE 0.35, HST 0.335 (0.29-0.43),
  Gaia 0.19 (0.16-0.23); non-zero orbits are scattered (median ~120 runs per aperture). Star-row rank:
  MUSE 18/21, HST 160/529, Gaia 285/289. Dropped-ridge share of an orbit's Laplace diagonal: max 5e-5
  (HST), 2e-12 (Gaia). Per-set statistic at `w_new` reproduces prod1 `sets_new` exactly.
- M4 (`m4_rowspace.py`, fork f2c4780, `pm_laplace_rowspace_kappa: 10`): rows reaching dsyrk
  HST 174,805 / 745,180, Gaia 16,015 / 43,210; all rows 210,793 / 808,363 (3.83x fewer, so ~3.8x less
  dsyrk work). The chi2 read gives prod1's per-set values to <= 3e-16 relative; reported bound
  E = 0.0040 (HST), 0.0022 (Gaia). Each read took 33-34 min at load ~180 (4 threads).
  First run (m4row) found a real bug: zero-A rows with b != 0 (cells no orbit reaches) were dropped
  with their b; fixed in f2c4780 with a test.
- M1 (`m1_syrk_bench.py`, one 2048-row block into a 45k x 45k Gram, best of 2, load ~160-165, ENV_mkl
  MKL): dsyrk 147 / 132 / 302 GFLOP/s at 4 / 8 / 16 threads (37 / 17 / 19 per thread); ssyrk 156 / 298 /
  527. The first repeat of each is slower (first touch of G's pages). Noisy under contention (8 threads
  slower than 4 for dsyrk), but 16 threads give ~2x over 4 and float32 ~1.75x at 16 threads.
  One production Gram build (~1.5e15 flop) is ~2.8 h of dsyrk at 4 threads at the measured rate, close to
  the ~4.5 h observed in the six-at-once production runs. With row-space rows (3.8x) and 16 threads (2x):
  ~20-25 min of dsyrk per build.

## 7. Options B and D1 implemented (2026-11-18 night)

Fork `feature/pm-fit-statistic`: 6605e3d (B), 8d5d965 (B guard), 9835e9c (D1). Tests:
`dev_tests/test_laplace_fused.py` (6), `dev_tests/test_laplace_working_set.py` (6); full suite unchanged
(same 45 pre-existing failures).

- **B, always on.** Each re-expansion pass's build forms `h_a = w_prev @ H_a` from the H_a it holds and expands
  about `mix(h_a) M_a` (`NNLS._lap_reexpand`), so the capture read is gone; `h0` is bitwise the two-read value
  (grouped streamed reads included). The final chi2 read is cell space: zero rows with `b = -Lt d`, no `H Lt'`
  (only when no vp gradient/line term needs the rows). `laplace_info["h0_pass"]` keeps the expansion points.
  The driver now keeps only the previous solver's `(z, u)`, freeing its p x p factor before the next build.
- **D1, `pm_laplace_working_set: FRAC`** (off by default), last re-expansion pass only. S = previous support plus the
  zero orbits with the largest ADMM dual u, `max(ceil(FRAC n), |support|)` orbits. G is accumulated over S only
  (`NormalEquationAccumulator(cols=S)`); column scaling and the ridge scale (`p_diag_mean`) come from every
  orbit's diagonal, so the restricted P is the full P's sub-matrix. The final chi2 read also returns the
  half-gradient of the pass's own objective for all orbits in cell space (`_lap_grad`: `H (C'C d2 + g2)`,
  C = R or Lt), giving `gamma = g + ridge - nu` (nu = median over the support) and the gap bound
  `2 M max(-gamma)_+` (chi2 units). Above `pm_laplace_working_set_gap` (0.01) the violators are added and the
  pass repeated (<= 4 rounds); `laplace_info["working_set"]` logs m, gap, violators, nu and its p10-p90 spread.
- Node acceptance run `s1a` (`experiments/fitstat/speed/s1_resolve.py`, orblib_001_001, kappa 10, FRAC 0.5,
  4 threads, `DYNAMITE_GRAM_PROFILE` on), launched 2026-11-18 ~22:20 local (UTC-3): first a cell-space chi2 read at
  prod1 `w_new` (must reproduce `sets_new`), then the driver. Results to follow here.
