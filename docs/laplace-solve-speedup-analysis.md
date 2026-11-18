# Laplace production solve: speedup analysis (2026-11-18)

Companion to `docs/laplace-solve-performance-record.md` (the measurements) and
`docs/superpowers/plans/2026-11-18-laplace-solve-speedup.md` (the first-pass list, S1-S7).
Code read: `~/research/dynamite-fitstat` at a464c01 (`weight_solvers.py`, `fitstat_pm.py`,
`kinematics.py`). Laptop scripts, logs and outputs: `experiments/fitstat/work/speed/`
(`common.py`, `m2_spectrum.py`, `m3_occupied.py`, `m4_expansion_point.py`,
`m5_trunc_worstcase.py`, `m6_rowspace.py`, `m7_c1b_rowspace.py`, each with a `.log`;
`m7_c1b_stored.log`; `m2_spectrum_{hst,gaia}.npz`).

Every number is tagged **[measured]** (from the record: logs, py-spy), **[computed]** (by me
on the laptop, from the local star rows and the prod1 model histograms) or **[estimated]**
(arithmetic on top of those; no run behind it). Nothing was run on the node and no code in
`dynamite-fitstat` was changed.

## 1. Summary

1. **The Laplace curvature has a rank cliff that the current rows ignore.** `F_a = M_a' D M_a`
   lives in the row space of the star matrix `M_a`, which does not depend on the expansion
   point. For HST that space has 0.25-0.31 of the 529 cells [computed]; everything else is
   the ridge alone. The whole ridge term is worth 0.16 of the HST statistic of 10 994 at the
   prod1 weights [computed]. Feeding only the row-space rows to `dsyrk` cuts HST rows by
   3.3x (no eigen-truncation) to 4.3x (dropping eigenvalues below 10 delta), with a
   rigorous, a-posteriori-checkable bound on the change of the minimum. This **removes the
   ridge from the solver's Hessian on 70-77% of the HST cell-space directions**; the
   reported chi2 keeps the full ridge. In a stress test with every cell free the minimum
   moves by < 1e-3 for the whole set, and on the C1b synthetic seeds the fitted dispersion
   moves by < 1e-4 km/s [computed].
   The record's premise that a 100x smaller ridge biases sigma by ~1 km/s does not survive
   the stored re-run (section 3.4): that shift was an optimiser failure of the folded-rows
   form, and with the offset-free form the same change moves sigma by <= 2e-4 km/s.
2. **The same fact removes the pass-2 read.** `B_a = H_a V_a` (orbit histograms projected
   on the row-space basis, ~33-45 GiB float32) cached during pass 1 is enough to build
   pass 2 exactly, and to evaluate model star probabilities and gradients in seconds.
3. **The solution is sparse**: 30-42% of the 45 000 orbits carry weight in the six prod
   solves [computed from the stored weights]. A working-set Gram over a superset of the
   support, verified by KKT on the excluded orbits, is exact at convergence and costs
   `(m/n)^2` of the `dsyrk`.
4. **Freezing F for pass 2 (S7) is dead**, with or without a low-rank correction [computed].
5. The capture read can be fused into the pass-2 build exactly (S3/S4).
6. Contention explains at most ~2x of the wall time; the rest is real flops (section 2).

Expected wall time at the same contention as prod2 [estimated, section 6]:
13.8 h -> ~4.7 h after phases 1-2, ~3 h after phase 3-4a, ~2 h after 4b/5.

Exactness labels used below:

| label | meaning |
|---|---|
| bitwise | same bits as now |
| roundoff | same mathematics, different summation order (rel ~1e-12) |
| certified | exact at convergence, with a computable certificate (KKT / gap) |
| bounded | not exact; error has a rigorous bound that the final read can evaluate |
| approximate | changes the result with no tight bound; needs calibration |

Section 5 of the record requires bitwise agreement on the existing paths, so **every option
is an opt-in setting whose default leaves the present code path untouched**.

## 2. Where the time goes, restated

Flop convention: `dsyrk` of a k x n block costs `k n^2` flops (multiply and add counted
separately, upper triangle only). One build: 790k rows x 45 000^2 = 1.6e15 flop, as in the
record.

- prod2 pass 1: 132 `dsyrk` samples x 120 s = 15 840 s -> 101 GFLOP/s on 4 threads =
  25 GFLOP/s per thread. Pass 2: 163 samples -> 82 GFLOP/s = 20 per thread [computed from
  the measured samples].
- A free Zen 4 core should reach roughly 45-50 GFLOP/s in `dsyrk` [estimated: AVX2 FMA,
  2 x 4 x 2 x ~3.7 GHz = 59 peak]. So the runs got ~45% of a core per thread, consistent
  with SMT sharing at load 160 and nice 19.
- Consequence: an uncontended run would be about 2x faster per thread, not 5-10x. One
  build alone on 16 free cores would take about 1.6e15 / (16 x 45e9) = 37 min [estimated].
  More threads per job shorten latency but not core-hours; with six jobs at once the
  16-core cap is already exceeded. **The cure has to be fewer flops.**

Stage budget, prod2, in py-spy samples of 2 min [measured]:

| stage | samples | dsyrk | Laplace block | reads | other |
|---|---|---|---|---|---|
| pass-1 build | 149 | 132 | 11 | 6 | |
| ADMM 1 | 5 | | | | 5 |
| capture read | 33 | | 19 | 12 | 2 |
| pass-2 build | 196 | 163 | 15 | 16 | 2 |
| ADMM 2 | 3 | | | | 3 |
| final chi2 | 14 | | ~6 | ~8 | |
| other | 16 | | | | 16 |
| total | 413-416 | 295 | ~51 | ~42 | ~28 |

The ~120 non-`dsyrk` samples (4 h) are the floor once `dsyrk` shrinks, so the read-side
items rank higher than their share today suggests.

## 3. Laptop measurements

Local inputs: `work/speed/{hst,gaia}_veldist_pm_2dhist.npz` (star rows; both have
`PM_star_sharpened = True`, so rows have negative side lobes, min entry -0.097) and
`work/model_hists_prod1.npz` (model histograms at the v4, E1 and prod1 weights of
orblib_001_001). `M_a` = row sum of the stored data histogram.

**Gate [computed].** The Laplace statistic recomputed locally from star rows and the stored
float32 model histograms reproduces the production numbers:

| set | local | prod1 `sets_new` | rel. diff |
|---|---|---|---|
| hst_pm at `w_new` | 10 994.476 | 10 994.4759 | -9e-10 |
| gaia_pm at `w_new` | 1 482.645 | 1 482.6449 | +2e-9 |
| hst_pm at v4 weights | 12 293.136 | 12 293.1358 (`sets_base`) | |

So the local pipeline is the production statistic, and the numbers below can be trusted at
that level.

### 3.1 Size of the ridge term (`m3_occupied.log`)

`sum_a delta_a |h_a - h0_a|^2`, the entire ridge contribution to the statistic [computed]:

| model | HST | Gaia |
|---|---|---|
| v4 weights | 0.142 | 0.379 |
| E1 `w_lap` | 0.164 | 0.110 |
| prod1 `w_new` | 0.159 | 0.142 |

A small value does not by itself show the ridge is irrelevant to the minimiser; sections
3.3 and 3.4 test that.

### 3.2 Spectrum of F_a in units of delta_a (`m2_spectrum.log`)

Rows kept if `lambda_i > kappa delta_a`, summed over apertures, as a fraction of all
cell rows; and the dropped quadratic `E = sum_dropped (lambda_i + delta)(u_i . d)^2` at
fixed models, expansion about the data h0 [computed]:

| kappa | HST rows kept | HST E at prod1 | Gaia rows kept | Gaia E at prod1 |
|---|---|---|---|---|
| 0.01 | 185 757 = 0.250 | 0.0020 | 23 596 = 0.548 | 6.7e-6 |
| 1 | 177 761 = 0.239 | 0.0025 | 18 461 = 0.429 | 1.5e-4 |
| 10 | 173 197 = 0.233 | 0.0042 | 15 727 = 0.365 | 0.0025 |
| 100 | 167 815 = 0.226 | 0.025 | 12 870 = 0.299 | 0.18 |
| 1000 | 161 320 = 0.217 | 0.39 | 9 456 = 0.220 | 21.8 |
| 1e4 | 151 841 = 0.204 | 10.2 | 4 981 = 0.116 | 283 |

- The kept-row counts at the floored prod1 model (the pass-2 expansion point) agree with
  those at the data h0 to within 0.5% at every kappa. E at the v4 and E1 models is within
  a factor 1.5 of the prod1 column.
- HST: 206 574 eigen-directions have `0 < lambda <= 0.01 delta` and ~351 000 are
  numerically zero; only 24 437 lie between 0.01 delta and 1000 delta. It is a cliff, so
  the choice of kappa between 0.01 and 100 barely changes the row count.
- HST apertures have a median of 242 cells touched by any star entry (0.458 of all cells),
  130 of numerical rank. Gaia's kernels (20 km/s errors on 6.6 km/s cells) touch all 289
  cells; its spectrum decays smoothly instead.
- Per HST aperture, rows kept at kappa = 10: min 79, median 121, max 181 (of 529).

### 3.3 Free-cell stress test of the truncation (`m5_trunc_worstcase.log`)

A fixed-model error does not show where a truncated solve would go: in the dropped
directions the linear term `2 g'd` has no curvature against it. Test: let every cell of an
aperture be a free parameter (`h >= 0`, `sum h = M_a`), minimise the truncated statistic by
QP (cvxopt), score with the exact one. This is a stress test, not an upper bound for orbit
models (a larger feasible set does not bound the gap on a smaller one); the bound is
`E(w~)` of option A. Excess
`Q(h~) - Q(h*)` summed and scaled to the full set [computed; 40 HST and 30 Gaia apertures]:

| kappa | HST excess (set) | Gaia excess (set) |
|---|---|---|
| 0.01 - 10 | < 1e-6 | < 4e-5 |
| 100 | 2e-6 | 0.027 |
| 1000 | 4e-4 | 231 |

Reason: for HST the dropped directions are cells with no stars, where `g = n/M_a > 0`
pushes the model to the positivity bound with or without the ridge. The unconstrained
excursion `|g.u| / (lambda + delta)` exceeds the aperture mass for 99.7% of the directions
below 10 delta (median 168-765 M_a). Positivity, not the ridge, holds those cells.

### 3.4 Does the ridge move the fit? C1b seeds (`m7_c1b_stored.log`, `m7_c1b_rowspace.log`)

The record takes "a 100x smaller ridge biased sigma by ~1 km/s" from C1b. The stored C1b
outputs (`work/c1b`, `work/c1b2`), `oracle_d01` (ridge x 0.01) minus `oracle`, sigma =
sqrt(s1 s2) [computed from stored runs, 360 fits per run, both modes, five tests]:

| run | statistic form | mean shift | max abs shift | d01 fits stuck at the start values |
|---|---|---|---|---|
| `work/c1b` (first) | rows with g folded in; objective carries a constant ~7e7 (7e9 for d01) | +0.14 to +0.31 km/s (HST tests) | 2.03 km/s | 124 of 320 HST-regime fits |
| `work/c1b2` (re-run) | offset-free, as DYNAMITE uses | 0.0000 | 0.0002 km/s | 0 |

In the first run the d01 fits return `s2 = 17.0, mu = 1.0`, the start values: L-BFGS-B
stopped on the 7e9 constant. The commit that produced the re-run (dda58a2) names it a
"Laplace offset artefact". With the offset-free statistic the ridge scale does not move
the Gaussian-family fit.

Same seeds, addback mode, oracle expansion point, offset-free `d'Pd + 2g'd` with reduced P,
paired against `P = F + delta I` [computed; 12 calib, 8 gaia, 8 tail, 8 edge seeds]:

| P | max abs sigma shift, HST-regime tests | Gaia-regime test |
|---|---|---|
| `F + 0.01 delta I` | < 5e-5 km/s | 1e-4 |
| row space of M only (tol 1e-8) | < 5e-5 | < 5e-5 |
| eigen-cut kappa = 10 | < 5e-5 | < 5e-5 |
| eigen-cut kappa = 100 | < 5e-5 | 4e-4 |
| eigen-cut kappa = 1000 | 1e-4 | 0.006 |

The threshold from C1c is 0.03 km/s. The model family here has three parameters
(s1, s2, mu), so this is the stiff end; section 3.3 is the loose end. Row-space rows and
kappa <= 100 pass both.

### 3.5 An expansion-independent basis (`m6_rowspace.log`)

`V_a` = right singular vectors of `M_a` with `s > tol s_max` [computed]:

| tol | HST rows | HST cache (45k orbits, float32) | worst missed curvature of F2, in delta | Gaia rows |
|---|---|---|---|---|
| 1e-4 | 185 551 = 0.249 | 31.1 GiB | 6e6 | 24 940 = 0.579 |
| 1e-6 | 199 186 = 0.268 | 33.4 GiB | 486 (median 0.07) | 34 203 = 0.794 |
| 1e-8 | 227 070 = 0.305 | 38.1 GiB | 1.9 (median 6e-4) | 42 243 = 0.981 |

F2 is F about the floored prod1 model. At tol 1e-8 the error of `d'F2 d` at the prod1 model
is 4e-4 of `delta |d|^2`, i.e. ~1e-4 of the 0.16 ridge. Gaia has no usable exact low rank;
its basis is the full cell space (7 GiB).

### 3.6 Pass-2 curvature against pass-1 curvature (`m2_spectrum.log`)

Proxy: F2 about the floored prod1 `w_new` model (the true pass-1 model is not stored
locally), F1 about the data h0 [computed, HST]:

- per-star weight ratio `(p_data / p_model)^2`: 1% / 5% / 50% / 95% / 99% =
  0.42 / 0.59 / 1.03 / 1.61 / 2.59, maximum 7.7e6;
- generalised eigenvalues of `(F2 + delta I, F1 + delta I)`: global range [0.052, 7.7e6];
  per-aperture maximum has median 10.6 and 95th percentile 1 479;
- correcting exactly the stars with `|ratio - 1| > 0.3` (179 034 stars, 29%) leaves
  [0.70, 1.30]; `> 0.1` (437 120 stars, 72%) leaves [0.90, 1.10].

### 3.7 Support of the solutions (`work/prod/*.npz`)

Orbits with weight > 0 [computed]:

| model | v4 weights | prod `w_new` | in both (> 1e-8 of total) | new only |
|---|---|---|---|---|
| 000_002 | 15 736 | 17 820 | 12 299 | 3 073 |
| 001_001 | 15 146 | 16 960 | 11 953 | 2 787 |
| 002_000 | 13 553 | 15 178 | 11 264 | 2 532 |
| 002_006 | 12 019 | 13 573 | 9 882 | 2 428 |
| 003_006 | 16 025 | 18 933 | 12 101 | 3 748 |
| 004_005 | 14 700 | 16 659 | 11 435 | 3 038 |

The final support is 30-42% of the library. The v4 (diagonal-statistic) support misses
about 20% of it. Pass-1 weights are not stored locally (`w_pass` is saved by `e1_solve.py`
on the node), so the pass-1 to pass-2 overlap is unmeasured.

### 3.8 Not settled locally

- Per-aperture orbit support (which orbits have any mass in aperture a): no orbit library
  on the laptop. See M3.
- Whether pass 1 could be replaced by an expansion about an existing model
  (`m4_expansion_point.log`): inconclusive. The exact per-star statistic at the stored
  model histograms meets non-positive star probabilities (562 HST stars over the points
  tested) because the sharpened rows have negative lobes, and the totals are dominated by
  those clips. Robust part: median per-aperture error of the quadratic model at the prod1
  model is 1.44 expanded about the data, 0.145 about the floored v4 model, 0.014 about the
  floored E1 model. Not used in the ranking. The non-positive probabilities themselves
  (stored model histograms against sharpened rows) were not followed up.

## 4. Options, ranked

`n` = 45 000 orbits, K2 = cells per aperture, `H_a` = orbit histograms (n x K2),
`Q_a = F_a + delta_a I`. Gram contribution of a Laplace set: `sum_a H_a Q_a H_a'`.

### A. Row-space rows (replaces S1) — bounded, a-posteriori certified

**Mathematics.** `F_a = M_a' diag(1/p0^2) M_a`, so `range(F_a) = rowspace(M_a)` for every
expansion point. With `V_a` an orthonormal basis of that row space (r_a columns),

    Q_a = V_a (V_a' F_a V_a + delta I_r) V_a'  +  delta (I - V_a V_a')
    H_a Q_a H_a' = B_a C_a' C_a B_a'  +  delta H_a (I - V_a V_a') H_a'
    B_a = H_a V_a (n x r_a),   C_a' C_a = V_a' F_a V_a + delta I_r

Feed the r_a rows `C_a B_a'` and drop the second term: the ridge on cell-space directions
no star touches. Optionally also drop eigen-directions of the small r x r matrix below
`kappa delta` (needed for Gaia, a small extra for HST). The linear term `g` is kept in full.

**Error bound.** Write the statistic as `Q = Q~ + E` with
`E(w) = sum_a sum_dropped (lambda_i + delta_a)(u_i . d_a(w))^2 >= 0`. If `w~` minimises the
truncated problem and `w*` the exact one over the same constraints,

    0 <= Q(w~) - Q(w*) <= E(w~) - E(w*) <= E(w~)

`E(w~)` is a per-aperture cell-space quantity that the final chi2 read evaluates for free,
so every solve reports its own bound. Expected size: 0.002-0.004 (HST), 0.003 (Gaia) at
kappa = 10 [computed, 3.2]; free-cell stress-test excess < 1e-4 [computed, 3.3]; no shift
of the fitted dispersion on the C1b seeds [computed, 3.4]. The reported
chi2 stays the full statistic (ridge included), evaluated in cell space at the final
weights.

**Speedup [estimated from computed row counts].** Rows per build:

| variant | HST | Gaia | MUSE + mass etc. | total | dsyrk factor |
|---|---|---|---|---|---|
| now | 743 774 | 43 061 | ~5 000 | ~792k | 1 |
| row space only, tol 1e-8 | 227 070 | 43 061 | ~5 000 | ~275k | 2.9x |
| + eigen-cut kappa = 10 | 173 197 | 15 727 | ~5 000 | ~194k | 4.1x |
| + eigen-cut kappa = 100 | 167 815 | 12 870 | ~5 000 | ~186k | 4.3x |

Recommended: kappa = 10 for both sets. The `H @ Lt'` product in the Laplace block shrinks
by the same factor (r instead of 529 columns). The MUSE set (166 x 21) is left alone.

**Memory.** None extra; blocks get smaller.

**Constraints.** Default path bitwise untouched (setting off). Econ-before-accumulation
holds trivially (Laplace rows have unit errors). The ridge stays on every kept direction.
One coupling needs care: `diag(G)` loses `delta_a (|H_a,j|^2 - |B_a,j|^2)` per orbit j, and
`col = sqrt(diag G)` sets the scale of the `ridge_lambda` penalty. The size of that loss was
not measured (it needs the orbit library); an orbit living mostly in star-free cells could
lose a large part of its PM-set diagonal. Fix: accumulate that dropped diagonal per orbit
(two row-norms per aperture, no n^2 work) and add it to `diag(G)` before `col` is taken, so
the column scaling and the `ridge_lambda` term are those of the full build. Then the bound
above holds for the objective the solver minimises, `ridge_lambda` term included. Not
bitwise against the full Laplace build, by construction.

**Implementation: small (~60 lines).** In `_laplace_pm_block`: SVD of `M_a` once per
aperture per solve (cache `V_a`), `eigh` of the r x r matrix, write the r_a rows into the
first r_a slots of the aperture's K2 rows and zeros after, record the keep mask. In the
chunk loops of `construct_gram_and_rhs_blockwise`: compact away the masked rows before
`acc.add` (dense BLAS does not skip zero rows). `con`/`econ` keep their length.

### B. Fuse the capture into the pass-2 build; cell-space chi2 (S3 + S4) — roundoff

`h_a = w1 @ H_a` is available when aperture a is loaded for the pass-2 build, and the new
`h0_a`, `F_a`, `C_a`, `g_a` depend only on that aperture. So compute `h_a` first, then the
rows, in the same loop: the capture read disappears (33 samples = 1.1 h). The final chi2
read needs only `h_a = w2 @ H_a` and `Q(h_a)` in cell space: no rows, no `H @ Lt'`
(~6 samples). `prod_one.py`'s `capture_base` / `capture_new` (18 + 29 min, outside `solve`)
take the same shortcut.

Exact: same quantities, per-aperture order unchanged; `h0` values identical to the
two-read version. Memory unchanged. ~50 lines. Acceptance: `_lap_h0` arrays identical to
the two-read path (bitwise expected), G / v / chi2 rel 1e-12.

### C. Projected-library cache `B_a = H_a V_a` (new) — same bound as A

Because `V_a` is expansion-independent, `B_a` stored during the pass-1 read gives, with no
second read:

- model star probabilities: `M_a h_a = (M_a V_a)(B_a' w)` (add the constant vector to
  `V_a` so `sum(h_a)` and the uniform floor of `laplace_reexpand_mix` are exact);
- pass-2 rows `C2_a B_a'`;
- the gradient of any pass's objective for all orbits: `B_a (C'C (B_a' w - V_a' h0)) + B_a V_a' g`.

Cost: 33-38 GiB (HST, tol 1e-6 to 1e-8) + 7 GiB (Gaia, full) float32 [computed, 3.5]; in
RAM (RSS 44 -> ~85 GiB while G is 16 GiB; ~55 GiB with option D) or an `np.memmap` on
scratch. Saves the pass-2 read (16 samples), the pass-2 float64 copies, and turns a
gradient evaluation from a 20-30 min read into ~2e10 flops (seconds). This is what makes D
cheap, and it is the natural base for more re-expansion passes. The final chi2 read stays
(it supplies the full-statistic value and `E(w~)`), at least until parity is established.

Not usable as is for relative vp mode, where the shift `S_c` acts on `H_a` before the
projection; option A's row reduction still applies there per pass.

Implementation: medium (~150 lines; cache object, pass-2 build from cache, tests).

### D. Working-set Gram with KKT verification (new) — certified

**Mathematics.** The QP is `min w'Gw - 2v'w` over `w >= 0`, `row0 . w = b0` (plus the
diagonal `ridge_lambda` in normalised units). If `w_S` solves the problem restricted to an
orbit subset S and the reduced gradient `gamma_j = 2 (G w - v)_j - nu` is `>= 0` for every
`j` outside S (nu = multiplier of the mass equality), then `w_S` padded with zeros is the
solution of the full problem. Column normalisation uses `diag(G)`, which is identical for
the kept columns, so the restricted P is exactly the sub-matrix of the full P.
Violators are added and the solve repeated; termination is finite. Certificate: by
convexity and complementarity on S, `f(w_S) - f(w*) <= max_j (-gamma_j)_+ x sum(w*)`, and
`sum(w*)` is fixed by the mass equality, so the largest violation gives a rigorous gap.

**Speedup [estimated].** `dsyrk` cost `(m/n)^2`. Final supports are 13.6k-18.9k
[computed, 3.7]; with a margin, m = 20-25k gives 0.20-0.31, i.e. 3-5x on the pass it is
used for, and G shrinks from 16 GiB to 3-5 GiB.

- **D1, pass 2 only.** S = support of w1 plus the orbits with the smallest pass-1 reduced
  gradients, taken from ADMM's dual variable for the bound `w >= 0` at the pass-1 solution
  (P itself is consumed by then). Verification uses the
  gradient from cache C, or the existing `grad_w` rhs-only read fused into the final read.
- **D2, pass 1 too.** Needs a seed (v4 weights for re-solves, a neighbouring model's
  support for new models; the v4 support misses ~20% of the final one) and several
  augmentation rounds. Only sensible with cache C, where a round costs a gradient (seconds),
  a border update `|new| x m x rows` and a re-solve at size m.

**Constraints.** Exact at convergence regardless of the seed. Memory falls. ADMM
factorisation and iterations also get cheaper (`m^3`, `m^2`).

**Implementation: medium (D1 ~120 lines), large (D2).** Unknown that decides the margin:
how much the support moves between pass 1 and pass 2 (node lookup L1).

### E. float32 products, float64 accumulation (refines S6) — roundoff at 1e-6, certified by the gradient

Compute each block's update with `ssyrk` into a float32 n x n scratch and add it to the
float64 G. Per-block relative error ~1e-6 of `|A|'|A|`, componentwise, so the column-norm
spread does not amplify it for non-negative rows; Laplace rows are signed, so cancellation
inside a block is possible and has to be measured, not argued. The reported chi2 is
computed in float64 from the model histograms, so the Gram error moves only the minimiser:
the effect on the reported minimum is second order, and the KKT residual from an exact
gradient certifies it; one refinement solve with the exact gradient removes it.

Speedup 2x on whatever `dsyrk` remains [estimated; M1 measures]. Memory +8 GiB scratch
(or tiled `sgemm`, no scratch). ~40 lines. After A and D little `dsyrk` is left, so this
comes last; the record's warning about a float32 Gram stands for a pure-float32 G, which is
not proposed.

### F. Threads, NUMA, BLAS (S2) — roundoff

Latency only (section 2). Worth a 10-minute measurement because 25 GFLOP/s per thread
leaves up to 2x on the table if it is placement, not load: pin to physical cores of one
socket (`numactl --cpunodebind --membind`, `OMP_PLACES=cores`, `OMP_PROC_BIND=close`),
check which code path MKL takes on Zen 4, try 8 and 16 threads for the build phase only.
No code.

### G. Orbit support per aperture (S5) — roundoff; gain unknown

If aperture a is reached by a fraction f_a of the orbits, its rows need only an
`f_a n` square of G: cost `f_a^2`. Orbit histograms are projected, so central apertures are
reached by most orbits that extend beyond them; only tightly bound orbits miss outer
apertures. If the library is ordered by energy the support is close to a contiguous suffix
and the update is a plain sub-block `dsyrk`. No factor is claimed until M3; it multiplies
with A and overlaps with D (both shrink the orbit dimension).

### Rejected or dominated

| idea | verdict | basis |
|---|---|---|
| Freeze F at pass 1, re-expand g only (S7) | no | generalised eigenvalues of pass-2 against pass-1 curvature span [0.05, 7.7e6] in cell space [computed, 3.6]; that range is an upper bound on the mismatch in orbit space, so a frozen-Hessian iteration is not guaranteed to contract, and the next row shows the repair costs a rebuild |
| G2 = G1 + low-rank update over the stars that changed | no | 29% of the stars (179k rows) must be corrected to reach +-30%, 72% for +-10%; a full row-space rebuild is 173-227k rows and exact, and G1 would have to be kept (+16 GiB) |
| Share the ridge Gram `delta sum H_a H_a'` across passes | no | it costs a full 744k-row build to form; its whole value is 0.16 and option A bounds what is dropped |
| Star-space rows `(1/p0_s) m_s' H_a'` | no | 611k rows plus ridge rows; A reaches the same rank r_a <= n_stars exactly and far lower |
| Matrix-free / first-order solver | no | with cache C a Gram matvec is ~3.5e10 flop (a few s, memory bound); the row-space Gram is 195k x n^2 = 3.9e14 flop, i.e. ~1e4 matvecs' flops, but ADMM's ~330 iterations each need a solve with `P + rho I`: inner CG at tol 1e-10 on a matrix with decades of column spread costs 10^2-10^3 matvecs per iteration [estimated], so 3e4-3e5 matvecs: more flops than the reduced Gram and memory-bound besides. The gradient alone (certificates, working set) is the useful matrix-free piece |
| Skip pass 1, expand about an existing model | not now | changes the definition of the result; local test inconclusive (3.7) |

## 5. Cheapest measurements before implementing

| id | confirms / kills | what | where, cost |
|---|---|---|---|
| done | A (rank, error at models, stress test, C1b seeds) | sections 3.1-3.5 | laptop |
| done | S7 and low-rank updates | section 3.6 | laptop |
| done | D feasibility (support size) | section 3.7 | laptop |
| L1 | D1 margin | support overlap of `w_pass[0]` and `w_pass[1]` in the E1 output npz files (`e1_solve.py` saves `w_pass`); also mass of pass-2-only orbits | node, file read only, seconds |
| M1 | E, F | `dsyrk` and `ssyrk` of one 8192 x 45 000 block into a 45k Gram at 4 / 8 / 16 threads, pinned and unpinned; report GFLOP/s per thread | node, ~20 GiB, ~15 min |
| M3 | G; also sizes D | in one aperture-group read (128 HST apertures): per aperture, fraction of orbits with any non-zero cell, whether the support is contiguous in orbit index, non-zero cell density | node, one group read, ~10 GiB, minutes |
| M4 | A end to end | after implementing A: one rhs-only read at prod1 `w_new` with the setting on; check that reported chi2 equals prod1 `sets_new` to 1e-6 relative and that `E(w_new)` matches 0.0042 (HST, kappa = 10) and 0.0025 (Gaia) | node, one read (~20-30 min) |
| M5 | E on real rows | float32 against float64 Gram on one aperture group restricted to 4 000 orbits spanning the column-norm range: max componentwise relative error of G | node, minutes |

M6 (node, with M3's read): per orbit, the fraction of its PM-set `diag(G)` that the
dropped ridge carries, `delta_a (|H_a,j|^2 - |B_a,j|^2) / diag(G)_j`. Confirms that the
diagonal carry-over in option A is a small correction.

## 6. Recommended order

Wall-time estimates scale the prod2 sample budget of section 2, at the same contention
[estimated]. Row factor for A: 4.1x.

| phase | change | label | samples after | wall | cumulative |
|---|---|---|---|---|---|
| 0 | L1, M1, M3, M6 on the node | | 413 | 13.8 h | 1x |
| 1 | B: fused capture, cell-space chi2 | roundoff | ~374 | 12.5 h | 1.1x |
| 2 | A: row-space rows, kappa = 10 | bounded | ~140 | 4.7 h | 2.9x |
| 3 | C: projected cache, pass 2 without a read | bounded (as A) | ~120 | 4.0 h | 3.4x |
| 4a | D1: working-set pass 2 | certified | ~90 | 3.0 h | 4.6x |
| 4b | D2: working-set pass 1 | certified | ~65 | 2.2 h | 6.4x |
| 5 | E: float32 products (only if `dsyrk` still leads) | roundoff 1e-6 + certificate | ~55 | 1.8 h | 7.5x |

Arithmetic for phase 2: `dsyrk` 295 -> 72 samples; Laplace block in the two builds
26 -> ~12 (the float64 copy of `H` stays, the product shrinks 4x). Phase 4 assumes
m/n = 0.5 (`dsyrk` x 0.25 on the affected pass). After phase 4 the remainder is reads
(~14 samples), ADMM (~8), setup and post (~16), Laplace block (~10): further `dsyrk` work
buys little, and the next target would be the bzip2 read path and the 24 min MGE setup.
F (threads) applies at any point and changes latency only.

### Acceptance tests and tolerances

Common to all phases: `dev_tests/test_gram_blockwise.py` and the five-read-path parity
tests pass unchanged with every new setting at its default.

| phase | test | tolerance |
|---|---|---|
| 1 | new path against current two-read path on the fork's small test library: `_lap_h0` | identical |
| 1 | same: G, v, per-tag chi2 | rel 1e-12 |
| 2 | identity test on the small library: `G_full - G_rowspace = sum_a delta_a H_a (I - V V') H_a'` (kappa off) | rel 1e-10 of `|G|` entries |
| 2 | with `F = diag`, `g = 0` and a full-rank basis the rows reproduce the diagonal baseline | rel 1e-12 |
| 2 | M4 at prod1 weights: chi2 per set equals `sets_new`; reported `E` | 1e-6 relative; `E` < 0.05 per set |
| 2 | `col` with the dropped diagonal carried against `col` of the full Laplace build (small library) | rel 1e-12 |
| 2 | node re-solve of orblib_001_001 (alone, stated threads, `DYNAMITE_GRAM_PROFILE` on): per-set chi2 against prod1 (10 994.48 / 1 482.64 / 557.23 / 1 654.61) | within 0.5 per set; `E(w~)` < 0.05 |
| 3 | pass-2 G and v from cache against pass-2 from a read (same kappa) | rel 1e-6 (float32 cache) ; chi2 within 0.01 |
| 4 | KKT on excluded orbits at the returned weights; chi2 against the phase-3 solve | `min gamma_j >= -1e-8` in normalised units; chi2 within 0.01 per set |
| 5 | KKT residual with the exact float64 gradient; chi2 against phase 4 | gap bound < 0.1; chi2 within 0.1 per set |

The 0.5 tolerance on the re-solve is looser than the bound because prod1 itself is an ADMM
solution at tolerance 1e-10 on a flat valley; the IMBH profile differences are hundreds
to thousands.

## 7. Caveats

- The pass-2 spectra and the F2/F1 comparison use the final prod1 model as a stand-in for
  the pass-1 model.
- Rank and error numbers are for orblib_001_001's models and this data pack; the rank
  depends only on the star rows (model-independent), the error values on the model.
- The MUSE Laplace set and the LVM / mass rows were not analysed (0.6% of rows).
- Stage sample counts for the Laplace block inside the capture and final reads are read
  off the record's table and are approximate (+-3 samples).
- The C1b check uses a three-parameter Gaussian family and the unsharpened synthetic rows;
  the stress test uses free cells. Neither is an orbit model; the per-solve `E(w~)` is the
  check that applies to the real thing.
- No timing was done on the laptop; all speedups are flop and row-count arithmetic on the
  measured prod2 profile.
