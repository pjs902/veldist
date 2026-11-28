# IMBH kinematic statistic: independent review (2026-11-28)

Context: TASKS.md entry "2026-11-27 LIBRARY NOISE + STATISTIC SYSTEMATIC". Paused 2026-11-28 for a couple of days.
Review by an independent (read-only) agent of the solver and the numbers; not yet verified per star.

## Diagnosis

- Weights are optimised on one objective and scored on another; no code bug found.
- `fitstat_pm.laplace_terms` gives the exact Hessian/gradient of -lnL at h0, so `pm_laplace_reexpand` is a full,
  undamped Newton step under w >= 0.
- Why it diverges: the per-star quadratic of -2 ln p about p0 costs only +3 at p = 0 (true cost infinite, ~15-20
  with the 1e-3 floor). The QP zeroes a star whenever that buys >3 elsewhere; recovery is slow (p -> 2p - a p^2).
  Pass 2 zeroed ~250 more HST stars (167 -> 417); at 15-20 each that is ~4000-5000, matching the measured
  +4800 / +5400. Unchecked per star.
- The floor only keeps 1/p0 finite at the expansion point; the QP sees unfloored h. `laplace_reexpand_mix`
  renormalises h (expansion point != iterate, minor). `default_delta` (~1e-3 of F) gives no damping.
- The Laplace-about-data statistic is in the wrong regime: HST ~435 stars per aperture over 529 cells, Gaia ~433
  over 289, i.e. about one star per cell, so the cubic term is not small. An ~800 offset over 610k stars is ordinary
  truncation error; it cannot be patched. Stage-3 ranking, the GP profile and the gold verdict all inherit it.

## Challenges to the current reading

- "Exact sign holds, ~3 sigma" is not supportable: -554 / -561 / -292 across libraries vs coherent library shifts
  ~300 gives 1.5-2 sigma at best, at unconverged weights.
- Floor sensitivity means a few very-low-p stars carry the signal; f acts as an undeclared background fraction. Fix
  it per aperture from the field density used in the fast-star term; stop scanning. Check that the top |d ln p|
  stars do not overlap the fast-star selection (double counting).
- The sphere of influence of 3e4 Msun is ~12"; headline Delta should be R < 20-30", with the outer Delta as a null
  test (Laplace R > 50" was +535).
- The 3.2e4 re-tunes start from nuisance values tuned at 1e4, so they can only move Delta toward 3.2e4; score them
  on the new statistic.

## Ranked plan

1. Damped re-expansion: existing QP as direction, backtrack w <- w + t (w_QP - w) on the floored exact objective
   (1-D convex, p and sum(h) linear in w), stop when the decrease is < ~1; report the floored exact statistic.
   ~30 lines in `_solve_laplace`; a few ~1 h passes per model.
2. Paired replicas: polish the six existing solves (3 libraries x 2 masses); quote the mean paired Delta inside 30";
   error from replica scatter. Add 2 pairs if marginal.
3. Re-score the three 3.2e4 re-tunes on the same statistic.
4. Likely fallback framing: kinematics do not separate 1e4 from 3.2e4 beyond ~2 sigma; fast stars set the lower
   bound; quote an interval.

Not recommended: EM / mix-SQP (cannot carry mass rows, LVM, regulariser; hundreds of orblib passes); binned Poisson
on histograms; Laplace-about-model as the reported statistic (same +3 pathology).
Cut: GP profile (LOO rms 421 vs ~300 signal), gold/stretch verdict on Laplace, working-set pass, kappa calibration,
HMC / marginalisation, further floor scans, undamped reexpand >= 2.

## Next experiment (offline, minutes, no node)

On the seed4243 pass-1 and pass-2 `hst_split` files (`experiments/fitstat/work/split/`, both masses): p(t) =
(1-t) p1 + t p2 per star (p, not log p; same for capt), floored exact -2lnL at f 1e-3 for t in {0, .05, .1, .2,
.35, .5, .75, 1}; LVM / mass rows as a quadratic in t if available. Print t*, gain at t*, Delta(3.2e4-1e4) at each
model's own t*, and whether the pass-2 damage sits on the newly zeroed stars.
- Interior t* with gains of tens to hundreds for both: implement step 1.
- t* ~ 0, no gain: pass-1 exact Delta is nearly final along that direction; do steps 2-3.
