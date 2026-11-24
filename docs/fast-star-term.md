# Fast-star likelihood term for the IMBH mass (plan)

Status: research and plan, 2026-11-24. Nothing implemented yet.

**Goal.** Add a separate likelihood term from the central fast stars to the M_IMBH profile only:

    -2 ln L_tot(M) = Delta kinchi2_profile(M) / kappa  +  chi2_fast(M)

It does not enter the GP search or the orbit-weight solve. Here kappa is the over-confidence factor of the kinematic statistic (see Calibration), and chi2_fast is computed after the solve, per finished model, from that model's own distribution function (DF).

## 1. What exists now (node `dynamite_analysis` a5bb4b5; the laptop copy is older)

- **`escape_feasibility.EscapeFeasibility`** is a generator constraint, not part of kinchi2 (the `vesc` Chi2Ext has weight 0):
  P(bound) = prod_i Phi((v_esc(r_i) - v_i)/e_i).
  - v_i is the 3D speed where a v_los exists (`fast_star_speeds`, linearised error), otherwise v2D.
  - r_i is the **projected** radius, i.e. z = 0.
  - The Plummer term sits at the mass-minimising offset (-0.38", 0.41") (`imbh_offset`); the rest of the potential is centred.
  - Thresholds: the incumbent needs P >= Phi(-1); proposals with P < Phi(-2) are never integrated.
- **`escape.fast_star_chi2`** also has `strict` (Haeberle's one-sided Gaussian) and `marginal` (z ~ N(0, sigma_z), with sigma_z from the stars' own radii). The grid uses neither.
- **Assumptions to replace:**
  - z = 0, and v_los = 0 for stars without an RV;
  - bound/unbound as the only information, which ignores how improbable a near-escape star is under the DF;
  - no selection term;
  - the IMBH placed where it most favours the result;
  - speeds at 5.43 kpc while the models use 5.494 kpc (`load_fast_stars` has a shortcut comment: v2D is 1.2% low).

**Why the all-star density is the wrong z weighting.** The right one is p(z | R, v_obs) ∝ ∫ f(R, z, v) dv_los. It is zero beyond the radius where v_esc = v and concentrated on the IMBH. The all-star density is flat over the ~90" core, so it would put the star at r >> R and make every model fail. A DF supplies this conditional automatically. Under a DF every star is bound, so the information becomes the phase-space density at (R, v), not "bound or not".

## 2. The likelihood: conditional on observed positions

Model only the velocities of an observed, velocity-blind sample. That removes crowding completeness and the absolute central density normalisation entirely.

- **Q:** Haeberle's high-quality subset within R_max (10"; the 3-10" ring has 2090 stars and no member fast star, which is information too).
- **Selection S:** mu_obs - 3 e_mu > 2.41 mas/yr. Selecting in mas/yr makes it distance-free; model velocities convert to PM at the model's distance.

```
chi2_fast = 2 [ sum_{j in Q} P_j(S) - sum_{i in fast} ln p_i ]          (Poisson limit of the Bernoulli terms)
P_j(S)   = ∫ p_mod(v_2D | R_j) Phi((mu(v) - 2.41 - 3 e_j)/e_j) dv       (each star's own e_j)
p_i      = (1-eps_i) ∫∫ p_mod(v_2D, v_los | R_i) N(obs_i; v, err_i) dv  +  eps_i p_bg(obs_i)
```

- **Stars with an RV** (A-E NIRSpec preliminary; F has MUSE 232.5 ± 4.0, i.e. -0.5 relative, which is **missing from `fast_stars.ecsv`**): use the joint (v2D, v_los) density.
- **Stars without an RV:** integrate v_los over the model's own conditional. Missing RVs are ignorable: availability depends on the NIRSpec/MUSE footprint and brightness, not on v_los.
- **Use the joint (v2D, v_los) density, not v3D.** The selection is on v2D, so the likelihood needs the joint density; it also removes the linearised 3D error. Star A's v_los error is ±24 km/s.
- **Background:** Haeberle's Milky Way density above the cut among quality stars is 0.0026 arcsec^-2, so about 0.8 expected inside 10". p_bg is the Besançon v2D distribution (`fast-stars/bella_fitting` has Besançon cutoffs). Contaminants are centred at v_los ≈ -233 relative, so a member RV drives eps_i to ~0. Test case: the 75.8 km/s star at 9.5" (Haeberle: consistent with foreground).
- **Double counting:** none. Fast stars are excluded from the main kinematic sets (user, 2026-11-24). s1's exact check found no central HST star with p <= 0 (all 121 are in apertures at 40-333"). The main per-star term is normalised over its velocity grid (±82.8 km/s per component for HST), so the in-grid/out-of-grid fraction it does not use is what this term adds.
- **Assumption to test:** the quality cuts are independent of velocity. Fast stars are faint and partly blended (B, G), so compare the cut pass rates against v2D in the 3-10" ring.

## 3. p_mod from the Schwarzschild model (post-hoc, no new solve)

- **New output set.** Fine annuli centred on the model centre: 0-10" in ~15 bins, 0.25" inside 3". Each annulus gets a 2D histogram in (v2D, |v_los|): v2D 0-200 km/s, |v_los| 0-200 km/s, ~4 km/s cells. No PSF. The 2D PM direction is dropped (a later option).
- **Fortran change** (`legacy_fortran/orblib_f_new_mirror.f90`): a histogram kind whose 2D slots are (sqrt(pmx^2+pmy^2), |losvel|). `project_n` already computes `vel2d` and `losvel` together (l. 1096-1118). Apertures are a fine boxed pixel grid (0.05") mapped to annular bins via bins.dat.
- **Post-hoc re-integration is possible.**
  - Each orbit's seed derives from the base seed and its index (`integrator_set_seed`, l. 153-219), the start conditions are saved (`datfil/begin.dat`, `beginbox.dat`), and chunked runs by orbit range already exist (`orblib.py` l. 715-746).
  - Re-integrating only the needed orbits with the extra set reproduces the original orbits, so the existing weights map one-to-one, with no new library, kinchi2 or solve.
  - Only orbits with non-zero weight are needed: s1 has 16,577 of 45,000 (99.9% of the mass in 10,209), restricted further to those reaching R < 10" (all boxes, inner tubes). That is ~1/4-1/3 of a library's integration per model.
  - **Gate:** for ~50 orbits, the re-integrated existing HST histograms must match the stored ones bit-for-bit (same binary; tube and box separately).
- **Then:** p_mod(v | annulus) = sum_k w_k h_k / sum_k w_k m_k, interpolated in R. The weights come from the model's `result.npz` / weight file, with the same column order as the library.
- **Energy floor.** The deepest shell is at r_circ = 10^-0.5 = 0.32". With a Kepler plus constant-cluster estimate (v_esc,cl 62-70), only **star A** is affected: 14-22% of its bound line-of-sight range is below the floor across M = 2.5e4-9.5e4. All other stars are unaffected. Keep `logrmin`, since lowering it means new libraries and weights. Report A's sensitivity with one deeper-shell library for the best model only (logrmin -1, nE 33), or check it with the analytic cusp (§5).
- **Monte Carlo noise of the tail.** Measure it by splitting the 27 dithers into halves and comparing p_i. It enters as an error on chi2_fast.

## 4. IMBH position: marginalised

- **Kinematics don't see it.** The innermost HST aperture is at 3.5" with 519 stars on 5" pixels. Sub-arcsec offsets change nothing there, so the BH stays at the model centre for kinchi2.
- **The central DF moves with the BH.** The cluster potential is flat across the ~90" core (Haeberle: v_esc flat to ~50"), so a shift Delta costs O((Delta/r_core)^2). Evaluate p_mod at R_i = |x_i - x_BH| for each candidate x_BH.
- **Marginalise:** chi2_fast(M) = -2 ln ∫ L(x_BH) pi(x_BH) d^2x on a grid (±2", 0.05"). Each grid point is just an interpolation in the annular tables.
- **Prior:** centred on the kinematic centre (AvdM10) with the centre's uncertainty. Haeberle's MCMC position, (-0.017, +0.011) pc ≈ (-0.64", +0.41"), is a cross-check, not a prior: it uses these same stars.
- **This replaces `imbh_offset`.** Example of why it matters: with its preliminary RV (-85 km/s, v3D ≈ 128), star C is unbound at M = 2.5e4 if the BH is at the centre.

## 5. Analytic cross-check: cusp DF, potential-only

- **Model:** isotropic f(E) ∝ |E|^(gamma-3/2) inside r_h = GM/sigma^2 (≈ 8" for M = 2e4, sigma = 20), in the model's own potential (Kepler plus cluster plus sBH). gamma in [0.5, 1.75] is a nuisance (Bahcall-Wolf 7/4).
- **Normalisation:** matched to the Schwarzschild model's density near r_h.
- **Why a cusp is needed:** the light MGE is cored. A core cannot host an isotropic DF around a BH (needs gamma >= 1/2; Tremaine et al. 1994), so an Eddington inversion of the MGE gives f < 0.
- **Use:**
  - a check of the orbit-library tail (the energy floor, Monte Carlo noise);
  - a cheap term for every profile model before any re-integration exists.

## 6. Calibration

The combination needs kappa: how far Delta(-2 ln L) of the kinematic statistic overstates the information on M_IMBH.

- **Known:** the per-star Laplace statistic is -2 ln L at fixed weights. It is nearly calibrated per aperture in C1 (`fit-statistic-design.md`).
- **Not included:**
  - the weights are profiled (optimised), not marginalised;
  - the noise between orbit libraries;
  - the GP noise of the profile;
  - model misspecification. Iteration-to-iteration differences of hundreds mean misspecification is likely to dominate.

Measure, in order:

1. **Library noise:** the best parset (orblib_006_002) re-integrated with 2 other `random_seed`s, then solved. The scatter of kinchi2 per set is sigma_lib. Cost: 2 libraries plus 2 solves, about 1 day on the node.
2. **Weight-profiling inflation:** the frozen-curvature bootstrap (paper §Uncertainties, still a `\ps`).
   - Resample stars per aperture (Poisson weights) and rebuild only the linear term and targets.
   - Re-run ADMM on the factored Gram matrix.
   - Delta kinchi2 between neighbouring profile models across replicates gives their variance.
   - The same replicate weights propagate weight uncertainty into chi2_fast cheaply (a histogram × w).
3. **End-to-end coverage:** the E2 mocks (`e2-mock-plan.md`). Mock stars are drawn from a known model, refit along M_IMBH, and the Delta(-2 ln L) distribution at the truth is compared with chi2_1, which gives kappa.

**Steps 1-2 give a provisional kappa; step 3 confirms it.** chi2_fast needs no kappa of its own: few parameters, an exact likelihood given the DF. Its Monte Carlo and weight noise come from §3 and step 2. Report the kinematics-only, fast-only and combined profiles side by side.

## 7. Data step 0 (before any modelling)

- **Rebuild Haeberle's quality sample** from `fast-stars/bella_fitting/astrometry.fits` plus F606W/F814W photometry, on the node (the laptop runs nothing heavy). Use their cuts, not Bella_fitting's:
  - nused >= 20, baseline >= 20 yr, (nfound - nused)/nfound < 0.15;
  - pm errors < 0.194 mas/yr, chi2x and chi2y < 10;
  - the CMD sequence.

  It must recover 157,320 of 241,133 within 90" and the 7 fast stars, with A, C, D, E, F passing 3 sigma and B, G not.
- **Fast-star table in PM units** (mas/yr), converted with each model's distance; add F's MUSE RV.
- **Flag star E:** MUSE +28.7 ± 2.7 vs NIRSpec +5.8 ± 6.8 relative (3.1 sigma), and the NIRSpec RVs are preliminary (Goold et al.). Run with and without E's RV and the NIRSpec zero point.
- **Positions:** (dx, dy) relative to the model centre (already in `fast_stars.ecsv`).

## 8. Implementation order

1. Data step 0 (quality sample, PM-unit table, background p_bg).
2. Analytic cusp term (§5) on the existing profile models, giving the first chi2_fast(M) curve. Code goes in `dynamite_analysis` (sync the laptop to a5bb4b5 first).
3. Fortran histogram kind plus annular aperture set in `dynamite-fitstat`, with the bit-for-bit gate on existing histograms.
4. Re-integration of non-zero-weight orbits for the best model, then the profile models (`launch.sh`, prechecked); p_mod tables; the dither-half noise.
5. chi2_fast(M) with IMBH-position marginalisation and the background mixture; compare it with the cusp version (energy floor, star A).
6. Calibration steps 1-2 (kappa), then the combined profile. E2 mocks for coverage.

**Tests:**
- the selection reproduces Haeberle's counts;
- on an analytic Kepler-plus-isotropic cusp mock (stars drawn with Haeberle-like errors and selection), chi2_fast recovers the true M;
- re-integration is bit-for-bit;
- p_mod built from a single orbit matches that orbit's sampled phase space.
