# M_IMBH status after stage 3 (2026-11-26)

## Results

**1. Kinematics alone barely constrain M_IMBH between ~6k and ~35k.**

- Stage 3's best model is orblib_007_005: M 1.0e4, kinchi2 13696.9, 346 below 006_002 (M 2.46e4).
- Most of that gain comes from the other parameters, not from M. At the same nuisance values (mscale 0.93,
  rscale 0.84, q 0.637, ml 2.93), M = 3.2e4 still scores 13975.8, below 006_002.
- GP refit on all 91 models (`r6_regrid_profile.py`; output `$W/gprun/r6_refit.pdf`):
  - LOO rms 421.
  - The profile minimum is at 8.5k.
  - Δ at the grid points:

    | M_IMBH | 3.2k | 5.7k | 10.1k | 17.9k | 31.9k | 56.6k |
    |---|---|---|---|---|---|---|
    | Δ | +596 | +121 | +16 | +223 | +40 | +1027 |

  - The profile is non-monotonic, and from 6k to 32k every Δ is below the surrogate error.
  - The enclosed mass M(<20") has its minimum at 17k.
- Enclosed mass (`enclosed_mass.py`): the sBH cluster adds only ~0.2k inside 2". The low-M family really does have
  less central point mass. It is not an IMBH ↔ sBH trade.
- Whether Δχ² of a few hundred is significant depends on the orbit-library noise and the calibration κ. **Neither has
  been measured.** The GP white-noise sd (~98) is only a lower bound.

**2. Fast stars, escape speed at r = R (z = 0, the most favourable case; `escape_check.py`).**

- With Haeberle 2D speeds only:
  - M = 1.0e4: strict χ² 7.1, from star C.
  - M ≥ 1.25e4: all bound.
  - M ≤ 5.5e3: excluded (χ² 87–193).
- With the preliminary 3D speeds in `PM_grid/.../escape_restart/fast_stars.ecsv`:
  - Strict χ² falls steadily with M: 94 at 1e4, 52 at 1.5e4, 25 at 2e4, ~10 at 2.4e4, 0.8 at 3.2e4.
  - This is driven by one star at 0.87" whose preliminary 3D speed needs a deeper potential than M ≲ 3e4 gives.
  - Those RVs are preliminary. Use Goold's final values before taking this at face value; their lower limit is ~16k.
- So the kinematic preference for ~1e4 is **not** in tension with the fast stars. It sits inside the kinematic noise
  band, and the fast stars set the lower bound.

**3. First real fast-star table (006_002).**

- Term (a) works on real files.
- The 2-cell-smoothed table inflates the tail across the 63 km/s cut: 16.5 expected selected members, against 2.5 raw
  and ~5.5 observed.
- So term (a) uses the raw v2D marginal (summed over |v_los|, which is already low-noise). Smoothing, if needed at all,
  goes along |v_los| only.

**4. Stretch (Υ reuse).**

- The gold library differs from the stretched solve by 715 (HST 499, LVM 261).
- That counts as a failure **only if** the library noise is ≲ 100. The verdict waits on the noise measurement.

## Path forward (in order)

1. **Library noise.** Re-integrating with the same seed is useless: the integration is deterministic and the gate
   reproduced the library bit for bit. Instead, perturb the start grid:
   - shift logrmin/logrmax by ±1/3 of an energy step, and make one replica with dithering 3 → 4;
   - do this for 007_005 (1e4) and 007_008 (3.2e4), 2–3 replicas each, so the noise is measured on a Δ.
   - Cost: ~24 h of integration (4 procs per library) plus a ~1 h solve each.
   - The scatter sets κ, the profile's significance and the gold verdict.
2. **Fast-star term along the profile.** Build raw tables for M ∈ {1e4, 1.5e4, 2.5e4, 3.2e4} at the 007 nuisance
   values (~18 h each, chunked). Then:
   - term (a) from the raw v2D marginal;
   - term (b) with Goold's final RVs (request the 102-star table and the final A–L values).
3. **Combine.** Kinematic profile / κ + fast-star χ² → M_IMBH. The strict escape χ² is the quick interim version.
4. **GP.** Optional stage 4 around the new nuisance region (mscale ~0.93, rscale ~0.84) to bring the LOO rms (421)
   down near the minimum.
5. **Paper.** Hold until steps 1–3 are done.
