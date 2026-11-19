# LOS-only fits: do the line-of-sight data predict the proper motions? (2026-11-19)

Suggested by D. Kranjcevic and D. Hogg: fit only the line-of-sight data, as in the extragalactic case, and see how
well the fitted orbit weights reproduce the proper motions that were not fit.

## Set-up

- DYNAMITE fork `feature/pm-fit-statistic`, setting `fit_exclude_sets: [hst_pm, gaia_pm]` (33ab210): the PM sets
  are not read and add no kinematic or projected-mass rows. Fit = MUSE (per-star Laplace, one re-expansion) + LVM
  (diagonal chi2) + the mass constraints, production ridge `ridge_lambda: 0.03` unless stated.
- Driver `experiments/fitstat/los_only.py`: LOS-only solve, then one read with every set at the LOS-only weights;
  the PM statistic there is the prediction (per-star Laplace about the data, the joint fit's own statistic).
  Profiles: `experiments/fitstat/profile_compare.py` on `model_hists.py` output.
- The six production libraries (the prod re-solves of the v4 grid, each its own potential and IMBH mass).
  Node results `$W/losonly/`, local copies `experiments/fitstat/work/losonly/`.
- One LOS-only solve takes 30-45 min at 4 threads (the joint fit: 14 h).

## Results

Statistic per set (lower is better); "joint" = the production fit of the same library to all four sets.

| M_IMBH | library | ridge | MUSE+LVM, LOS-only (MUSE / LVM) | MUSE+LVM, joint | HST predicted / joint | Gaia predicted / joint |
|---|---|---|---|---|---|---|
| 5 513 | 003_006 | 0.03 | 2371 (1477 / 894) | 2130 | 137 439 / 15 742 | 5777 / 1818 |
| 10 676 | 002_000 | 0.03 | 1887 (662 / 1225) | 2143 | 45 338 / 11 664 | 3787 / 1607 |
| 14 159 | 000_002 | 0.03 | 1942 (830 / 1112) | 2145 | 69 829 / 12 516 | 5162 / 1763 |
| 20 000 | 001_001 | 0.03 | 1920 (708 / 1212) | 2212 | 56 972 / 10 994 | 4365 / 1483 |
| 20 000 | 001_001 | 0.003 | 1356 (562 / 794) | 2212 | 94 151 / 10 994 | 5377 / 1483 |
| 20 000 | 001_001 | 0.0003* | 1188 (536 / 653) | 2212 | 122 690 / 10 994 | 5652 / 1483 |
| 34 486 | 004_005 | 0.03 | 1829 (806 / 1023) | 2116 | 80 843 / 13 022 | 6247 / 1719 |
| 56 368 | 002_006 | 0.03 | 1923 (857 / 1067) | 2207 | 67 509 / 16 260 | 4909 / 2085 |

Ridge 0.003 at every library:

| M_IMBH | library | ridge | MUSE+LVM, LOS-only (MUSE / LVM) | MUSE+LVM, joint | HST predicted / joint | Gaia predicted / joint |
|---|---|---|---|---|---|---|
| 5 513 | 003_006 | 0.003 | 1832 (1232 / 600) | 2130 | 175 481 / 15 742 | 7682 / 1818 |
| 10 676 | 002_000 | 0.003 | 1295 (500 / 795) | 2143 | 88 448 / 11 664 | 5006 / 1607 |
| 14 159 | 000_002 | 0.003 | 1359 (646 / 713) | 2145 | 111 933 / 12 516 | 6842 / 1763 |
| 20 000 | 001_001 | 0.003 | 1356 (562 / 794) | 2212 | 94 151 / 10 994 | 5377 / 1483 |
| 34 486 | 004_005 | 0.003 | 1255 (602 / 653) | 2116 | 119 533 / 13 022 | 8786 / 1719 |
| 56 368 | 002_006 | 0.003 | 1271 (597 / 674) | 2207 | 121 559 / 16 260 | 6932 / 2085 |

\* ADMM stopped at 4000 iterations without reaching its tolerance (126 min); the trend is the point.

Profiles for 001_001 at ridge 0.03 (`experiments/fitstat/figs/los_only_001_001.png`):

- HST: sigma_T/sigma_R stays ~1.0 out to 260" (data 0.87, joint 0.85); rotation slightly low.
- Gaia: the sigma_T/sigma_R gradient is reversed (1.12 -> 0.85 from 486" to 1365"; data 0.88 -> 1.05); rotation
  too low outside 600" (0.4 vs 1.8 km/s at 1365").

## Reading

1. The LOS data do not predict the proper motions. At every potential the predicted HST statistic is 3-9x the
   joint fit's and the Gaia statistic 2-4x; the predicted anisotropy profile has the wrong shape in the Gaia
   field.
2. The prediction is set by the regulariser, not the data. Lowering the ridge tenfold fits the LOS data much
   better (MUSE 562, as good as the joint fit's 557; LVM 794) and predicts the PMs worse (HST 94k, Gaia 5.4k);
   another tenfold (not converged) continues the trend (MUSE 536, LVM 653; HST 123k, Gaia 5.7k): the orbit-weight
   directions the PMs constrain are left to the ridge.
3. The LOS-only ranking of the potentials is nearly flat from 10.7k to 56.4k and rejects only 5.5k, at both
   ridges: 1829-1942 at 0.03 (best 34.5k), 1255-1359 at 0.003 (best 34.5k, then 56.4k, 10.7k; 20.0k fourth). The
   joint fit prefers 20.0k by 700+ over its neighbours. The LOS data alone weakly favour the heavier IMBHs and do
   not reproduce the joint fit's choice; the order of the middle libraries changes with the ridge.
4. At ridge 0.003 every library predicts the PMs worse than at 0.03 (HST 88k-176k, Gaia 5.0k-8.8k), as for 001_001.

## Open

- The same exclude-and-predict test for HST alone (fit MUSE+LVM+Gaia, predict HST) and for Gaia alone.
