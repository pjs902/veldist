#!/bin/bash
# Relative-PM pilot: turn a c3_private_copy.sh private config into the relative arm.
#   rel_config.sh PRIV_DIR REL_PACK_DIR rows|norows MAX_ROUNDS
# hst_pm -> relative: True, statistic: laplace, start_mean (Gaia-tied), + rotation_rows (arm B) or none (arm C);
# input_directory -> REL_PACK_DIR; weight_solver_settings += pm_recentre_mode vp, round cap. Other sets unchanged (as E1).
set -euo pipefail
PRIV=$1; PACK=$2; ARM=$3; NR=$4
C=$PRIV/config.yaml
sed -i "s|input_directory: \".*\"|input_directory: \"$PACK/\"|" "$C"
ROWS=""
[ "$ARM" = rows ] && ROWS='                rotation_rows:\n                    datafile: "hst_pm_rot_rows.ecsv"\n                    weightfile: "hst_pm_rot_weights.ecsv"\n                    normalisation: pass0\n'
sed -i "s|                binfile: \"hst_veldist_bins.dat\"|                binfile: \"hst_veldist_bins.dat\"\n                statistic: laplace\n                relative: True\n                start_mean: \"hst_start_mean_gaia_tied.ecsv\"\n$ROWS|" "$C"
sed -i "s|    ridge_lambda: 0.03|    ridge_lambda: 0.03\n    pm_recentre_mode: \"vp\"\n    pm_recentre_max_rounds: $NR|" "$C"
grep -q "relative: True" "$C" && grep -q "pm_recentre_mode: \"vp\"" "$C" && grep -q "input_directory: \"$PACK/\"" "$C"
[ "$ARM" = rows ] && grep -q rotation_rows "$C" || [ "$ARM" = norows ]
echo "config ok: $C ($ARM, max_rounds $NR)"
