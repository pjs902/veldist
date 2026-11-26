#!/bin/bash
# Orbit-library noise replica of one gprun model (docs/imbh-status-2026-11-26.md, step 1), private under $W/noise/.
#   noise_setup.sh W ORBLIB REP KEY=VALUE [KEY=VALUE ...]
# e.g. noise_setup.sh $W orblib_007_005 seed4243 random_seed=4243
#      noise_setup.sh $W orblib_007_005 logr+ logrmin=-0.4517 logrmax=3.7483
# Makes $W/noise/ORBLIB_REP/: config.yaml = gprun/config.yaml with output_directory -> NGC5139_output/ and the given
# orblib_settings replaced (each must change exactly one line); NGC5139_output/all_models.ecsv = gprun's. The model
# directory is left empty, so noise_build.py integrates a fresh library under the new settings.
# Then: noise_build.py config.yaml ORBLIB/MLDIR, and regrid_one.py config.yaml ORBLIB/MLDIR result.npz.
set -euo pipefail
W=$1; ORB=$2; REP=$3; shift 3
D=$W/noise/${ORB}_$REP; PO=$D/NGC5139_output
[ -e "$D/config.yaml" ] && { echo "exists: $D/config.yaml"; exit 1; }
mkdir -p "$PO/models"
sed -e "s|output_directory: \".*\"|output_directory: \"$PO/\"|" "$W/gprun/config.yaml" > "$D/config.yaml"
n=1
for kv in "$@"; do
    k=${kv%%=*}; v=${kv#*=}
    grep -qE "^    $k: " "$D/config.yaml" || { echo "no orblib setting $k"; exit 1; }
    sed -i -E "s|^(    $k: )[^#]*|\1$v |" "$D/config.yaml"
    n=$((n + 1))
done
[ "$(diff "$W/gprun/config.yaml" "$D/config.yaml" | grep -c '^>')" = "$n" ]  # output_directory + each setting
cp "$W/gprun/NGC5139_output/all_models.ecsv" "$PO/all_models.ecsv"
grep -q "$ORB/" "$PO/all_models.ecsv"
diff "$W/gprun/config.yaml" "$D/config.yaml" | grep "^>" || true  # diff exits 1 on a difference
echo "ready: $D"
