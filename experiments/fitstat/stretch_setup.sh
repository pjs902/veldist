#!/bin/bash
# One stretched-data solve of orblib_001_001 (docs/ml-reuse-investigation.md), private and separate from regrid/.
#   stretch_setup.sh W NAME
# Needs $W/stretch/NAME/replace/ (rsynced from the laptop: ml_stretch_pack.py output, the four replaced files and
# stretch_meta.npz). Makes in $W/stretch/NAME/:
#   input/          = prod/input/NGC5139_perstar_xeast with exactly the four files of replace/ swapped in
#                     (md5 of every other file equal to production, of the four equal to replace/);
#   config.yaml     = regrid/config.yaml (production statistic, regrid speed settings) with input_directory ->
#                     input/ and output_directory -> NGC5139_output/;
#   NGC5139_output/ = regrid's all_models.ecsv and a private copy of orblib_001_001 as regrid_setup.sh (large
#                     read-only orbit files symlinked to the live v4 grid, the rest copied).
# Then: regrid_one.py $W/stretch/NAME/config.yaml orblib_001_001/ml02.96 $W/stretch/NAME/result.npz
set -euo pipefail
W=$1; NAME=$2
LIVE=/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/sBH_GP_grid_v4
PROD=$W/prod/input/NGC5139_perstar_xeast
D=$W/stretch/$NAME; PO=$D/NGC5139_output; MD=orblib_001_001/ml02.96
[ -d "$D/replace" ] || { echo "no $D/replace"; exit 1; }
[ -e "$D/config.yaml" ] && { echo "exists: $D/config.yaml"; exit 1; }
REPL="hst_veldist_pm_2dhist.npz gaia_veldist_pm_2dhist.npz veldist_kinematics_stars.npz lvm_kins.ecsv"
cp -r "$PROD" "$D/input"
for f in $REPL; do cp "$D/replace/$f" "$D/input/$f"; done
for f in "$PROD"/*; do
    b=$(basename "$f")
    if [[ " $REPL " == *" $b "* ]]; then
        [ "$(md5sum < "$D/input/$b")" = "$(md5sum < "$D/replace/$b")" ] || { echo "replace mismatch $b"; exit 1; }
    else
        [ "$(md5sum < "$D/input/$b")" = "$(md5sum < "$f")" ] || { echo "prod mismatch $b"; exit 1; }
    fi
done
sed -e "s|output_directory: \".*\"|output_directory: \"$PO/\"|" \
    -e "s|input_directory: \".*\"|input_directory: \"$D/input/\"|" "$W/regrid/config.yaml" > "$D/config.yaml"
grep -q "output_directory: \"$PO/\"" "$D/config.yaml"
grep -q "input_directory: \"$D/input/\"" "$D/config.yaml"
[ "$(diff "$W/regrid/config.yaml" "$D/config.yaml" | grep -c '^>')" = 2 ]  # only those two lines differ
mkdir -p "$PO/models"
cp "$W/regrid/NGC5139_output/all_models.ecsv" "$PO/all_models.ecsv"
orb=${MD%%/*}; ml=$(basename "$MD"); src=$LIVE/NGC5139_output/models/$orb; dst=$PO/models/$orb
mkdir -p "$dst/datfil" "$dst/$ml"
cp -r "$src/infil" "$dst/"
cp "$src/interpolgrid" "$src/fort.30" "$dst/"
cp -r "$src/$ml/." "$dst/$ml/"
rm -f "$dst/$ml/orbit_weights.ecsv"  # solved fresh here
for f in "$src"/datfil/*; do
    b=$(basename "$f")
    case "$b" in
        *_p[0-9]*.dat) ;;
        *.bz2|*.npz|begin*.dat|*orbclass.out) ln -s "$f" "$dst/datfil/$b" ;;
        *) cp "$f" "$dst/datfil/$b" ;;
    esac
done
echo "ready: $D (stretch $(python3 -c "import numpy as np; print(float(np.load('$D/replace/stretch_meta.npz')['stretch']))" 2>/dev/null || echo '?'))"
