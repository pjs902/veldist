#!/bin/bash
# Gold-standard solve (docs/ml-reuse-investigation.md step 5), private and separate from the live stage-3 gprun/.
#   gold_setup.sh W
# orblib_007_012 = orblib_001_001's potential with every mass x 0.93086 (Υ 2.760), integrated by stage 3. Solved on
# the production data with the production statistic (regrid/config.yaml), to compare with the ml2.76 stretched solve.
# Makes in $W/stretch/gold/:
#   config.yaml     = regrid/config.yaml with output_directory -> NGC5139_output/ (input stays production);
#   NGC5139_output/ = gprun's all_models.ecsv and a private copy of orblib_007_012 (large orbit files symlinked to
#                     gprun, the rest copied), so regrid_one's orbit_weights.ecsv never lands in the live gprun.
# Then: regrid_one.py $W/stretch/gold/config.yaml orblib_007_012/ml02.76 $W/stretch/gold/result.npz
set -euo pipefail
W=$1
D=$W/stretch/gold; PO=$D/NGC5139_output; MD=orblib_007_012/ml02.76
SRC_OUT=$W/gprun/NGC5139_output
[ -e "$D/config.yaml" ] && { echo "exists: $D/config.yaml"; exit 1; }
orb=${MD%%/*}; ml=$(basename "$MD"); src=$SRC_OUT/models/$orb; dst=$PO/models/$orb
[ -e "$src/datfil/tube_box_done" ] || { echo "library not finished: $src"; exit 1; }
mkdir -p "$D"
sed -e "s|output_directory: \".*\"|output_directory: \"$PO/\"|" "$W/regrid/config.yaml" > "$D/config.yaml"
grep -q "output_directory: \"$PO/\"" "$D/config.yaml"
[ "$(diff "$W/regrid/config.yaml" "$D/config.yaml" | grep -c '^>')" = 1 ]  # only that line differs
mkdir -p "$PO/models" "$dst/datfil" "$dst/$ml"
cp "$SRC_OUT/all_models.ecsv" "$PO/all_models.ecsv"
grep -q "$MD/" "$PO/all_models.ecsv"
cp -r "$src/infil" "$dst/"
cp "$src/interpolgrid" "$src/fort.30" "$dst/"
cp -r "$src/$ml/." "$dst/$ml/"
rm -f "$dst/$ml/orbit_weights.ecsv"
for f in "$src"/datfil/*; do
    b=$(basename "$f")
    case "$b" in
        *_p[0-9]*.dat|*.lock) ;;
        *.bz2|*.npz|begin*.dat|*orbclass.out) ln -s "$f" "$dst/datfil/$b" ;;
        *) cp "$f" "$dst/datfil/$b" ;;
    esac
done
echo "ready: $D"
