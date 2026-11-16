#!/bin/bash
# C3: private, read-safe copy of finished v4 models for rescoring (the live grid is running).
#   c3_private_copy.sh LIVE_GRID_DIR PRIV_DIR MODEL_DIR [MODEL_DIR ...]
# MODEL_DIR is the all_models 'directory' value, e.g. orblib_003_007/ml03.11/.
# Large read-only inputs (*.bz2, *.npz caches, begin*.dat, *orbclass.out) are SYMLINKED:
# dynamite never writes into them in place (its cache writes go to a temp name
# that is then renamed, which replaces the symlink, not its target). Everything else is copied.
# Process-unique temp decompressions (*_p<pid>.dat) are skipped.
set -euo pipefail
LIVE=$1; PRIV=$2; shift 2
LO=$LIVE/NGC5139_output; PO=$PRIV/NGC5139_output
mkdir -p "$PO/models"
cp "$LO/all_models.ecsv" "$PO/all_models.ecsv"
sed "s|output_directory: \"NGC5139_output/\"|output_directory: \"$PO/\"|; s|input_directory: \"input/|input_directory: \"$LIVE/input/|" \
    "$LIVE/config.yaml" > "$PRIV/config.yaml"
grep -q "output_directory: \"$PO/\"" "$PRIV/config.yaml"
grep -q "input_directory: \"$LIVE/input/" "$PRIV/config.yaml"
for md in "$@"; do
    orb=${md%%/*}; ml=$(basename "$md")
    src=$LO/models/$orb; dst=$PO/models/$orb
    [ -e "$dst/datfil/tube_box_done" ] && { echo "exists: $dst"; continue; }
    mkdir -p "$dst/datfil" "$dst/$ml"
    cp -r "$src/infil" "$dst/"
    cp "$src/interpolgrid" "$src/fort.30" "$dst/"
    cp "$src/$ml/"* "$dst/$ml/"
    for f in "$src"/datfil/*; do
        b=$(basename "$f")
        case "$b" in
            *_p[0-9]*.dat) ;;  # another process's temp decompress
            *.bz2|*.npz|begin*.dat|*orbclass.out) ln -s "$f" "$dst/datfil/$b" ;;
            *) cp "$f" "$dst/datfil/$b" ;;
        esac
    done
    echo "copied $md"
done
