#!/bin/bash
# Full-statistic re-solve of the v4 grid: private copy of every model with an orbit library (run once).
#   regrid_setup.sh W
# W = $W (veldist-fitstat). Makes $W/regrid/ with
#   config.yaml       = $W/prodone/config.yaml (production statistic) with output_directory -> regrid and the accepted
#                       speed settings (pm_laplace_rowspace_kappa 10, pm_laplace_working_set 0.6) added;
#   NGC5139_output/   = all_models.ecsv of the live v4 grid and models/ as c3_private_copy.sh: large read-only
#                       inputs (*.bz2, *.npz, begin*.dat, *orbclass.out) symlinked, the rest copied;
#   models.txt        = the model directories (all_models 'directory'), one per line.
# The live grid (PM_grid/sBH_GP_grid_v4) is only read.
set -euo pipefail
W=$1
LIVE=/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/sBH_GP_grid_v4
R=$W/regrid; LO=$LIVE/NGC5139_output; PO=$R/NGC5139_output
[ -e "$R/config.yaml" ] && { echo "exists: $R/config.yaml"; exit 1; }
mkdir -p "$PO/models" "$R/results" "$R/logs"
cp "$LO/all_models.ecsv" "$PO/all_models.ecsv"
sed "s|output_directory: \".*\"|output_directory: \"$PO/\"|" "$W/prodone/config.yaml" > "$R/config.yaml"
grep -q "output_directory: \"$PO/\"" "$R/config.yaml"
# the two accepted speed settings, at the indentation of pm_laplace_reexpand
ind=$(grep -o "^ *pm_laplace_reexpand:" "$R/config.yaml" | sed 's/pm_laplace_reexpand://')
sed -i "s|^\( *pm_laplace_reexpand:.*\)$|\1\n${ind}pm_laplace_rowspace_kappa: 10.0\n${ind}pm_laplace_working_set: 0.6|" "$R/config.yaml"
[ "$(grep -c 'pm_laplace_rowspace_kappa: 10.0\|pm_laplace_working_set: 0.6' "$R/config.yaml")" = 2 ]
"$W/../ENV_mkl/bin/python" - "$PO/all_models.ecsv" > "$R/models.txt" <<'EOF'
import sys
from astropy.table import Table
for d in Table.read(sys.argv[1])["directory"]:
    print(d)
EOF
while read -r md; do
    orb=${md%%/*}; ml=$(basename "$md")
    src=$LO/models/$orb; dst=$PO/models/$orb
    [ -e "$src/datfil/orblib_pm_hist.dat.bz2" ] || { echo "no orbit library: $md"; continue; }
    mkdir -p "$dst/datfil" "$dst/$ml"
    cp -r "$src/infil" "$dst/"
    cp "$src/interpolgrid" "$src/fort.30" "$dst/"
    cp -r "$src/$ml/." "$dst/$ml/"
    for f in "$src"/datfil/*; do
        b=$(basename "$f")
        case "$b" in
            *_p[0-9]*.dat) ;;  # another process's temp decompress
            *.bz2|*.npz|begin*.dat|*orbclass.out) ln -s "$f" "$dst/datfil/$b" ;;
            *) cp "$f" "$dst/datfil/$b" ;;
        esac
    done
    echo "copied $md"
done < "$R/models.txt"
echo "models: $(wc -l < "$R/models.txt"), copied: $(ls -d "$PO"/models/*/ | wc -l)"
