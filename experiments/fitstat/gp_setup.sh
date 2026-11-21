#!/bin/bash
# GP search under the per-star (Laplace) statistic, warm-started from the 74 regrid models (run once, after the
# regrid lanes have finished).
#   gp_setup.sh W
# Makes $W/gprun/ with
#   config.yaml      = $W/regrid/config.yaml with output_directory -> gprun, 12 libraries x 2 chunks = 48 cores,
#                      3 concurrent solves (16 MKL threads each, set by the launch), n_max_mods 98 (74 + 2 batches of 12),
#                      n_max_iter 2, warmup_mode sobol (74 done >= n_initial_random: no warm-up batch, GP from the start);
#   NGC5139_output/  = a copy of the regrid output (cp -a keeps the symlinks to the v4 orbit libraries) whose
#                      all_models.ecsv carries the re-solved statistic: kinchi2 = chi2_kin, chi2 = chi2_tot,
#                      chi2_ext_added = 0 (the escape term has weight 0); the 12 profile-fill rows
#                      are marked done (their weights come from the regrid solve).
# The regrid and the v4 grid are only read.
set -euo pipefail
W=$1
R=$W/regrid; G=$W/gprun; PO=$G/NGC5139_output
[ -e "$G" ] && { echo "exists: $G"; exit 1; }
n=$(ls "$R"/results/*.npz | wc -l)
[ "$n" = 74 ] || { echo "regrid not finished: $n/74 results"; exit 1; }
tmux list-windows -t veldist-fitstat -F '#W' | grep -q '^regrid_l' && { echo "regrid lanes still open"; exit 1; }
mkdir -p "$G"
cp -a "$R/NGC5139_output" "$PO"
sed -e "s|output_directory: \".*\"|output_directory: \"$PO/\"|" \
    -e 's|^\( *\)total_cores: 96$|\1total_cores: 48|' \
    -e 's|^\( *\)ncpus: 24 .*$|\1ncpus: 12  # concurrent orbit libraries (gprun: 12 x 2 chunks x 2 families = 48)|' \
    -e 's|^\( *\)ncpus_weights: 6$|\1ncpus_weights: 3  # gprun: 3 x 16 MKL threads = 48 cores, ~220 GB|' \
    -e 's|^\( *\)n_max_mods: 180$|\1n_max_mods: 98  # gprun: 74 warm start + 2 batches of 12|' \
    -e 's|^\( *\)n_max_iter: 100$|\1n_max_iter: 2|' \
    -e 's|^\( *\)warmup_mode: "initial_guess"$|\1warmup_mode: "sobol"  # gprun: 74 done >= n_initial_random 24 -> straight to GP (initial_guess would replay leftover axial points)|' \
    "$R/config.yaml" > "$G/config.yaml"
for k in "output_directory: \"$PO/\"" "total_cores: 48" "ncpus: 12 " "ncpus_weights: 3 " "n_max_mods: 98 " "n_max_iter: 2$" 'warmup_mode: "sobol"'; do
    grep -q "$k" "$G/config.yaml" || { echo "config rewrite failed: $k"; exit 1; }
done
"$W/../ENV_mkl/bin/python" - "$PO/all_models.ecsv" "$R/results" <<'EOF'
import glob, sys
import numpy as np
from astropy.table import Table
path, res = sys.argv[1:3]
t = Table.read(path)
d = [str(x).strip("/") for x in t["directory"]]
seen = set()
for f in glob.glob(f"{res}/*.npz"):
    z = np.load(f)
    i = d.index(str(z["directory"]).strip("/"))
    t["kinchi2"][i] = float(z["chi2_kin"])  # kinmapchi2 untouched (NaN for these kinematics)
    t["chi2"][i] = float(z["chi2_tot"])
    # the 12 profile-fill rows never got v4 weights; regrid_one's solve wrote their orbit_weights.ecsv
    t["weights_done"][i] = t["all_done"][i] = True
    t["chi2_ext_added"][i] = 0.0
    seen.add(i)
assert len(seen) == len(t) == 74, (len(seen), len(t))
assert np.all(t["orblib_done"])
t.write(path, format="ascii.ecsv", overwrite=True)
print("warm-start table:", len(t), "models; best kinchi2 %.1f at" % np.min(t["kinchi2"]),
      t["directory"][int(np.argmin(t["kinchi2"]))])
EOF
echo "gprun ready: $G"
