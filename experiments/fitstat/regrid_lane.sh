#!/bin/bash
# One lane of the full-statistic re-solve: the models of LIST, one after another (16 threads each).
#   regrid_lane.sh W LIST FORK_WORKTREE
# Skips a model whose result exists; a failed model is logged and the lane goes on.
set -uo pipefail
W=$1; LIST=$2; FORK=$3
R=$W/regrid
export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 PYTHONPATH=$FORK
while read -r md; do
    tag=$(echo "$md" | tr '/' '_' | sed 's/_$//')
    out=$R/results/$tag.npz
    [ -e "$out" ] && { echo "done already: $md"; continue; }
    echo "### $(date -u +%FT%T) $md"
    "$W/../ENV_mkl/bin/python" -u "$W/veldist/experiments/fitstat/regrid_one.py" "$R/config.yaml" "$md" "$out" \
        > "$R/logs/$tag.log" 2>&1 || echo "FAILED $md (exit $?): $R/logs/$tag.log"
    grep "solve .* h" "$R/logs/$tag.log" | cut -c1-300
done < "$LIST"
