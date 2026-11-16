"""Summarise fitstat results: bias, scatter, claimed error, calibration ratio.

usage: python summarise.py results_dir
Per test and group (N for calib, delta for addback, f for tail), for each
statistic and parameter: mean bias, scatter across seeds, median claimed
error, scatter/claimed. Baseline = diag_floor.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PARAMS = {"calib": ["s1", "s2"], "addback": ["s1", "s2"], "tail": ["s", "f"],
          "edge": ["s", "f"], "outlier": ["s"]}
GROUP = {"calib": "N", "addback": "delta", "tail": "f", "edge": "f", "outlier": "N"}


def truth_of(t, p):
    return t[p]


for f in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    if not rows:
        continue
    test = rows[0]["test"]
    groups = defaultdict(list)
    for r in rows:
        groups[r["truth"][GROUP[test]]].append(r)
    print(f"\n##### {test}  ({len(rows)} apertures; median secs "
          f"{np.median([r['secs'] for r in rows]):.0f})")
    for g, rs in sorted(groups.items()):
        print(f"--- {GROUP[test]} = {g}  (n={len(rs)})")
        stats = list(rs[0]["fits"])
        print(f"{'statistic':16s}{'param':>6s}{'bias':>9s}{'scatter':>9s}{'claimed':>9s}{'ratio':>7s}")
        for st in stats:
            for i, p in enumerate(PARAMS[test]):
                x = np.array([r["fits"][st][0][i] for r in rs])
                e = np.array([r["fits"][st][1][i] for r in rs])
                tr = np.array([truth_of(r["truth"], p) for r in rs])
                b = x - tr
                sc = b.std(ddof=1) if len(b) > 1 else np.nan
                cl = np.nanmedian(e)
                print(f"{st:16s}{p:>6s}{b.mean():9.3f}{sc:9.3f}{cl:9.3f}{sc / cl if cl else np.nan:7.1f}")
