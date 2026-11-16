"""Summarise C1 results: per test / group / version, bias and calibration.

usage: python summarise_c1.py RESULTS_DIR
Columns per parameter:
  bias      mean(fit - truth)                     (+- its standard error)
  scatter   std of fit across seeds
  claimed   median Hessian error at Delta = 1
  r1        scatter / claimed                     (1 = calibrated at Delta = 1)
  rprod     scatter / (claimed * (2 N_tot)^(1/4)) (the same, under production's
            sqrt(2 N_tot) unit for a parameter shared by all 1406 HST apertures,
            N_tot = 1406 * 529; < 1 means production's unit is conservative)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

A_HST, NC_HST = 1406, 529
PROD = (2 * A_HST * NC_HST) ** 0.25
PNAMES = {"addback": ["s1", "s2", "mu"], "relative": ["s1", "s2", "mu", "m1", "m2"]}
GROUP = {"calib": "N", "gaia": "N", "addback": "delta", "tail": "f", "edge": "f"}

for f in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    rows, n_garbled = [], 0
    for x in f.read_text().splitlines():
        try:
            rows.append(json.loads(x))
        except json.JSONDecodeError:  # a line still being written (or interleaved)
            n_garbled += 1
    bad = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    if not rows:
        continue
    test = rows[0]["test"]
    seeds = sorted(r["seed"] for r in rows)
    dup = len(seeds) - len(set(seeds))
    print(f"\n##### {test}: {len(rows)} apertures ({len(bad)} errors, {dup} duplicate seeds, {n_garbled} garbled lines), "
          f"median {np.median([r['secs'] for r in rows]):.0f} s")
    groups = defaultdict(list)
    for r in rows:
        groups[r["truth"][GROUP[test]]].append(r)
    for g, rs in sorted(groups.items()):
        print(f"--- {GROUP[test]} = {g} (n = {len(rs)})")
        print(f"{'version':18s}{'par':>5s}{'bias':>8s}{'+-':>6s}{'scatter':>8s}{'claimed':>8s}{'r1':>6s}{'rprod':>7s}")
        for v in rs[0]["fits"]:
            mode = v.split("_")[0]
            names = PNAMES[mode] + (["f"] if test in ("tail", "edge") else [])
            for i, p in enumerate(names):
                x = np.array([r["fits"][v]["x"][i] for r in rs])
                e = np.array([r["fits"][v]["err"][i] for r in rs])
                if p in ("s1", "s2"):
                    tr = np.array([r["truth"][p] for r in rs])
                elif p == "mu":
                    tr = np.ones(len(rs))
                elif p == "m1":
                    tr = np.array([r["truth"]["delta"] for r in rs])
                elif p == "m2":
                    tr = np.zeros(len(rs))
                else:
                    tr = np.array([r["truth"]["f"] for r in rs])
                b = x - tr
                sc = b.std(ddof=1) if len(b) > 1 else np.nan
                cl = np.nanmedian(e)
                print(f"{v:18s}{p:>5s}{b.mean():8.3f}{sc / np.sqrt(len(b)):6.3f}{sc:8.3f}{cl:8.3f}"
                      f"{sc / cl if cl else np.nan:6.2f}{sc / (cl * PROD) if cl else np.nan:7.3f}")
