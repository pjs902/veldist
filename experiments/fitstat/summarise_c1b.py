"""Summarise C1b: per test and mode, each variant's s1/s2 bias against truth and paired against exact.

usage: python summarise_c1b.py C1B_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

for f in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    rs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    rs = [r for r in rs if "fits" in r]
    if not rs:
        continue
    print(f"\n##### {f.stem}: n = {len(rs)}")
    for mode in ("addback", "relative"):
        names = [k.split("_", 1)[1] for k in rs[0]["fits"] if k.startswith(mode)]
        print(f"--- {mode}: {'variant':12s} {'s1-truth':>14s} {'s2-truth':>14s} {'s1-exact':>14s} {'s2-exact':>14s}"
              + ("  m1-truth" if mode == "relative" else ""))
        for v in names:
            cols = []
            for i, p in ((0, "s1"), (1, "s2")):
                b = np.array([r["fits"][f"{mode}_{v}"]["x"][i] - r["truth"][p] for r in rs])
                cols.append(f"{b.mean():+7.3f}±{b.std(ddof=1) / np.sqrt(len(b)):.3f}")
            for i in (0, 1):
                d = np.array([r["fits"][f"{mode}_{v}"]["x"][i] - r["fits"][f"{mode}_exact"]["x"][i] for r in rs])
                cols.append(f"{d.mean():+7.3f}±{d.std(ddof=1) / np.sqrt(len(d)):.3f}")
            extra = ""
            if mode == "relative":
                m = np.array([r["fits"][f"{mode}_{v}"]["x"][3] - r["truth"]["delta"] for r in rs])
                extra = f"  {m.mean():+.2f}±{m.std(ddof=1) / np.sqrt(len(m)):.2f}"
            print(f"    {'':9s}{v:12s} " + " ".join(f"{c:>14s}" for c in cols) + extra)
