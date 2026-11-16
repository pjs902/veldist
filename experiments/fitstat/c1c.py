"""C1c: C1 rerun with the fixed Laplace (offset-free) and the paper candidates.

Same seeds, data and veldist NUTS fit as C1 (c1.py). Versions per mode
(addback / relative):
  diag      baseline (as C1)
  exact     exact per-star, flat-in-cell M (as C1)
  exact_S   exact with M S (veldist.fitstat.sharpen_matrix)
  lap       Laplace, h0 = posterior mean (as C1), offset-free
  lap_S     the same with M S
  lap_S_it1 lap_S re-expanded once at its own fitted model (the paper candidate)

usage: python c1c.py TEST SEED [SEED ...]   (one @@-prefixed JSON line per seed)
"""
import json
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from c1 import GAIA, family, fit, simulate, spec  # noqa: E402

from veldist.fitstat import (laplace_rows, laplace_terms, sharpen_matrix, stat_diag,  # noqa: E402
                             stat_exact, stat_laplace)
from veldist.veldist import _cell_mass_correction  # noqa: E402
from veldist.veldist2d import KinematicSolver2D, precompute_design_matrix_2d  # noqa: E402


def lap(M, h0, n):
    F, g = laplace_terms(M, h0, n, 1.0)
    Lt, tg = laplace_rows(F, g, h0, n_stars=n)
    return lambda h: stat_laplace(h, Lt, tg, h0=h0)


def run(test, seed):
    t0 = time.time()
    reg, N, s1, s2, delta, f, v0 = spec(test, seed)
    rng = np.random.default_rng([seed, zlib.crc32(test.encode())])
    v = simulate(rng, reg, N, s1, s2, delta, f, v0)
    cov = np.tile(np.eye(2) * reg["err"] ** 2, (len(v), 1, 1))
    sol = KinematicSolver2D()
    sol.setup_grid(center=(0.0, 0.0), width=(2 * reg["half"],) * 2, n_bins=reg["K"])
    sol.add_data(v[:, 0], v[:, 1], cov)
    sol.run(num_samples=3000, seed=seed)
    sol.clip_uncertainties()
    e = np.asarray(sol.grid["edges_x"], float)
    k2 = reg["K"] ** 2
    med, sig = sol.clipped_samples["pdf_median"], sol.clipped_samples["pdf_uncertainty"]
    d, sd = med / med.sum(), sig / med.sum()
    draws = _cell_mass_correction(np.asarray(sol.samples["intrinsic_pdf"], float), sol.grid["shape"])
    mean = draws.mean(0)
    h0 = 0.99 * mean / mean.sum() + 0.01 / k2
    M = precompute_design_matrix_2d(v[:, 0], v[:, 1], cov, sol.grid).astype(float)
    M = M[M.sum(axis=1) >= 1e-3]
    MS = M @ sharpen_matrix(reg["K"])
    n = len(M)
    stats = {"diag": lambda h: stat_diag(h, d, sd),
             "exact": lambda h: stat_exact(h, M, n, 1.0),
             "exact_S": lambda h: stat_exact(h, MS, n, 1.0),
             "lap": lap(M, h0, n),
             "lap_S": lap(MS, h0, n)}
    fits = {}
    tail = f > 0 or test in ("tail", "edge")
    for mode in ("addback", "relative"):
        x0, b, hf = family(e, mode == "relative", tail, v0 or 60.0)
        for name, st in stats.items():
            fits[f"{mode}_{name}"] = fit(lambda t, st=st, hf=hf: st(hf(t)), x0, b)
        hm = hf(fits[f"{mode}_lap_S"]["x"])
        st = lap(MS, 0.99 * hm / hm.sum() + 0.01 / k2, n)
        fits[f"{mode}_lap_S_it1"] = fit(lambda t, st=st, hf=hf: st(hf(t)), x0, b)
    return dict(test=test, seed=seed, truth=dict(regime="gaia" if reg is GAIA else "hst", N=N,
                s1=s1, s2=s2, delta=delta, f=f, v0=v0), secs=round(time.time() - t0, 1), fits=fits)


if __name__ == "__main__":
    test = sys.argv[1]
    for seed in map(int, sys.argv[2:]):
        try:
            out = run(test, seed)
        except Exception as exc:
            out = dict(test=test, seed=seed, error=repr(exc), tb=traceback.format_exc()[-2000:])
        print("@@" + json.dumps(out), flush=True)
