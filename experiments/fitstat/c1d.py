"""C1d: sensitivity of the paper set-up (sharpened Laplace, h0 = posterior mean, 1 re-expansion).

Same seeds / data / NUTS fit as C1 and C1c. Every version uses the
sharpened rows M S and the offset-free statistic; per mode:
  exact_S    reference
  it1        h0 = 0.99 mean + 0.01 u, delta = 1e-3 N / mass^2, 1 re-expansion  (= C1c lap_S_it1)
  it1_nomix  h0 = posterior mean as is (strictly positive: softmax), re-expansion point mixed
  it1_d01    delta x 0.1
  it1_d10    delta x 10
  it2        2 re-expansions
  it1_mix1e3 re-expansion point mixed at 0.001 instead of 0.01

usage: python c1d.py TEST SEED [SEED ...]   (one @@-prefixed JSON line per seed)
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

from veldist.fitstat import laplace_rows, laplace_terms, sharpen_matrix, stat_exact, stat_laplace  # noqa: E402
from veldist.veldist import _cell_mass_correction  # noqa: E402
from veldist.veldist2d import KinematicSolver2D, precompute_design_matrix_2d  # noqa: E402


def lap(M, h0, n, dscale=1.0):
    F, g = laplace_terms(M, h0, n, 1.0)
    Lt, tg = laplace_rows(F, g, h0, delta=1e-3 * n * dscale, n_stars=n)
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
    k2 = reg["K"] ** 2
    draws = _cell_mass_correction(np.asarray(sol.samples["intrinsic_pdf"], float), sol.grid["shape"])
    mean = draws.mean(0)
    mean = mean / mean.sum()
    e = np.asarray(sol.grid["edges_x"], float)
    M = precompute_design_matrix_2d(v[:, 0], v[:, 1], cov, sol.grid).astype(float)
    M = M[M.sum(axis=1) >= 1e-3]
    MS = M @ sharpen_matrix(reg["K"])
    n = len(M)
    mix = lambda h, a=0.01: (1 - a) * h / h.sum() + a / k2
    tail = f > 0 or test in ("tail", "edge")
    fits = {}
    for mode in ("addback", "relative"):
        x0, b, hf = family(e, mode == "relative", tail, v0 or 60.0)
        F = lambda st: fit(lambda t: st(hf(t)), x0, b)
        fits[f"{mode}_exact_S"] = F(lambda h: stat_exact(h, MS, n, 1.0))
        for name, h0, ds, n_it, a in (("it1", mix(mean), 1.0, 1, 0.01),
                                      ("it1_nomix", mean, 1.0, 1, 0.01),
                                      ("it1_d01", mix(mean), 0.1, 1, 0.01),
                                      ("it1_d10", mix(mean), 10.0, 1, 0.01),
                                      ("it2", mix(mean), 1.0, 2, 0.01),
                                      ("it1_mix1e3", mix(mean), 1.0, 1, 0.001)):
            r = F(lap(MS, h0, n, ds))
            for _ in range(n_it):
                r = F(lap(MS, mix(hf(r["x"]), a), n, ds))
            fits[f"{mode}_{name}"] = r
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
