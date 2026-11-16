"""C1b: why is Laplace biased low against exact, and does a better h0 fix it?

Re-runs C1 seeds without NUTS (stars and M are regenerated from the seed; the
exact fit does not depend on the veldist posterior) and fits Laplace with
other expansion points h0. Judge by the paired difference to exact.

  exact        as in C1 (check: reproduces the stored C1 exact fit)
  oracle       h0 = 0.99 truth + 0.01 u            (is the bias h0-driven?)
  oracle_pure  h0 = truth (+1e-12)                  (the 1% uniform mix)
  oracle_d01   oracle with delta x 0.01             (the ridge)
  npmle        h0 = 0.99 EM deconvolution + 0.01 u  (data-only; one QP pass)
  iter1/iter2  h0 re-expanded at the previous Laplace fit, starting from npmle
               (= IRLS: one extra rebuild per iteration in DYNAMITE)
  npmle_warm   npmle, but the fit also starts from the exact fit (local minimum?)
  exact_S      exact with M -> M S, S = [-1/12, 7/6, -1/12] per axis: removes the
               h^2/6 the flat-in-cell density adds to model cell masses
  exact_fine   exact on a 3x finer grid (discretisation reference)

usage: python c1b.py TEST SEED [SEED ...]   (one JSON line per seed, @@-prefixed)
"""
import json
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from c1 import family, fit, gmass, ring, simulate, spec  # noqa: E402

from veldist.fitstat import laplace_rows, laplace_terms, stat_exact, stat_laplace  # noqa: E402
from veldist.veldist2d import KinematicSolver2D, precompute_design_matrix_2d  # noqa: E402


def npmle(M, n_stars, iters=2000):
    """EM for the extended-Poisson MLE over cells: h_c <- h_c sum_s M_sc / p_s / N."""
    h = np.full(M.shape[1], 1.0 / M.shape[1])
    for _ in range(iters):
        h *= M.T @ (1.0 / np.maximum(M @ h, 1e-300)) / n_stars
    return h


def sharpen(M, K):
    """M S: S is [-1/12, 7/6, -1/12] along each grid axis (sum-preserving inside)."""
    T = np.diag(np.full(K, 7 / 6)) + np.diag(np.full(K - 1, -1 / 12), 1) + np.diag(np.full(K - 1, -1 / 12), -1)
    return M @ np.kron(T, T)


def lap_stat(M, h0, n_stars, delta_scale=1.0):
    F, g = laplace_terms(M, h0, n_stars, 1.0)
    Lt, tg = laplace_rows(F, g, h0, delta=1e-3 * n_stars * delta_scale, n_stars=n_stars)
    r = tg - Lt @ h0  # ||Lt h - tg||^2 - ||r||^2 = q.q - 2 r.q with q = Lt (h - h0):
    # the constant ||r||^2 (~1e8) made L-BFGS-B stop early and its finite differences noisy

    def st(h):
        q = Lt @ (h - h0)
        return float(q @ q - 2 * r @ q)
    assert np.isclose(st(h0 * 1.01), stat_laplace(h0 * 1.01, Lt, tg) - stat_laplace(h0, Lt, tg),
                      rtol=1e-5, atol=1e-3)
    return st


def run(test, seed):
    t0 = time.time()
    reg, N, s1, s2, delta, f, v0 = spec(test, seed)
    rng = np.random.default_rng([seed, zlib.crc32(test.encode())])
    v = simulate(rng, reg, N, s1, s2, delta, f, v0)
    cov = np.tile(np.eye(2) * reg["err"] ** 2, (len(v), 1, 1))
    sol = KinematicSolver2D()
    sol.setup_grid(center=(0.0, 0.0), width=(2 * reg["half"],) * 2, n_bins=reg["K"])
    e = np.asarray(sol.grid["edges_x"], float)
    k2 = reg["K"] ** 2
    M = precompute_design_matrix_2d(v[:, 0], v[:, 1], cov, sol.grid).astype(float)
    M = M[M.sum(axis=1) >= 1e-3]
    Ns = len(M)
    tail = f > 0 or test in ("tail", "edge")
    truth = gmass(e, s1, s2, delta, 0.0)
    if f > 0:
        truth = (1 - f) * truth + f * ring(e, v0)
    u = np.full(k2, 1.0 / k2)
    hn = npmle(M, Ns)
    stats = {"exact": lambda h: stat_exact(h, M, Ns, 1.0),
             "oracle": lap_stat(M, 0.99 * truth + 0.01 * u, Ns),
             "oracle_pure": lap_stat(M, truth + 1e-12, Ns),
             "oracle_d01": lap_stat(M, 0.99 * truth + 0.01 * u, Ns, 0.01),
             "npmle": lap_stat(M, 0.99 * hn / hn.sum() + 0.01 * u, Ns)}
    stats["exact_S"] = (lambda MS: lambda h: stat_exact(h, MS, Ns, 1.0))(sharpen(M, reg["K"]))
    fine = KinematicSolver2D()
    fine.setup_grid(center=(0.0, 0.0), width=(2 * reg["half"],) * 2, n_bins=3 * reg["K"])
    ef = np.asarray(fine.grid["edges_x"], float)
    Mf = precompute_design_matrix_2d(v[:, 0], v[:, 1], cov, fine.grid).astype(float)
    Mf = Mf[Mf.sum(axis=1) >= 1e-3]
    fits = {}
    for mode in ("addback", "relative"):
        x0, b, hf = family(e, mode == "relative", tail, v0 or 60.0)
        for name, st in stats.items():
            fits[f"{mode}_{name}"] = fit(lambda t, st=st, hf=hf: st(hf(t)), x0, b)
        ex = fits[f"{mode}_exact"]["x"]
        fits[f"{mode}_npmle_warm"] = min(
            (fits[f"{mode}_npmle"], fit(lambda t, st=stats["npmle"], hf=hf: st(hf(t)), ex, b)),
            key=lambda r: r["fun"])
        x0f, bf, hff = family(ef, mode == "relative", tail, v0 or 60.0)
        fits[f"{mode}_exact_fine"] = fit(lambda t: stat_exact(hff(t), Mf, len(Mf), 1.0), x0f, bf)
        prev = fits[f"{mode}_npmle"]
        for it in (1, 2):
            hm = hf(prev["x"])
            st = lap_stat(M, 0.99 * hm / hm.sum() + 0.01 * u, Ns)
            prev = fits[f"{mode}_iter{it}"] = fit(lambda t, st=st, hf=hf: st(hf(t)), x0, b)
    return dict(test=test, seed=seed, truth=dict(N=N, s1=s1, s2=s2, delta=delta, f=f, v0=v0),
                secs=round(time.time() - t0, 1), fits=fits)


if __name__ == "__main__":
    test = sys.argv[1]
    for seed in map(int, sys.argv[2:]):
        try:
            out = run(test, seed)
        except Exception as exc:
            out = dict(test=test, seed=seed, error=repr(exc), tb=traceback.format_exc()[-2000:])
        print("@@" + json.dumps(out), flush=True)
