"""C1: synthetic comparison of the six fit-statistic versions (plan Part C).

Versions = {addback, relative} x {diag, laplace, exact}:
  addback  the model mean is fixed at 0 (the data may carry an addback error delta)
  relative the model mean is free (the exact profile over the shift)
  diag     DYNAMITE's baseline chi2 on veldist's median / floored sigma
  laplace  veldist.fitstat Laplace rows (closed grid, h0 = 0.99 mean + 0.01 u)
  exact    veldist.fitstat per-star extended Poisson (closed grid; stars with
           < 1e-3 of their kernel on the grid dropped, as the fit drops them)
The model family is a Gaussian (s1, s2) with an aperture-mass factor mu
(models are not normalised; the MGE mass is 1), plus (m1, m2) in relative
mode and a ring fraction f in the tail/edge tests.

One JSON line per aperture; a failure writes {"error": ...} and the run goes on.
usage: python c1.py TEST SEED [SEED ...]
"""
import json
import sys
import time
import traceback
import zlib

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from veldist.fitstat import laplace_rows, laplace_terms, stat_diag, stat_exact, stat_laplace
from veldist.veldist import _cell_mass_correction
from veldist.veldist2d import KinematicSolver2D, precompute_design_matrix_2d

HST = dict(K=23, half=82.80638794003434, err=1.5)
GAIA = dict(K=17, half=56.45657373000141, err=20.0)


def spec(test, seed):
    """(regime, N, s1, s2, delta, f, v0) for this aperture."""
    if test == "calib":
        return HST, [176, 430, 700][seed % 3], 18.0, 16.0, 0.0, 0.0, 0.0
    if test == "gaia":
        return GAIA, 430, 18.0, 16.0, 0.0, 0.0, 0.0
    if test == "addback":
        return HST, 430, 18.0, 16.0, [0.0, 1.0, 2.0, 4.0][seed % 4], 0.0, 0.0
    if test == "tail":
        return HST, 430, 17.0, 17.0, 0.0, [0.0, 0.01][seed % 2], 60.0
    if test == "edge":
        return HST, 430, 17.0, 17.0, 0.0, 0.01, 90.0
    raise ValueError(test)


def gmass(e, s1, s2, m1=0.0, m2=0.0):
    """Gaussian cell masses on the closed grid, renormalised to sum 1 (the
    intrinsic distribution is confined to the grid, as in the veldist fit)."""
    h = np.outer(np.diff(ndtr((e - m1) / s1)), np.diff(ndtr((e - m2) / s2))).ravel()
    return h / h.sum()


def ring(e, v0, dv=4.0):
    """Ring cell masses on the closed grid (sum 1) by fine quadrature."""
    x = np.linspace(-v0 - 6 * dv, v0 + 6 * dv, 801)
    X, Y = np.meshgrid(x, x, indexing="ij")
    p = np.exp(-0.5 * ((np.hypot(X, Y) - v0) / dv) ** 2)
    p /= p.sum()
    K = len(e) - 1
    ix = np.searchsorted(e, X.ravel()) - 1
    iy = np.searchsorted(e, Y.ravel()) - 1
    ok = (ix >= 0) & (ix < K) & (iy >= 0) & (iy < K)
    h = np.zeros(K * K)
    np.add.at(h, ix[ok] * K + iy[ok], p.ravel()[ok])
    return h / max(h.sum(), 1e-300)


def simulate(rng, reg, N, s1, s2, delta, f, v0):
    nt = rng.binomial(N, f) if f > 0 else 0
    v = rng.normal(0, 1, (N - nt, 2)) * [s1, s2] + [delta, 0.0]
    if nt:
        a = rng.uniform(0, 2 * np.pi, nt)
        r = rng.normal(v0, 4.0, nt)
        v = np.r_[v, np.c_[r * np.cos(a), r * np.sin(a)]]
    return v + rng.normal(0, reg["err"], v.shape)


def family(e, relative, tail, v0):
    """(x0, bounds, h(theta)) with theta = (s1, s2, mu[, m1, m2][, f])."""
    r = ring(e, v0) if tail else None

    def h(t):
        s1, s2, mu = t[0], t[1], t[2]
        m1, m2 = (t[3], t[4]) if relative else (0.0, 0.0)
        base = gmass(e, s1, s2, m1, m2)
        if tail:
            fr = t[-1]
            base = (1 - fr) * base + fr * r
        return mu * base

    x0 = [17.0, 17.0, 1.0] + ([0.0, 0.0] if relative else []) + ([0.003] if tail else [])
    b = [(8, 30), (8, 30), (0.5, 1.5)] + ([(-8, 8), (-8, 8)] if relative else []) + (
        [(0.0, 0.1)] if tail else [])
    return x0, b, h


def hessian(f, x, bounds, rel=1e-3):
    x = np.asarray(x, float)
    n = len(x)
    h = np.array([max(rel * (b[1] - b[0]), 1e-6) for b in bounds])
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.eye(n)[i] * h[i]
        if x[i] - h[i] < bounds[i][0]:
            H[i, i] = (f(x + 2 * ei) - 2 * f(x + ei) + f0) / h[i] ** 2
        else:
            H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i + 1, n):
            ej = np.eye(n)[j] * h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej)
                                 + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def fit(stat, x0, bounds):
    best = None
    for start in (x0, [0.5 * (lo + hi) for lo, hi in bounds]):
        r = minimize(stat, start, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    try:
        err = np.sqrt(np.clip(np.diag(np.linalg.inv(hessian(stat, best.x, bounds) / 2)), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(len(best.x), np.nan)
    return dict(x=best.x.tolist(), err=err.tolist(), fun=float(best.fun))


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
    d, sd = med / med.sum(), sig / med.sum()  # DYNAMITE normalise(), MGE mass 1
    draws = _cell_mass_correction(np.asarray(sol.samples["intrinsic_pdf"], float), sol.grid["shape"])
    mean = draws.mean(0)
    h0 = 0.99 * mean / mean.sum() + 0.01 / k2
    M = precompute_design_matrix_2d(v[:, 0], v[:, 1], cov, sol.grid).astype(float)
    M = M[M.sum(axis=1) >= 1e-3]
    Ns = len(M)
    F, g = laplace_terms(M, h0, Ns, 1.0)
    Lt, tg = laplace_rows(F, g, h0, n_stars=Ns)
    stats = {
        "diag": lambda h: stat_diag(h, d, sd),
        "laplace": lambda h: stat_laplace(h, Lt, tg),
        "exact": lambda h: stat_exact(h, M, Ns, 1.0),
    }
    fits = {}
    for mode in ("addback", "relative"):
        x0, b, hf = family(e, mode == "relative", f > 0 or test in ("tail", "edge"), v0 or 60.0)
        for name, st in stats.items():
            fits[f"{mode}_{name}"] = fit(lambda t, st=st, hf=hf: st(hf(t)), x0, b)
    return dict(test=test, seed=seed, truth=dict(regime="gaia" if reg is GAIA else "hst", N=N,
                s1=s1, s2=s2, delta=delta, f=f, v0=v0), n_cells=k2,
                at_floor=float(np.isclose(sig, sig.min()).mean()), secs=round(time.time() - t0, 1),
                fits=fits)


if __name__ == "__main__":
    test = sys.argv[1]
    for seed in map(int, sys.argv[2:]):
        try:
            out = run(test, seed)
        except Exception as exc:  # keep going; the gap is visible in the line count
            out = dict(test=test, seed=seed, error=repr(exc), tb=traceback.format_exc()[-2000:])
        print("@@" + json.dumps(out), flush=True)
