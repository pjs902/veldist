"""Phase-1 fit-statistic tests: addback + diagonal floored chi2 is the baseline.

One synthetic HST-like aperture per (test, seed): simulate, fit veldist once,
then fit a small parametric model family by minimising each statistic and
record best-fit parameters and claimed (Hessian) errors. One JSON line per
aperture goes to stdout.

Statistics (all on the -2 log L scale):
  diag_floor  dynamite's current chi2 on veldist median / floored sigma (baseline)
  fullcov     chi2 with the Ledoit-Wolf posterior covariance (pinv)
  exact       per-star -2 sum log(M h), closed grid
  exact_eps   same with a 1e-3 uniform background per star
  laplace     quadratic expansion of `exact` about the veldist posterior mean
  laplace1    one IRLS round: re-expanded about the laplace best fit
  exact_open  per-star with open (semi-infinite) border cells and the model's
              overflow in the border cells, i.e. what DYNAMITE's orbit
              histograms contain (test `edge` only)

Tests:
  calib   aniso Gaussian (18, 16), mean 0, N in {180, 430, 700}; fit (s1, s2)
  addback mean offset delta in {0,1,2,4} km/s on axis 1; fit (s1, s2) with the
          mean fixed at 0 (addback) and, as `*_rel`, with the mean free
          (relative: the shift is profiled out)
  tail    Gaussian 17 + f ring at 60 km/s, f in {0, 0.01}; fit (s, f>=0)
  outlier Gaussian 17 + 2 contaminants uniform in +-80; fit s
  edge    Gaussian 17 + 1% ring at 90 km/s (beyond the +-82.8 grid); fit (s, f)

Usage: python fitstat.py TEST SEED [SEED ...]
"""
import json
import zlib
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from sklearn.covariance import LedoitWolf
from veldist.veldist2d import KinematicSolver2D

K, W, ERR, SIG0 = 23, 165.61277588006868, 1.5, 17.0
SUB = 9  # sub-samples per cell axis for non-separable (ring) cell masses
EPS = 1e-3


def edges_open(e):
    o = e.copy()
    o[0], o[-1] = -np.inf, np.inf
    return o


def gmass(e, s1, s2, m1=0.0, m2=0.0):
    cx = np.diff(ndtr((e - m1) / s1))
    cy = np.diff(ndtr((e - m2) / s2))
    return np.outer(cx, cy).ravel()


def ring_mass(e, v0, dv, open_edges=False):
    """Cell masses of a 2D ring (radius v0, Gaussian width dv) by sub-cell
    quadrature, normalised over the plane. With open_edges, mass beyond the
    grid is added to the border cells (DYNAMITE's histogram_velbin)."""
    lo, hi = -v0 - 6 * dv, v0 + 6 * dv
    n = 801
    x = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    p = np.exp(-0.5 * ((np.hypot(X, Y) - v0) / dv) ** 2)
    p /= p.sum()
    if open_edges:
        ix = np.clip(np.searchsorted(e, X.ravel()) - 1, 0, K - 1)
        iy = np.clip(np.searchsorted(e, Y.ravel()) - 1, 0, K - 1)
        keep = np.ones(ix.size, bool)
    else:
        ix = np.searchsorted(e, X.ravel()) - 1
        iy = np.searchsorted(e, Y.ravel()) - 1
        keep = (ix >= 0) & (ix < K) & (iy >= 0) & (iy < K)
    h = np.zeros(K * K)
    np.add.at(h, ix[keep] * K + iy[keep], p.ravel()[keep])
    return h


def design(v, e):
    """Per-star cell masses for isotropic diagonal errors (exact erf)."""
    ax = np.diff(ndtr((e[None, :] - v[:, 0:1]) / ERR), axis=1)
    ay = np.diff(ndtr((e[None, :] - v[:, 1:2]) / ERR), axis=1)
    return (ax[:, :, None] * ay[:, None, :]).reshape(len(v), -1)


def simulate(test, seed, rng):
    """Returns velocities (N,2), truth dict, model family spec."""
    if test == "calib":
        N = [180, 430, 700][seed % 3]
        v = rng.normal(0, 1, (N, 2)) * [18.0, 16.0]
        truth = dict(N=N, s1=18.0, s2=16.0)
    elif test == "addback":
        delta = [0.0, 1.0, 2.0, 4.0][seed % 4]
        N = 430
        v = rng.normal(0, 1, (N, 2)) * [18.0, 16.0] + [delta, 0.0]
        truth = dict(N=N, s1=18.0, s2=16.0, delta=delta)
    elif test in ("tail", "edge"):
        f = (0.0 if seed % 2 == 0 else 0.01) if test == "tail" else 0.01
        v0 = 60.0 if test == "tail" else 90.0
        N = 430
        nt = rng.binomial(N, f)
        core = rng.normal(0, SIG0, (N - nt, 2))
        ang = rng.uniform(0, 2 * np.pi, nt)
        rad = rng.normal(v0, 4.0, nt)
        ring = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
        v = np.r_[core, ring]
        truth = dict(N=N, s=SIG0, f=f, v0=v0, n_tail=int(nt))
    elif test == "outlier":
        N = 430
        v = np.r_[rng.normal(0, SIG0, (N - 2, 2)), rng.uniform(-80, 80, (2, 2))]
        truth = dict(N=N, s=SIG0, n_out=2)
    else:
        raise ValueError(test)
    v = v + rng.normal(0, ERR, v.shape)
    return v, truth


def model_family(test, e, open_edges=False):
    """Returns (x0, bounds, h(theta)) with h normalised to 1."""
    eg = edges_open(e) if open_edges else e

    def norm(h):
        return h / h.sum()

    if test in ("calib", "addback"):
        return [17.0, 17.0], [(10, 25), (10, 25)], lambda t: norm(gmass(eg, t[0], t[1]))
    if test == "addback_rel":
        return ([17.0, 17.0, 0.0, 0.0], [(10, 25), (10, 25), (-8, 8), (-8, 8)],
                lambda t: norm(gmass(eg, t[0], t[1], t[2], t[3])))
    if test == "outlier":
        return [17.0], [(10, 25)], lambda t: norm(gmass(eg, t[0], t[0]))
    if test in ("tail", "edge"):
        v0 = 60.0 if test == "tail" else 90.0
        r = ring_mass(e, v0, 4.0, open_edges=open_edges)
        return ([17.0, 0.003], [(10, 25), (0.0, 0.1)],
                lambda t: norm((1 - t[1]) * gmass(eg, t[0], t[0]) + t[1] * r))
    raise ValueError(test)


def hessian(f, x, bounds, rel=1e-3):
    x = np.asarray(x, float)
    n = len(x)
    h = np.array([max(rel * (b[1] - b[0]), 1e-6) for b in bounds])
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h[i], np.eye(n)[j] * h[j]
            if i == j:
                # one-sided at a lower bound (e.g. f = 0)
                if x[i] - h[i] < bounds[i][0]:
                    H[i, i] = (f(x + 2 * ei) - 2 * f(x + ei) + f0) / h[i] ** 2
                else:
                    H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
            else:
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej)
                                     + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def fit(stat, x0, bounds):
    best = None
    for start in (x0, [0.5 * (b[0] + b[1]) for b in bounds]):
        r = minimize(stat, start, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    H = hessian(stat, best.x, bounds)
    try:
        err = np.sqrt(np.clip(np.diag(np.linalg.inv(H / 2)), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(len(best.x), np.nan)
    return best.x.tolist(), err.tolist(), float(best.fun)


def run(test, seed):
    t0 = time.time()
    rng = np.random.default_rng([seed, zlib.crc32(test.encode())])
    v, truth = simulate(test, seed, rng)
    s = KinematicSolver2D()
    s.setup_grid(center=(0.0, 0.0), width=(W, W), n_bins=K)
    s.add_data(v[:, 0], v[:, 1], np.tile(np.eye(2) * ERR**2, (len(v), 1, 1)))
    s.run(num_samples=3000, seed=seed)
    s.clip_uncertainties()
    e = np.asarray(s.grid["edges_x"], float)
    draws = np.asarray(s.samples["intrinsic_pdf"], float)
    med, sig = s.clipped_samples["pdf_median"], s.clipped_samples["pdf_uncertainty"]
    sc = med.sum()
    d, sd = med / sc, sig / sc
    P = np.linalg.pinv(LedoitWolf().fit(draws).covariance_, rcond=1e-10, hermitian=True)
    pm = draws.mean(0)

    Mc = design(v, e)
    Mc = Mc[Mc.sum(1) >= 1e-3]  # veldist drops these stars too
    Mo = design(v, edges_open(e))
    bc = Mc.sum(1) / (K * K)  # uniform background per star

    def laplace_at(h0):
        p0 = Mc @ h0
        g = -(Mc.T @ (1 / p0))
        F = Mc.T @ ((1 / p0**2)[:, None] * Mc)
        return lambda h: 2 * (g @ (h - h0) + 0.5 * (h - h0) @ F @ (h - h0))

    def stats_for(hf, hf_open=None):
        lap = laplace_at(pm)
        # defensive expansion point: tail stars no longer sit at p0 ~ 0, where
        # one quadratic step can at most double their probability
        lapm = laplace_at(0.99 * pm + 0.01 / (K * K))
        out = {
            "laplace_mix": lambda t: lapm(hf(t)),
            "diag_floor": lambda t: np.sum(((hf(t) - d) / sd) ** 2),
            "fullcov": lambda t: (hf(t) - d) @ P @ (hf(t) - d),
            "exact": lambda t: -2 * np.sum(np.log(np.maximum(Mc @ hf(t), 1e-300))),
            "exact_eps": lambda t: -2 * np.sum(np.log((1 - EPS) * (Mc @ hf(t)) + EPS * bc)),
            "laplace": lambda t: lap(hf(t)),
        }
        if hf_open is not None:
            out["exact_open"] = lambda t: -2 * np.sum(np.log(np.maximum(Mo @ hf_open(t), 1e-300)))
        return out

    results = {}
    fams = [test] + (["addback_rel"] if test == "addback" else [])
    for fam in fams:
        x0, bnd, hf = model_family(fam, e)
        hf_open = model_family(fam, e, open_edges=True)[2] if test == "edge" else None
        for name, st in stats_for(hf, hf_open).items():
            key = name if fam == test else f"{name}_rel"
            results[key] = fit(st, x0, bnd)
        # one IRLS round: re-expand about the laplace best fit
        lap1 = laplace_at(hf(np.array(results["laplace" if fam == test else "laplace_rel"][0])))
        key = "laplace1" if fam == test else "laplace1_rel"
        results[key] = fit(lambda t: lap1(hf(t)), x0, bnd)

    return dict(test=test, seed=seed, truth=truth, n_used=int(len(Mc)),
                at_floor=float(np.isclose(sig, sig.min()).mean()),
                secs=round(time.time() - t0, 1), fits=results)


if __name__ == "__main__":
    test = sys.argv[1]
    for seed in map(int, sys.argv[2:]):
        print(json.dumps(run(test, seed)), flush=True)
