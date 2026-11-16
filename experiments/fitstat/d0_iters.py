"""D0 (third number): evaluations a first-order exact solve needs, on a small synthetic library.

Toy: n_orb orbits, n_ap apertures of a K x K grid, each orbit a Gaussian
blob per aperture; stars drawn from w_true; a soft mass row per aperture (as
DYNAMITE's projected masses). Compares
  laplace_it1  Laplace rows + one re-expansion, solved as NNLS (2 solves)
  exact        -2 log L + mass rows, L-BFGS-B over w >= 0, counting f+grad
               evaluations to reach the optimum to 0.1 (from a cold start and
               from the Laplace solution).
Iteration counts depend on conditioning more than on size; the full-scale
cost is (evaluations here) x (D0's t_eval), an extrapolation.

usage: python d0_iters.py [n_orb n_ap K stars_per_ap]
"""
import sys
import time

import numpy as np
from scipy.optimize import minimize, nnls

sys.path.insert(0, "src")
from veldist.fitstat import laplace_terms, laplace_rows  # noqa: E402

n_orb, n_ap, K, n_st = (int(x) for x in (sys.argv[1:5] or (1500, 30, 15, 300)))
rng = np.random.default_rng(1)
c = np.linspace(-1, 1, K)
X, Y = np.meshgrid(c, c, indexing="ij")
H = np.empty((n_orb, n_ap, K * K))
for o in range(n_orb):
    mx, my = rng.normal(0, 0.3, (2, n_ap))
    s = rng.uniform(0.15, 0.5, n_ap)
    occ = rng.uniform(0, 1, n_ap) ** 3  # orbits visit apertures unevenly
    for a in range(n_ap):
        g = np.exp(-0.5 * ((X - mx[a]) ** 2 + (Y - my[a]) ** 2) / s[a] ** 2).ravel()
        H[o, a] = occ[a] * g / g.sum()
w_true = rng.gamma(0.5, 1.0, n_orb)
h_true = np.einsum("o,oak->ak", w_true, H)
Ma = h_true.sum(1)
Ms = []
err = 0.08
for a in range(n_ap):  # stars, then closed-grid design rows (flat cells)
    idx = rng.choice(K * K, n_st, p=h_true[a] / Ma[a])
    v = np.c_[X.ravel()[idx], Y.ravel()[idx]] + rng.normal(0, err, (n_st, 2))
    e = np.r_[c - (c[1] - c[0]) / 2, c[-1] + (c[1] - c[0]) / 2]
    from scipy.special import ndtr
    px = np.diff(ndtr((e[None] - v[:, :1]) / err), axis=1)
    py = np.diff(ndtr((e[None] - v[:, 1:]) / err), axis=1)
    M = (px[:, :, None] * py[:, None, :]).reshape(n_st, -1)
    Ms.append(M[M.sum(1) >= 1e-3])
emass = 0.01 * Ma  # 1% mass rows
Pm = H.sum(2).T / emass[:, None]  # (n_ap, n_orb)
bm = Ma / emass


def exact(w):
    f, g = 0.0, np.zeros(n_orb)
    for a in range(n_ap):
        h = w @ H[:, a]
        p = np.maximum(Ms[a] @ h, 1e-300)
        n = len(Ms[a])
        f += 2 * (-np.log(p).sum() + n / Ma[a] * h.sum())
        g += 2 * H[:, a] @ (n / Ma[a] - Ms[a].T @ (1 / p))
    r = Pm @ w - bm
    return f + r @ r, g + 2 * Pm.T @ r


def laplace_solve(h0s):
    A, b = [Pm], [bm]
    for a in range(n_ap):
        n = len(Ms[a])
        F, g = laplace_terms(Ms[a], h0s[a], n, Ma[a])
        Lt, tg = laplace_rows(F, g, h0s[a], n_stars=n, mass=Ma[a])
        A.append(Lt @ H[:, a].T)
        b.append(tg)
    A, b = np.vstack(A), np.concatenate(b)
    sc = np.linalg.norm(A, axis=0) + 1e-30
    return nnls(A / sc, b, maxiter=50 * n_orb)[0] / sc


t0 = time.time()
mix = lambda h: (0.99 * h / h.sum() + 0.01 / h.size) * h.sum()
w1 = laplace_solve([mix(h_true[a] * (1 + 0.1 * rng.standard_normal(K * K)).clip(0.2)) for a in range(n_ap)])
w2 = laplace_solve([mix(w1 @ H[:, a]) for a in range(n_ap)])
t_lap = time.time() - t0

ref = minimize(exact, w2, jac=True, method="L-BFGS-B", bounds=[(0, None)] * n_orb,
               options=dict(maxfun=200000, maxiter=200000, ftol=1e-15, gtol=1e-10))
fstar = ref.fun
print(f"n_orb {n_orb} n_ap {n_ap} K {K} stars/ap {n_st}; laplace_it1 two solves {t_lap:.1f} s; "
      f"exact(laplace_it1) - exact_min = {exact(w2)[0] - fstar:.2f}", flush=True)
for name, w0 in (("cold (uniform)", np.full(n_orb, Ma.sum() / n_orb / H.sum(2).mean() / n_ap)),
                 ("from laplace_it1", w2)):
    hist = []

    def f(w):
        v = exact(w)
        hist.append(v[0])
        return v
    minimize(f, w0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * n_orb,
             options=dict(maxfun=200000, maxiter=200000, ftol=1e-15, gtol=1e-10))
    hist = np.minimum.accumulate(hist) - fstar
    n01 = int(np.argmax(hist < 0.1)) + 1 if (hist < 0.1).any() else -1
    n1 = int(np.argmax(hist < 1.0)) + 1 if (hist < 1.0).any() else -1
    print(f"exact L-BFGS-B {name}: evals to within 1.0: {n1}, within 0.1: {n01} (of {len(hist)})", flush=True)
