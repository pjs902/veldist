"""Profile likelihood in one parameter from scattered models: GP surrogate over all parameters, minimised over the
other parameters' OBSERVED values.

The grid models are Bayesian-optimisation samples in 7 parameters, so the statistic at a given M_IMBH mixes the IMBH
with the six others. gp_profile fits a GP (Matern 5/2, ARD, white noise) to the statistic over all parameters scaled to
the unit box, then at each value of the profiled parameter takes the minimum of the GP mean over the models' own values
of the other parameters: profile(x) = min_i mean(x, theta_i). Minimising over the whole box instead lands in empty
corners (with few models in 7-D the GP is nearly linear there) and runs below every model. loo_rmse gives the
leave-one-out prediction error that says whether the surrogate can be trusted at the scale of the profile.

    python gp_profile.py      self-check on a synthetic quadratic
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def _fit(U, z, seed=0):
    d = U.shape[1]
    kern = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(np.ones(d), (0.05, 100.0), nu=2.5) + WhiteKernel(1e-2, (1e-6, 1.0))
    return GaussianProcessRegressor(kern, n_restarts_optimizer=5, random_state=seed).fit(U, z)


def _unit(X):
    lo, hi = X.min(0), X.max(0)
    return lo, np.where(hi > lo, hi - lo, 1.0)


def gp_profile(X, y, j, grid):
    """X [n, d] parameters, y [n] statistic, j = column profiled, grid = values of column j (units of X).
    Returns (profile [n_grid], sd at the minimiser [n_grid], fitted GP)."""
    lo, span = _unit(X)
    U = (X - lo) / span
    ym, ys = y.mean(), y.std()
    gp = _fit(U, (y - ym) / ys)
    prof, sd = [], []
    for g in (np.asarray(grid) - lo[j]) / span[j]:
        V = U.copy()
        V[:, j] = g
        m, s = gp.predict(V, return_std=True)
        i = np.argmin(m)
        prof.append(m[i] * ys + ym)
        sd.append(s[i] * ys)
    return np.array(prof), np.array(sd), gp


def loo_rmse(X, y):
    """Leave-one-out RMSE of the GP prediction (refit without each model)."""
    lo, span = _unit(X)
    U = (X - lo) / span
    err = []
    for k in range(len(y)):
        keep = np.arange(len(y)) != k
        ym, ys = y[keep].mean(), y[keep].std()
        gp = _fit(U[keep], (y[keep] - ym) / ys)
        err.append(gp.predict(U[k:k + 1])[0] * ys + ym - y[k])
    return float(np.sqrt(np.mean(np.square(err)))), np.array(err)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (60, 4))
    y = 100 * (X[:, 0] - 0.3) ** 2 + 50 * (X[:, 1] - 0.6) ** 2 + 30 * X[:, 2] + rng.normal(0, 0.1, 60)
    g = np.linspace(0.1, 0.9, 9)
    p, s, _ = gp_profile(X, y, 0, g)
    dev = (p - p.min()) - 100 * (g - 0.3) ** 2   # up to a constant (the best observed nuisance values)
    print(np.round(dev, 1), "argmin", g[np.argmin(p)], "loo rmse %.2f" % loo_rmse(X, y)[0])
    assert abs(g[np.argmin(p)] - 0.3) < 1e-9 and np.max(abs(dev)) < 5, dev
    print("ok")
