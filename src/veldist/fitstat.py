"""Reference statistics for scoring model PM histograms against the stars.

Offline yardstick for the DYNAMITE fit statistic (``docs/fit-statistic-design.md``):

- ``stat_diag``: DYNAMITE's current diagonal chi2 on veldist's median /
  floored sigma (the baseline).
- ``stat_exact``: the exact per-star likelihood, extended-Poisson form in
  mass units (expected count ``n_stars * sum(h) / mass``).
- ``laplace_terms`` / ``laplace_rows`` / ``stat_laplace``: its quadratic
  expansion about ``h0``, written as least-squares rows so DYNAMITE's
  Gram / admm path can take it unchanged.
- ``shift_matrix`` / ``shift_hist`` / ``profile_shift``: relative mode (a free
  per-aperture shift of the model histogram).

Every function returns values on the ``-2 log L`` scale (chi2-like), and every
``h`` is a flat row-major histogram of cell **masses** (not densities).
"""

import numpy as np
from scipy import optimize
from scipy.interpolate import CubicSpline

__all__ = [
    "laplace_terms",
    "laplace_rows",
    "stat_laplace",
    "stat_exact",
    "stat_diag",
    "shift_matrix",
    "shift_hist",
    "profile_shift",
]


def laplace_terms(M, h0, n_stars, mass):
    """Fisher matrix F and gradient g of ``-log L`` at ``h0``.

    ``-log L(h) = -sum_s log(m_s . h) + (n_stars / mass) * sum(h)``, so with
    ``p0 = M h0``: ``F = M^T diag(1/p0^2) M`` and ``g = (n_stars/mass) 1 - M^T (1/p0)``.
    """
    M = np.asarray(M, dtype=np.float64)
    p0 = M @ h0
    F = M.T @ ((1.0 / p0**2)[:, None] * M)
    g = n_stars / mass - M.T @ (1.0 / p0)
    return F, g


def laplace_rows(F, g, h0, delta=None, n_stars=None, mass=1.0):
    """Least-squares form of the quadratic ``(h-h0)' F (h-h0) + 2 g'(h-h0)``.

    Returns ``(Lt, target)`` with ``||Lt h - target||^2`` equal to that form
    plus ``delta ||h - h0||^2`` plus a constant. ``Lt = L^T`` for
    ``F + delta I = L L^T``. The ridge makes the empty cells (``F ~ 0``)
    invertible and carries their linear penalty. ``delta=None`` uses
    ``1e-3 * n_stars / mass**2`` (F scales as n_stars / mass**2, so the
    ridge is a fixed small fraction of the information per unit mass^2).
    """
    d = np.diag(F)
    if delta is None:
        if n_stars is None:
            raise ValueError("laplace_rows: give delta or n_stars")
        delta = 1e-3 * n_stars / mass**2
    Fd = F + delta * np.eye(len(d))
    L = np.linalg.cholesky(Fd)
    Lt = L.T
    target = Lt @ (h0 - np.linalg.solve(Fd, g))
    return Lt, target


def stat_laplace(h, Lt, target):
    """``||Lt h - target||^2``: the Laplace statistic up to a constant."""
    r = Lt @ h - target
    return float(r @ r)


def stat_exact(h, M, n_stars, mass):
    """``2 * (-log L)``, extended Poisson in mass units (see ``laplace_terms``)."""
    p = np.asarray(M, dtype=np.float64) @ h
    return float(2.0 * (-np.sum(np.log(np.maximum(p, 1e-300))) + n_stars / mass * np.sum(h)))


def stat_diag(h, d, sd):
    """DYNAMITE's diagonal chi2 against data ``d`` with errors ``sd``."""
    return float(np.sum(((h - d) / sd) ** 2))


def _cdf_regrid_matrix(src, dst):
    # Port of DYNAMITE kinematics.cdf_regrid_matrix (kind='spline'), fork commit c1b0acb.
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src) - 1
    cum = np.tril(np.ones((n + 1, n)), -1)
    at_dst = np.clip(dst, src[0], src[-1])
    interp = CubicSpline(src, np.eye(n + 1), bc_type="not-a-knot")(at_dst)
    return np.diff(interp @ cum, axis=0)


def shift_matrix(edges, m):
    """``S`` with ``S @ masses`` = the masses of the distribution shifted by ``-m``.

    Port of DYNAMITE ``kinematics.pm_shift_matrix`` (fork commit c1b0acb):
    spline-CDF regrid, columns renormalised so mass pushed out of the window
    stays in the edge bins. Entries can be negative (spline overshoot).
    """
    r = _cdf_regrid_matrix(np.asarray(edges) - m, edges)
    col = r.sum(axis=0)
    return np.divide(r, col, out=r, where=col > 0)


def shift_hist(h, edges_x, edges_y, m):
    """Histogram ``h`` (flat, row-major) shifted by ``-m`` (DYNAMITE convention)."""
    kx, ky = len(edges_x) - 1, len(edges_y) - 1
    H = np.asarray(h, dtype=float).reshape(kx, ky)
    return (shift_matrix(edges_x, m[0]) @ H @ shift_matrix(edges_y, m[1]).T).ravel()


def profile_shift(stat_of_h, h, edges_x, edges_y, cap):
    """Best shift ``s`` of the model ``h``: minimises ``stat_of_h(shift_hist(h, ..., -s))``.

    ``s`` is the offset of the data from the model (the model moved by ``+s``).
    Local Nelder-Mead from 0, capped at ``cap`` in norm as DYNAMITE's
    ``_pm_fit_shift`` is. Returns ``(s, value)``.
    """

    def f(s):
        return stat_of_h(shift_hist(h, edges_x, edges_y, -np.asarray(s)))

    d = 0.1 * cap
    r = optimize.minimize(f, np.zeros(2), method="Nelder-Mead", options=dict(
        xatol=1e-5, fatol=1e-9, initial_simplex=np.array([[0.0, 0.0], [d, 0.0], [0.0, d]])))
    s = r.x if r.fun < f(np.zeros(2)) else np.zeros(2)
    n = np.hypot(*s)
    if n > cap:
        s = s * (cap / n)
    return s, f(s)
