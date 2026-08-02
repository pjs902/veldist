# file tests/test_veldist2d.py
"""
Fast (no-MCMC) tests for the 2D solver's grid, design matrix, and GMRF
prior machinery -- see PLAN.md Part 3 (§3.1-§3.3). The slow tilted-Gaussian
recovery / SBC tests are explicitly out of scope for this module.
"""

import numpy as np
import pytest
from scipy import integrate

import jax
import jax.numpy as jnp

from veldist.veldist import precompute_design_matrix
from veldist.veldist2d import (
    build_gmrf_precision,
    precompute_design_matrix_2d,
    setup_grid_2d,
)


# ==============================================================================
# GMRF Q properties
# ==============================================================================


def test_gmrf_Q_properties():
    k = 6
    Q, Q_reg = build_gmrf_precision(k)

    # Symmetric
    np.testing.assert_allclose(Q, Q.T, atol=1e-12)

    # Row sums ~= 0 and Q @ ones ~= 0
    row_sums = Q.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)
    ones = np.ones(k * k)
    np.testing.assert_allclose(Q @ ones, 0.0, atol=1e-10)

    # Connected graph: second-smallest eigenvalue > 0
    eigvals = np.linalg.eigvalsh(Q)
    eigvals_sorted = np.sort(eigvals)
    assert eigvals_sorted[0] == pytest.approx(0.0, abs=1e-8)
    assert eigvals_sorted[1] > 1e-8

    # Q_reg positive definite -> Cholesky succeeds
    L = np.linalg.cholesky(Q_reg)
    assert np.all(np.isfinite(L))

    # diag(Q) takes exactly 3 distinct values for a square grid K>2
    # (corner=3 neighbours, edge=5, interior=8 -- weighted by edge/diag
    # weights, still exactly 3 distinct sums). Catches periodic-boundary leak.
    diag_vals = np.diag(Q)
    distinct = np.unique(np.round(diag_vals, 8))
    assert len(distinct) == 3, f"expected 3 distinct diag values, got {distinct}"


# ==============================================================================
# GMRF prior isotropy (row-major transposition catcher)
# ==============================================================================


def test_gmrf_prior_is_isotropic():
    k = 16
    Q, Q_reg = build_gmrf_precision(k)
    L = np.linalg.cholesky(Q_reg)
    assert np.all(np.isfinite(L))

    n_cells = k * k
    n_draws = 4000
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n_draws, n_cells))

    # x = L^-T z, per draw
    Lt = L.T
    x = np.linalg.solve(Lt, z.T).T  # (n_draws, n_cells)
    x = x.reshape(n_draws, k, k)

    def autocorr_lag1(field, axis):
        a = field
        b = np.roll(field, -1, axis=axis)
        # drop wrapped edge
        if axis == 1:
            a = a[:, :-1, :]
            b = b[:, :-1, :]
        else:
            a = a[:, :, :-1]
            b = b[:, :, :-1]
        a_flat = a.reshape(a.shape[0], -1)
        b_flat = b.reshape(b.shape[0], -1)
        num = np.mean(a_flat * b_flat)
        denom = np.std(a_flat) * np.std(b_flat)
        return num / denom

    corr_x = autocorr_lag1(x, axis=1)
    corr_y = autocorr_lag1(x, axis=2)

    # Diagonal correlations
    def diag_corr(field, sign):
        vals_a = []
        vals_b = []
        for i in range(k - 1):
            for j in range(k - 1):
                # main diagonal: (i,j) vs (i+1,j+1); anti: (i,j+1) vs (i+1,j)
                if sign > 0:
                    vals_a.append(field[:, i, j])
                    vals_b.append(field[:, i + 1, j + 1])
                else:
                    vals_a.append(field[:, i, j + 1])
                    vals_b.append(field[:, i + 1, j])
        vals_a = np.concatenate(vals_a)
        vals_b = np.concatenate(vals_b)
        return np.mean(vals_a * vals_b) / (np.std(vals_a) * np.std(vals_b))

    corr_diag_main = diag_corr(x, 1)
    corr_diag_anti = diag_corr(x, -1)

    # All four directional correlations should agree within Monte Carlo error.
    corrs = np.array([corr_x, corr_y, corr_diag_main, corr_diag_anti])
    assert np.std(corrs) < 0.05, f"anisotropic prior: {corrs}"
    # x and y (edge neighbours) should be close to each other specifically,
    # and diagonals close to each other -- catches an axis transposition,
    # which would show up as corr_x != corr_y despite the grid being square.
    assert abs(corr_x - corr_y) < 0.03, f"x/y correlation mismatch: {corr_x} vs {corr_y}"
    assert (
        abs(corr_diag_main - corr_diag_anti) < 0.03
    ), f"diagonal correlation mismatch: {corr_diag_main} vs {corr_diag_anti}"


# ==============================================================================
# solve_triangular direction check
# ==============================================================================


def test_solve_triangular_direction():
    k = 4
    Q, Q_reg = build_gmrf_precision(k)
    n_cells = k * k
    L = jnp.asarray(np.linalg.cholesky(Q_reg))

    sigma = 1.0
    n_draws = 20000
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (n_draws, n_cells))

    def transform(zi):
        return sigma * jax.scipy.linalg.solve_triangular(L.T, zi, lower=False)

    x = jax.vmap(transform)(z)
    x = np.asarray(x)

    empirical_cov = np.cov(x.T)
    expected_cov = sigma**2 * np.linalg.pinv(np.asarray(Q_reg))

    # Compare only off the null-space direction is fine here since Q_reg is
    # already regularised (full-rank); compare directly with a loose-ish
    # tolerance appropriate for a Monte Carlo covariance estimate.
    np.testing.assert_allclose(empirical_cov, expected_cov, atol=0.15, rtol=0.25)


# ==============================================================================
# Design matrix factorisation / quadrature
# ==============================================================================


def test_design_matrix_2d_factorises():
    grid = setup_grid_2d(center=(0.0, 0.0), width=(20.0, 20.0), n_bins=10)

    rng = np.random.default_rng(1)
    n = 15
    pm1 = rng.uniform(-8, 8, n)
    pm2 = rng.uniform(-8, 8, n)
    sx = rng.uniform(0.5, 2.0, n)
    sy = rng.uniform(0.5, 2.0, n)

    cov = np.zeros((n, 2, 2))
    cov[:, 0, 0] = sx**2
    cov[:, 1, 1] = sy**2

    M2d = precompute_design_matrix_2d(pm1, pm2, cov, grid)

    Mx = np.asarray(
        precompute_design_matrix(pm1, sx, grid["centers_x"], bin_width=grid["width_x"])
    )
    My = np.asarray(
        precompute_design_matrix(pm2, sy, grid["centers_y"], bin_width=grid["width_y"])
    )
    expected = (Mx[:, :, None] * My[:, None, :]).reshape(n, grid["n_cells"])

    np.testing.assert_allclose(M2d, expected, atol=1e-6, rtol=1e-4)


def test_design_matrix_2d_vs_quadrature():
    # The 2x2 Gauss-Legendre sub-cell rule (used for correlated covariance)
    # is an approximation whose accuracy depends on cell size relative to
    # sigma -- PLAN.md §3.1 describes it as "kills most of the error", not
    # exact. Use a grid fine enough (bin << sigma) that the approximation is
    # tight, matching the regime the rule is actually meant for.
    grid = setup_grid_2d(center=(0.0, 0.0), width=(12.0, 12.0), n_bins=24)

    # One diagonal-covariance star, one correlated-covariance star.
    stars = [
        {"mu": (1.0, -0.5), "cov": np.array([[1.0, 0.0], [0.0, 1.5**2]])},
        {"mu": (-1.5, 2.0), "cov": np.array([[1.2**2, 0.6], [0.6, 0.9**2]])},
    ]

    for star in stars:
        mu1, mu2 = star["mu"]
        cov = star["cov"]
        M = precompute_design_matrix_2d(
            np.array([mu1]), np.array([mu2]), cov[None, :, :], grid
        )[0]

        cov_inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        norm = 1.0 / (2 * np.pi * np.sqrt(det))

        def pdf(x, y, mu1=mu1, mu2=mu2, cov_inv=cov_inv, norm=norm):
            dx = x - mu1
            dy = y - mu2
            quad = (
                cov_inv[0, 0] * dx * dx
                + 2 * cov_inv[0, 1] * dx * dy
                + cov_inv[1, 1] * dy * dy
            )
            return norm * np.exp(-0.5 * quad)

        # Check a handful of cells (not all, for speed): near the star's mean
        # (where quadrature accuracy matters most) plus a far corner cell.
        ix_c = np.searchsorted(grid["edges_x"], mu1) - 1
        iy_c = np.searchsorted(grid["edges_y"], mu2) - 1
        near_cells = [
            np.ravel_multi_index((ix_c, iy_c), grid["shape"]),
            np.ravel_multi_index((ix_c + 1, iy_c), grid["shape"]),
            np.ravel_multi_index((ix_c, iy_c + 1), grid["shape"]),
        ]
        cell_idxs = [*near_cells, 0, grid["n_cells"] - 1]
        for m in cell_idxs:
            ix, iy = np.unravel_index(m, grid["shape"])
            x0, x1 = grid["edges_x"][ix], grid["edges_x"][ix + 1]
            y0, y1 = grid["edges_y"][iy], grid["edges_y"][iy + 1]
            expected, _ = integrate.dblquad(
                lambda y, x: pdf(x, y), x0, x1, y0, y1, epsabs=1e-10, epsrel=1e-8
            )
            assert M[m] == pytest.approx(expected, abs=1e-4, rel=1e-2)
