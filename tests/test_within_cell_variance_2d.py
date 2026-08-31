"""Fast, numpy/analytic-only tests for the within-cell-variance fix.

Verifies the three-way bookkeeping fix described in
``calibration2d._moments_from_pdf_samples_2d`` and
``calibration2d._discretised_truth_moments``:

1. The likelihood implicitly spreads each cell's probability mass
   uniformly across the cell (piecewise-constant assumption), so the
   continuous variance it fits is ``sum p (v-mu)^2 + h^2/12``.
2. ``_moments_from_pdf_samples_2d`` must now report that same continuous
   quantity (point-mass moment + ``h^2/12`` per axis), not the raw
   point-mass moment.
3. ``_discretised_truth_moments`` must return the CONTINUOUS truth
   (grid-independent), not the Sheppard-inflated (``V + h^2/12``)
   cell-centre moment of the discretised truth mass.

No MCMC/NUTS anywhere in this file -- everything is exact arithmetic on a
known bivariate Gaussian and its exact cell masses.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from veldist.calibration2d import (
    _discretised_truth_moments,
    _moments_from_pdf_samples_2d,
)
from veldist.veldist2d import setup_grid_2d


def _exact_cell_masses(mean, cov, edges_x, edges_y):
    """Exact per-cell probability mass of a bivariate Gaussian, same
    convention (row-major, m = ix*ky + iy) as _discretised_truth_moments."""
    mvn = multivariate_normal(mean=mean, cov=cov)
    kx = len(edges_x) - 1
    ky = len(edges_y) - 1
    mass = np.empty(kx * ky)
    for ix in range(kx):
        for iy in range(ky):
            mass[ix * ky + iy] = (
                mvn.cdf([edges_x[ix + 1], edges_y[iy + 1]])
                - mvn.cdf([edges_x[ix], edges_y[iy + 1]])
                - mvn.cdf([edges_x[ix + 1], edges_y[iy]])
                + mvn.cdf([edges_x[ix], edges_y[iy]])
            )
    mass /= mass.sum()
    return mass


@pytest.mark.parametrize("K", [61, 101, 201])
def test_moments_from_pdf_samples_matches_sheppard_and_correction(K):
    """Exact cell masses of a known Gaussian: the WITHOUT-correction
    (raw point-mass) moment recovers V + h^2/12 (Sheppard), and the
    function as fixed (WITH the h^2/12 add-back) recovers V + h^2/6.

    Both are arithmetic facts about the estimator on exact cell masses --
    independent of any fit -- but Sheppard's h^2/12 is itself the leading
    term of an asymptotic (small-h) series (the next term is O(h^4)), so
    the match is only tight for a reasonably fine grid; hence the large K
    values and the 2% tolerance rather than exact equality.
    """
    sigma = 10.0
    mean = [0.0, 0.0]
    cov = [[sigma**2, 0.0], [0.0, sigma**2]]
    width = 7.0 * sigma
    grid = setup_grid_2d(center=(0.0, 0.0), width=(width, width), n_bins=K)
    centers_2d = grid["centers_2d"]
    h = grid["width_x"]
    assert grid["width_x"] == pytest.approx(grid["width_y"])

    mass = _exact_cell_masses(mean, cov, grid["edges_x"], grid["edges_y"])

    # Raw point-mass moment (what the corrected function would return
    # without the h^2/12 add-back): reconstruct it directly.
    cx, cy = centers_2d[:, 0], centers_2d[:, 1]
    mx = mass @ cx
    var_x_raw = mass @ (cx - mx) ** 2
    assert var_x_raw == pytest.approx(sigma**2 + h**2 / 12.0, rel=0.02)

    # Fixed function: point-mass moment + h^2/12 -> V + h^2/6.
    _, _, sigma_x, sigma_y, _ = _moments_from_pdf_samples_2d(mass[None, :], centers_2d, grid)
    var_x_corrected = float(sigma_x[0]) ** 2
    var_y_corrected = float(sigma_y[0]) ** 2
    assert var_x_corrected == pytest.approx(sigma**2 + h**2 / 6.0, rel=0.02)
    assert var_y_corrected == pytest.approx(sigma**2 + h**2 / 6.0, rel=0.02)


@pytest.mark.parametrize("K", [11, 25])
def test_correction_increases_sigma_by_expected_amount(K):
    """The h^2/12 add-back must INCREASE the reported sigma (not decrease
    or leave it unchanged), by approximately h^2/24/sigma for a
    well-resolved grid (first-order Taylor expansion of
    sqrt(V + h^2/12) around sqrt(V))."""
    sigma = 12.0
    mean = [1.0, -2.0]
    cov = [[sigma**2, 0.0], [0.0, sigma**2]]
    width = 7.0 * sigma
    grid = setup_grid_2d(center=(0.0, 0.0), width=(width, width), n_bins=K)
    centers_2d = grid["centers_2d"]
    h = grid["width_x"]

    mass = _exact_cell_masses(mean, cov, grid["edges_x"], grid["edges_y"])

    cx = centers_2d[:, 0]
    mx = mass @ cx
    sigma_x_raw = np.sqrt(mass @ (cx - mx) ** 2)

    _, _, sigma_x_corr, _, _ = _moments_from_pdf_samples_2d(mass[None, :], centers_2d, grid)
    sigma_x_corr = float(sigma_x_corr[0])

    delta = sigma_x_corr - sigma_x_raw
    assert delta > 0
    expected = h**2 / 24.0 / sigma
    assert delta == pytest.approx(expected, rel=0.05)


def test_rho_recomputed_from_corrected_variances():
    """rho must come from the CORRECTED (h^2/12-added) variances, not the
    raw point-mass ones -- correcting variances upward with unchanged
    covariance must reduce |rho|."""
    sx, sy, rho_true = 10.0, 6.0, 0.5
    mean = [0.0, 0.0]
    cov = [[sx**2, rho_true * sx * sy], [rho_true * sx * sy, sy**2]]
    K = 15
    width_x, width_y = 7.0 * sx, 7.0 * sy
    grid = setup_grid_2d(center=(0.0, 0.0), width=(width_x, width_y), n_bins=K)
    centers_2d = grid["centers_2d"]

    mass = _exact_cell_masses(mean, cov, grid["edges_x"], grid["edges_y"])

    cx, cy = centers_2d[:, 0], centers_2d[:, 1]
    mx, my = mass @ cx, mass @ cy
    var_x_raw = mass @ (cx - mx) ** 2
    var_y_raw = mass @ (cy - my) ** 2
    cov_xy = mass @ ((cx - mx) * (cy - my))
    rho_raw = cov_xy / np.sqrt(var_x_raw * var_y_raw)

    _, _, sigma_x, sigma_y, rho = _moments_from_pdf_samples_2d(mass[None, :], centers_2d, grid)
    rho = float(rho[0])

    assert abs(rho) < abs(rho_raw)
    # Directly recomputed from the corrected variances via the returned sigma.
    expected_rho = cov_xy / (float(sigma_x[0]) * float(sigma_y[0]))
    assert rho == pytest.approx(expected_rho, rel=1e-10)


@pytest.mark.parametrize("K", [9, 15, 25, 41])
def test_discretised_truth_moments_is_grid_independent(K):
    """The truth function must return the CONTINUOUS analytic moments --
    i.e. matching the analytic sx/sy/rho to tight tolerance regardless of
    K, with no grid-resolution dependence at all."""
    t = dict(mux=1.5, muy=-0.7, sx=9.0, sy=13.0, rho=0.3)
    sigma_ref = max(t["sx"], t["sy"])
    width = 7.0 * sigma_ref
    grid = setup_grid_2d(center=(0.0, 0.0), width=(width, width), n_bins=K)

    moments, mass = _discretised_truth_moments(t, grid["edges_x"], grid["edges_y"], grid["centers_2d"])

    assert moments["mean_x"] == pytest.approx(t["mux"], abs=1e-12)
    assert moments["mean_y"] == pytest.approx(t["muy"], abs=1e-12)
    assert moments["sigma_x"] == pytest.approx(t["sx"], abs=1e-12)
    assert moments["sigma_y"] == pytest.approx(t["sy"], abs=1e-12)
    assert moments["rho"] == pytest.approx(t["rho"], abs=1e-12)
    assert mass.shape[0] == K * K
    assert mass.sum() == pytest.approx(1.0, rel=1e-10)
