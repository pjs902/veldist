"""open_edges option of precompute_design_matrix_2d (export-only, used by the
DYNAMITE Laplace statistic): border cells integrate each star's kernel to
+-infinity, matching DYNAMITE's orbit histograms, which pile out-of-range
velocities into the border cells."""

from pathlib import Path

import numpy as np

from veldist.veldist2d import precompute_design_matrix_2d, setup_grid_2d

REF = Path(__file__).parent / "data" / "design2d_closed_ref.npy"


def _stars():
    # inside, near the edge, and 10 sigma off-grid; half diagonal, half correlated
    pm1 = np.array([0.0, 7.0, 19.0, 21.0, 60.0, -3.0, 18.5, -60.0])
    pm2 = np.array([0.0, -4.0, 2.0, -20.5, 0.0, 5.0, 19.0, -60.0])
    s = np.array([1.5, 2.0, 1.0, 1.5, 2.0, 3.0, 1.0, 2.5])
    cov = np.zeros((8, 2, 2))
    cov[:, 0, 0] = s**2
    cov[:, 1, 1] = (1.3 * s) ** 2
    cov[4:, 0, 1] = cov[4:, 1, 0] = 0.4 * s[4:] * 1.3 * s[4:]
    return pm1, pm2, cov


def _grid():
    return setup_grid_2d(center=(0.0, 0.0), width=(40.0, 40.0), n_bins=9)


def test_open_edges_rows_sum_to_one():
    pm1, pm2, cov = _stars()
    M = precompute_design_matrix_2d(pm1, pm2, cov, _grid(), open_edges=True)
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-6)


def test_open_edges_interior_unchanged():
    pm1, pm2, cov = _stars()
    g = _grid()
    Mo = precompute_design_matrix_2d(pm1, pm2, cov, g, open_edges=True).reshape(-1, 9, 9)
    Mc = precompute_design_matrix_2d(pm1, pm2, cov, g).reshape(-1, 9, 9)
    np.testing.assert_array_equal(Mo[:, 1:-1, 1:-1], Mc[:, 1:-1, 1:-1])


def test_closed_default_unchanged():
    pm1, pm2, cov = _stars()
    M = precompute_design_matrix_2d(pm1, pm2, cov, _grid())
    np.testing.assert_array_equal(M, np.load(REF))
