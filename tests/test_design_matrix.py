"""Tests for veldist.veldist.precompute_design_matrix correctness.

See PLAN.md section 1.4 "Design-matrix correctness (fast)".
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from veldist.veldist import precompute_design_matrix


def _grid(center=0.0, width=10.0, n_bins=21):
    """Return (bin_centers, bin_width, edges) for a simple regular grid."""
    edges = center + np.linspace(-width / 2, width / 2, n_bins + 1)
    bin_width = edges[1] - edges[0]
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, bin_width, edges


def test_design_matrix_vs_quadrature():
    """Each M[i, j] should equal the exact Gaussian mass in [edge_j, edge_{j+1}]."""
    centers, bin_width, edges = _grid()

    obs_val = np.array([-3.2, 0.5, 2.1])
    obs_err = np.array([0.8, 1.5, 0.3])

    M = np.asarray(precompute_design_matrix(obs_val, obs_err, centers, bin_width))

    for i, (y, sig) in enumerate(zip(obs_val, obs_err)):
        for j in range(len(centers)):
            expected, _ = quad(
                lambda x: norm.pdf(x, loc=y, scale=sig),
                edges[j],
                edges[j + 1],
            )
            # function adds a 1e-30 floor on top of the true mass
            # jax computes in float32 by default (~1e-7 relative precision),
            # so we can't hit the plan's aspirational 1e-10; 1e-6 absolute
            # tolerance is appropriate for float32 while still catching any
            # bin-alignment (off-by-half-a-bin) error.
            assert M[i, j] == pytest.approx(expected, abs=1e-6)


def test_design_matrix_rows_sum_to_one_for_interior_stars():
    """A star well inside the grid with err << bin width should sum to ~1."""
    centers, bin_width, edges = _grid()

    obs_val = np.array([0.0, 1.3, -2.0])
    obs_err = np.array([0.05, 0.05, 0.05]) * bin_width  # tiny errors

    M = np.asarray(precompute_design_matrix(obs_val, obs_err, centers, bin_width))

    row_sums = M.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_design_matrix_small_error_limit():
    """err = 0.01 * bin_width should put >=99% of mass in the containing bin."""
    centers, bin_width, edges = _grid()

    # place a star exactly at a bin center so there's no ambiguity about
    # which bin "contains" it
    target_bin = 10
    obs_val = np.array([centers[target_bin]])
    obs_err = np.array([0.01 * bin_width])

    M = np.asarray(precompute_design_matrix(obs_val, obs_err, centers, bin_width))

    assert M[0, target_bin] >= 0.99


def test_design_matrix_offgrid_star_has_negligible_mass():
    """A star far outside the grid range has near-zero (epsilon-floor) mass everywhere.

    This documents the grid-truncation behavior: precompute_design_matrix does
    not extend the grid to cover the star's support, so the row sum for an
    off-grid star with a small error is essentially zero (just the N * 1e-30
    epsilon floor), not 1. Callers must ensure the grid covers the data.
    """
    centers, bin_width, edges = _grid()

    # far outside [edges[0], edges[-1]], with an error much smaller than the
    # distance to the grid, so essentially zero real Gaussian mass falls in
    # any bin
    obs_val = np.array([1000.0])
    obs_err = np.array([1.0])

    M = np.asarray(precompute_design_matrix(obs_val, obs_err, centers, bin_width))

    row_sum = M[0].sum()
    n_bins = len(centers)
    # essentially just the epsilon floor accumulated over all bins
    assert row_sum < n_bins * 1e-25
    assert row_sum >= 0.0
