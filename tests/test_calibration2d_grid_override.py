"""Unit tests for recovery_curve_2d's grid-override support.

Fast only -- no MCMC/NUTS. See calibration2d.py's recovery_curve_2d /
square_cell_grid / _resolve_grid / _validate_grid_override docstrings for
the rationale.
"""

import numpy as np
import pytest

from veldist.calibration2d import (
    HST_BRIGHT,
    _discretised_truth_moments,
    _resolve_grid,
    _validate_grid_override,
    square_cell_grid,
    truths_for,
)
from veldist.veldist2d import KinematicSolver2D, setup_grid_2d


# ---------------------------------------------------------------------------
# square_cell_grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sigma_ref,hx,hy,cell_sigma",
    [
        (17.0, 3.5, 2.0, 0.25),
        (8.0, 3.5, 2.0, 0.19),
        (1.0, 5.0, 5.0, 0.5),
        (10.0, 1.0, 4.0, 0.33),
    ],
)
def test_square_cell_grid_returns_exactly_square_cells(sigma_ref, hx, hy, cell_sigma):
    g = square_cell_grid(sigma_ref, hx, hy, cell_sigma)
    wx, wy = g["width"]
    kx, ky = g["n_bins"]
    assert wx / kx == pytest.approx(wy / ky, rel=1e-12)


@pytest.mark.parametrize(
    "sigma_ref,hx,hy,cell_sigma",
    [
        (17.0, 3.5, 2.0, 0.25),
        (8.0, 3.5, 2.0, 0.19),
        (1.0, 5.0, 5.0, 0.5),
    ],
)
def test_square_cell_grid_gives_odd_counts_both_axes(sigma_ref, hx, hy, cell_sigma):
    g = square_cell_grid(sigma_ref, hx, hy, cell_sigma)
    kx, ky = g["n_bins"]
    assert kx % 2 == 1
    assert ky % 2 == 1


def test_square_cell_grid_worked_case_a():
    """cell_sigma=0.25, half extents (3.5, 2.0) sigma."""
    g = square_cell_grid(1.0, 3.5, 2.0, 0.25)
    assert g["n_bins"] == (29, 17)


def test_square_cell_grid_worked_case_b():
    """cell_sigma=0.19, half extents (3.5, 2.0) sigma.

    Note: hand-derived expectation in the task was n_bins=(37, 21); this
    implementation (ceil(full_width/h) then round up to odd) gives (37, 23)
    -- see the report for the arithmetic. This test pins the implementation's
    actual, verified output, not the hand estimate.
    """
    g = square_cell_grid(1.0, 3.5, 2.0, 0.19)
    assert g["n_bins"] == (37, 23)


@pytest.mark.parametrize(
    "sigma_ref,hx,hy,cell_sigma",
    [
        (17.0, 3.5, 2.0, 0.25),
        (8.0, 3.5, 2.0, 0.19),
        (1.0, 5.0, 5.0, 0.5),
        (10.0, 1.0, 4.0, 0.33),
    ],
)
def test_square_cell_grid_extents_never_smaller_than_requested(sigma_ref, hx, hy, cell_sigma):
    g = square_cell_grid(sigma_ref, hx, hy, cell_sigma)
    wx, wy = g["width"]
    assert wx >= 2.0 * hx * sigma_ref - 1e-9
    assert wy >= 2.0 * hy * sigma_ref - 1e-9


# ---------------------------------------------------------------------------
# _validate_grid_override
# ---------------------------------------------------------------------------

def test_validate_grid_override_rejects_even_bin_count_scalar():
    with pytest.raises(ValueError, match="even"):
        _validate_grid_override({"width": 10.0, "n_bins": 10})


def test_validate_grid_override_rejects_even_bin_count_tuple():
    with pytest.raises(ValueError, match="even"):
        _validate_grid_override({"width": (10.0, 10.0), "n_bins": (11, 12)})


def test_validate_grid_override_rejects_nonpositive_width_scalar():
    with pytest.raises(ValueError, match="not positive"):
        _validate_grid_override({"width": 0.0, "n_bins": 11})


def test_validate_grid_override_rejects_nonpositive_width_tuple():
    with pytest.raises(ValueError, match="not positive"):
        _validate_grid_override({"width": (10.0, -1.0), "n_bins": (11, 13)})


def test_validate_grid_override_accepts_good_grid():
    # Should not raise.
    _validate_grid_override({"width": (49.0, 28.0), "n_bins": (37, 21)})


# ---------------------------------------------------------------------------
# _resolve_grid: default vs override consistency
# ---------------------------------------------------------------------------

def test_resolve_grid_default_matches_profile_derived_square_grid():
    profile = HST_BRIGHT
    center, width, n_bins = _resolve_grid(profile, None)
    assert center == (0.0, 0.0)
    assert width == (profile.grid_width, profile.grid_width)
    assert n_bins == profile.n_bins


def test_resolve_grid_override_is_used_verbatim():
    profile = HST_BRIGHT
    override = {"width": (49.0, 28.0), "n_bins": (37, 21)}
    center, width, n_bins = _resolve_grid(profile, override)
    assert center == (0.0, 0.0)
    assert width == (49.0, 28.0)
    assert n_bins == (37, 21)


def test_resolve_grid_override_runs_validation():
    with pytest.raises(ValueError, match="even"):
        _resolve_grid(HST_BRIGHT, {"width": (10.0, 10.0), "n_bins": (10, 11)})


def test_grid_plumbing_solver_and_truth_moments_share_identical_edges():
    """Without running any inference: verify that with a grid override, the
    edges used to build the KinematicSolver2D and the edges fed to
    _discretised_truth_moments are identical (byte-for-byte), for both a
    square (default-shaped) and a rectangular override grid.

    This mirrors exactly what recovery_curve_2d does internally (solver0 /
    the per-realisation solver / _discretised_truth_moments must all read
    from one place), just without the NUTS run.
    """
    profile = HST_BRIGHT
    truth = truths_for(profile.sigma_ref)["isotropic"]

    for grid_override in (None, {"width": (49.0, 28.0), "n_bins": (37, 21)}):
        center, width, n_bins = _resolve_grid(profile, grid_override)

        solver = KinematicSolver2D()
        solver.setup_grid(center=center, width=width, n_bins=n_bins)
        edges_x = solver.grid["edges_x"]
        edges_y = solver.grid["edges_y"]
        centers_2d = solver.grid["centers_2d"]

        # Independently constructed grid (as the per-realisation solver in
        # the loop would build), using the same resolved inputs.
        grid2 = setup_grid_2d(center=center, width=width, n_bins=n_bins)

        np.testing.assert_array_equal(edges_x, grid2["edges_x"])
        np.testing.assert_array_equal(edges_y, grid2["edges_y"])
        np.testing.assert_array_equal(centers_2d, grid2["centers_2d"])

        # And _discretised_truth_moments must run without error against
        # these same edges, including for a rectangular (kx != ky) grid.
        moments, mass = _discretised_truth_moments(truth, edges_x, edges_y, centers_2d)
        assert mass.shape[0] == centers_2d.shape[0]
        assert np.isfinite(moments["sigma_x"])
        assert np.isfinite(moments["sigma_y"])
