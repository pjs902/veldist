"""The 2D grid recommendation, and the measured profiles behind it.

These pin the SHAPE of the rule (which quantity drives which dimension) and
the measured constants, both of which have been wrong before in ways that
looked fine: the profiles were hand-picked guesses for months, and
cell_per_sigma carried a value derived from a buggy estimator.
"""

import dataclasses

import pytest

from veldist.calibration2d import (
    GAIA_OUTER_MEASURED,
    HST_MEASURED,
    ObservingProfile2D,
    recommend_grid_2d,
)

MEASURED = [GAIA_OUTER_MEASURED, HST_MEASURED]


@pytest.mark.parametrize("profile", MEASURED, ids=lambda p: p.name)
def test_extent_covers_the_widest_bin_plus_rotation(profile):
    """One shared grid serves every spatial bin, so it must hold the widest
    LOSVD AND the mean-velocity offset of the fastest-rotating bins."""
    needed = 2.0 * profile.n_sigma_grid * profile.sigma_hi + profile.rotation_span
    assert recommend_grid_2d(profile)["width"] == pytest.approx(needed)


@pytest.mark.parametrize("profile", MEASURED, ids=lambda p: p.name)
def test_extent_uses_sigma_max_not_sigma_ref(profile):
    """Sizing on the median dispersion would truncate the widest bins.

    Guards the specific regression: sigma_ref sits between sigma_min and
    sigma_max, so a grid built on it looks plausible and clips silently.
    """
    on_ref = dataclasses.replace(profile, sigma_min=None, sigma_max=None)
    assert recommend_grid_2d(profile)["width"] > recommend_grid_2d(on_ref)["width"]


@pytest.mark.parametrize("profile", MEASURED, ids=lambda p: p.name)
def test_rotation_span_is_measured_not_zero(profile):
    """Defaulting rotation_span to 0 asserts the cluster does not rotate, in
    a pipeline whose purpose is measuring its rotation."""
    assert profile.rotation_span > 5.0


@pytest.mark.parametrize("profile", MEASURED, ids=lambda p: p.name)
def test_n_bins_is_odd(profile):
    """DYNAMITE's ProperMotions reader raises ValueError on even counts."""
    assert recommend_grid_2d(profile)["n_bins"] % 2 == 1


def test_cell_width_is_set_by_the_narrowest_bin():
    """Resolution must serve sigma_min: the widest bin is easy, the narrowest
    is what a shared grid struggles to resolve."""
    p = dataclasses.replace(GAIA_OUTER_MEASURED, sigma_min=4.0)
    assert p.cell_width < GAIA_OUTER_MEASURED.cell_width


def test_errors_dominating_is_warned_not_hidden():
    """When the measurement errors exceed what the cells resolve, no grid
    choice fixes it -- the caller has to be told rather than handed a grid
    that looks fine."""
    p = dataclasses.replace(GAIA_OUTER_MEASURED, err_median=50.0)
    warnings = recommend_grid_2d(p)["warnings"]
    assert any("errors dominate" in w for w in warnings)


def test_sparse_grid_is_warned_against_the_measured_floor():
    """0.70 stars/cell is the sparsest point the 2026-09-01 sweep actually
    measured; below that the recommendation is extrapolation."""
    p = dataclasses.replace(GAIA_OUTER_MEASURED, n_stars=50)
    assert any("extrapolation" in w for w in recommend_grid_2d(p)["warnings"])


def test_clean_profile_warns_about_nothing():
    """A profile inside measured territory should come back silent, or the
    warnings are noise and will be ignored when they matter."""
    p = ObservingProfile2D(
        name="clean",
        sigma_ref=16.0,
        err_median=1.5,
        err_cut=8.0,
        n_stars=2000,
        sigma_min=14.0,
        sigma_max=18.0,
        rotation_span=10.0,
    )
    assert recommend_grid_2d(p)["warnings"] == []
