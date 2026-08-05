"""Unit tests for the 2D proper-motion observing profiles."""

import numpy as np
import pytest

from veldist.calibration2d import (
    GAIA_OUTER,
    HST_BRIGHT,
    HST_FAINT,
    PROFILES_2D,
    ObservingProfile2D,
)


@pytest.mark.parametrize("profile", [HST_BRIGHT, HST_FAINT, GAIA_OUTER])
def test_n_bins_is_always_odd(profile):
    """DYNAMITE's ProperMotions reader raises ValueError on even bin counts,
    so an even n_bins is unshippable regardless of how it measures."""
    assert profile.n_bins % 2 == 1, f"{profile.name}: n_bins={profile.n_bins} is even"
    assert profile.n_bins >= 5


def test_n_bins_is_odd_for_every_cell_per_sigma():
    """The rounding must land on odd for any cell size, not just the defaults."""
    for cps in np.arange(0.30, 1.01, 0.01):
        p = ObservingProfile2D(name="t", sigma_ref=17.0, err_median=1.0,
                               err_cut=6.4, n_stars=400, cell_per_sigma=float(cps))
        assert p.n_bins % 2 == 1, f"cell_per_sigma={cps:.2f} gave {p.n_bins}"


def test_grid_width_spans_the_requested_sigma_range():
    p = HST_BRIGHT
    assert p.grid_width == pytest.approx(2 * p.n_sigma_grid * p.sigma_ref)


def test_err_over_sigma_matches_the_documented_regimes():
    """These are the numbers the whole 2D calibration argument rests on."""
    assert HST_BRIGHT.err_over_sigma == pytest.approx(0.014, abs=0.002)
    assert HST_FAINT.err_over_sigma == pytest.approx(0.147, abs=0.002)
    assert GAIA_OUTER.err_over_sigma == pytest.approx(0.625, abs=0.002)


@pytest.mark.parametrize("profile", [HST_BRIGHT, HST_FAINT, GAIA_OUTER])
def test_draw_errors_respects_the_quality_cut(profile):
    rng = np.random.default_rng(0)
    e = profile.draw_errors(5000, rng)
    assert e.shape == (5000,)
    assert (e > 0).all()
    assert (e <= profile.err_cut).all(), "errors must not exceed the quality cut"
    # The median should land near the stated one (log-normal, so use the median
    # not the mean, and allow for the upper truncation pulling it down).
    assert np.median(e) == pytest.approx(profile.err_median, rel=0.25)


def test_profiles_2d_registry_is_consistent():
    assert set(PROFILES_2D) == {"hst_bright", "hst_faint", "gaia_outer"}
    for key, p in PROFILES_2D.items():
        assert p.name == key


def test_report_mentions_the_numbers_that_matter():
    r = HST_BRIGHT.report()
    for token in ["hst_bright", "err/sigma", "n_bins", "stars"]:
        assert token in r
