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
    # HST_BRIGHT: sigma_ref=17.0, n_sigma_grid=3.5 -> +/-3.5 sigma each side.
    # Hardcoded, not recomputed from the property's own formula, so a
    # regression in that formula would actually be caught.
    p = HST_BRIGHT
    assert p.grid_width == pytest.approx(119.0)


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
    assert set(PROFILES_2D) == {
        "hst_bright",
        "hst_faint",
        "gaia_outer",
        "gaia_outer_measured",
        "hst_measured",
    }
    for key, p in PROFILES_2D.items():
        assert p.name == key


def test_report_mentions_the_numbers_that_matter():
    r = HST_BRIGHT.report()
    for token in ["hst_bright", "err/sigma", "n_bins", "stars"]:
        assert token in r


def _synthetic_pm_field(rng, n_bins=30, n_per_bin=400, sigma_ref=17.0, err_median=1.0, err_cut=6.4):
    pm1, pm2, err1, err2, bin_ids = [], [], [], [], []
    for i in range(n_bins):
        e = np.clip(rng.lognormal(np.log(err_median), 0.3, n_per_bin), 1e-3, err_cut)
        p1 = rng.normal(0.0, sigma_ref, n_per_bin) + rng.normal(0.0, e)
        p2 = rng.normal(0.0, sigma_ref, n_per_bin) + rng.normal(0.0, e)
        pm1.append(p1)
        pm2.append(p2)
        err1.append(e)
        err2.append(e)
        bin_ids.append(np.full(n_per_bin, i))
    return (
        np.concatenate(pm1),
        np.concatenate(pm2),
        np.concatenate(err1),
        np.concatenate(err2),
        np.concatenate(bin_ids),
    )


def test_from_data_recovers_known_profile():
    rng = np.random.default_rng(20260831)
    pm1, pm2, err1, err2, bin_ids = _synthetic_pm_field(rng)

    p = ObservingProfile2D.from_data(pm1, pm2, err1, err2, bin_ids, err_cut=6.4, name="synthetic")

    assert p.sigma_ref == pytest.approx(17.0, rel=0.1)
    assert p.err_median == pytest.approx(1.0, rel=0.3)
    assert p.n_stars == 400
    assert p.err_cut == 6.4
    assert p.name == "synthetic"


def test_from_data_excludes_bins_below_min_stars_consistently():
    rng = np.random.default_rng(1)
    pm1, pm2, err1, err2, bin_ids = _synthetic_pm_field(rng, n_bins=5, n_per_bin=400)
    # Add one undersized bin that should be excluded from every statistic.
    small = np.full(3, 1000.0)
    pm1 = np.concatenate([pm1, small])
    pm2 = np.concatenate([pm2, small])
    err1 = np.concatenate([err1, np.full(3, 1.0)])
    err2 = np.concatenate([err2, np.full(3, 1.0)])
    bin_ids = np.concatenate([bin_ids, np.full(3, 99)])

    p = ObservingProfile2D.from_data(pm1, pm2, err1, err2, bin_ids, err_cut=6.4, min_stars=10)

    # The undersized bin's wildly offset values must not leak into sigma_ref.
    assert p.sigma_ref == pytest.approx(17.0, rel=0.1)


def test_from_data_rejects_too_few_surviving_bins():
    rng = np.random.default_rng(2)
    pm1, pm2, err1, err2, bin_ids = _synthetic_pm_field(rng, n_bins=1, n_per_bin=400)
    with pytest.raises(ValueError, match="at least 2 bins"):
        ObservingProfile2D.from_data(pm1, pm2, err1, err2, bin_ids, err_cut=6.4)
