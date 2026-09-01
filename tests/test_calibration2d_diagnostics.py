"""Unit tests for the 2D grid diagnostics (cells_per_sigma/extent_in_sigma)
and the paired McNemar comparison on RecoveryCurve2D.

Fast unit tests only -- no MCMC/NUTS sampling.
"""

import pytest

from veldist.calibration2d import HST_BRIGHT, ObservingProfile2D, RecoveryCurve2D


def test_cells_per_sigma_matches_sigma_ref_up_to_rounding():
    """cells_per_sigma(sigma_ref) should reproduce cell_per_sigma, modulo the
    odd-integer rounding baked into n_bins."""
    p = HST_BRIGHT
    assert p.cells_per_sigma(p.sigma_ref) == pytest.approx(
        p.grid_width / p.n_bins / p.sigma_ref
    )
    # Close to the nominal target, not exact due to rounding. Read the target
    # off cells_per_sigma_target, not the raw cell_per_sigma field: the latter
    # is None whenever the profile derives its resolution from its own error
    # regime, which is now the default.
    assert p.cells_per_sigma(p.sigma_ref) == pytest.approx(
        p.cells_per_sigma_target, rel=0.15
    )


def test_extent_in_sigma_matches_n_sigma_grid_at_sigma_ref():
    p = HST_BRIGHT
    assert p.extent_in_sigma(p.sigma_ref) == pytest.approx(p.n_sigma_grid)


def test_cells_per_sigma_hand_computed():
    p = ObservingProfile2D(
        name="t", sigma_ref=10.0, err_median=1.0, err_cut=6.4, n_stars=100,
        n_sigma_grid=3.5, cell_per_sigma=0.5,
    )
    cell_width = p.grid_width / p.n_bins
    assert p.cells_per_sigma(10.0) == pytest.approx(cell_width / 10.0)
    assert p.cells_per_sigma(5.0) == pytest.approx(cell_width / 5.0)
    assert p.extent_in_sigma(5.0) == pytest.approx(p.grid_width / 2.0 / 5.0)


def test_narrow_axis_is_coarser_wide_axis_is_finer():
    """Anisotropic truth: sx = 1.18*sigma (wide), sy = 0.76*sigma (narrow).
    cells_per_sigma is a cell WIDTH in that axis's sigma units, so a smaller
    number means finer resolution. The narrow axis (sy) should therefore get
    a LARGER cells_per_sigma (coarser) than sigma_ref does, and the wide
    axis (sx) a SMALLER one (finer)."""
    p = HST_BRIGHT
    sigma = p.sigma_ref
    sx, sy = 1.18 * sigma, 0.76 * sigma

    cps_ref = p.cells_per_sigma(sigma)
    cps_wide = p.cells_per_sigma(sx)
    cps_narrow = p.cells_per_sigma(sy)

    assert cps_narrow > cps_ref, "narrow axis should be coarser (larger cells/sigma)"
    assert cps_wide < cps_ref, "wide axis should be finer (smaller cells/sigma)"


def test_report_mentions_anisotropic_axis_diagnostics():
    r = HST_BRIGHT.report()
    assert "anisotropic truth" in r
    assert "x-axis" in r and "y-axis" in r


def _make_row(metric, n_stars, hits):
    return {
        "n_stars": float(n_stars),
        "truth": "isotropic",
        "metric": metric,
        "bias": 0.0,
        "coverage": sum(hits) / len(hits),
        "ci_width": 1.0,
        "cr_bound": 1.0,
        "hits": list(hits),
    }


def test_mcnemar_recovers_known_discordant_counts():
    # self hits, other misses -> b; other hits, self misses -> c.
    hits_self = [True, True, False, False, True, False, True, True]
    hits_other = [True, False, False, True, True, True, False, True]
    # discordant pairs: idx1 (T,F)->b, idx2 none, idx3 (F,T)->c, idx5(F,T)->c, idx6(T,F)->b
    row_self = _make_row("mean_x", 100, hits_self)
    row_other = _make_row("mean_x", 100, hits_other)

    curve_self = RecoveryCurve2D(profile=None, truth_name="isotropic", rows=[row_self], n_real=8)
    curve_other = RecoveryCurve2D(profile=None, truth_name="isotropic", rows=[row_other], n_real=8)

    b, c, p = curve_self.mcnemar(curve_other, "mean_x", 100)

    expected_b = sum(1 for hs, ho in zip(hits_self, hits_other) if hs and not ho)
    expected_c = sum(1 for hs, ho in zip(hits_self, hits_other) if ho and not hs)
    assert b == expected_b
    assert c == expected_c
    assert 0.0 <= p <= 1.0


def test_mcnemar_perfectly_symmetric_discordance_gives_p_one():
    hits_self = [True, False, True, False]
    hits_other = [False, True, False, True]
    row_self = _make_row("sigma_x", 200, hits_self)
    row_other = _make_row("sigma_x", 200, hits_other)

    curve_self = RecoveryCurve2D(profile=None, truth_name="isotropic", rows=[row_self], n_real=4)
    curve_other = RecoveryCurve2D(profile=None, truth_name="isotropic", rows=[row_other], n_real=4)

    b, c, p = curve_self.mcnemar(curve_other, "sigma_x", 200)
    assert b == 2
    assert c == 2
    assert p == pytest.approx(1.0)


def test_mcnemar_raises_on_mismatched_lengths():
    row_self = _make_row("mean_x", 100, [True, False, True])
    row_other = _make_row("mean_x", 100, [True, False])

    curve_self = RecoveryCurve2D(profile=None, truth_name="isotropic", rows=[row_self], n_real=3)
    curve_other = RecoveryCurve2D(profile=None, truth_name="isotropic", rows=[row_other], n_real=2)

    with pytest.raises(ValueError, match="length mismatch"):
        curve_self.mcnemar(curve_other, "mean_x", 100)


def test_mcnemar_raises_on_missing_n_stars():
    row_self = _make_row("mean_x", 100, [True, False, True])
    curve_self = RecoveryCurve2D(profile=None, truth_name="isotropic", rows=[row_self], n_real=3)
    curve_other = RecoveryCurve2D(profile=None, truth_name="isotropic", rows=[], n_real=3)

    with pytest.raises(ValueError, match="no row"):
        curve_self.mcnemar(curve_other, "mean_x", 100)

    with pytest.raises(ValueError, match="no row"):
        curve_self.mcnemar(curve_other, "mean_x", 999)
