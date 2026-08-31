import numpy as np
import pytest

from veldist.calibration2d import GAIA_OUTER, RecoveryCurve2D, coverage_floor


def _rows(n_stars_cov_pairs, metric="rho", truth="anisotropic"):
    return [
        {
            "n_stars": n,
            "truth": truth,
            "metric": metric,
            "bias": 0.0,
            "coverage": cov,
            "ci_width": 1.0,
            "cr_bound": 1.0,
        }
        for n, cov in n_stars_cov_pairs
    ]


def test_threshold_returns_first_sufficient_n_stars():
    curve = RecoveryCurve2D(
        profile=GAIA_OUTER, truth_name="anisotropic",
        rows=_rows([(100, 0.20), (400, 0.45), (800, 0.70), (1600, 0.72)]),
    )
    assert curve.threshold("rho") == pytest.approx(800)


def test_threshold_requires_all_higher_n_stars_to_also_pass():
    curve = RecoveryCurve2D(
        profile=GAIA_OUTER, truth_name="anisotropic",
        rows=_rows([(100, 0.70), (400, 0.30), (800, 0.70)]),
    )
    assert curve.threshold("rho") == pytest.approx(800)


def test_threshold_is_none_when_never_calibrated():
    curve = RecoveryCurve2D(
        profile=GAIA_OUTER, truth_name="anisotropic",
        rows=_rows([(100, 0.2), (800, 0.3)]),
    )
    assert curve.threshold("rho") is None


def test_threshold_rejects_inefficient_points():
    rows = _rows([(800, 0.70)])
    rows[0]["ci_width"] = 10.0
    rows[0]["cr_bound"] = 1.0
    curve = RecoveryCurve2D(profile=GAIA_OUTER, truth_name="anisotropic", rows=rows)
    assert curve.threshold("rho", max_ci_ratio=1.5) is None


def test_threshold_is_none_when_top_n_stars_fails():
    curve = RecoveryCurve2D(
        profile=GAIA_OUTER, truth_name="anisotropic",
        rows=_rows([(100, 0.20), (800, 0.70), (3200, 0.20)]),
    )
    assert curve.threshold("rho") is None


def test_threshold_unknown_metric_raises():
    curve = RecoveryCurve2D(
        profile=GAIA_OUTER, truth_name="anisotropic", rows=_rows([(800, 0.7)])
    )
    with pytest.raises(ValueError, match="no rows for metric"):
        curve.threshold("sigma_x")


def test_threshold_uses_coverage_floor_from_n_real():
    n_real = 25
    floor = coverage_floor(n_real, band=0.99)
    curve = RecoveryCurve2D(
        profile=GAIA_OUTER, truth_name="anisotropic",
        rows=_rows([(800, floor + 0.01)]), n_real=n_real,
    )
    assert curve.threshold("rho") == pytest.approx(800)

    curve_low = RecoveryCurve2D(
        profile=GAIA_OUTER, truth_name="anisotropic",
        rows=_rows([(800, floor - 0.2)]), n_real=n_real,
    )
    assert curve_low.threshold("rho") is None


def test_report_mentions_profile_and_truth_name():
    curve = RecoveryCurve2D(
        profile=GAIA_OUTER, truth_name="anisotropic",
        rows=_rows([(800, 0.70)]),
    )
    r = curve.report()
    assert GAIA_OUTER.name in r
    assert "anisotropic" in r
    assert "rho" in r


@pytest.mark.slow
def test_recovery_curve_2d_smoke():
    """Tiny, fast-as-possible real run: confirms the function executes end
    to end and returns a well-formed RecoveryCurve2D, not that any
    particular n_stars threshold holds (that requires a real campaign)."""
    from veldist.calibration2d import GAIA_OUTER, recovery_curve_2d

    curve = recovery_curve_2d(
        profile=GAIA_OUTER,
        truth_name="isotropic",
        n_stars_values=[100],
        n_real=2,
        num_warmup=50,
        num_samples=100,
    )

    assert curve.profile is GAIA_OUTER
    assert curve.truth_name == "isotropic"
    assert curve.n_real == 2
    metrics = {r["metric"] for r in curve.rows}
    assert metrics == {"mean_x", "mean_y", "sigma_x", "sigma_y", "rho"}
    for r in curve.rows:
        assert r["n_stars"] == 100
        assert np.isfinite(r["coverage"])
        assert np.isfinite(r["ci_width"])
        assert np.isfinite(r["cr_bound"])
        assert np.isfinite(r["bias"])
