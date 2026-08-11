import pytest

from veldist.calibration import (
    NOMINAL_BAND,
    OMEGACAT,
    RecoveryCurve,
    coverage_floor,
    make_truths,
    recovery_curve,
)


def _rows(ivar_cov_pairs, metric="sigma", truth="gaussian"):
    return [
        {
            "ivar": iv,
            "truth": truth,
            "metric": metric,
            "bias": 0.0,
            "coverage": cov,
            "ci_width": 1.0,
            "cr_bound": 1.0,
            "baseline_ci_width": 1.0,
        }
        for iv, cov in ivar_cov_pairs
    ]


def test_threshold_returns_first_sufficient_ivar():
    curve = RecoveryCurve(
        profile=OMEGACAT, sigma=20.0, rows=_rows([(0.1, 0.20), (0.5, 0.45), (1.0, 0.70), (2.0, 0.72)])
    )
    assert curve.threshold("sigma") == pytest.approx(1.0)


def test_threshold_requires_all_higher_ivar_to_also_pass():
    """A single lucky point at low ivar is not a threshold."""
    curve = RecoveryCurve(profile=OMEGACAT, sigma=20.0, rows=_rows([(0.1, 0.70), (0.5, 0.30), (1.0, 0.70)]))
    assert curve.threshold("sigma") == pytest.approx(1.0)


def test_threshold_is_none_when_never_calibrated():
    curve = RecoveryCurve(profile=OMEGACAT, sigma=20.0, rows=_rows([(0.1, 0.2), (1.0, 0.3)]))
    assert curve.threshold("sigma") is None


def test_threshold_rejects_inefficient_points():
    """Coverage alone is not enough: an interval 10x too wide covers
    everything and tells you nothing."""
    rows = _rows([(1.0, 0.70)])
    rows[0]["ci_width"] = 10.0
    rows[0]["cr_bound"] = 1.0
    curve = RecoveryCurve(profile=OMEGACAT, sigma=20.0, rows=rows)
    assert curve.threshold("sigma", max_ci_ratio=1.5) is None


def test_threshold_is_none_when_top_ivar_fails():
    """A non-contiguous passing set where the top ivar fails is not a threshold.

    ivars [0.1, 1.0, 4.0] with ok = [False, True, False]: 1.0 passes on its
    own, but the top point (4.0) fails, so the run of passes from the top is
    broken immediately and the correct answer is None, not the passing 1.0.
    """
    curve = RecoveryCurve(profile=OMEGACAT, sigma=20.0, rows=_rows([(0.1, 0.20), (1.0, 0.70), (4.0, 0.20)]))
    assert curve.threshold("sigma") is None


def test_threshold_unknown_metric_raises():
    curve = RecoveryCurve(profile=OMEGACAT, sigma=20.0, rows=_rows([(1.0, 0.7)]))
    with pytest.raises(ValueError, match="no rows for metric"):
        curve.threshold("nonexistent")


def test_report_mentions_every_metric():
    curve = RecoveryCurve(
        profile=OMEGACAT,
        sigma=20.0,
        rows=_rows([(1.0, 0.7)], metric="v_mean") + _rows([(1.0, 0.7)], metric="sigma"),
    )
    text = curve.report()
    assert "v_mean" in text
    assert "sigma" in text


def test_report_annotates_edge_pinned_threshold():
    """An edge-pinned threshold (at the bottom or top of the swept range) is
    flagged in the report, since it means the sweep did not bracket the true
    value."""
    curve = RecoveryCurve(profile=OMEGACAT, sigma=20.0, rows=_rows([(0.1, 0.70), (1.0, 0.70)]))
    text = curve.report()
    assert "at the bottom of the swept range" in text


def test_coverage_floor_matches_nominal_band_at_n25():
    """coverage_floor(25) must reproduce the repo's own documented
    NOMINAL_BAND[0] = 0.440, since that constant is the established
    convention this helper generalises."""
    assert coverage_floor(25) == pytest.approx(NOMINAL_BAND[0], abs=1e-9)


def test_coverage_floor_at_n40():
    assert coverage_floor(40) == pytest.approx(0.475, abs=1e-3)


def test_threshold_explicit_min_coverage_overrides_n_real():
    curve = RecoveryCurve(profile=OMEGACAT, sigma=20.0, rows=_rows([(1.0, 0.55), (2.0, 0.55)]), n_real=40)
    # 0.55 is below the n_real=40 binomial floor (~0.475 is below it, so this
    # would pass) -- use a value that fails only the explicit floor.
    assert curve.threshold("sigma", min_coverage=0.60) is None
    assert curve.threshold("sigma") == pytest.approx(1.0)


def test_threshold_falls_back_to_060_when_n_real_is_none():
    curve = RecoveryCurve(profile=OMEGACAT, sigma=20.0, rows=_rows([(1.0, 0.55)]))
    assert curve.threshold("sigma") is None
    curve2 = RecoveryCurve(profile=OMEGACAT, sigma=20.0, rows=_rows([(1.0, 0.62)]))
    assert curve2.threshold("sigma") == pytest.approx(1.0)


def test_cell_below_060_but_inside_band_passes_with_n_real_set():
    """The regression this whole fix exists for: a coverage value below the
    old fixed 0.60 floor, but inside the proper binomial band for n_real=40,
    must now PASS when n_real is set, and must still FAIL under the old
    fixed min_coverage=0.60."""
    coverage = 0.55  # below 0.60, above coverage_floor(40) ~= 0.475
    curve = RecoveryCurve(profile=OMEGACAT, sigma=20.0, rows=_rows([(1.0, coverage)]), n_real=40)
    assert curve.threshold("sigma") == pytest.approx(1.0)
    assert curve.threshold("sigma", min_coverage=0.60) is None


def test_report_mentions_coverage_floor():
    curve = RecoveryCurve(profile=OMEGACAT, sigma=20.0, rows=_rows([(1.0, 0.7)]), n_real=40)
    text = curve.report()
    assert "coverage floor" in text


@pytest.mark.slow
def test_recovery_curve_v_mean_threshold_is_finite():
    """The real sweep, cut down to something that finishes in minutes.

    Asserts only that a v_mean threshold exists and is not absurd. The
    scientific result is the printed table, produced by running this at full
    size (see the module docstring), not this assertion.
    """
    truths = [t for t in make_truths() if t.name == "gaussian"]
    curve = recovery_curve(
        OMEGACAT,
        truths,
        ivar_values=[0.25, 1.0, 4.0],
        sigma=20.0,
        n_real=12,
        num_warmup=200,
        num_samples=400,
    )
    t_vmean = curve.threshold("v_mean")

    assert t_vmean is not None
    assert 0.1 < t_vmean < 100.0
    print(curve.report())
