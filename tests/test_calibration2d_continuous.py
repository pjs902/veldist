"""Tests for the continuous calibration statistics (rms_z, mean_abs_z,
mean_z) and RecoveryCurve2D.aa_noise -- see calibration2d.py's docstrings
for the noise-floor rationale.

All rows are hand-built dataclass instances; no inference is run.
"""

import numpy as np
import pytest

from veldist.calibration2d import GAIA_OUTER, RecoveryCurve2D


def _row(n_stars, metric, z, truth="isotropic", **overrides):
    z = list(z)
    finite = np.asarray([np.isfinite(v) for v in z])
    z_ok = np.asarray(z, dtype=float)[finite]
    if z_ok.size > 0:
        rms_z = float(np.sqrt(np.mean(z_ok**2)))
        mean_abs_z = float(np.mean(np.abs(z_ok)))
        mean_z = float(np.mean(z_ok))
    else:
        rms_z = mean_abs_z = mean_z = float("nan")
    row = {
        "n_stars": float(n_stars),
        "truth": truth,
        "metric": metric,
        "bias": 0.0,
        "coverage": 0.5,
        "ci_width": 1.0,
        "cr_bound": 1.0,
        "hits": [True] * len(z),
        "z": z,
        "rms_z": rms_z,
        "mean_abs_z": mean_abs_z,
        "mean_z": mean_z,
        "n_z_excluded": int(np.sum(~finite)),
    }
    row.update(overrides)
    return row


def test_rms_z_mean_abs_z_mean_z_computed_correctly():
    z = [1.0, -1.0, 2.0, -2.0]
    row = _row(100, "sigma_x", z)
    assert row["rms_z"] == pytest.approx(np.sqrt((1 + 1 + 4 + 4) / 4))
    assert row["mean_abs_z"] == pytest.approx((1 + 1 + 2 + 2) / 4)
    assert row["mean_z"] == pytest.approx(0.0)
    assert row["n_z_excluded"] == 0


def test_rms_z_near_one_for_standard_normal_like_samples():
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 1.0, size=5000).tolist()
    row = _row(100, "sigma_x", z)
    assert row["rms_z"] == pytest.approx(1.0, abs=0.05)
    assert row["mean_abs_z"] == pytest.approx(np.sqrt(2 / np.pi), abs=0.05)
    assert row["mean_z"] == pytest.approx(0.0, abs=0.05)


def test_degenerate_half68_excluded_and_counted():
    # Simulates z values computed with zero/nan/inf half68: those entries
    # are nan in the z vector, excluded from the aggregates, and counted.
    z = [1.0, -1.0, float("nan"), float("inf"), 0.5]
    row = _row(100, "sigma_x", z)
    assert row["n_z_excluded"] == 2
    assert np.isfinite(row["rms_z"])
    assert np.isfinite(row["mean_abs_z"])
    assert np.isfinite(row["mean_z"])
    finite_z = np.array([1.0, -1.0, 0.5])
    assert row["rms_z"] == pytest.approx(np.sqrt(np.mean(finite_z**2)))


def test_all_degenerate_half68_gives_finite_nan_not_crash():
    z = [float("nan"), float("inf"), float("-inf")]
    row = _row(100, "sigma_x", z)
    assert row["n_z_excluded"] == 3
    assert np.isnan(row["rms_z"])
    assert np.isnan(row["mean_abs_z"])
    assert np.isnan(row["mean_z"])


def test_aa_noise_returns_correct_differences():
    row_a = _row(800, "sigma_x", [1.0, -1.0, 2.0], coverage=0.70, bias=0.10)
    row_b = _row(800, "sigma_y", [0.5, -0.5, 1.0], coverage=0.56, bias=0.05)
    curve = RecoveryCurve2D(
        profile=GAIA_OUTER, truth_name="isotropic", rows=[row_a, row_b],
    )
    result = curve.aa_noise("sigma_x", "sigma_y", 800)
    assert result["d_coverage"] == pytest.approx(abs(0.70 - 0.56))
    assert result["d_bias"] == pytest.approx(abs(0.10 - 0.05))
    assert result["d_rms_z"] == pytest.approx(abs(row_a["rms_z"] - row_b["rms_z"]))


def test_aa_noise_raises_on_non_isotropic_truth():
    row_a = _row(800, "sigma_x", [1.0, -1.0], truth="anisotropic")
    row_b = _row(800, "sigma_y", [1.0, -1.0], truth="anisotropic")
    curve = RecoveryCurve2D(
        profile=GAIA_OUTER, truth_name="anisotropic", rows=[row_a, row_b],
    )
    with pytest.raises(ValueError, match="isotropic"):
        curve.aa_noise("sigma_x", "sigma_y", 800)


def test_aa_noise_raises_when_row_missing():
    row_a = _row(800, "sigma_x", [1.0, -1.0])
    curve = RecoveryCurve2D(
        profile=GAIA_OUTER, truth_name="isotropic", rows=[row_a],
    )
    with pytest.raises(ValueError, match="no row"):
        curve.aa_noise("sigma_x", "sigma_y", 800)


def test_report_includes_rms_z_and_mean_z_columns():
    row = _row(800, "rho", [1.0, -1.0, 0.5], truth="isotropic")
    curve = RecoveryCurve2D(profile=GAIA_OUTER, truth_name="isotropic", rows=[row])
    r = curve.report()
    assert "rms_z" in r
    assert "mean_z" in r
