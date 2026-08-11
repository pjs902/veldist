import numpy as np
import pytest

from veldist.baseline import gaussian_mle


def test_gaussian_mle_recovers_known_parameters():
    """With errors much smaller than sigma, the MLE must find the truth."""
    rng = np.random.default_rng(42)
    n = 20000
    true_v, true_sigma = 12.0, 20.0
    err = np.full(n, 2.0)
    obs = rng.normal(true_v, true_sigma, size=n) + rng.normal(0.0, err)

    fit = gaussian_mle(obs, err)

    # 4-sigma tolerance on the analytic standard errors.
    assert fit["v_mean"] == pytest.approx(true_v, abs=4 * true_sigma / np.sqrt(n))
    assert fit["sigma"] == pytest.approx(true_sigma, abs=4 * true_sigma / np.sqrt(2 * n))


def test_gaussian_mle_deconvolves_errors():
    """The point of deconvolution: recover the *intrinsic* sigma when the
    measurement errors are comparable to it. A naive np.std would return
    sqrt(sigma^2 + err^2) = sqrt(200) ~ 14.1 here, not 10."""
    rng = np.random.default_rng(7)
    n = 20000
    true_sigma, err_scale = 10.0, 10.0
    err = np.full(n, err_scale)
    obs = rng.normal(0.0, true_sigma, size=n) + rng.normal(0.0, err)

    fit = gaussian_mle(obs, err)

    assert fit["sigma"] == pytest.approx(true_sigma, rel=0.05)
    assert np.std(obs) > 13.0  # the naive estimate really is badly biased


def test_gaussian_mle_errors_match_cramer_rao():
    """Reported uncertainties must match the analytic bounds for equal errors."""
    rng = np.random.default_rng(3)
    n = 5000
    true_sigma = 15.0
    err = np.full(n, 1.0)
    obs = rng.normal(0.0, true_sigma, size=n) + rng.normal(0.0, err)

    fit = gaussian_mle(obs, err)
    total = np.sqrt(true_sigma**2 + 1.0)

    assert fit["v_mean_err"] == pytest.approx(total / np.sqrt(n), rel=0.1)
    assert fit["sigma_err"] == pytest.approx(total / np.sqrt(2 * n), rel=0.15)


def test_gaussian_mle_handles_zero_intrinsic_dispersion():
    """sigma is bounded below at 0; a delta-function truth must not crash or
    return a negative sigma."""
    rng = np.random.default_rng(11)
    n = 500
    err = np.full(n, 3.0)
    obs = np.zeros(n) + rng.normal(0.0, err)

    fit = gaussian_mle(obs, err)

    assert fit["sigma"] >= 0.0
    assert np.isfinite(fit["v_mean"])


def test_gaussian_mle_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        gaussian_mle(np.zeros(5), np.ones(4))
