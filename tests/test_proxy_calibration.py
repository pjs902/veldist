import numpy as np
import pytest

from veldist.calibration import OMEGACAT, make_truths, measure_proxy_to_gh


def test_proxy_to_gh_slopes_are_positive_and_finite():
    """Bowley skew and h3 share a sign convention, as do excess Moors
    kurtosis and h4, so both slopes must be positive. A negative slope means
    a convention has been broken somewhere."""
    result = measure_proxy_to_gh(
        make_truths(),
        sigma=OMEGACAT.sigma_max,
        n_bins=OMEGACAT.n_bins,
        grid_width=OMEGACAT.grid_width,
    )

    skew_slope, skew_scatter = result["skew_pct_to_h3"]
    kurt_slope, kurt_scatter = result["kurtosis_pct_to_h4"]

    assert skew_slope > 0
    assert kurt_slope > 0
    assert np.isfinite(skew_scatter)
    assert np.isfinite(kurt_scatter)


def test_proxy_to_gh_is_deterministic():
    """No sampling involved: truths are analytic, so the mapping must be
    reproducible to the last digit."""
    kwargs = {"sigma": OMEGACAT.sigma_max, "n_bins": OMEGACAT.n_bins, "grid_width": OMEGACAT.grid_width}
    a = measure_proxy_to_gh(make_truths(), **kwargs)
    b = measure_proxy_to_gh(make_truths(), **kwargs)

    assert a["skew_pct_to_h3"][0] == b["skew_pct_to_h3"][0]


def test_proxy_to_gh_needs_at_least_two_truths():
    with pytest.raises(ValueError, match="at least 2"):
        measure_proxy_to_gh(
            make_truths()[:1],
            sigma=OMEGACAT.sigma_max,
            n_bins=OMEGACAT.n_bins,
            grid_width=OMEGACAT.grid_width,
        )


def test_kurtosis_slope_matches_realistic_amplitude_ratio():
    """A few large-amplitude, strongly non-Gaussian truths (h4 outside the
    envelope) must not be allowed to set the slope: it should track the
    per-truth ratio seen among the small-amplitude truths that resemble real
    data, not an origin-fit dominated by outliers."""
    from veldist.analysis import compute_percentile_summary, gauss_hermite_fit

    truths = make_truths()
    sigma, n_bins, grid_width = OMEGACAT.sigma_max, OMEGACAT.n_bins, OMEGACAT.grid_width
    max_h4 = 0.10

    edges = np.linspace(-grid_width / 2.0, grid_width / 2.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ratios = []
    for t in truths:
        pdf, _ = t.scaled(sigma)
        mass = np.asarray(pdf(centers), dtype=float)
        mass = mass / mass.sum()
        row = np.tile(mass[None, :], (2, 1))
        pct = compute_percentile_summary(row, centers)
        gh = gauss_hermite_fit(row, centers, n_draws=2)
        h4, x = gh["h4"][0], pct["kurtosis_pct"][0]
        if np.isfinite(h4) and abs(h4) <= max_h4 and abs(x) > 1e-3:
            ratios.append(h4 / x)

    median_ratio = float(np.median(ratios))

    result = measure_proxy_to_gh(truths, sigma=sigma, n_bins=n_bins, grid_width=grid_width, max_h4=max_h4)
    slope, _ = result["kurtosis_pct_to_h4"]

    assert 0.5 < slope / median_ratio < 1.5


def test_out_of_envelope_truth_is_excluded():
    """bimodal_counter_rotation's h4 is far outside the default envelope. If
    it were included in the h4 fit, its large amplitude would dominate the
    origin-fit slope, so the slope must be identical with and without it."""
    truths = make_truths()
    without_bimodal = [t for t in truths if t.name != "bimodal_counter_rotation"]
    kwargs = {"sigma": OMEGACAT.sigma_max, "n_bins": OMEGACAT.n_bins, "grid_width": OMEGACAT.grid_width}

    with_it = measure_proxy_to_gh(truths, **kwargs)
    without_it = measure_proxy_to_gh(without_bimodal, **kwargs)

    assert with_it["kurtosis_pct_to_h4"][0] == without_it["kurtosis_pct_to_h4"][0]
