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
