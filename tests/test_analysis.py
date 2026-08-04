"""
Analytic unit tests for veldist.analysis.

These tests build exact probability mass functions (PMFs) on a fine velocity
grid by discretising known closed-form distributions (Gaussian, skew-normal,
Student-t, uniform, mixtures), then check `analysis.py`'s outputs against the
analytic values. No MCMC / sampling is involved anywhere in this file, so the
whole suite runs in well under a second.
"""

import numpy as np
import pytest
from scipy import stats

from veldist.analysis import (
    bimodality_score,
    cdf_percentile,
    compute_percentile_summary,
    compute_summary,
    compute_summary_maps,
    half_68ci,
    truncate_pdf_samples,
)

from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def analytic_pmf(dist, centers, width):
    """
    Discretise a scipy.stats distribution onto a fixed grid.

    Parameters
    ----------
    dist : scipy.stats frozen distribution (or object with .cdf)
    centers : array-like, shape (n_bins,)
        Bin centres (evenly spaced, spacing = width).
    width : float
        Bin width.

    Returns
    -------
    ndarray, shape (1, n_bins)
        Renormalised PMF as a single-row "sample set", ready to feed to
        compute_summary / compute_summary_maps stubs.
    """
    centers = np.asarray(centers, dtype=float)
    edges = np.concatenate([centers - width / 2.0, centers[-1:] + width / 2.0])
    pmf = np.diff(dist.cdf(edges))
    pmf = pmf / pmf.sum()
    return pmf[np.newaxis, :]


def make_grid(lo, hi, n_bins):
    width = (hi - lo) / n_bins
    centers = lo + width * (np.arange(n_bins) + 0.5)
    return centers, width


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


def test_moments_gaussian():
    dist = stats.norm(loc=12, scale=8)
    centers, width = make_grid(-60, 84, 4000)
    pmf = analytic_pmf(dist, centers, width)

    summary = compute_summary(pmf, centers)

    assert summary["v_mean"][0] == pytest.approx(12, abs=1e-2)
    assert summary["sigma"][0] == pytest.approx(8, abs=1e-2)
    assert summary["skewness"][0] == pytest.approx(0, abs=1e-2)
    assert summary["kurtosis"][0] == pytest.approx(0, abs=1e-2)


def test_moments_skewnormal():
    a = 4
    dist = stats.skewnorm(a=a, loc=0, scale=5)
    centers, width = make_grid(-40, 40, 4000)
    pmf = analytic_pmf(dist, centers, width)

    summary = compute_summary(pmf, centers)

    true_skew = dist.stats(moments="s")
    assert summary["skewness"][0] == pytest.approx(float(true_skew), rel=0.02)


def test_percentile_summary_gaussian():
    dist = stats.norm(loc=12, scale=8)
    centers, width = make_grid(-60, 84, 4000)
    pmf = analytic_pmf(dist, centers, width)

    summary = compute_percentile_summary(pmf, centers)

    assert summary["median"][0] == pytest.approx(12, rel=0.01)
    assert summary["sigma_pct"][0] == pytest.approx(8, rel=0.01)
    assert summary["skew_pct"][0] == pytest.approx(0, abs=1e-2)
    # Excess Moors kurtosis: zeroed on a Gaussian, like GH's h4.
    assert summary["kurtosis_pct"][0] == pytest.approx(0, abs=1e-2)


def test_percentile_summary_skewnormal_sign():
    dist = stats.skewnorm(a=4, loc=0, scale=5)
    centers, width = make_grid(-40, 40, 4000)
    pmf = analytic_pmf(dist, centers, width)

    summary = compute_percentile_summary(pmf, centers)

    # Right-skewed distribution -> positive Bowley skewness, same sign
    # convention as compute_summary's moment-based skewness.
    assert summary["skew_pct"][0] > 0


def test_percentile_summary_kurtosis_student_t_sign():
    dist = stats.t(df=6)
    centers, width = make_grid(-80, 80, 4000)
    pmf = analytic_pmf(dist, centers, width)

    summary = compute_percentile_summary(pmf, centers)

    # Heavy-tailed (leptokurtic) distribution -> positive excess kurtosis_pct,
    # same sign convention as compute_summary's moment-based kurtosis and h4.
    assert summary["kurtosis_pct"][0] > 0


def test_kurtosis_student_t():
    # Excess kurtosis of Student-t(nu) = 6/(nu-4) for nu > 4.
    nu = 6
    true_kurt = 6.0 / (nu - 4)
    dist = stats.t(df=nu)
    centers, width = make_grid(-80, 80, 4000)
    pmf = analytic_pmf(dist, centers, width)

    summary = compute_summary(pmf, centers)
    kurt = summary["kurtosis"][0]

    # The plan flags this as approximate: grid truncation (finite range,
    # finite bins) biases the estimate low relative to the true nu=6 value
    # of 3.0. We assert the correct sign (positive / leptokurtic) and
    # order-of-magnitude agreement rather than a tight percentage tolerance,
    # per the plan's own caveat.
    assert kurt > 0
    assert kurt == pytest.approx(true_kurt, rel=0.5)


# ---------------------------------------------------------------------------
# Tail weight
# ---------------------------------------------------------------------------


def test_tail_weight_gaussian():
    from math import erf, sqrt

    dist = stats.norm(loc=0, scale=1)
    centers, width = make_grid(-10, 10, 4000)
    pmf = analytic_pmf(dist, centers, width)

    summary = compute_summary(pmf, centers)
    expected = 1 - erf(1 / sqrt(2))
    assert summary["tail_weight"][0] == pytest.approx(expected, abs=1e-3)


def test_tail_weight_ordering():
    # NOTE: the plan's suggested ordering `tw(uniform) < tw(gauss) < tw(t6)`
    # does not hold numerically. tail_weight is the mass strictly *outside*
    # exactly 1 empirical sigma, which is a single-point cut, not a general
    # heavy-tail indicator. A t(6) distribution has a very concentrated peak
    # (more mass packed inside 1 sigma than a Gaussian) even though its
    # extreme tails beyond several sigma are much fatter; a hard-edged
    # uniform distribution has no concentration at all, so a larger fraction
    # of its mass sits outside 1 sigma than either. Verified directly via
    # scipy: gauss=0.3173, t(6)=0.2666, uniform=0.4226. We test the ordering
    # that actually holds: t6 < gauss < uniform.
    centers, width = make_grid(-30, 30, 6000)

    gauss = analytic_pmf(stats.norm(loc=0, scale=3), centers, width)
    t6 = analytic_pmf(stats.t(df=6, scale=3), centers, width)
    uniform = analytic_pmf(stats.uniform(loc=-3 * np.sqrt(3), scale=6 * np.sqrt(3)), centers, width)

    tw_gauss = compute_summary(gauss, centers)["tail_weight"][0]
    tw_t6 = compute_summary(t6, centers)["tail_weight"][0]
    tw_uniform = compute_summary(uniform, centers)["tail_weight"][0]

    assert tw_t6 < tw_gauss < tw_uniform


# ---------------------------------------------------------------------------
# sigma_iqr
# ---------------------------------------------------------------------------


def test_sigma_iqr_gaussian():
    dist = stats.norm(loc=0, scale=5)
    centers, width = make_grid(-40, 40, 4000)
    pmf = analytic_pmf(dist, centers, width)

    summary = compute_summary(pmf, centers)
    assert summary["sigma_iqr"][0] == pytest.approx(5, rel=0.01)


def test_sigma_iqr_heavy_tails():
    dist = stats.t(df=6, scale=5)
    centers, width = make_grid(-80, 80, 4000)
    pmf = analytic_pmf(dist, centers, width)

    summary = compute_summary(pmf, centers)
    assert summary["sigma_iqr"][0] < summary["sigma"][0]


# ---------------------------------------------------------------------------
# cdf_percentile
# ---------------------------------------------------------------------------


def test_cdf_percentile_recovers_quantiles():
    dist = stats.norm(loc=3, scale=6)
    centers, width = make_grid(-60, 66, 4000)
    pmf = analytic_pmf(dist, centers, width)

    q = cdf_percentile(pmf, centers, [0.25, 0.5, 0.75])
    true_q = dist.ppf([0.25, 0.5, 0.75])

    # cdf_percentile linearly interpolates the discretised CDF between
    # bin-centre nodes spaced `width` apart, so the achievable accuracy is
    # bounded by one full bin width, not half of one.
    assert np.allclose(q[0], true_q, atol=width)


# ---------------------------------------------------------------------------
# v_asymmetry
# ---------------------------------------------------------------------------


def test_v_asymmetry_sign():
    # Right-skewed mixture: a narrow main peak with a trailing tail to
    # higher velocities.
    centers, width = make_grid(-30, 60, 4000)

    def make_mix_pmf(sign):
        d1 = stats.norm(loc=0, scale=3)
        d2 = stats.norm(loc=sign * 15, scale=8)
        edges = np.concatenate([centers - width / 2, centers[-1:] + width / 2])
        pmf = 0.7 * np.diff(d1.cdf(edges)) + 0.3 * np.diff(d2.cdf(edges))
        pmf = pmf / pmf.sum()
        return pmf[np.newaxis, :]

    pmf_pos = make_mix_pmf(+1)
    summary_pos = compute_summary(pmf_pos, centers)
    assert summary_pos["v_asymmetry"][0] > 0

    pmf_neg = make_mix_pmf(-1)
    summary_neg = compute_summary(pmf_neg, -centers[::-1])
    # v -> -v flips the sign of asymmetry; verify using the flipped grid and
    # a mirrored pmf, which is equivalent to negating all velocities.
    assert summary_neg["v_asymmetry"][0] < 0


# ---------------------------------------------------------------------------
# bimodality_score
# ---------------------------------------------------------------------------


def test_bimodality_score_counts():
    centers, width = make_grid(-60, 60, 2000)
    edges = np.concatenate([centers - width / 2, centers[-1:] + width / 2])

    def mixture_pmf(means, weights, scale=2.0):
        pmf = np.zeros(len(centers))
        for m, w in zip(means, weights):
            pmf += w * np.diff(stats.norm(loc=m, scale=scale).cdf(edges))
        return (pmf / pmf.sum())[np.newaxis, :]

    pmf1 = mixture_pmf([0], [1.0])
    pmf2 = mixture_pmf([-20, 20], [0.5, 0.5])
    pmf3 = mixture_pmf([-30, 0, 30], [1 / 3, 1 / 3, 1 / 3])

    assert bimodality_score(pmf1) == 1
    assert bimodality_score(pmf2) == 2
    assert bimodality_score(pmf3) == 3


def test_bimodality_ignores_tail_noise():
    centers, width = make_grid(-60, 60, 2000)
    edges = np.concatenate([centers - width / 2, centers[-1:] + width / 2])

    main = np.diff(stats.norm(loc=0, scale=5).cdf(edges))
    main = main / main.sum()

    # Add a tiny ripple (0.5% of max amplitude) far in the tail.
    ripple_idx = np.argmin(np.abs(centers - 45))
    ripple = np.zeros_like(main)
    ripple[ripple_idx - 1] = 0.003 * main.max()
    ripple[ripple_idx] = 0.005 * main.max()
    ripple[ripple_idx + 1] = 0.003 * main.max()

    pmf = main + ripple
    pmf = pmf / pmf.sum()
    pmf = pmf[np.newaxis, :]

    assert bimodality_score(pmf) == 1


# ---------------------------------------------------------------------------
# half_68ci
# ---------------------------------------------------------------------------


def test_half_68ci_gaussian():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=0, scale=1, size=200_000)
    assert half_68ci(samples) == pytest.approx(1.0, rel=0.02)


# ---------------------------------------------------------------------------
# Uncertainty ordering with posterior spread
# ---------------------------------------------------------------------------


def test_summary_uncertainty_shrinks_with_n():
    centers, width = make_grid(-40, 40, 2000)
    edges = np.concatenate([centers - width / 2, centers[-1:] + width / 2])
    base = np.diff(stats.norm(loc=0, scale=5).cdf(edges))
    base = base / base.sum()

    rng = np.random.default_rng(1)
    n_samples = 500

    def jittered_pmfs(mean_jitter):
        rows = []
        for _ in range(n_samples):
            shift = rng.normal(0, mean_jitter)
            edges_shift = edges + shift
            row = np.diff(stats.norm(loc=0, scale=5).cdf(edges_shift))
            row = row / row.sum()
            rows.append(row)
        return np.array(rows)

    narrow = jittered_pmfs(mean_jitter=0.1)
    wide = jittered_pmfs(mean_jitter=2.0)

    summary_narrow = compute_summary(narrow, centers)
    summary_wide = compute_summary(wide, centers)

    assert summary_narrow["v_mean"][1] < summary_wide["v_mean"][1]


# ---------------------------------------------------------------------------
# truncate_pdf_samples
# ---------------------------------------------------------------------------


def test_truncate_pdf_samples_noop_when_all_mass_interior():
    # Grid spans only +/-5 sigma; use a large n_sigma cut (10) so the true
    # (infinite-tailed) Gaussian's residual mass beyond the cut is
    # negligible to floating-point precision, making this a genuine no-op
    # check rather than one that trips on the distribution's real tail.
    dist = stats.norm(loc=0, scale=1)
    centers, width = make_grid(-5, 5, 500)
    pmf = analytic_pmf(dist, centers, width)

    truncated = truncate_pdf_samples(pmf, centers, n_sigma=10.0)

    assert np.allclose(truncated, pmf, atol=1e-10)
    assert np.sum(truncated, axis=1) == pytest.approx(1.0, abs=1e-10)


def test_truncate_pdf_samples_removes_far_edge_mass():
    centers, width = make_grid(-40, 40, 200)
    dist = stats.norm(loc=0, scale=3)
    edges = np.concatenate([centers - width / 2, centers[-1:] + width / 2])
    pmf = np.diff(dist.cdf(edges))
    pmf = pmf / pmf.sum()

    # Artificially plant leaked mass in the far-edge bin (well beyond
    # n_sigma of the bulk dispersion of ~3).
    pmf = pmf * 0.99
    pmf[0] += 0.01
    pmf = pmf[np.newaxis, :]

    truncated = truncate_pdf_samples(pmf, centers, n_sigma=4.0)

    # The far-edge bin's mass should be zeroed.
    assert truncated[0, 0] == pytest.approx(0.0, abs=1e-12)
    # Rows still sum to 1 after renormalisation.
    assert np.sum(truncated, axis=1) == pytest.approx(1.0, abs=1e-10)
    # Remaining bins should be redistributed proportionally: the ratio
    # between any two surviving bins should be unchanged by renormalisation.
    interior_mask = np.abs(centers - 0.0) <= 4.0 * 3.0
    interior_mask[0] = False  # exclude the leaked bin itself (already 0)
    orig_interior = pmf[0][interior_mask]
    new_interior = truncated[0][interior_mask]
    ratio = new_interior / orig_interior
    assert np.allclose(ratio, ratio[0], rtol=1e-8)


def test_truncate_pdf_samples_per_row_uses_own_dispersion():
    # Two rows with very different dispersions; a fixed global threshold
    # would truncate one row's legitimate tail while under-truncating the
    # other's leaked mass. Per-row truncation should treat each fairly.
    centers, width = make_grid(-20, 20, 400)
    narrow = analytic_pmf(stats.norm(loc=0, scale=1), centers, width)
    wide = analytic_pmf(stats.norm(loc=0, scale=8), centers, width)
    pmf = np.vstack([narrow, wide])

    truncated = truncate_pdf_samples(pmf, centers, n_sigma=10.0)

    # Each row still sums to 1.
    assert np.sum(truncated, axis=1) == pytest.approx([1.0, 1.0], abs=1e-8)
    # Narrow row (scale=1) should have essentially no truncation at
    # n_sigma=10, since the grid only spans +/-20 = +/-20 sigma for it.
    assert np.allclose(truncated[0], pmf[0], atol=1e-8)


# ---------------------------------------------------------------------------
# compute_summary keys
# ---------------------------------------------------------------------------


def test_compute_summary_keys():
    dist = stats.norm(loc=0, scale=5)
    centers, width = make_grid(-40, 40, 500)
    pmf = analytic_pmf(dist, centers, width)

    summary = compute_summary(pmf, centers)

    expected_keys = {
        "v_mean",
        "v_median",
        "v_asymmetry",
        "sigma",
        "iqr",
        "sigma_iqr",
        "skewness",
        "kurtosis",
        "tail_weight",
        "bimodality_score",
    }
    assert set(summary.keys()) == expected_keys

    for key, value in summary.items():
        if key == "bimodality_score":
            assert isinstance(value, int)
        else:
            assert isinstance(value, tuple)
            assert len(value) == 2
            assert all(isinstance(v, float) for v in value)


# ---------------------------------------------------------------------------
# compute_summary_maps
# ---------------------------------------------------------------------------


def test_compute_summary_maps_nan_masking():
    dist = stats.norm(loc=0, scale=5)
    centers, width = make_grid(-40, 40, 500)
    pmf = analytic_pmf(dist, centers, width)

    solver = SimpleNamespace(
        samples={"intrinsic_pdf": pmf}, grid={"centers": centers}
    )
    solvers = [solver, None, solver]

    maps = compute_summary_maps(solvers)

    for key, sub in maps.items():
        assert np.isnan(sub["median"][1])
        assert np.isnan(sub["uncertainty"][1])
        assert not np.isnan(sub["median"][0])
        assert not np.isnan(sub["median"][2])

    # bimodality_score has no posterior CI, so uncertainty is all-NaN.
    assert np.all(np.isnan(maps["bimodality_score"]["uncertainty"]))


def test_compute_summary_maps_all_none():
    with pytest.raises(ValueError):
        compute_summary_maps([None, None])


# ---------------------------------------------------------------------------
# Gauss-Hermite conversion consistency
# ---------------------------------------------------------------------------


def test_gh_conversion_consistency():
    """
    The docstring for compute_summary claims h3 ~= -skewness/sqrt(6) and
    h4 ~= kurtosis/sqrt(24), valid for |h| <~ 0.2. Estimate h3, h4 directly
    from the analytic PMF via projection onto the orthonormal Gauss-Hermite
    basis (van der Marel & Franx 1993 convention) and check the conversion
    holds for a mildly non-Gaussian distribution.

    The GH basis functions used here,
        H3(y) = -(y^3 - 3y) / sqrt(6)
        H4(y) =  (y^4 - 6y^2 + 3) / sqrt(24)
    are the probabilists'-Hermite functions orthonormalised under the
    standard-normal weight (verified numerically: <H3^2> = <H4^2> = 1,
    <H3> = <H4> = <H3 H4> = 0). Projecting the PMF onto this basis
    (h_n = integral of pdf(v) * H_n(y) dv) recovers the Edgeworth-series
    coefficients exactly to leading order, which is the cleanest way to
    check the docstring's claimed approximation without the extra noise of
    a nonlinear curve fit.
    """
    dist = stats.skewnorm(a=2, loc=0, scale=5)
    centers, width = make_grid(-40, 40, 4000)
    pmf = analytic_pmf(dist, centers, width)

    summary = compute_summary(pmf, centers)
    skew = summary["skewness"][0]
    kurt = summary["kurtosis"][0]
    v_mean = summary["v_mean"][0]
    sigma = summary["sigma"][0]

    y = (centers - v_mean) / sigma
    h3_basis = -(y**3 - 3 * y) / np.sqrt(6)
    h4_basis = (y**4 - 6 * y**2 + 3) / np.sqrt(24)

    h3_fit = float(np.sum(pmf[0] * h3_basis))
    h4_fit = float(np.sum(pmf[0] * h4_basis))

    h3_predicted = -skew / np.sqrt(6)
    h4_predicted = kurt / np.sqrt(24)

    # Only meaningful within the documented validity range |h| <~ 0.2.
    assert abs(h3_fit) < 0.3
    assert abs(h4_fit) < 0.3
    assert h3_fit == pytest.approx(h3_predicted, abs=0.01)
    assert h4_fit == pytest.approx(h4_predicted, abs=0.01)
