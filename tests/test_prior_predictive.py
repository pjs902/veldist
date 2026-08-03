"""Prior-predictive tests for the latent-curve priors.

These are the regression tests for the "uniform null space" bug: the RW1
prior's infinite-smoothing limit is a *uniform* LOSVD over the whole
velocity grid, which deposits probability mass at +/-4-5 sigma wherever the
data is weak and biases sigma high and kurtosis high.

A prior whose null space is Gaussian (Merritt 1997, AJ 114, 228) does not
have this failure mode. These tests assert that property directly on the
prior, with NO data and NO MCMC, so they run in seconds and pin the root
cause rather than a downstream symptom.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer import Predictive

from veldist.veldist import model, model_gaussian_core, generate_gaussian_core_curve

# Grid used throughout: 400 km/s wide, centred on 200.
GRID_CENTER = 200.0
GRID_WIDTH = 400.0

# A uniform distribution over a grid of width W has sigma = W/sqrt(12) and
# excess kurtosis -1.2. These are the values a flat-null-space prior
# reproduces, and the values a Gaussian-null-space prior must NOT.
UNIFORM_SIGMA = GRID_WIDTH / np.sqrt(12.0)
UNIFORM_EXCESS_KURTOSIS = -1.2


def _grid(n_bins):
    """Return (centers, bin_width) for the standard test grid."""
    bin_width = GRID_WIDTH / n_bins
    centers = GRID_CENTER - GRID_WIDTH / 2 + (np.arange(n_bins) + 0.5) * bin_width
    return centers, bin_width


def _prior_predictive_pdfs(model_fn, n_bins, centers, bin_width, num_samples=2000, seed=0):
    """Draw `num_samples` LOSVDs from `model_fn`'s prior, with no data.

    A dummy one-star design matrix is required rather than `matrix=None`.
    `numpyro.factor` evaluates its argument eagerly when the model is
    traced, so the likelihood expression `jnp.dot(matrix, intrinsic_pdf)`
    runs even under `Predictive` and raises `TypeError` on None. The
    *value* of the factor is irrelevant here -- `Predictive` with no
    conditioning forward-simulates through `sample` sites and never uses
    it -- so a uniform dummy row is sufficient and cannot influence the
    draws.
    """
    dummy_matrix = np.full((1, n_bins), 1.0 / n_bins)
    kwargs = dict(matrix=dummy_matrix, n_bins=n_bins, bin_width=bin_width)
    if model_fn is model_gaussian_core:
        kwargs["centers"] = jnp.asarray(centers)
    predictive = Predictive(model_fn, num_samples=num_samples)
    draws = predictive(jax.random.PRNGKey(seed), **kwargs)
    return np.asarray(draws["intrinsic_pdf"])


def _moments(pdf_samples, centers):
    """Per-draw sigma and excess kurtosis from PMF samples."""
    pdf_samples = np.asarray(pdf_samples, dtype=float)
    centers = np.asarray(centers, dtype=float)
    mean = pdf_samples @ centers
    delta = centers[np.newaxis, :] - mean[:, np.newaxis]
    var = np.einsum("ij,ij->i", pdf_samples, delta**2)
    m4 = np.einsum("ij,ij->i", pdf_samples, delta**4)
    with np.errstate(divide="ignore", invalid="ignore"):
        excess_kurtosis = m4 / var**2 - 3.0
    return np.sqrt(var), excess_kurtosis


@pytest.mark.parametrize("n_bins", [20, 40, 80])
def test_gaussian_core_prior_is_not_uniform(n_bins):
    """The Gaussian-core prior must not default to spreading mass over the
    whole grid.

    This is the direct regression test for the bug. The RW1 prior fails it:
    its prior-predictive median sigma is ~111.5 on this grid vs. the uniform
    value of 115.5, i.e. it IS the uniform distribution to within a few
    percent. A Gaussian-null-space prior sits far below that.
    """
    centers, bin_width = _grid(n_bins)
    pdfs = _prior_predictive_pdfs(model_gaussian_core, n_bins, centers, bin_width)
    sigma, _ = _moments(pdfs, centers)

    finite = np.isfinite(sigma)
    assert finite.mean() > 0.9, (
        f"{(~finite).sum()}/{len(sigma)} prior draws produced a non-finite "
        "sigma; the prior is generating degenerate LOSVDs"
    )
    median_sigma = float(np.median(sigma[finite]))

    assert median_sigma < 0.6 * UNIFORM_SIGMA, (
        f"n_bins={n_bins}: prior-predictive median sigma is {median_sigma:.1f}, "
        f"vs {UNIFORM_SIGMA:.1f} for a uniform LOSVD over this grid. The "
        "prior's null space is flat, not Gaussian -- this is the bias bug. "
        "See docs/superpowers/plans/2026-08-03-gaussian-null-space-prior.md"
    )

    # Lower bound: without this the test passes VACUOUSLY for a prior that
    # has collapsed to a delta function (median sigma 0.00), which is
    # exactly what a HalfNormal/HalfCauchy prior on s0 produces, and what a
    # mis-scaled sigma3 produces by saturating the softmax. A degenerate
    # prior is not a fix; it is a different failure.
    assert median_sigma > 0.05 * UNIFORM_SIGMA, (
        f"n_bins={n_bins}: prior-predictive median sigma is {median_sigma:.2f}, "
        "which is degenerate -- the prior has collapsed onto a delta "
        "function rather than producing plausible LOSVDs. Check the s0 prior "
        "and the (bin_width / span) ** 2.5 scaling on sigma3."
    )

    degenerate_fraction = float(np.mean(sigma[finite] < GRID_WIDTH / n_bins))
    assert degenerate_fraction < 0.25, (
        f"n_bins={n_bins}: {degenerate_fraction:.1%} of prior draws put all " "their mass within a single bin"
    )


def test_rw1_prior_is_uniform_documents_the_bug():
    """Negative control: the OLD RW1 prior really is uniform-like.

    This documents the defect rather than asserting it is acceptable. If this
    test ever starts failing, the RW1 prior changed and the motivation
    recorded in the plan needs revisiting.
    """
    n_bins = 40
    centers, bin_width = _grid(n_bins)
    pdfs = _prior_predictive_pdfs(model, n_bins, centers, bin_width)
    sigma, excess_kurtosis = _moments(pdfs, centers)

    median_sigma = float(np.median(sigma[np.isfinite(sigma)]))
    median_kurt = float(np.median(excess_kurtosis[np.isfinite(excess_kurtosis)]))

    assert median_sigma > 0.85 * UNIFORM_SIGMA, (
        f"RW1 prior-predictive median sigma {median_sigma:.1f} is no longer "
        f"close to the uniform value {UNIFORM_SIGMA:.1f}"
    )
    assert median_kurt < UNIFORM_EXCESS_KURTOSIS + 0.5, (
        f"RW1 prior-predictive median excess kurtosis {median_kurt:.2f} is no "
        f"longer close to the uniform value {UNIFORM_EXCESS_KURTOSIS}"
    )


def test_gaussian_core_curve_is_exactly_quadratic_when_deviation_is_zero():
    """With the deviation forced to zero, the latent curve must be exactly
    the log of a Gaussian, i.e. exactly quadratic in velocity.

    This is the defining property of the prior: the Gaussian is the free,
    unpenalised null-space element (Merritt 1997). A quadratic has zero
    third derivative, so fitting a parabola through the curve must leave
    residuals at floating-point noise level.
    """
    import numpyro
    from numpyro.handlers import seed, substitute, trace

    from veldist.veldist import generate_gaussian_core_curve

    n_bins = 40
    centers, bin_width = _grid(n_bins)

    fixed = {
        "v0": 210.0,
        "s0": 45.0,
        "sigma3": 0.0,  # kills the deviation entirely
        "d3": np.zeros(n_bins),
    }
    model_fn = substitute(seed(generate_gaussian_core_curve, jax.random.PRNGKey(0)), data=fixed)
    curve = np.asarray(model_fn(n_bins, jnp.asarray(centers), bin_width))

    # Fit a parabola in velocity and require an essentially perfect fit.
    coeffs = np.polyfit(centers, curve, 2)
    residual = curve - np.polyval(coeffs, centers)
    assert np.max(np.abs(residual)) < 1e-6, (
        "latent curve is not quadratic when the deviation is switched off; "
        f"max parabola residual {np.max(np.abs(residual)):.2e}"
    )


def test_gaussian_core_deviation_is_orthogonal_to_quadratics():
    """The deviation term must carry no constant, linear, or quadratic
    component -- those belong to the Gaussian core.

    If it does carry them, the core and the deviation can trade off against
    each other, `v0`/`s0` become unidentifiable, and NUTS will diverge.
    """
    import numpyro
    from numpyro.handlers import seed, substitute

    from veldist.veldist import generate_gaussian_core_curve

    n_bins = 40
    centers, bin_width = _grid(n_bins)
    rng = np.random.default_rng(0)

    # Zero out the core so only the deviation remains: s0 -> huge makes the
    # quadratic core flat to within floating point over this grid.
    fixed = {
        "v0": GRID_CENTER,
        "s0": 1e8,
        "sigma3": 1.0,
        "d3": rng.normal(size=n_bins),
    }
    model_fn = substitute(seed(generate_gaussian_core_curve, jax.random.PRNGKey(1)), data=fixed)
    curve = np.asarray(model_fn(n_bins, jnp.asarray(centers), bin_width))

    u = (centers - centers.mean()) / (centers.max() - centers.min())
    basis = np.stack([np.ones_like(u), u, u**2], axis=1)
    projection = basis @ np.linalg.lstsq(basis, curve, rcond=None)[0]
    assert np.max(np.abs(projection)) < 1e-5 * max(1.0, np.max(np.abs(curve))), (
        "deviation term has a non-zero projection onto {1, u, u^2}; the QR " "projection is not working"
    )


def test_gaussian_core_prior_is_resolution_invariant():
    """Refining the velocity grid must not change what the prior means.

    The repo already guarantees this for the RW1 prior via a sqrt(bin_width)
    step scaling. The Gaussian-core prior instead standardises the deviation
    to unit generalised variance (Sorbye & Rue), which is resolution-invariant
    by construction. If that standardisation is wrong, the prior-predictive
    dispersion will drift systematically as n_bins changes.
    """
    medians = []
    for n_bins in (20, 40, 80):
        centers, bin_width = _grid(n_bins)
        pdfs = _prior_predictive_pdfs(model_gaussian_core, n_bins, centers, bin_width, num_samples=3000)
        sigma, _ = _moments(pdfs, centers)
        medians.append(float(np.median(sigma[np.isfinite(sigma)])))

    spread = (max(medians) - min(medians)) / np.mean(medians)
    assert spread < 0.15, (
        "prior-predictive median sigma drifts with grid resolution "
        f"({[round(m, 1) for m in medians]} for n_bins=20/40/80, relative "
        f"spread {spread:.2f}); the Sorbye-Rue generalised-variance "
        "standardisation is probably wrong -- check _rw3_deviation_scale"
    )


def _rw3_scale_via_structure_matrix(n_bins):
    """Independent route to the Sorbye-Rue constant, via pinv(D3' D3).

    The implementation builds the constrained covariance as (L^3)(L^3)' with
    L lower-triangular ones. This builds it instead as the pseudo-inverse of
    the third-difference structure matrix. Same object, different arithmetic,
    so agreement is not circular.
    """
    d3 = np.zeros((n_bins - 3, n_bins))
    for i in range(n_bins - 3):
        d3[i, i : i + 4] = [-1.0, 3.0, -3.0, 1.0]
    r = d3.T @ d3
    idx = np.arange(n_bins, dtype=float)
    u = (idx - idx.mean()) / (n_bins - 1)
    basis = np.stack([np.ones_like(u), u, u**2], axis=1)
    q, _ = np.linalg.qr(basis)
    proj = np.eye(n_bins) - q @ q.T
    cov = proj @ np.linalg.pinv(r) @ proj.T
    var = np.clip(np.diag(cov), 1e-300, None)
    return float(1.0 / np.sqrt(np.exp(np.mean(np.log(var)))))


@pytest.mark.parametrize("n_bins", [20, 40, 55, 80])
def test_rw3_deviation_scale_matches_structure_matrix(n_bins):
    """The scaling constant must equal the Sorbye-Rue constant for a
    constrained RW3 intrinsic GMRF, computed independently."""
    from veldist.veldist import _rw3_deviation_scale

    got = _rw3_deviation_scale(n_bins)
    want = _rw3_scale_via_structure_matrix(n_bins)
    assert np.isclose(got, want, rtol=1e-6), f"n_bins={n_bins}: scale {got:.6g} != structure-matrix value {want:.6g}"


def test_rw3_deviation_scale_is_cached():
    """The constant costs an O(n^3) pinv/QR; it must not be recomputed."""
    from veldist.veldist import _rw3_deviation_scale

    _rw3_deviation_scale.cache_clear()
    _rw3_deviation_scale(40)
    _rw3_deviation_scale(40)
    assert _rw3_deviation_scale.cache_info().hits >= 1


@pytest.mark.parametrize("n_bins", [20, 40, 80])
def test_gaussian_core_deviation_has_unit_marginal_sd(n_bins):
    """With sigma3 = 1, the deviation must have marginal SD ~1 in
    log-density units.

    This is the regression test for the collapse bug: before the Sorbye-Rue
    scaling landed, the deviation's marginal SD was ~0.0036, far too small
    for the likelihood to see, so the posterior reverted to the pure
    Gaussian null space no matter what the data said.
    """
    import jax
    from numpyro.handlers import seed, substitute

    from veldist.veldist import generate_gaussian_core_curve

    centers, bin_width = _grid(n_bins)
    rng = np.random.default_rng(0)
    n_draws = 400

    deviations = np.empty((n_draws, n_bins))
    for i in range(n_draws):
        fixed = {
            "v0": GRID_CENTER,
            "s0": 1e8,  # flattens the quadratic core to ~0 over this grid
            "sigma3": 1.0,
            "d3": rng.normal(size=n_bins),
        }
        model_fn = substitute(seed(generate_gaussian_core_curve, jax.random.PRNGKey(i)), data=fixed)
        deviations[i] = np.asarray(model_fn(n_bins, jnp.asarray(centers), bin_width))

    per_bin_var = deviations.var(axis=0)
    generalised_sd = np.sqrt(np.exp(np.mean(np.log(per_bin_var))))
    assert 0.8 < generalised_sd < 1.25, (
        f"n_bins={n_bins}: deviation generalised SD is {generalised_sd:.4g}, "
        "expected ~1. If this is ~0.004 the Sorbye-Rue scaling is not wired in."
    )


def test_gaussian_core_prior_spans_nongaussian_shapes():
    """The PC prior must make non-Gaussian LOSVDs reachable a priori.

    Two failure modes bracketed here. Too tight (the original bug): every
    prior draw is a Gaussian, excess kurtosis is ~0 everywhere, and the
    likelihood can never pull the posterior away from Gaussian. Too loose:
    draws saturate the softmax into spiky near-delta functions, which shows
    up as an enormous kurtosis tail.

    This test pins SIGMA3_RATE. If it fails, change SIGMA3_RATE rather than
    the bounds, and record the implied P(sigma3 > 1) in its comment.
    """
    n_bins = 40
    centers, bin_width = _grid(n_bins)
    pdfs = _prior_predictive_pdfs(model_gaussian_core, n_bins, centers, bin_width, num_samples=4000)
    _, excess_kurtosis = _moments(pdfs, centers)
    finite = excess_kurtosis[np.isfinite(excess_kurtosis)]
    p90 = float(np.percentile(np.abs(finite), 90))

    assert p90 > 0.3, (
        f"prior-predictive |excess kurtosis| p90 = {p90:.4g}; the prior is too "
        "tight to represent non-Gaussian LOSVDs at all"
    )
    assert p90 < 10.0, (
        f"prior-predictive |excess kurtosis| p90 = {p90:.4g}; the prior is so "
        "loose it is producing near-delta functions"
    )
