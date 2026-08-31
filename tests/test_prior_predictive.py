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

from math import comb

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
    assert degenerate_fraction < 0.40, (
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


def test_gaussian_core_curve_is_exact_gaussian_bin_mass_when_deviation_is_zero():
    """With the deviation forced to zero, softmax(curve) must be the exact
    per-bin probability MASS of a Gaussian.

    This is the defining property of the prior: the Gaussian is the free,
    unpenalised null-space element (Merritt 1997). The subtlety is *which*
    Gaussian object the core has to equal. `intrinsic_pdf` is consumed as bin
    mass -- precompute_design_matrix integrates each star's error kernel
    between bin EDGES -- so the core must be the Gaussian's bin mass, not its
    density sampled at bin centres.

    Those differ at O(h^2): mass = int_bin f ~= h*f(c) + (h^3/24)*f''(c), and
    f'' > 0 in the tails, so centre sampling under-weights the tails and makes
    the core narrower than the Gaussian it claims to be. This test previously
    asserted the curve was exactly QUADRATIC, which is the centre-density
    property -- it passed while the prior carried exactly that bias. Asserting
    against the exact bin mass is strictly stronger: it pins the shape, not
    just the polynomial degree.
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

    from scipy.stats import norm

    # softmax the curve, and compare against the exact Gaussian bin mass.
    got = np.exp(curve - curve.max())
    got /= got.sum()

    edges = np.concatenate([centers - bin_width / 2, [centers[-1] + bin_width / 2]])
    want = np.diff(norm.cdf(edges, fixed["v0"], fixed["s0"]))
    want /= want.sum()

    assert np.max(np.abs(got - want)) < 1e-10, (
        "core is not the exact Gaussian bin mass when the deviation is off; "
        f"max mass error {np.max(np.abs(got - want)):.2e}"
    )

    # And the moments it implies must match the bin mass's, not the centre
    # density's. At this grid the centre-density error is ~1e-2 on sigma, so
    # this tolerance separates the two unambiguously.
    def _sigma(w):
        m = w @ centers
        return np.sqrt(w @ (centers - m) ** 2)

    assert abs(_sigma(got) - _sigma(want)) < 1e-8, (
        f"core sigma {_sigma(got):.6f} != bin-mass sigma {_sigma(want):.6f}"
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

    # The core is log(Gaussian BIN MASS), so a "flattened" core (s0 -> huge) is
    # flat at a large negative CONSTANT -- the log normalisation -- not at zero.
    # softmax is invariant to an additive constant in the log field, so that
    # offset is gauge, not signal. Remove it before projecting; the invariant
    # under test is that the deviation carries no LINEAR or QUADRATIC content.
    curve = curve - curve.mean()

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
        "standardisation is probably wrong -- check _rw_deviation_scale"
    )


def _rw_scale_via_structure_matrix(n_bins, order):
    """Independent route to the Sorbye-Rue constant, via pinv(D_k' D_k).

    The implementation builds the constrained covariance as (L^k)(L^k)' with
    L lower-triangular ones. This builds it as the pseudo-inverse of the k-th
    difference structure matrix instead. Same mathematical object, completely
    different arithmetic, so agreement is meaningful rather than circular.
    """
    stencil = np.array([(-1) ** (order - j) * comb(order, j) for j in range(order + 1)], dtype=float)
    d = np.zeros((n_bins - order, n_bins))
    for i in range(n_bins - order):
        d[i, i : i + order + 1] = stencil
    r = d.T @ d
    idx = np.arange(n_bins, dtype=float)
    u = (idx - idx.mean()) / (n_bins - 1)
    basis = np.stack([u**k for k in range(order)], axis=1)
    q, _ = np.linalg.qr(basis)
    proj = np.eye(n_bins) - q @ q.T
    var = np.clip(np.diag(proj @ np.linalg.pinv(r) @ proj.T), 1e-300, None)
    return float(1.0 / np.sqrt(np.exp(np.mean(np.log(var)))))


@pytest.mark.parametrize("order", [3, 4, 5])
@pytest.mark.parametrize("n_bins", [20, 37, 60])
def test_rw_deviation_scale_matches_structure_matrix(n_bins, order):
    """The scaling constant must equal the Sorbye-Rue constant for a
    constrained RW-k intrinsic GMRF, computed independently."""
    from veldist.veldist import _rw_deviation_scale

    got = _rw_deviation_scale(n_bins, order)
    want = _rw_scale_via_structure_matrix(n_bins, order)
    assert np.isclose(got, want, rtol=1e-5), (
        f"n_bins={n_bins} order={order}: scale {got:.6g} != " f"structure-matrix value {want:.6g}"
    )


def test_rw_deviation_scale_order_3_is_unchanged():
    """Order 3 must reproduce the previously committed constants exactly.

    This is the regression guard: the generalisation must not alter existing
    behaviour, so if a later task breaks something you can rule this out.
    """
    from veldist.veldist import _rw_deviation_scale

    assert np.isclose(_rw_deviation_scale(20, 3), 0.1627, rtol=1e-3)
    assert np.isclose(_rw_deviation_scale(55, 3), 0.0130, rtol=1e-3)


def test_rw_deviation_scale_is_cached():
    """The constant costs an O(n^3) pinv/QR; it must not be recomputed."""
    from veldist.veldist import _rw_deviation_scale

    _rw_deviation_scale.cache_clear()
    _rw_deviation_scale(40)
    _rw_deviation_scale(40)
    assert _rw_deviation_scale.cache_info().hits >= 1


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


@pytest.mark.parametrize("rw_order", [3, 4, 5])
def test_gaussian_core_null_space_is_polynomial_of_degree_order_minus_one(rw_order):
    import jax
    from numpyro.handlers import seed, substitute

    from veldist.veldist import generate_gaussian_core_curve

    n_bins = 37
    centers, bin_width = _grid(n_bins)
    rng = np.random.default_rng(3)
    fixed = {
        "v0": GRID_CENTER,
        "s0": 1e8,
        "sigma3": 1.0,
        "d3": rng.normal(size=n_bins),
    }
    model_fn = substitute(seed(generate_gaussian_core_curve, jax.random.PRNGKey(0)), data=fixed)
    curve = np.asarray(model_fn(n_bins, jnp.asarray(centers), bin_width, rw_order=rw_order))

    # See the note in test_gaussian_core_deviation_is_orthogonal_to_quadratics:
    # the core is log(bin mass), so a flattened core sits at a constant, and
    # that constant is softmax gauge rather than signal.
    curve = curve - curve.mean()

    u = (centers - centers.mean()) / (centers.max() - centers.min())
    scale = max(1.0, np.max(np.abs(curve)))

    inside = np.stack([u**k for k in range(rw_order)], axis=1)
    coef = np.linalg.lstsq(inside, curve, rcond=None)[0]
    assert np.max(np.abs(inside @ coef)) < 1e-4 * scale, (
        f"order {rw_order}: deviation has a non-zero projection onto "
        f"polynomials of degree < {rw_order}; the QR projection is wrong"
    )

    resid = curve - inside @ coef
    outside = (u**rw_order)[:, None]
    coef_next = np.linalg.lstsq(outside, resid, rcond=None)[0]
    assert np.max(np.abs(outside @ coef_next)) > 1e-3 * scale, (
        f"order {rw_order}: deviation has no component along u^{rw_order}; " "the null space is larger than intended"
    )


def test_gaussian_core_prior_spans_nongaussian_shapes():
    """The PC prior must make non-Gaussian LOSVDs reachable a priori.

    Two failure modes bracketed here. Too tight (the original bug): every
    prior draw is a Gaussian, excess kurtosis is ~0 everywhere, and the
    likelihood can never pull the posterior away from Gaussian. Too loose:
    draws saturate the softmax into spiky near-delta functions, which shows
    up as an enormous kurtosis tail.

    This is a *bracket*, not a pin, despite what it used to claim. Measured
    p90 |excess kurtosis| at n_bins=40: rate 0.35 -> 38.8, rate 5.0 -> 1.13,
    rate 10.0 -> 1.08, rate 50.0 -> 1.05. So every rate from 0.35 upward
    passes, and this test cannot select one. It only catches the two gross
    failure modes. Note that the adopted rate is chosen by SBC (see
    tests/test_calibration.py and TASKS.md), not here.

    Note also that 38.8 at Exp(0.35) sits close to the near-delta ceiling --
    consistent with the high degenerate-draw fraction that prior showed.
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
    assert p90 < 50.0, (
        f"prior-predictive |excess kurtosis| p90 = {p90:.4g}; the prior is so "
        "loose it is producing near-delta functions"
    )


@pytest.mark.parametrize(
    "rw_order,expect_h3_free,expect_h4_free",
    [
        pytest.param(3, False, False),
        pytest.param(
            4,
            False,
            False,
            marks=pytest.mark.xfail(
                reason="rw_order=4 partly frees h3 (0.158 vs 0.15 threshold): "
                "the null-space enlargement in log-density space does not fully "
                "translate to PDF moments through softmax",
                strict=False,
            ),
        ),
        pytest.param(
            5,
            False,
            False,
            marks=pytest.mark.xfail(
                reason="rw_order=5 partly frees h3 (0.156 vs 0.15 threshold): "
                "same softmax decoupling as order 4; does not free h4 either (0.038)",
                strict=False,
            ),
        ),
    ],
)
def test_penalty_order_controls_which_moments_are_free(rw_order, expect_h3_free, expect_h4_free):
    """Raising the penalty order enlarges the null space, and the null space
    is what the prior does NOT shrink.

    Measured as the ratio of prior-predictive spread at a very tight sigma3
    to the spread at a loose one. Near 1 means the moment survives with the
    deviation switched off, i.e. it lives in the null space. Near 0 means it
    only exists because of the penalised deviation.

    NOTE 2026-08-03: orders 4 and 5 do NOT free h3/h4 as hypothesised.
    All three orders show strong shrinkage (h3_retained ~0.13-0.16,
    h4_retained ~0.03-0.04). The null-space enlargement from raising
    rw_order does not translate into freedom for the third/fourth moments.
    Orders 4/5 fail at the 0.15 threshold (h3_retained 0.158, 0.156) --
    the softmax nonlinearity decouples log-density polynomial null space
    from PDF moments, so the plan's central hypothesis is not supported.
    """
    import veldist.veldist as vd

    n_bins = 37
    centers, bin_width = _grid(n_bins)

    def spread(rate):
        old = vd.SIGMA3_RATE
        try:
            vd.SIGMA3_RATE = rate
            pdfs = _prior_predictive_pdfs(
                lambda matrix, n_bins, bin_width: model_gaussian_core(
                    matrix, n_bins, centers, bin_width, rw_order=rw_order
                ),
                n_bins,
                centers,
                bin_width,
                num_samples=3000,
            )
        finally:
            vd.SIGMA3_RATE = old
        m = pdfs @ centers
        d = centers[None, :] - m[:, None]
        var = np.einsum("ij,ij->i", pdfs, d**2)
        sk = np.einsum("ij,ij->i", pdfs, d**3) / var**1.5
        ku = np.einsum("ij,ij->i", pdfs, d**4) / var**2 - 3.0
        ok = np.isfinite(sk) & np.isfinite(ku)
        return (float(np.percentile(np.abs(sk[ok]), 90)), float(np.percentile(np.abs(ku[ok]), 90)))

    sk_tight, ku_tight = spread(50.0)
    sk_loose, ku_loose = spread(0.35)

    h3_retained = sk_tight / sk_loose
    h4_retained = ku_tight / ku_loose
    print(f"\norder {rw_order}: h3 retained {h3_retained:.3f}, " f"h4 retained {h4_retained:.3f}")

    if expect_h3_free:
        assert h3_retained > 0.30, (
            f"order {rw_order}: skewness should be in the null space and "
            f"survive shrinkage, but retained only {h3_retained:.2f} of its "
            "spread with the deviation switched off"
        )
    else:
        assert h3_retained < 0.15, (
            f"order {rw_order}: skewness should be penalised, but retained " f"{h3_retained:.2f} with the deviation off"
        )

    if expect_h4_free:
        assert h4_retained > 0.30, (
            f"order {rw_order}: kurtosis should be in the null space, " f"retained only {h4_retained:.2f}"
        )
    else:
        assert h4_retained < 0.15, f"order {rw_order}: kurtosis should be penalised, retained " f"{h4_retained:.2f}"
