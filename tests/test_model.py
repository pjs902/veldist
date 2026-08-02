"""Tests for the NumPyro model definition itself (veldist.veldist.model).

Covers PLAN.md §0.1 (no dead total_flux parameter), §0.2 (translation-
invariant random-walk prior), and §0.3 (grid-resolution-invariant smoothness
prior).
"""

import jax
import numpy as np
import pytest
from numpyro.infer import MCMC, NUTS
from scipy.stats import skewnorm

from veldist import KinematicSolver, precompute_design_matrix
from veldist.veldist import model


def _run_model(matrix, n_bins, bin_width, seed, num_warmup=100, num_samples=200):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), matrix=matrix, n_bins=n_bins, bin_width=bin_width)
    return mcmc.get_samples()


def test_model_has_no_dead_parameters():
    """total_flux was removed; only x, smoothness_sigma, intrinsic_pdf remain."""
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 5, 40)
    err = np.full(40, 1.0)
    centers = np.linspace(-15, 15, 20)
    matrix = precompute_design_matrix(obs, err, centers)

    samples = _run_model(matrix, 20, bin_width=1.5, seed=0, num_warmup=50, num_samples=50)

    assert set(samples.keys()) == {"x", "smoothness_sigma", "intrinsic_pdf"}


def test_intrinsic_pdf_is_mass_not_density():
    """Every posterior row must sum to 1 regardless of bin width."""
    rng = np.random.default_rng(1)
    obs = rng.normal(0, 5, 40)
    err = np.full(40, 1.0)
    centers = np.linspace(-15, 15, 20)
    matrix = precompute_design_matrix(obs, err, centers)

    samples = _run_model(matrix, 20, bin_width=1.5, seed=1, num_warmup=50, num_samples=50)

    pdf = np.asarray(samples["intrinsic_pdf"])
    np.testing.assert_allclose(pdf.sum(axis=1), 1.0, atol=1e-5)


def test_smoothness_sigma_prior_normalisation_not_collapsed():
    """The -(K-1) log(sigma_step) term must be present, or sigma collapses to ~0."""
    rng = np.random.default_rng(2)
    obs = rng.normal(0, 5, 200)
    err = np.full(200, 1.0)
    centers = np.linspace(-20, 20, 30)
    matrix = precompute_design_matrix(obs, err, centers)

    samples = _run_model(matrix, 30, bin_width=1.33, seed=2, num_warmup=300, num_samples=400)

    sigma = np.asarray(samples["smoothness_sigma"])
    # A collapsed prior drives sigma to ~1e-8 or below; a healthy posterior
    # should sit well above the HalfNormal(0.1) prior's numerical floor.
    assert np.median(sigma) > 1e-3


@pytest.mark.slow
def test_prior_is_symmetric_under_velocity_reversal():
    """Fitting skewed data forwards vs. mirrored should give equal-and-opposite
    skewness — confirms the RW1 prior has no bin-index asymmetry (PLAN.md §0.2).
    """
    rng = np.random.default_rng(7)
    true_v = skewnorm.rvs(a=6, loc=-8, scale=10, size=400, random_state=rng)
    err = np.full(400, 2.5)
    obs = true_v + rng.normal(0, err)

    def fit(vel):
        s = KinematicSolver()
        s.setup_grid(center=0, width=80, n_bins=30)
        s.add_data(vel=vel, err=err)
        return s, s.run(num_warmup=400, num_samples=800, gpu=False, seed=11)

    s_fwd, samp_fwd = fit(obs)
    _, samp_rev = fit(-obs)

    def skewness(samp, centers):
        pdf = np.asarray(samp["intrinsic_pdf"])
        m = pdf @ centers
        d = centers[None, :] - m[:, None]
        var = np.einsum("ij,ij->i", pdf, d**2)
        std = np.sqrt(var)
        return np.einsum("ij,ij->i", pdf, d**3) / std**3

    centers = s_fwd.grid["centers"]
    sk_fwd = np.median(skewness(samp_fwd, centers))
    sk_rev = np.median(skewness(samp_rev, centers))

    # Under a translation-invariant prior these should cancel; under the old
    # pinned-cumsum prior they would not.
    assert abs(sk_fwd + sk_rev) < 0.15


@pytest.mark.slow
def test_smoothness_prior_is_resolution_invariant():
    """Same mock data fit on two grid resolutions should agree within
    posterior uncertainty (PLAN.md §0.3 / §1's resolution-consistency test).
    """
    rng = np.random.default_rng(3)
    true_v = rng.normal(5, 12, 500)
    err = np.full(500, 3.0)
    obs = true_v + rng.normal(0, err)

    def fit(n_bins):
        s = KinematicSolver()
        s.setup_grid(center=5, width=100, n_bins=n_bins)
        s.add_data(vel=obs, err=err)
        samp = s.run(num_warmup=400, num_samples=800, gpu=False, seed=21)
        return s, samp

    def stats(s, samp):
        pdf = np.asarray(samp["intrinsic_pdf"])
        c = s.grid["centers"]
        m = pdf @ c
        d = c[None, :] - m[:, None]
        var = np.einsum("ij,ij->i", pdf, d**2)
        return m.mean(), m.std(), np.sqrt(var).mean(), np.sqrt(var).std()

    s40, samp40 = fit(40)
    s80, samp80 = fit(80)

    m40, dm40, sig40, dsig40 = stats(s40, samp40)
    m80, dm80, sig80, dsig80 = stats(s80, samp80)

    tol_m = max(dm40, dm80)
    tol_sig = max(dsig40, dsig80)
    assert abs(m40 - m80) < 2 * tol_m
    assert abs(sig40 - sig80) < 2 * tol_sig
