"""
Test helpers and utilities for veldist tests.
"""

import numpy as np
from scipy import stats


def make_gaussian_distribution(mean, std):
    """Create a Gaussian PDF function."""

    def pdf(x):
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    return pdf


def make_uniform_distribution(low, high):
    """Create a uniform PDF function."""

    def pdf(x):
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        mask = (x >= low) & (x <= high)
        result[mask] = 1.0 / (high - low)
        return result

    return pdf


def make_bimodal_distribution(mean1, std1, mean2, std2, weight1=0.5):
    """Create a bimodal PDF (mixture of two Gaussians)."""
    pdf1 = make_gaussian_distribution(mean1, std1)
    pdf2 = make_gaussian_distribution(mean2, std2)
    weight2 = 1.0 - weight1

    def pdf(x):
        return weight1 * pdf1(x) + weight2 * pdf2(x)

    return pdf


def make_mixture_distribution(components):
    """
    Create a mixture distribution from multiple Gaussian components.

    Parameters
    ----------
    components : list of dict
        Each dict should contain 'mean', 'std', and 'weight'.
        Weights should sum to 1.0.
    """
    total_weight = sum(c["weight"] for c in components)
    if not np.isclose(total_weight, 1.0):
        msg = f"Component weights must sum to 1.0, got {total_weight}"
        raise ValueError(msg)

    pdfs = [make_gaussian_distribution(c["mean"], c["std"]) for c in components]
    weights = [c["weight"] for c in components]

    def pdf(x):
        result = np.zeros_like(x, dtype=float)
        for w, p in zip(weights, pdfs):
            result += w * p(x)
        return result

    return pdf


def generate_mock_data(true_pdf, n_stars, obs_errors, grid_range=(-50, 50), seed=None):
    """
    Generate synthetic observations from a true distribution.

    Parameters
    ----------
    true_pdf : callable
        Function that evaluates the true PDF at given points.
    n_stars : int
        Number of stars to generate.
    obs_errors : float or array-like
        Measurement errors. If float, all stars have same error.
        If array, must have length n_stars.
    grid_range : tuple, optional
        (min, max) range for sampling.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        - 'true_velocities': Intrinsic velocities
        - 'observed_velocities': True velocities + noise
        - 'errors': Measurement errors
        - 'true_pdf': The PDF function
    """
    if seed is not None:
        np.random.seed(seed)

    x_min, x_max = grid_range

    # Rejection sampling
    x_test = np.linspace(x_min, x_max, 10000)
    pdf_max = np.max(true_pdf(x_test)) * 1.1

    true_velocities = []
    while len(true_velocities) < n_stars:
        x_proposal = np.random.uniform(x_min, x_max, n_stars * 2)
        u = np.random.uniform(0, pdf_max, n_stars * 2)
        accepted = x_proposal[u < true_pdf(x_proposal)]
        true_velocities.extend(accepted)

    true_velocities = np.array(true_velocities[:n_stars])

    # Handle errors
    if np.isscalar(obs_errors):
        errors = np.ones(n_stars) * obs_errors
    else:
        errors = np.asarray(obs_errors)
        if len(errors) != n_stars:
            msg = f"obs_errors must be scalar or have length {n_stars}"
            raise ValueError(msg)

    # Add noise
    observed_velocities = true_velocities + np.random.normal(0, errors)

    return {
        "true_velocities": true_velocities,
        "observed_velocities": observed_velocities,
        "errors": errors,
        "true_pdf": true_pdf,
    }


def compute_ks_test(inferred_samples, true_samples):
    """
    Perform KS test between inferred and true velocity samples.

    Uses scipy.stats.ks_2samp.

    Parameters
    ----------
    inferred_samples : array-like
        Samples from the inferred distribution.
    true_samples : array-like
        Samples from the true distribution.

    Returns
    -------
    tuple
        (statistic, p_value) from the KS test.
    """
    return stats.ks_2samp(inferred_samples, true_samples)


def check_moment_recovery(inferred_moments, true_mean, true_std, tolerance_sigma=2):
    """
    Check if inferred moments match true values within tolerance.

    Parameters
    ----------
    inferred_moments : dict
        Output from compute_moments().
    true_mean : float
        True mean of the distribution.
    true_std : float
        True standard deviation.
    tolerance_sigma : float
        Number of standard deviations for tolerance.

    Returns
    -------
    dict
        - 'mean_ok': bool
        - 'std_ok': bool
        - 'mean_z_score': float
        - 'std_z_score': float
    """
    mean_inferred, mean_uncertainty = inferred_moments["mean"]
    std_inferred, std_uncertainty = inferred_moments["std"]

    mean_z = (
        abs(mean_inferred - true_mean) / mean_uncertainty
        if mean_uncertainty > 0
        else np.inf
    )
    std_z = (
        abs(std_inferred - true_std) / std_uncertainty
        if std_uncertainty > 0
        else np.inf
    )

    return {
        "mean_ok": mean_z < tolerance_sigma,
        "std_ok": std_z < tolerance_sigma,
        "mean_z_score": mean_z,
        "std_z_score": std_z,
    }
