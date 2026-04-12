"""
Statistical analysis utilities for inferred velocity distributions.
"""

import numpy as np


__all__ = [
    "compute_moments",
]


def compute_moments(pdf_samples, grid_centers):
    """
    Compute statistical moments from PDF samples.

    Parameters
    ----------
    pdf_samples : array-like (n_samples, n_bins)
        MCMC samples of the probability mass function (each row sums to 1).
    grid_centers : array-like (n_bins,)
        Centers of the velocity bins.

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean': (mean, std) tuple
        - 'std': (mean, std) tuple
        - 'skewness': (mean, std) tuple
        - 'kurtosis': (mean, std) tuple for excess kurtosis
    """
    pdf_samples = np.asarray(pdf_samples)  # (n_samples, n_bins)
    grid_centers = np.asarray(grid_centers)  # (n_bins,)

    # --- Mean velocity for each posterior sample ---
    # Shape: (n_samples,)
    means = pdf_samples @ grid_centers

    # --- Centered residuals ---
    # Broadcast: (n_samples, n_bins) - (n_samples, 1)  ->  (n_samples, n_bins)
    delta = grid_centers[np.newaxis, :] - means[:, np.newaxis]

    # --- Variance and standard deviation ---
    # einsum 'ij,ij->i' is a row-wise dot product: sum_j(pdf[i,j] * delta[i,j]^2)
    variance = np.einsum("ij,ij->i", pdf_samples, delta**2)
    stds = np.sqrt(variance)

    # --- Skewness and excess kurtosis ---
    # Divide by std^3 / std^4 only where std > 0; set 0 elsewhere.
    safe_stds = np.where(stds > 0, stds, 1.0)

    skews = np.einsum("ij,ij->i", pdf_samples, delta**3) / safe_stds**3
    skews = np.where(stds > 0, skews, 0.0)

    kurts = np.einsum("ij,ij->i", pdf_samples, delta**4) / safe_stds**4 - 3
    kurts = np.where(stds > 0, kurts, 0.0)

    return {
        "mean": (np.mean(means), np.std(means)),
        "std": (np.mean(stds), np.std(stds)),
        "skewness": (np.mean(skews), np.std(skews)),
        "kurtosis": (np.mean(kurts), np.std(kurts)),
    }
