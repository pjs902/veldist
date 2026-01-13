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
    pdf_samples = np.asarray(pdf_samples)
    grid_centers = np.asarray(grid_centers)

    n_samples = pdf_samples.shape[0]
    means = np.zeros(n_samples)
    stds = np.zeros(n_samples)
    skews = np.zeros(n_samples)
    kurts = np.zeros(n_samples)

    for i in range(n_samples):
        pdf = pdf_samples[i]
        # Mean
        mean = np.sum(grid_centers * pdf)
        means[i] = mean

        # Variance and std
        variance = np.sum(((grid_centers - mean) ** 2) * pdf)
        stds[i] = np.sqrt(variance)

        # Skewness
        if stds[i] > 0:
            skews[i] = np.sum(((grid_centers - mean) ** 3) * pdf) / (stds[i] ** 3)
        else:
            skews[i] = 0

        # Excess kurtosis
        if stds[i] > 0:
            kurts[i] = (np.sum(((grid_centers - mean) ** 4) * pdf) / (stds[i] ** 4)) - 3
        else:
            kurts[i] = 0

    return {
        "mean": (np.mean(means), np.std(means)),
        "std": (np.mean(stds), np.std(stds)),
        "skewness": (np.mean(skews), np.std(skews)),
        "kurtosis": (np.mean(kurts), np.std(kurts)),
    }
