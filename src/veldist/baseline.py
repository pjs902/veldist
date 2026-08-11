"""Reference estimators that veldist must beat, or at least match.

A method with a 1500-line probabilistic model and an MCMC sampler needs to
justify itself against the simplest thing that could work. For a Gaussian
LOSVD observed with known per-star Gaussian errors, that simplest thing is a
two-parameter maximum-likelihood fit, and it is not merely a strawman: it is
*exactly optimal* in that case. veldist cannot beat it on Gaussian truths.
Matching it there, and beating it on non-Gaussian truths, is the whole claim.

This module is deliberately free of JAX and NumPyro so it stays fast enough
to call once per spatial bin over a real catalogue (see
``ObservingProfile.from_data``).
"""

import numpy as np
from scipy import optimize

__all__ = ["gaussian_mle"]


def _neg_log_likelihood(params, vel, err_sq):
    """Negative log-likelihood of a Gaussian LOSVD convolved with per-star errors.

    ``params`` is ``(mu, log_sigma)``. Fitting ``log_sigma`` rather than
    ``sigma`` keeps the dispersion positive without a constrained optimiser,
    at the cost of being unable to represent sigma exactly 0 (see
    ``gaussian_mle``'s floor).
    """
    mu, log_sigma = params
    var = np.exp(2.0 * log_sigma) + err_sq
    return 0.5 * np.sum(np.log(var) + (vel - mu) ** 2 / var)


def gaussian_mle(vel, err):
    """Maximum-likelihood Gaussian deconvolution of discrete velocities.

    Maximises ``sum_i log N(v_i | mu, sqrt(sigma^2 + err_i^2))`` over
    ``(mu, sigma)``. Because each star's error enters its own variance, this
    is a true deconvolution: it returns the *intrinsic* dispersion, not the
    observed scatter, and stays correct when the errors are heteroscedastic.

    Parameters
    ----------
    vel : array-like, shape (n_stars,)
        Observed line-of-sight velocities, km/s.
    err : array-like, shape (n_stars,)
        Per-star measurement uncertainties (1 sigma), km/s. Must be positive.

    Returns
    -------
    dict
        ``'v_mean'``, ``'sigma'`` : the ML point estimates, km/s.
        ``'v_mean_err'``, ``'sigma_err'`` : 1 sigma uncertainties from the
        inverse Hessian at the optimum. For equal errors these reduce to the
        familiar ``sigma_tot/sqrt(N)`` and ``sigma_tot/sqrt(2N)``.

    Raises
    ------
    ValueError
        If *vel* and *err* differ in length, if fewer than 2 stars are given,
        or if any error is non-positive.

    Notes
    -----
    ``sigma`` is floored at 0. When the data are consistent with a delta
    function the optimiser drives ``log_sigma`` to negative infinity; the
    floor turns that into a clean 0 with a large reported ``sigma_err``.
    """
    vel = np.asarray(vel, dtype=float)
    err = np.asarray(err, dtype=float)

    if vel.shape != err.shape:
        msg = "vel and err must have the same length"
        raise ValueError(msg)
    if vel.size < 2:
        msg = "at least 2 stars are required"
        raise ValueError(msg)
    if np.any(err <= 0):
        msg = "all measurement errors must be positive"
        raise ValueError(msg)

    err_sq = err**2

    # Moment-based starting point: the observed variance minus the mean error
    # variance is an unbiased (if noisy, and possibly negative) estimate of
    # the intrinsic variance.
    mu0 = float(np.mean(vel))
    var0 = float(np.var(vel) - np.mean(err_sq))
    sigma0 = np.sqrt(var0) if var0 > 0 else 0.1 * float(np.std(vel) + np.mean(err))
    sigma0 = max(sigma0, 1e-3)

    res = optimize.minimize(
        _neg_log_likelihood,
        x0=np.array([mu0, np.log(sigma0)]),
        args=(vel, err_sq),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 10000},
    )

    mu_hat, log_sigma_hat = res.x
    sigma_hat = float(np.exp(log_sigma_hat))

    # Analytic Fisher information for (mu, sigma), evaluated at the optimum.
    # d2L/dmu2 = sum 1/v_i ; d2L/dsigma2 = sum 2*sigma^2/v_i^2 with v_i the
    # per-star total variance. The cross term vanishes for symmetric errors.
    var_i = sigma_hat**2 + err_sq
    fisher_mu = float(np.sum(1.0 / var_i))
    fisher_sigma = float(np.sum(2.0 * sigma_hat**2 / var_i**2))

    v_mean_err = 1.0 / np.sqrt(fisher_mu)
    sigma_err = 1.0 / np.sqrt(fisher_sigma) if fisher_sigma > 0 else np.inf

    return {
        "v_mean": float(mu_hat),
        "sigma": max(sigma_hat, 0.0),
        "v_mean_err": float(v_mean_err),
        "sigma_err": float(sigma_err),
    }
