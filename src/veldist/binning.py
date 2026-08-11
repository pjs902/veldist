"""Spatial binning targets based on information content rather than star count.

Voronoi binning conventionally equalises signal-to-noise across bins, and for
discrete tracers that is usually implemented as equalising the star count.
Star count is a proxy for information, and it is the wrong proxy whenever
measurement errors vary across the field: a hundred stars measured to 20 km/s
constrain a 10 km/s dispersion far less than a hundred measured to 1 km/s.

This module does not implement tessellation. ``vorbin``'s
``voronoi_2d_binning`` already accepts an ``sn_func(index, signal, noise)``
callback, documented as computing "any quantity the user wishes to equalize
across the bins", so all that is needed is the right callback and the right
target value.

Usage::

    from vorbin.voronoi_2d_binning import voronoi_2d_binning
    from veldist.binning import make_ivar_sn_func, target_ivar

    bin_id, *_ = voronoi_2d_binning(
        x, y,
        signal=np.zeros_like(err),   # unused by this sn_func
        noise=err,                   # per-star velocity errors, km/s
        target_sn=target_ivar(1.0),  # want v_mean to 1 km/s
        sn_func=make_ivar_sn_func(sigma=10.0),
        plot=False,
    )

Choosing ``sigma``: use a representative LOSVD dispersion for the field. The
result is insensitive to it wherever ``sigma`` dominates the errors, which is
exactly where ivar binning and star-count binning agree anyway. It matters in
the outer regions where errors approach the dispersion, which is the regime
this module exists for.
"""

import numpy as np

__all__ = ["make_ivar_sn_func", "target_ivar"]


def make_ivar_sn_func(sigma):
    """Build a ``vorbin`` ``sn_func`` that equalises information content.

    The returned callback computes ``sum_i 1/(sigma^2 + err_i^2)`` over the
    stars in a bin. Note the ``sigma^2``: a star's velocity constrains the
    LOSVD centroid only up to the intrinsic spread it was drawn from, so the
    relevant variance is that of the *observed* velocity. Using ``1/err_i^2``
    instead would claim unbounded information from perfectly measured stars
    and produce absurdly small bins.

    Parameters
    ----------
    sigma : float
        Representative LOSVD dispersion for the field, km/s. Must be positive.

    Returns
    -------
    callable
        ``f(index, signal, noise) -> float``, matching vorbin's ``sn_func``
        protocol. ``signal`` is accepted and ignored.

    Raises
    ------
    ValueError
        If *sigma* is not positive.
    """
    if sigma <= 0:
        msg = "sigma must be positive"
        raise ValueError(msg)

    var = float(sigma) ** 2

    def ivar_sn_func(index, signal, noise):  # noqa: ARG001 - vorbin's protocol
        err = np.asarray(noise, dtype=float)[index]
        return float(np.sum(1.0 / (var + err**2)))

    return ivar_sn_func


def target_ivar(precision):
    """Information content needed for a given precision on ``v_mean``.

    The Cramer-Rao bound on the mean velocity is ``1/sqrt(ivar)``, so this is
    just its inverse. The point is that it turns a science requirement
    ("mean velocities good to 1 km/s") into the number handed to vorbin,
    instead of a star count picked by eye.

    This bounds ``v_mean`` only. ``sigma`` and the shape statistics need more
    information, by a factor that depends on the LOSVD shape and is not
    analytic for this model. Use ``calibration.recovery_curve`` and its
    ``threshold`` method to measure it.

    Parameters
    ----------
    precision : float
        Desired 1 sigma uncertainty on ``v_mean``, km/s. Must be positive.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If *precision* is not positive.
    """
    if precision <= 0:
        msg = "precision must be positive"
        raise ValueError(msg)
    return 1.0 / float(precision) ** 2
