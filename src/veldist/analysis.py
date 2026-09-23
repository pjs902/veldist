"""
Statistical analysis utilities for inferred velocity distributions.
"""

import numpy as np
from scipy import optimize

__all__ = [
    "compute_moments",
    "within_cell_variance",
    "cdf_percentile",
    "tail_weight",
    "bimodality_score",
    "half_68ci",
    "truncate_pdf_samples",
    "compute_summary",
    "compute_summary_maps",
    "compute_percentile_summary",
    "compute_percentile_summary_maps",
    "gauss_hermite_fit",
]

# ---------------------------------------------------------------------------
# Within-cell (Sheppard) correction
# ---------------------------------------------------------------------------


def within_cell_variance(grid_centers, bin_width=None):
    """Return ``h**2 / 12``, the variance of a uniform distribution over one cell.

    The model infers probability MASS ``p_m`` per cell of width ``h``. The
    design matrix (``precompute_design_matrix`` in ``veldist.py``) integrates
    each star's error kernel over the cell, which treats the density as
    constant inside it, ``p(v) ~= p_m / h``. The density actually fitted
    therefore has

        ``Var(q) = sum_m p_m * (v_m - mu)**2 + h**2 / 12``

    Treating ``p_m`` as a point mass at the cell centre drops the ``h**2/12``
    term and biases sigma low by about ``h**2 / (12 * sigma)``. This is the 1D
    counterpart of the fix in ``calibration2d.py::_moments_from_pdf_samples_2d``
    (commit 2e2cbf8), whose docstring has the full derivation and the check
    (predicted and measured sigma bias agree within 4% at every resolution
    tested).

    Parameters
    ----------
    grid_centers : array-like, shape (n_bins,)
        Bin centres in ascending order. Used only to derive the cell width
        when *bin_width* is not given, which requires uniform spacing (checked
        to a relative tolerance of ``1e-6``).
    bin_width : float or None, optional
        Cell width ``h``. If ``None`` (default), it is taken as
        ``grid_centers[1] - grid_centers[0]``. Pass it explicitly when you have
        the grid dict (e.g. ``solver.grid["width"]``); that is exact whether or
        not the grid is uniform.

    Returns
    -------
    float
        ``bin_width**2 / 12``.

    Raises
    ------
    ValueError
        If *bin_width* is ``None`` and *grid_centers* has fewer than 2 points
        or is not uniformly spaced.
    """
    if bin_width is not None:
        return float(bin_width) ** 2 / 12.0

    grid_centers = np.asarray(grid_centers, dtype=float)
    if grid_centers.size < 2:
        msg = (
            "cannot derive bin_width from grid_centers with fewer than 2 "
            "points; pass bin_width explicitly"
        )
        raise ValueError(msg)

    diffs = np.diff(grid_centers)
    h = diffs[0]
    if not np.allclose(diffs, h, rtol=1e-6, atol=1e-9):
        msg = (
            "grid_centers is not uniformly spaced, so bin_width cannot be "
            "safely derived from grid_centers[1] - grid_centers[0]; pass "
            "bin_width explicitly (e.g. solver.grid['width'] for a uniform "
            "cell size, or a per-cell value)"
        )
        raise ValueError(msg)
    return float(h) ** 2 / 12.0


# ---------------------------------------------------------------------------
# Legacy API (kept for backward compatibility)
# ---------------------------------------------------------------------------


def compute_moments(pdf_samples, grid_centers, bin_width=None):
    """Compute moments from posterior LOSVD samples.

    .. deprecated::
        Kept for backward compatibility. New code should use
        :func:`compute_summary`, which returns the same quantities plus robust
        alternatives (median, IQR, tail weight, bimodality score) and reports
        the half-width of the 68% interval instead of the posterior standard
        deviation.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        Posterior samples of the probability mass per bin; each row sums to 1.
    grid_centers : array-like, shape (n_bins,)
        Bin centres, in km/s or another consistent unit.
    bin_width : float or None, optional
        Cell width ``h``, used to add the within-cell variance ``h**2/12``
        (see :func:`within_cell_variance`) so that ``std`` estimates the
        continuous dispersion the likelihood fits instead of being biased low
        by about ``h**2/(12*sigma)``. If ``None`` (default), ``h`` is derived
        from *grid_centers*, which must then be uniform.

    Returns
    -------
    dict
        Each value is ``(posterior_mean, posterior_std)``:

        ``'mean'``
            Mean velocity.
        ``'std'``
            Velocity dispersion.
        ``'skewness'``
            Skewness.
        ``'kurtosis'``
            Excess kurtosis.
    """
    pdf_samples = np.asarray(pdf_samples, dtype=float)
    grid_centers = np.asarray(grid_centers, dtype=float)
    cell_var = within_cell_variance(grid_centers, bin_width)

    # Mean (1st moment)
    means = pdf_samples @ grid_centers  # (n_samples,)

    # Central moments (vectorised)
    delta = grid_centers[np.newaxis, :] - means[:, np.newaxis]  # (n_s, n_bins)
    variance = np.einsum("ij,ij->i", pdf_samples, delta**2) + cell_var  # (n_samples,)
    stds = np.sqrt(variance)  # (n_samples,)

    # Skewness and excess kurtosis; guard against zero-dispersion samples
    safe_stds = np.where(stds > 0, stds, 1.0)
    skews = np.einsum("ij,ij->i", pdf_samples, delta**3) / safe_stds**3
    skews = np.where(stds > 0, skews, 0.0)
    kurts = (np.einsum("ij,ij->i", pdf_samples, delta**4) / safe_stds**4) - 3.0
    kurts = np.where(stds > 0, kurts, 0.0)

    return {
        "mean": (float(np.mean(means)), float(np.std(means))),
        "std": (float(np.mean(stds)), float(np.std(stds))),
        "skewness": (float(np.mean(skews)), float(np.std(skews))),
        "kurtosis": (float(np.mean(kurts)), float(np.std(kurts))),
    }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def cdf_percentile(pdf_samples, grid_centers, p):
    """Compute CDF percentile(s) for each posterior draw.

    Each draw is treated as a discrete distribution on *grid_centers*, and
    its cumulative distribution is interpolated at level(s) *p*.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        Posterior samples of the probability mass per bin; each row sums to 1.
    grid_centers : array-like, shape (n_bins,)
        Bin centres in ascending order.
    p : float or array-like
        Cumulative probability level(s) in [0, 1]. A scalar gives a 1-D
        result, an array a 2-D result.

    Returns
    -------
    ndarray
        Shape ``(n_samples,)`` for scalar *p*, or ``(n_samples, len(p))``.

    Examples
    --------
    Posterior median velocity for each draw:

    >>> v_median_samples = cdf_percentile(pdf_samples, grid_centers, 0.5)

    Q25 and Q75 together:

    >>> q25_q75 = cdf_percentile(pdf_samples, grid_centers, [0.25, 0.75])
    >>> iqr_samples = q25_q75[:, 1] - q25_q75[:, 0]

    No within-cell (``h**2/12``) correction is applied; that correction is
    specific to variances (see :func:`compute_percentile_summary`).
    """
    pdf_samples = np.asarray(pdf_samples, dtype=float)
    grid_centers = np.asarray(grid_centers, dtype=float)
    cdf = np.cumsum(pdf_samples, axis=1)  # (n_samples, n_bins)
    scalar_p = np.ndim(p) == 0
    p_arr = np.atleast_1d(np.asarray(p, dtype=float))
    # np.interp is not vectorised over the xp axis, so loop over samples.
    result = np.array(
        [np.interp(p_arr, cdf[s], grid_centers) for s in range(len(pdf_samples))]
    )  # (n_samples, len(p_arr))
    return result[:, 0] if scalar_p else result


def tail_weight(pdf_samples, grid_centers, means, stds):
    """Fraction of probability mass more than 1 sigma from the mean, per draw.

    A direct, model-free measure of how heavy the tails are, needing no series
    expansion; the non-parametric counterpart of the Gauss-Hermite *h4*. For a
    Gaussian it is ``1 - erf(1/sqrt(2)) ~ 0.3173``.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        Posterior samples of the probability mass per bin; each row sums to 1.
    grid_centers : array-like, shape (n_bins,)
        Bin centres.
    means : array-like, shape (n_samples,)
        Mean velocity of each draw, e.g. ``pdf_samples @ grid_centers``.
    stds : array-like, shape (n_samples,)
        Dispersion of each draw. This function only uses *stds* as the
        threshold; it computes no second moment itself. Pass the
        within-cell-corrected sigma (as :func:`compute_summary` does), so that
        "1 sigma" means the same sigma ``compute_summary`` reports. The
        threshold then moves out slightly (by a fraction of about
        ``h**2/(24*sigma**2)``), which is the intended effect.

    Returns
    -------
    ndarray, shape (n_samples,)
        Tail weight of each draw. Above 0.317 means heavier tails than a
        Gaussian (associated with radial anisotropy); below means lighter tails
        or a flat top (tangential anisotropy).

    Examples
    --------
    >>> means = pdf_samples @ grid_centers
    >>> stds  = np.sqrt(np.einsum('ij,ij->i', pdf_samples,
    ...                           (grid_centers - means[:, None])**2))
    >>> tw_samples = tail_weight(pdf_samples, grid_centers, means, stds)
    >>> print(f"tail weight = {np.median(tw_samples):.4f}")
    """
    pdf_samples = np.asarray(pdf_samples, dtype=float)
    grid_centers = np.asarray(grid_centers, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    delta = grid_centers[np.newaxis, :] - means[:, np.newaxis]  # (n_s, n_bins)
    outside = np.abs(delta) > stds[:, np.newaxis]  # bool (n_s, n_bins)
    return np.sum(pdf_samples * outside, axis=1)  # (n_samples,)


def bimodality_score(pdf_samples):
    """Count the peaks in the smoothed posterior-mean LOSVD.

    A diagnostic integer, not a posterior quantity: it is computed from the
    posterior-mean LOSVD, so it has no credible interval. A score of 2 or more
    means the distribution may be multimodal, in which case the mean, sigma,
    skewness and kurtosis can mislead and the full histogram should be
    inspected.

    The LOSVD is smoothed with a 3-point boxcar to ignore single-bin noise,
    and a peak must exceed 1% of the global maximum, to ignore spurious peaks
    in the tails.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        Posterior samples of the probability mass per bin; each row sums to 1.

    Returns
    -------
    int
        Number of local maxima found:

        ``1``
            Unimodal (the normal case).
        ``2``
            Bimodal: possibly counter-rotation, two kinematic components, or
            a contaminating population.
        ``>= 3``
            Irregular; look at it before interpreting any scalar summary.

    Notes
    -----
    Counting peaks in every draw and reporting a distribution over the count
    would be more principled. That is left for later; the posterior-mean
    version is enough to flag bins that need a closer look.

    Not affected by the within-cell (``h**2/12``) correction: it counts peaks
    in the raw mass and uses no dispersion.

    Examples
    --------
    >>> score = bimodality_score(solver.samples["intrinsic_pdf"])
    >>> if score >= 2:
    ...     print("Multimodal; inspect full histogram before trusting moments")
    """
    pdf_samples = np.asarray(pdf_samples, dtype=float)
    mean_pdf = np.mean(pdf_samples, axis=0)
    smoothed = np.convolve(mean_pdf, np.full(3, 1.0 / 3), mode="same")
    # Require each peak to exceed 1% of the global maximum to suppress
    # noise peaks in poorly constrained tail bins.
    min_height = 0.01 * smoothed.max()
    interior = smoothed[1:-1]
    left = smoothed[:-2]
    right = smoothed[2:]
    n_peaks = int(np.sum((interior > left) & (interior > right) & (interior > min_height)))
    return n_peaks


def half_68ci(samples):
    """Half-width of the 68% posterior credible interval.

    Returns ``(p84 - p16) / 2``, the symmetric uncertainty veldist reports for
    both LOSVD values and scalar summaries. It follows the BayesLOSVD
    convention and is the non-parametric counterpart of a 1-sigma error bar.

    Parameters
    ----------
    samples : array-like, shape (n_samples,)
        Posterior samples of a scalar, e.g. one metric evaluated on every
        draw.

    Returns
    -------
    float
        ``(p84 - p16) / 2``, in the units of *samples*.

    Notes
    -----
    For a Gaussian posterior this equals the posterior standard deviation.
    For skewed or heavy-tailed posteriors it can differ a lot, but it always
    means that the true value lies within +/- half_68ci of the median with
    about 68% posterior probability.

    Not affected by the within-cell (``h**2/12``) correction: it is a
    generic percentile spread of whatever samples it is given.

    Examples
    --------
    >>> v_mean_samples = pdf_samples @ grid_centers
    >>> uncertainty = half_68ci(v_mean_samples)
    >>> median = float(np.median(v_mean_samples))
    >>> print(f"v_mean = {median:.1f} +/- {uncertainty:.1f} km/s")
    """
    samples = np.asarray(samples, dtype=float)
    p16, p84 = np.percentile(samples, [16, 84])
    return float((p84 - p16) / 2.0)


def truncate_pdf_samples(pdf_samples, grid_centers, n_sigma=4.0):
    """Zero the far tails of each posterior draw and renormalise.

    The raw-sample counterpart of
    :meth:`~veldist.veldist.KinematicSolver.truncate_losvd`. The RW1 prior
    leaks a little posterior mass into bins far from the bulk of the
    distribution. Moments that weight residuals by a high power are very
    sensitive to this: for kurtosis, a bin at 5 sigma counts about 625 times
    as much as one at 1 sigma.

    ``truncate_losvd`` works on the per-bin summary in ``clipped_samples`` and
    does not renormalise. This function works on the full
    ``(n_samples, n_bins)`` array that :func:`compute_summary` uses, and
    renormalises each row after truncating so that moment calculations stay
    valid.

    Each draw is truncated using its own mean and dispersion rather than one
    global threshold, because the draws differ in both.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        Posterior samples of the probability mass per bin; each row sums to 1.
    grid_centers : array-like, shape (n_bins,)
        Bin centres.
    n_sigma : float, optional
        Zero the mass beyond this many (per-draw) dispersions from the mean.
        Default 4.0.

    Returns
    -------
    ndarray, shape (n_samples, n_bins)
        Truncated, renormalised samples. Every row still sums to 1, except a
        row whose mass all lies beyond the cut: renormalising zeros is
        undefined, so that row is returned unchanged.

    Examples
    --------
    >>> truncated = truncate_pdf_samples(solver.samples["intrinsic_pdf"],
    ...                                   solver.grid["centers"], n_sigma=4.0)
    >>> summary = compute_summary(truncated, solver.grid["centers"])
    """
    pdf_samples = np.asarray(pdf_samples, dtype=float)  # (n_samples, n_bins)
    grid_centers = np.asarray(grid_centers, dtype=float)  # (n_bins,)

    means = pdf_samples @ grid_centers  # (n_samples,)
    delta = grid_centers[np.newaxis, :] - means[:, np.newaxis]  # (n_s, n_bins)
    variance = np.einsum("ij,ij->i", pdf_samples, delta**2)  # (n_samples,)
    stds = np.sqrt(variance)  # (n_samples,)

    truncation_mask = np.abs(delta) > n_sigma * stds[:, np.newaxis]  # (n_s, n_bins)

    truncated = np.where(truncation_mask, 0.0, pdf_samples)
    row_sums = truncated.sum(axis=1, keepdims=True)

    # Guard against degenerate rows (entire mass truncated, or zero-mass
    # rows to begin with) where renormalisation is undefined. Leave those
    # rows unmodified rather than dividing by zero.
    safe = row_sums > 0
    renormalised = np.where(safe, truncated / np.where(safe, row_sums, 1.0), pdf_samples)

    return renormalised


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------


def compute_summary(pdf_samples, grid_centers, n_sigma_truncate=None, bin_width=None):
    """Compute scalar summaries of posterior LOSVD samples, suitable for maps.

    This is the main function for turning :class:`~veldist.KinematicSolver`
    output into kinematic maps. Every metric except ``bimodality_score`` is
    computed on each posterior draw separately, so the full uncertainty
    (measurement noise, finite star count, prior) carries through with no
    bootstrap or error propagation.

    Each metric is reported as ``(posterior_median, half_68ci)``, with the
    uncertainty ``(p84 - p16) / 2`` as for the LOSVD itself.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        Posterior samples of the probability mass per bin; each row sums to 1.
    grid_centers : array-like, shape (n_bins,)
        Bin centres, in km/s or another consistent unit.
    n_sigma_truncate : float or None, optional
        If given, apply :func:`truncate_pdf_samples` with this ``n_sigma``
        before computing moments, to reduce the RW1 tail-leakage bias (see the
        ``kurtosis`` note below). Default ``None``, no truncation.

        It is opt-in because truncation is a lossy repair that depends on the
        threshold. It throws away mass beyond the cut, which is right if that
        mass is prior leakage but wrong if the distribution really extends
        there, and truncating by default could bias results for users who
        have not checked for leakage.

        On Gaussian mocks with a 20-bin grid (``PLAN.md`` §1.3),
        ``n_sigma_truncate=3.0`` brings the median excess kurtosis from +1.78
        to +0.05, removing the bias. Looser cuts help less (4.0 gives +0.81,
        5.0 gives +1.63). In the coverage tests (``tests/test_coverage.py``, 25
        realisations per truth) it fixes calibration for a Gaussian truth
        (kurtosis coverage 0.000 to 0.840) and a mildly skewed one (0.000 to
        0.800). **It does not fix heavy-tailed or multimodal truths**: a
        Student-t (df = 6) truth stays at 0.000 and a bimodal
        counter-rotating truth at 0.080, because a fixed cut removes real tail
        mass along with the leakage. Use it only if you expect the LOSVD to be
        close to Gaussian or mildly skewed.
    bin_width : float or None, optional
        Cell width ``h``. ``h**2/12`` (see :func:`within_cell_variance`) is
        added to the point-mass variance so that ``sigma``, and the
        normalisation of ``skewness`` and ``kurtosis``, estimate the
        continuous quantity the likelihood fits instead of being biased low by
        about ``h**2/(12*sigma)``. If ``None`` (default), ``h`` is derived from
        *grid_centers*, which must be uniform (an error is raised if not).
        Passing ``solver.grid["width"]`` is exact and preferred. The
        correction is **not** applied to ``iqr`` or ``sigma_iqr``, which are
        percentiles (see :func:`compute_percentile_summary`).

    Returns
    -------
    dict
        Each key maps to a ``(median, half_68ci)`` tuple of floats, in the
        units of *grid_centers* for velocities and dimensionless for shapes,
        **except** ``'bimodality_score'``, which is a plain ``int``.

        **Location**

        ``'v_mean'``
            Mean velocity; GH counterpart *V*. Sensitive to tail
            contamination, so compare it with ``v_median``.
        ``'v_median'``
            Median velocity (CDF = 0.5). Robust to edge-bin contamination and
            heavy tails.
        ``'v_asymmetry'``
            Mean minus median. Near zero for a symmetric LOSVD, positive when
            a tail toward high velocities pulls the mean above the median.
            Closely related to *h3*, without needing higher moments.

        **Dispersion**

        ``'sigma'``
            Standard deviation of the LOSVD; GH counterpart *sigma*.
        ``'iqr'``
            Interquartile range Q75 - Q25, a dispersion measure insensitive to
            the tails.
        ``'sigma_iqr'``
            IQR / 1.3490, the Gaussian-equivalent dispersion from the IQR. For
            a Gaussian ``sigma_iqr ~= sigma``. ``sigma_iqr < sigma`` means
            heavy tails (radial anisotropy); ``sigma_iqr > sigma`` means a
            flat top (tangential anisotropy).

        **Shape**

        ``'skewness'``
            Standardised third central moment *gamma1*; zero for a symmetric
            distribution. GH counterpart: *h3* ~= -*gamma1* / sqrt(6). Note the
            sign: *gamma1* > 0 (a tail toward high velocities) gives *h3* < 0.
        ``'kurtosis'``
            Excess kurtosis *kappa*, the fourth central moment over sigma^4,
            minus 3; zero for a Gaussian. GH counterpart: *h4* ~= *kappa* /
            sqrt(24). Positive (peaked, heavy-tailed) suggests radial
            anisotropy; negative (flat-topped) suggests tangential anisotropy.

            .. note::
                The kurtosis bias came from the RW1 prior's flat null space,
                and the default prior no longer has it. Since commit 4b3bca2
                (2026-08-03), ``KinematicSolver.run()`` uses
                ``prior="gaussian_core"`` (Merritt 1997, AJ, 114, 228), whose
                infinite-smoothing limit is a Gaussian rather than a uniform
                distribution. This removes the +1.1 excess-kurtosis and +4-8%
                dispersion biases: for a Gaussian truth the kurtosis bias is
                0.00 and the sigma bias is within 3%. The prior-predictive
                median sigma is about 45 km/s on a 400 km/s grid, against
                about 115 km/s for a uniform distribution.

                ``n_sigma_truncate`` is therefore only needed with
                ``prior="rw1"``. With the default prior, heavy-tailed and
                multimodal truths still under-cover in kurtosis; the cause is
                still open (see ``docs/validation.md``,
                ``tests/test_coverage.py`` and ``tests/test_moment_bias.py``).
        ``'tail_weight'``
            Fraction of mass more than 1 *sigma* from the mean; 0.3173 for a
            Gaussian. A more direct anisotropy diagnostic than *h4*, because it
            assumes no expansion and stays meaningful for non-Gaussian shapes.
            See :func:`tail_weight`.

        **Diagnostic**

        ``'bimodality_score'``
            Number of peaks in the smoothed posterior-mean LOSVD (see
            :func:`bimodality_score`). 1 means unimodal; 2 or more means look
            at the histogram. No uncertainty is given.

    Notes
    -----
    Approximate Gauss-Hermite conversions, for ``|h3|`` and ``|h4|`` below
    about 0.2:

    .. code-block:: text

        h3 ~= -skewness / sqrt(6)
        h4 ~=  kurtosis / sqrt(24)

    These allow rough comparison with GH-based models and published maps; for
    literature-comparable values use :func:`gauss_hermite_fit`.

    Where ``bimodality_score >= 2``, treat the mean, sigma, skewness and
    kurtosis with care: the mean falls between the peaks, sigma is inflated by
    their separation, and the skewness mostly reflects which peak is taller.

    Examples
    --------
    >>> solver = KinematicSolver()
    >>> solver.setup_grid(center=0.0, width=200.0, n_bins=50)
    >>> solver.add_data(velocities, uncertainties)
    >>> solver.run()
    >>> summary = compute_summary(solver.samples["intrinsic_pdf"],
    ...                           solver.grid["centers"])
    >>> v, dv = summary["v_mean"]
    >>> s, ds = summary["sigma"]
    >>> print(f"V = {v:.1f} +/- {dv:.1f}  sigma = {s:.1f} +/- {ds:.1f}  km/s")
    """
    pdf_samples = np.asarray(pdf_samples, dtype=float)  # (n_samples, n_bins)
    grid_centers = np.asarray(grid_centers, dtype=float)  # (n_bins,)
    cell_var = within_cell_variance(grid_centers, bin_width)
    # kappa_4 of Uniform(cell) = -h**4/120, i.e. -(12*cell_var)**2/120
    cell_var_kurt = -((12.0 * cell_var) ** 2) / 120.0

    if n_sigma_truncate is not None:
        pdf_samples = truncate_pdf_samples(pdf_samples, grid_centers, n_sigma=n_sigma_truncate)

    # ------------------------------------------------------------------
    # Moment-based quantities (fully vectorised)
    # ------------------------------------------------------------------
    means = pdf_samples @ grid_centers  # (n_samples,)
    delta = grid_centers[np.newaxis, :] - means[:, np.newaxis]  # (n_s, n_bins)

    # Point-mass second moment, plus the within-cell (Sheppard) term so this
    # estimates Var(q) of the continuous piecewise-constant density the
    # likelihood actually fits, not the point-mass-at-centres quantity.
    variance = np.einsum("ij,ij->i", pdf_samples, delta**2) + cell_var  # (n_samples,)
    stds = np.sqrt(variance)  # (n_samples,)
    safe_stds = np.where(stds > 0, stds, 1.0)

    # skewness: the 3rd-cumulant within-cell correction is exactly zero by
    # symmetry of the uniform kernel, so the point-mass numerator is already
    # the right one. It is normalised by the CORRECTED sigma, since skewness
    # is defined relative to *the* sigma of the distribution.
    skews = np.einsum("ij,ij->i", pdf_samples, delta**3) / safe_stds**3
    skews = np.where(stds > 0, skews, 0.0)

    # kurtosis: correct the 4th CUMULANT, not the 4th moment. Binning to cell
    # centres is a convolution with Uniform(cell), and cumulants add under
    # convolution, so kappa_4(continuous) = kappa_4(point-mass) - h**4/120
    # exactly -- the same additive structure as the h**2/12 term above, and
    # NOT shape-dependent at leading order (the residual of the asymptotic
    # expansion is, and is an order smaller; measured in
    # docs/handoff-2d-tilt-recovery.md).
    #
    # This function previously divided an UNCORRECTED 4th moment by the
    # CORRECTED sigma**4. Those two choices are inconsistent, and the
    # inconsistency is not small: -6*k2*(h**2/12) in the numerator, i.e.
    # roughly -0.5 * (h/sigma)**2 in excess kurtosis. On MUSE's grid at its
    # narrowest bins (h/sigma = 0.78) it reported -0.272 excess kurtosis for
    # an exact, noiseless Gaussian -- which matched the -0.268 "bias" the
    # MUSE recovery curve was attributing to prior shrinkage.
    m4 = np.einsum("ij,ij->i", pdf_samples, delta**4)
    var_pm = variance - cell_var  # point-mass 2nd central moment
    k4 = (m4 - 3.0 * var_pm**2) + cell_var_kurt
    kurts = k4 / safe_stds**4
    kurts = np.where(stds > 0, kurts, 0.0)

    tw = tail_weight(pdf_samples, grid_centers, means, stds)  # (n_samples,)

    # ------------------------------------------------------------------
    # CDF-based quantities  (loop over samples; fast for ~1 000 draws)
    # ------------------------------------------------------------------
    # Single call returns (n_samples, 3) for [Q25, Q50, Q75]
    pctls = cdf_percentile(pdf_samples, grid_centers, np.array([0.25, 0.50, 0.75]))
    q25, medians, q75 = pctls[:, 0], pctls[:, 1], pctls[:, 2]

    iqr = q75 - q25  # (n_samples,)
    sigma_iqr = iqr / 1.3490  # Gaussian-equivalent sigma
    v_asym = means - medians  # (n_samples,)

    # ------------------------------------------------------------------
    # Bimodality score (scalar, from posterior mean, not per-sample)
    # ------------------------------------------------------------------
    bscore = bimodality_score(pdf_samples)

    # ------------------------------------------------------------------
    # Summarise each per-sample array as (median, half_68ci)
    # ------------------------------------------------------------------
    def _summarise(arr):
        p16, p50, p84 = np.percentile(arr, [16, 50, 84])
        return (float(p50), float((p84 - p16) / 2.0))

    return {
        "v_mean": _summarise(means),
        "v_median": _summarise(medians),
        "v_asymmetry": _summarise(v_asym),
        "sigma": _summarise(stds),
        "iqr": _summarise(iqr),
        "sigma_iqr": _summarise(sigma_iqr),
        "skewness": _summarise(skews),
        "kurtosis": _summarise(kurts),
        "tail_weight": _summarise(tw),
        "bimodality_score": bscore,
    }


def compute_summary_maps(solvers):
    """Compute :func:`compute_summary` for every bin returned by
    :func:`~veldist.fit_all_bins`.

    Runs :func:`compute_summary` on each :class:`~veldist.KinematicSolver`
    and collects the results into arrays of shape ``(n_bins,)``, one entry per
    spatial bin, ready for plotting as maps.

    Skipped bins (``None`` in *solvers*, e.g. below ``min_stars``) give
    ``NaN`` in every array, so the spatial indexing is kept.

    Parameters
    ----------
    solvers : list of :class:`~veldist.KinematicSolver` or None
        As returned by :func:`~veldist.fit_all_bins`. ``None`` entries become
        ``NaN``. At least one entry must be non-``None``.

    Returns
    -------
    dict
        One key per metric from :func:`compute_summary`, each holding:

        ``'median'`` : ndarray, shape (n_bins,)
            Posterior median; ``NaN`` for skipped bins. ``bimodality_score``
            is cast to float.
        ``'uncertainty'`` : ndarray, shape (n_bins,)
            Half-width of the 68% interval; ``NaN`` for skipped bins and for
            ``bimodality_score``, which has no interval.

    Raises
    ------
    ValueError
        If every entry in *solvers* is ``None``.
    """
    n_bins = len(solvers)
    # Determine metric names from the first non-None solver
    metrics = None
    for s in solvers:
        if s is not None:
            summary = compute_summary(
                s.samples["intrinsic_pdf"], s.grid["centers"], bin_width=s.grid.get("width")
            )
            metrics = list(summary.keys())
            break

    if metrics is None:
        raise ValueError("all solvers are None; no data to build maps from")

    # Build dict of arrays, NaN-filled
    maps = {}
    for m in metrics:
        maps[m] = {"median": np.full(n_bins, np.nan), "uncertainty": np.full(n_bins, np.nan)}

    for i, solver in enumerate(solvers):
        if solver is not None:
            summary = compute_summary(
                solver.samples["intrinsic_pdf"],
                solver.grid["centers"],
                bin_width=solver.grid.get("width"),
            )
            for m in metrics:
                if isinstance(summary[m], tuple):
                    maps[m]["median"][i], maps[m]["uncertainty"][i] = summary[m]
                else:
                    maps[m]["median"][i] = float(summary[m])  # e.g. bimodality_score
                    maps[m]["uncertainty"][i] = np.nan

    return maps


# Moors (1988) octile kurtosis of a standard Gaussian: ((P87.5-P62.5)+(P37.5-P12.5))/(P75-P25)
# evaluated at scipy.stats.norm.ppf. Exact analytic value (not a fit), used
# to zero kurtosis_pct on a Gaussian so it reads on the same scale as GH's h4
# (0 = Gaussian, not Moors' raw ~1.23).
_MOORS_KURTOSIS_GAUSSIAN = 1.2330951154852172


def compute_percentile_summary(pdf_samples, grid_centers):
    """Compute percentile-based shape statistics for each posterior draw.

    The ``skewness`` and ``kurtosis`` from ``compute_summary`` are ordinary
    moments, which a few stars in the tails can dominate. The statistics here
    are built only from CDF percentiles, so a single outlying star moves them
    by at most one bin width. They are not moments and will not match GH
    h3/h4 or ``compute_summary``'s values numerically; they are a more robust
    view of the same asymmetry and peakedness. They are computed per draw, so
    uncertainties carry through as elsewhere in this module.

    No within-cell (``h**2/12``) correction is applied, unlike in
    :func:`compute_summary`. That correction is Sheppard's correction for a
    *variance*: it comes from
    ``Var(uniform in cell) = Var(point mass at centre) + h**2/12``, and there
    is no equivalent for a percentile. The CDF of the piecewise-constant
    density the likelihood fits is piecewise *linear* between cell edges, so
    an exact percentile would interpolate against the edges. This function,
    via :func:`cdf_percentile`, interpolates the cumulative mass between
    *centres* instead. That is a separate, smaller bias (at most one cell) and
    is not addressed here; fixing it would mean changing ``cdf_percentile``,
    not adding a term. (Adding ``h**2/12`` to a percentile would not even have
    the right units.)

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        Posterior samples of the probability mass per bin; each row sums to 1.
    grid_centers : array-like, shape (n_bins,)
        Bin centres in ascending order.

    Returns
    -------
    dict
        Four keys, each ``(posterior_median, half_68ci)``:

        ``'median'``
            The 50th percentile, i.e. the LOSVD median.
        ``'sigma_pct'``
            ``(P84 - P16) / 2``, the percentile counterpart of the standard
            deviation (exact for a Gaussian).
        ``'skew_pct'``
            Bowley skewness ``((P75-P50)-(P50-P25)) / (P75-P25)``, in
            ``[-1, 1]``. Zero for a symmetric distribution, with the same sign
            convention as ``compute_summary``'s ``skewness``.
        ``'kurtosis_pct'``
            *Excess* Moors (1988) octile kurtosis,
            ``((P87.5-P62.5)+(P37.5-P12.5)) / (P75-P25) - 1.233...``, i.e. the
            Moors statistic minus its Gaussian value. Zero for a Gaussian and
            positive for peaked, heavy-tailed distributions, the same
            convention as GH's ``h4``. (The raw Moors statistic is about 1.23
            for a Gaussian.)
    """
    pdf_samples = np.asarray(pdf_samples, dtype=float)
    grid_centers = np.asarray(grid_centers, dtype=float)

    levels = [0.125, 0.16, 0.25, 0.375, 0.5, 0.625, 0.75, 0.84, 0.875]
    q = cdf_percentile(pdf_samples, grid_centers, levels)
    p125, p16, p25, p375, p50, p625, p75, p84, p875 = (q[:, i] for i in range(len(levels)))

    median_samples = p50
    sigma_pct_samples = (p84 - p16) / 2.0

    iqr = p75 - p25
    safe_iqr = np.where(iqr > 0, iqr, np.nan)
    skew_pct_samples = ((p75 - p50) - (p50 - p25)) / safe_iqr
    kurt_pct_samples = ((p875 - p625) + (p375 - p125)) / safe_iqr - _MOORS_KURTOSIS_GAUSSIAN

    return {
        "median": (float(np.median(median_samples)), half_68ci(median_samples)),
        "sigma_pct": (float(np.median(sigma_pct_samples)), half_68ci(sigma_pct_samples)),
        "skew_pct": (float(np.nanmedian(skew_pct_samples)), half_68ci(skew_pct_samples[~np.isnan(skew_pct_samples)])),
        "kurtosis_pct": (float(np.nanmedian(kurt_pct_samples)), half_68ci(kurt_pct_samples[~np.isnan(kurt_pct_samples)])),
    }


def compute_percentile_summary_maps(solvers):
    """Compute :func:`compute_percentile_summary` for every bin, the percentile
    counterpart of :func:`compute_summary_maps`.

    Parameters
    ----------
    solvers : list of :class:`~veldist.KinematicSolver` or None
        As returned by :func:`~veldist.fit_all_bins`. ``None`` entries become
        ``NaN``.

    Returns
    -------
    dict
        Keys ``'median'``, ``'sigma_pct'``, ``'skew_pct'``, ``'kurtosis_pct'``,
        each a dict of ``'median'`` and ``'uncertainty'`` arrays of shape
        ``(n_bins,)``, ``NaN`` for skipped bins.
    """
    n_bins = len(solvers)
    metrics = ["median", "sigma_pct", "skew_pct", "kurtosis_pct"]
    maps = {m: {"median": np.full(n_bins, np.nan), "uncertainty": np.full(n_bins, np.nan)} for m in metrics}

    any_solved = False
    for i, solver in enumerate(solvers):
        if solver is not None:
            any_solved = True
            summary = compute_percentile_summary(solver.samples["intrinsic_pdf"], solver.grid["centers"])
            for m in metrics:
                maps[m]["median"][i], maps[m]["uncertainty"][i] = summary[m]

    if not any_solved:
        msg = "all solvers are None; no data to build maps from"
        raise ValueError(msg)

    return maps


def _gh_basis(y):
    """Normalised Hermite polynomials H_3 and H_4 (van der Marel & Franx 1993).

    The normalisation makes the Gauss-Hermite series orthonormal under the
    Gaussian weight, so h3 and h4 are directly comparable with published
    values, not just proportional to them.
    """
    hermite3 = (2.0 * np.sqrt(2.0) * y**3 - 3.0 * np.sqrt(2.0) * y) / np.sqrt(6.0)
    hermite4 = (4.0 * y**4 - 12.0 * y**2 + 3.0) / np.sqrt(24.0)
    return hermite3, hermite4


def _gh_model(params, centers):
    """Gauss-Hermite probability mass on *centers* for ``(v, log_sigma, h3, h4)``."""
    v, log_sigma, h3, h4 = params
    sigma = np.exp(log_sigma)
    y = (centers - v) / sigma
    hermite3, hermite4 = _gh_basis(y)
    dens = np.exp(-0.5 * y**2) * (1.0 + h3 * hermite3 + h4 * hermite4)
    dens = np.clip(dens, 0.0, None)
    total = dens.sum()
    return dens / total if total > 0 else dens


def gauss_hermite_fit(pdf_samples, grid_centers, n_draws=500, seed=0):
    """Fit a Gauss-Hermite series to each posterior draw of the LOSVD.

    ``compute_percentile_summary`` gives robust shape statistics, but not on
    the Gauss-Hermite scale used in the dynamical-modelling literature. This
    function gives literature-comparable ``h3`` and ``h4`` with a posterior.
    It fits ``exp(-y^2/2) [1 + h3 H3(y) + h4 H4(y)]``, with
    ``y = (v - V)/sigma``, to each draw by least squares and summarises the
    resulting parameter samples.

    It fits each draw rather than the posterior mean because the mean of many
    LOSVDs is smoother than any one of them, which would bias ``h4`` low and
    understate the uncertainty on both coefficients.

    No within-cell (``h**2/12``) correction is applied, on purpose. This
    function never forms a variance from point masses; ``sigma_gh`` is a shape
    parameter of a continuous curve, fitted by matching the model's normalised
    density at the bin centres to the normalised masses. For a smooth density
    on a reasonably fine grid, ``p_m ~= q(v_m) * h`` to ``O(h**3)`` (midpoint
    rule), so both sides of the fit carry the same discretisation at leading
    order and it cancels. Adding ``h**2/12`` afterwards would correct for an
    error the fit does not make; the tests confirm that it recovers a planted
    ``sigma`` to within a few percent without it.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        Posterior samples of the probability mass per bin; each row sums to 1.
    grid_centers : array-like, shape (n_bins,)
        Bin centres in ascending order.
    n_draws : int
        Number of draws to fit. Each is a separate non-linear least-squares
        fit, so the cost is linear in this. If there are more samples, a
        random subset is used. The default of 500 gives a half-68% interval
        stable to a few percent; raise it only if that is the dominant
        uncertainty.
    seed : int
        Seed for choosing the subset, for reproducibility.

    Returns
    -------
    dict
        ``'v_gh'``, ``'sigma_gh'``, ``'h3'``, ``'h4'``, each
        ``(posterior_median, half_68ci)``.

        ``v_gh`` and ``sigma_gh`` are the Gauss-Hermite location and width.
        They differ from ``compute_summary``'s ``v_mean`` and ``sigma``
        (ordinary moments) whenever ``h4`` is non-zero. Report them together
        with ``h3``/``h4`` or not at all; mixing GH coefficients with moment
        widths is a common source of confusion in the literature.

    Notes
    -----
    Draws whose fit does not converge are dropped. If more than half fail,
    the LOSVD is probably not well described by a low-order GH series, and
    ``bimodality_score`` is the better diagnostic.
    """
    pdf_samples = np.asarray(pdf_samples, dtype=float)
    grid_centers = np.asarray(grid_centers, dtype=float)

    n_samples = pdf_samples.shape[0]
    if n_draws < n_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_samples, size=n_draws, replace=False)
    else:
        idx = np.arange(n_samples)

    fits = []
    for i in idx:
        target = pdf_samples[i]
        # Moment starting point. A bad start makes the solver find a
        # reflected solution with the sign of h3 flipped.
        mean0 = float(np.sum(target * grid_centers))
        var0 = float(np.sum(target * (grid_centers - mean0) ** 2))
        if not np.isfinite(var0) or var0 <= 0:
            continue
        x0 = np.array([mean0, 0.5 * np.log(var0), 0.0, 0.0])

        try:
            res = optimize.least_squares(
                lambda p, t=target: _gh_model(p, grid_centers) - t,
                x0=x0,
                method="lm",
                max_nfev=2000,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not res.success:
            continue
        v, log_sigma, h3, h4 = res.x
        fits.append((v, np.exp(log_sigma), h3, h4))

    if len(fits) < 2:
        nan_pair = (float("nan"), float("nan"))
        return {"v_gh": nan_pair, "sigma_gh": nan_pair, "h3": nan_pair, "h4": nan_pair}

    fits = np.asarray(fits)
    names = ["v_gh", "sigma_gh", "h3", "h4"]
    return {name: (float(np.median(fits[:, j])), half_68ci(fits[:, j])) for j, name in enumerate(names)}
