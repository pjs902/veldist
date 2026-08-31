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
    """
    ``h**2 / 12``, the exact variance of a Uniform(cell) distribution.

    The velocity distribution the model actually infers is per-cell
    probability MASS ``p_m`` on a grid of cell width ``h``. The likelihood's
    design matrix (see ``precompute_design_matrix`` in ``veldist.py``)
    INTEGRATES each star's error kernel over the cell -- a piecewise-constant
    assumption for the density, ``p(v) ~= p_m / h`` within cell ``m``. The
    density the model actually fits therefore has

        ``Var(q) = sum_m p_m * (v_m - mu)**2 + h**2 / 12``

    Treating ``p_m`` as a point mass at the cell centre (``sum_m p_m *
    (v_m - mu)**2`` alone, with no within-cell term) omits the ``h**2/12``
    piece and is biased low by approximately ``h**2 / (12 * sigma)`` in
    sigma. This is the 1D counterpart of the fix already applied in
    ``calibration2d.py::_moments_from_pdf_samples_2d`` (commit 2e2cbf8); see
    that function's docstring for the full three-way derivation (likelihood
    vs. point-mass estimator vs. discretised-truth target) and the
    confirming numbers (predicted vs. measured sigma bias within 4% at
    every tested resolution).

    Parameters
    ----------
    grid_centers : array-like, shape (n_bins,)
        Bin-centre velocities, in ascending order. Used only to derive the
        cell width when *bin_width* is not given; the grid must be uniformly
        spaced for that derivation to be valid (checked to a relative
        tolerance of ``1e-6``).
    bin_width : float or None, optional
        Cell width ``h``. If ``None`` (default), derived as
        ``grid_centers[1] - grid_centers[0]``. Passing it explicitly is
        preferable whenever the caller already has the grid dict (e.g.
        ``solver.grid["width"]``), since that is exact regardless of grid
        uniformity; the derived value assumes -- and checks -- uniform
        spacing.

    Returns
    -------
    float
        ``bin_width**2 / 12``.

    Raises
    ------
    ValueError
        If *bin_width* is ``None`` and *grid_centers* has fewer than 2
        points, or is not uniformly spaced.
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
    """
    Compute statistical moments from posterior LOSVD samples.

    .. deprecated::
        This function is retained for backward compatibility.  For new code,
        prefer :func:`compute_summary`, which returns the same quantities plus
        robust alternatives (median, IQR, tail weight, bimodality score) and
        uses the posterior half-68% CI as its uncertainty convention instead
        of the posterior standard deviation.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        MCMC samples of the probability mass function.  Each row must sum to 1.
    grid_centers : array-like, shape (n_bins,)
        Centres of the velocity bins (km/s or consistent velocity unit).
    bin_width : float or None, optional
        Grid cell width ``h``, used to add the within-cell variance
        ``h**2/12`` (see :func:`within_cell_variance`) so ``std`` estimates
        the same continuous quantity the likelihood fits rather than being
        biased low by ``~h**2/(12*sigma)``. Defaults to ``None``, which
        derives ``h`` from *grid_centers* (requires uniform spacing); pass
        it explicitly for a non-uniform grid or to avoid the derivation.

    Returns
    -------
    dict
        Keys and values:

        ``'mean'`` : (float, float)
            ``(posterior_mean, posterior_std)`` of the flux-weighted mean
            velocity across MCMC samples.
        ``'std'`` : (float, float)
            ``(posterior_mean, posterior_std)`` of the velocity dispersion.
        ``'skewness'`` : (float, float)
            ``(posterior_mean, posterior_std)`` of the skewness.
        ``'kurtosis'`` : (float, float)
            ``(posterior_mean, posterior_std)`` of the excess kurtosis.
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
    """
    Compute the CDF percentile(s) for each posterior draw.

    For each MCMC sample, treats the probability mass as a discrete
    distribution over *grid_centers* and interpolates the cumulative
    distribution at level(s) *p*.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        MCMC samples of the probability mass function.  Each row must sum to 1
        and *grid_centers* must be monotonically increasing.
    grid_centers : array-like, shape (n_bins,)
        Bin-centre velocities in ascending order.
    p : float or array-like
        Cumulative probability level(s) in the interval [0, 1].  Scalar input
        returns a 1-D result; array input returns a 2-D result.

    Returns
    -------
    ndarray
        Shape ``(n_samples,)`` when *p* is a scalar, or
        ``(n_samples, len(p))`` when *p* is array-like.

    Examples
    --------
    Compute the posterior median velocity for each MCMC sample:

    >>> v_median_samples = cdf_percentile(pdf_samples, grid_centers, 0.5)

    Compute Q25 and Q75 simultaneously:

    >>> q25_q75 = cdf_percentile(pdf_samples, grid_centers, [0.25, 0.75])
    >>> iqr_samples = q25_q75[:, 1] - q25_q75[:, 0]

    No within-cell (``h**2/12``) correction is applied here; see
    :func:`compute_percentile_summary`'s docstring for why that correction
    -- specific to variances -- does not apply to percentiles at all.
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
    """
    Fraction of probability mass outside +/-1 sigma of the mean, per MCMC sample.

    This is a direct, model-free measure of tail heaviness that can be
    interpreted without any expansion assumption.  It is the non-parametric
    analogue of the Gauss-Hermite *h4* coefficient.  For a Gaussian the value
    is exactly ``1 - erf(1/sqrt(2)) ~ 0.3173``.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        MCMC samples of the probability mass function.  Each row sums to 1.
    grid_centers : array-like, shape (n_bins,)
        Bin-centre velocities.
    means : array-like, shape (n_samples,)
        Per-sample flux-weighted mean velocities.  Compute as
        ``pdf_samples @ grid_centers`` if not already available.
    stds : array-like, shape (n_samples,)
        Per-sample velocity dispersions (standard deviations of the LOSVD).
        This function performs no second-moment computation itself -- it
        only thresholds against whatever *stds* the caller passes in. Pass
        the within-cell-corrected sigma (e.g. from :func:`compute_summary`,
        which is what it is called with internally) so "outside +/-1 sigma"
        means the same physical sigma that ``compute_summary`` reports; the
        threshold shifts outward accordingly (fractionally by
        ``~h**2/(24*sigma**2)``), which is the correct, if small,
        consequence of using the corrected sigma, not a separate bug.

    Returns
    -------
    ndarray, shape (n_samples,)
        Tail-weight value for each MCMC sample.  Values greater than 0.317
        indicate heavier tails than Gaussian (associated with radial velocity
        anisotropy); values below 0.317 indicate lighter tails / flat top
        (tangential anisotropy).

    Notes
    -----
    The Gaussian reference value can be computed as::

        from math import erf, sqrt
        gaussian_tail_weight = 1 - erf(1 / sqrt(2))   # = 0.3173

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
    """
    Count the number of peaks in the smoothed posterior-mean LOSVD.

    This is a diagnostic integer flag, not a posterior quantity.  It is
    computed from the posterior-mean LOSVD (not from individual samples), so
    no credible interval is available.  Bins with a score of 2 or more
    indicate a potentially multimodal distribution where GH-analogue summary
    statistics (mean, sigma, skewness, kurtosis) may be misleading and the
    full histogram shape should be inspected.

    Peak detection uses a 3-point boxcar to smooth over single-bin noise, and
    requires each candidate peak to exceed 1% of the global maximum to
    suppress spurious detections in low-probability tails.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        MCMC samples of the probability mass function.  Each row sums to 1.

    Returns
    -------
    int
        Number of local maxima detected.  Typical values:

        ``1``
            Unimodal distribution (normal case).
        ``2``
            Bimodal; could indicate counter-rotation, two kinematic
            components, or a contaminating population.
        ``>= 3``
            Highly irregular; inspect visually before interpreting any
            scalar summaries.

    Notes
    -----
    A more principled treatment would count peaks on every individual MCMC
    sample and return a distribution over the count.  This is deferred as a
    future improvement; the posterior-mean approach suffices to identify bins
    that require visual inspection.

    Unaffected by the within-cell (``h**2/12``) variance correction
    elsewhere in this module: this is a peak count on the raw PMF shape, not
    a second moment, and takes no dispersion estimate as input.

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
    """
    Half-width of the 68% posterior credible interval.

    Computes ``(p84 - p16) / 2``, the symmetric +/- uncertainty reported
    throughout veldist for both LOSVD values and scalar summary metrics.
    This matches the BayesLOSVD convention and is the natural non-parametric
    analogue of a 1-sigma Gaussian error bar.

    Parameters
    ----------
    samples : array-like, shape (n_samples,)
        Posterior samples of a scalar quantity (e.g., one metric evaluated
        over all MCMC draws).

    Returns
    -------
    float
        ``(p84 - p16) / 2``, in the same units as *samples*.

    Notes
    -----
    For a Gaussian posterior this equals the posterior standard deviation.
    For skewed or heavy-tailed posteriors it can differ substantially, but
    it always has the interpretation "the true value lies within +/-half_68ci
    of the median with approximately 68% posterior probability."

    Unaffected by the within-cell (``h**2/12``) variance correction
    elsewhere in this module: it is a generic percentile-spread operator on
    whatever scalar *samples* it is handed (already-corrected sigma
    samples, GH parameters, anything), not a second-moment computation
    itself.

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
    """
    Zero far-edge tail mass in each posterior draw and renormalise.

    This is the raw-sample analogue of
    :meth:`~veldist.veldist.KinematicSolver.truncate_losvd`. The RW1
    smoothness prior used during inference leaks a small amount of posterior
    mass into velocity-grid bins far from the bulk of the distribution.  For
    moments that weight residuals by a high power (e.g. kurtosis's fourth
    power), even a tiny amount of far-edge mass can produce a large bias (a
    bin at 5 sigma carries ~625x the weight of a bin at 1 sigma).

    Unlike ``truncate_losvd``, which operates on an already-fixed scalar
    ``clipped_samples`` summary and never renormalises, this function
    operates on the full ``(n_samples, n_bins)`` PMF array that
    :func:`compute_summary` consumes, where every row is assumed to sum to
    1.  Each row is renormalised after truncation so downstream moment
    calculations (which divide by the total probability implicitly, via
    ``pdf_samples @ grid_centers``, etc.) remain valid.

    Truncation is applied **per draw**, using that draw's own mean and
    dispersion, rather than a single global threshold computed from (e.g.)
    the posterior-mean LOSVD.  Posterior draws vary in their bulk mean and
    dispersion; per-row truncation tracks each draw's own scale.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        MCMC samples of the probability mass function.  Each row must sum
        to 1.
    grid_centers : array-like, shape (n_bins,)
        Centres of the velocity bins.
    n_sigma : float, optional
        Number of (per-draw) velocity dispersions beyond which to zero the
        probability mass.  Default 4.0.

    Returns
    -------
    ndarray, shape (n_samples, n_bins)
        Truncated and renormalised PMF samples.  Each row still sums to 1
        (to floating-point precision), except for the degenerate case of a
        row whose entire mass falls outside ``n_sigma`` sigma of its own
        mean, which is returned unmodified (renormalising an all-zero row
        is undefined) since that indicates every bin was truncated, not
        that a residual leak was suppressed.

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
    """
    Compute spatially mappable scalar summaries from posterior LOSVD samples.

    This is the primary analysis function for extracting kinematic maps from
    :class:`~veldist.KinematicSolver` output.  All metrics except
    ``bimodality_score`` are evaluated independently on every MCMC sample,
    so the full posterior uncertainty (measurement noise, finite star count,
    prior regularisation) is propagated automatically with no separate
    bootstrap or error-propagation step.

    Each metric is summarised as ``(posterior_median, half_68ci)`` using the
    same convention as the LOSVD itself: the reported uncertainty is
    ``(p84 - p16) / 2``.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        MCMC samples of the probability mass function.  Each row must sum to 1.
    grid_centers : array-like, shape (n_bins,)
        Centres of the velocity bins (km/s or consistent velocity unit).
    n_sigma_truncate : float or None, optional
        If given, apply :func:`truncate_pdf_samples` with this ``n_sigma``
        to *pdf_samples* before computing moment-based quantities, to
        mitigate the RW1 tail-leakage bias described in the ``kurtosis``
        warning below.  Default ``None`` (no truncation, fully backward-
        compatible with existing callers).  This is opt-in rather than on-by-
        default because truncation is a lossy, threshold-dependent repair:
        it discards posterior mass at the chosen cut, which is desirable when
        that mass is known prior leakage but undesirable if a distribution
        has real support there (e.g. a deliberately wide grid
        margin for a heavy-tailed truth). Silently truncating by default
        would risk quietly biasing results for users who have not diagnosed
        whether leakage is present in their setup. Empirical testing (mock
        Gaussian realisations on a 20-bin grid, see ``PLAN.md`` §1.3) found
        ``n_sigma_truncate=3.0`` reduces median excess kurtosis from +1.78
        (untruncated) to +0.05; i.e. it removes the observed bias for a
        Gaussian truth. Looser cuts recover less of the bias (4.0 -> +0.81,
        5.0 -> +1.63, barely better than untruncated). Frequentist coverage
        testing (``tests/test_coverage.py``, n=25 realisations per truth)
        confirms this fixes calibration for a Gaussian truth (kurtosis
        coverage 0.000 -> 0.840) and for a mildly skewed truth (skew-normal:
        0.000 -> 0.800), both inside the nominal 68% band. **It does not fix
        heavy-tailed or multimodal truths**: a Student-t(df=6) truth (true
        excess kurtosis 2.82) stays at 0.000 coverage, and a bimodal counter-
        rotation truth stays at 0.080, because a fixed ``n_sigma`` cut
        removes some of their real tail mass along with the leaked mass.
        Use ``n_sigma_truncate`` when you have reason to believe your true
        LOSVD is close to Gaussian or mildly skewed; it is not a general cure
        for kurtosis bias on heavy-tailed or multimodal truths.
    bin_width : float or None, optional
        Grid cell width ``h``, passed to :func:`within_cell_variance` and
        added to the point-mass ``sigma**2`` so ``sigma`` (and the
        ``skewness``/``kurtosis`` normalisation) estimates the continuous
        quantity the likelihood actually fits rather than being biased low
        by ``~h**2/(12*sigma)`` (see that function's docstring). Defaults to
        ``None``, which derives ``h`` from *grid_centers* assuming uniform
        spacing (validated; raises if the grid is not uniform). Prefer
        passing ``solver.grid["width"]`` explicitly when available -- it is
        exact regardless of grid uniformity. This correction is **not**
        applied to ``iqr``/``sigma_iqr`` (percentile-based; see
        :func:`compute_percentile_summary`'s docstring for why a within-cell
        term is not the right correction for order statistics).

    Returns
    -------
    dict
        Each key maps to a ``(median, half_68ci)`` tuple of floats, in the
        same units as *grid_centers* for velocity quantities and dimensionless
        for shape metrics, **except** ``'bimodality_score'``, which is a
        plain ``int``.

        **Location**

        ``'v_mean'``
            Flux-weighted mean velocity.  GH analogue: *V*.  Sensitive to
            tail contamination; compare with ``v_median`` as a cross-check.
        ``'v_median'``
            Median velocity (CDF = 0.5).  Robust against edge-bin
            contamination and heavy tails.
        ``'v_asymmetry'``
            Mean minus median.  Near zero for symmetric LOSVDs; the sign
            mirrors that of the low-velocity tail (positive = mean pulled
            toward higher velocities by a trailing tail).  Closely related
            to *h3*, but does not require computing higher-order moments.

        **Dispersion**

        ``'sigma'``
            Standard deviation of the LOSVD.  GH analogue: *sigma*.
        ``'iqr'``
            Interquartile range Q75 - Q25.  Robust dispersion estimate
            insensitive to tail contamination.
        ``'sigma_iqr'``
            IQR / 1.3490, the Gaussian-equivalent dispersion derived from
            the IQR.  For a Gaussian, ``sigma_iqr ~= sigma``.
            ``sigma_iqr < sigma`` implies heavy tails (radial anisotropy);
            ``sigma_iqr > sigma`` implies flat top (tangential anisotropy).

        **Shape**

        ``'skewness'``
            Normalised third central moment *gamma1*.  Zero for symmetric
            distributions.  GH analogue: *h3* ~= -*gamma1* / sqrt(6).  Note
            sign: *gamma1* > 0 (right-skewed, trailing tail) implies *h3* <
            0, the expected signal on the receding side of a rotating system.
        ``'kurtosis'``
            Excess kurtosis *kappa* = fourth central moment / sigma4 - 3.
            Zero for a Gaussian.  GH analogue: *h4* ~= *kappa* / sqrt(24).
            Positive (leptokurtic) implies radially anisotropic; negative
            (platykurtic) implies tangentially anisotropic / flat-topped.

            .. warning::
                **The root cause of the kurtosis bias, the RW1 prior's flat
                null space, has been fixed in the default prior.** As of
                commit 4b3bca2 (2026-08-03), ``KinematicSolver.run()``
                defaults to ``prior="gaussian_core"`` (Merritt 1997, AJ, 114,
                228), which replaces the RW1 prior's uniform-over-grid limit
                with a Gaussian null space. This eliminates the +1.1 excess-
                kurtosis and +4-8% velocity-dispersion biases at source.
                Prior-predictive median sigma is ~45 km/s on a 400 km/s grid
                (vs. ~115 for uniform). Kurtosis bias for a Gaussian truth is
                0.00; sigma bias is within 3%.

                The ``n_sigma_truncate`` mitigation is now a **legacy option
                needed only when using** ``prior="rw1"``. With the default
                Gaussian-core prior, truncation is not required for Gaussian
                or mildly skewed truths. Heavy-tailed and multimodal truths
                still show under-coverage in kurtosis even with the Gaussian-
                core prior: this is an inherent finite-data limitation of any
                smoothness prior at N=150, not the flat-null-space bug. See
                ``tests/test_coverage.py``, ``tests/test_moment_bias.py``,
                and ``PLAN.md`` §1.3 for numbers.
        ``'tail_weight'``
            Fraction of probability mass outside +/-1*sigma* of the mean.
            Gaussian reference: 0.3173.  A more direct anisotropy diagnostic
            than *h4* because it makes no expansion assumption and remains
            interpretable for non-Gaussian shapes.  See also :func:`tail_weight`.

        **Diagnostic**

        ``'bimodality_score'``
            Integer number of peaks in the smoothed posterior-mean LOSVD
            (see :func:`bimodality_score`).  Score 1 = unimodal; >= 2 =
            inspect visually.  No uncertainty is returned for this metric.

    Notes
    -----
    The approximate Gauss-Hermite conversions (valid for ``|h3|``, ``|h4|``
    less than about 0.2):

    .. code-block:: text

        h3 ~= -skewness / sqrt(6)
        h4 ~=  kurtosis / sqrt(24)

    These allow cross-validation against GH-based models and literature maps.

    For spatial bins where ``bimodality_score >= 2``, the mean, sigma,
    skewness, and kurtosis should be treated with caution: the mean lands
    between two peaks, sigma is inflated by their separation, and skewness
    reflects which peak is taller rather than any genuine asymmetry.

    Examples
    --------
    >>> solver = KinematicSolver(v_grid)
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

    # skewness/kurtosis: the 3rd/4th central-moment *numerators* are left as
    # plain point-mass sums. A rigorous within-cell (Sheppard) correction
    # exists for these too, but by symmetry of the uniform-within-cell
    # kernel the 3rd-moment correction is exactly zero, and the 4th-moment
    # correction is a distribution-shape-dependent term this codebase has
    # not derived or validated anywhere (the reference fix in
    # calibration2d.py only covers 2nd moments/covariance). We do use the
    # corrected `stds` (which includes h**2/12) as the normalising sigma in
    # the denominator, since skewness/kurtosis are defined relative to *the*
    # sigma of the distribution, and reporting them against the biased-low
    # point-mass sigma would be inconsistent with the `sigma` this function
    # now returns.
    skews = np.einsum("ij,ij->i", pdf_samples, delta**3) / safe_stds**3
    skews = np.where(stds > 0, skews, 0.0)

    kurts = (np.einsum("ij,ij->i", pdf_samples, delta**4) / safe_stds**4) - 3.0
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
    """
    Compute summary statistics for all solved bins from :func:`~veldist.fit_all_bins`.

    Iterates over a list of :class:`~veldist.KinematicSolver` instances,
    calls :func:`compute_summary` on each one, and assembles the results into
    arrays shaped ``(n_bins,)``, one entry per spatial bin, ready to pass
    directly to a spatial map plotting function.

    Bins that were skipped during inference (``None`` entries in *solvers*, as
    returned for bins below ``min_stars`` by :func:`~veldist.fit_all_bins`)
    produce ``NaN`` in all output arrays so that the spatial indexing is
    preserved.

    Parameters
    ----------
    solvers : list of :class:`~veldist.KinematicSolver` or None
        As returned by :func:`~veldist.fit_all_bins`.  ``None`` entries
        (skipped bins) are silently mapped to ``NaN``.  The list must
        contain at least one non-``None`` entry.

    Returns
    -------
    dict
        One key per metric name from :func:`compute_summary`.  Each value is
        a sub-dict with two keys:

        ``'median'`` : ndarray, shape (n_bins,)
            Posterior median of the metric.  ``NaN`` for skipped bins.  For
            ``bimodality_score``, the integer score is cast to float.
        ``'uncertainty'`` : ndarray, shape (n_bins,)
            Half-width of the 68% credible interval.  ``NaN`` for skipped
            bins and for ``bimodality_score`` (which has no posterior CI).

    Raises
    ------
    ValueError
        If all entries in *solvers* are ``None``.
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
    """
    Compute percentile-based (order-statistic) moment analogues per sample.

    ``compute_summary``'s ``skewness``/``kurtosis`` are ordinary moments of
    the posterior mean/variance, which a few heavy stars can dominate.
    These analogues are built entirely from CDF percentiles instead, so a
    single outlier star moves them by at most the width of one velocity bin.
    They are not moments and are not expected to match GH h3/h4 or
    ``compute_summary``'s skewness/kurtosis numerically; they are a more
    robust *alternative* view of the same asymmetry/peakedness, evaluated
    per posterior sample so uncertainty propagates the same way as
    everywhere else in this module.

    No within-cell (``h**2/12``) correction is applied anywhere in this
    function, unlike :func:`compute_summary`. That correction is a Sheppard
    correction to a *moment* -- it comes from the algebraic identity
    ``Var(uniform-in-cell) = Var(point-mass-at-centre) + h**2/12``, which is
    specific to variances (second moments) and has no analogue for a
    percentile. A percentile of the piecewise-constant density the
    likelihood fits is not "a percentile of point masses at centres plus a
    correction term" -- it is a different computation entirely: the CDF of a
    piecewise-constant density is piecewise-*linear* between cell edges, so
    its exact percentile requires interpolating against the cell *edges*
    with the within-cell mass spread uniformly, not against centres. This
    function (and :func:`cdf_percentile`, which it's built on) already
    approximates that by linearly interpolating the cumulative point mass
    between adjacent *centres*, which is a distinct, smaller bias (bounded
    by one grid cell, per :func:`cdf_percentile`'s docstring/tests) from the
    variance bias this fix addresses, and is out of scope here: fixing it
    would mean reworking ``cdf_percentile`` to interpolate against edges,
    not adding an ``h**2/12`` term. Adding ``h**2/12`` to a percentile
    output would not even be dimensionally sensible on its own (a percentile
    is a location, not a squared-deviation).

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        MCMC samples of the probability mass function, each row summing to 1.
    grid_centers : array-like, shape (n_bins,)
        Bin-centre velocities in ascending order.

    Returns
    -------
    dict
        Four keys, each ``(posterior_median, half_68ci)``:

        ``'median'``
            CDF P50 (this *is* the LOSVD median, not a moment).
        ``'sigma_pct'``
            ``(P84 - P16) / 2``, the percentile analogue of the standard
            deviation (exact for a Gaussian).
        ``'skew_pct'``
            Bowley skewness ``((P75-P50)-(P50-P25)) / (P75-P25)``, in
            ``[-1, 1]``; 0 for a symmetric distribution, matching sign
            convention with ``compute_summary``'s ``skewness``.
        ``'kurtosis_pct'``
            *Excess* Moors (1988) octile kurtosis:
            ``((P87.5-P62.5)+(P37.5-P12.5)) / (P75-P25) - 1.233...``, the
            raw Moors statistic minus its exact Gaussian reference value.
            Zeroed on a Gaussian, positive for leptokurtic (heavy-tailed,
            peaked) distributions -- the same convention as GH's ``h4``,
            unlike the raw Moors statistic (which is ~1.23, not 0, for a
            Gaussian).
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
    """
    Percentile-based summary maps, the ``compute_percentile_summary`` analogue
    of :func:`compute_summary_maps`.

    Parameters
    ----------
    solvers : list of :class:`~veldist.KinematicSolver` or None
        As returned by :func:`~veldist.fit_all_bins`. ``None`` entries
        (skipped bins) are silently mapped to ``NaN``.

    Returns
    -------
    dict
        Keys ``'median'``, ``'sigma_pct'``, ``'skew_pct'``, ``'kurtosis_pct'``.
        Each value is a sub-dict with ``'median'`` and ``'uncertainty'``
        arrays of shape ``(n_bins,)``, ``NaN`` for skipped bins.
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
    """Normalised Hermite polynomials H_3 and H_4, van der Marel & Franx (1993).

    Normalised so that the Gauss-Hermite series is orthonormal against the
    Gaussian weight, which is what makes h3 and h4 directly comparable to
    published values rather than merely proportional to them.
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

    ``compute_percentile_summary`` gives robust shape statistics, but they are
    not on the Gauss-Hermite scale that the dynamical-modelling literature
    reports. This function produces literature-comparable ``h3`` and ``h4``,
    with a posterior, by fitting the four-parameter series
    ``exp(-y^2/2) [1 + h3 H3(y) + h4 H4(y)]``, ``y = (v - V)/sigma``, to each
    draw by least squares and summarising the resulting parameter samples.

    Fitting is per draw rather than to the posterior mean because the mean of
    a set of LOSVDs is smoother than any of them, which would bias ``h4``
    low and understate the uncertainty on both coefficients.

    No within-cell (``h**2/12``) correction is applied here, deliberately.
    Unlike :func:`compute_summary`, this function never sums point masses
    against ``(v - mu)**2`` to get a variance -- ``sigma_gh`` is a *fitted
    shape parameter* of a continuous analytic curve, obtained by
    least-squares matching ``_gh_model``'s normalised density samples
    against the (normalised) target masses at each grid centre. For a
    reasonably well-resolved grid, midpoint-rule integration means
    ``p_m ~= q(v_m) * h`` to ``O(h**3)`` for smooth ``q``, so both sides of
    the fit -- the model's sampled density and the target's per-cell mass --
    carry the *same* discretisation to leading order, and that error
    cancels rather than propagating one-sidedly into the fit the way it does
    in a raw point-mass moment sum. Adding ``h**2/12`` to ``sigma_gh`` post
    hoc would double-count an effect the fit is not making in the first
    place (verified by :func:`gauss_hermite_fit`'s own tests: it recovers
    planted ``sigma`` to within a few percent from GH-shaped grids without
    any such correction).

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        MCMC samples of the probability mass function, each row summing to 1.
    grid_centers : array-like, shape (n_bins,)
        Bin-centre velocities in ascending order.
    n_draws : int
        Number of posterior draws to fit. Each fit is an independent
        non-linear least squares solve, so cost is linear in this. Draws are
        thinned by random selection when ``n_samples`` exceeds it. The default
        of 500 gives a half-68CI stable to a few percent; raise it only if
        that is the dominant uncertainty in a result.
    seed : int
        Seed for the thinning selection, so results are reproducible.

    Returns
    -------
    dict
        ``'v_gh'``, ``'sigma_gh'``, ``'h3'``, ``'h4'``, each
        ``(posterior_median, half_68ci)``.

        ``v_gh`` and ``sigma_gh`` are the *Gauss-Hermite* location and width,
        which differ from ``compute_summary``'s ``v_mean`` and ``sigma``
        (ordinary moments) whenever ``h4`` is non-zero. Report them as a pair
        with ``h3``/``h4`` or not at all; mixing GH coefficients with moment
        widths is a known source of confusion in the literature.

    Notes
    -----
    Draws whose fit fails to converge are dropped. If more than half fail,
    the LOSVD is probably not well described by a low-order GH series at all,
    which ``bimodality_score`` is the right diagnostic for.
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
