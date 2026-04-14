"""
Statistical analysis utilities for inferred velocity distributions.
"""

import numpy as np

__all__ = [
    "compute_moments",
    "cdf_percentile",
    "tail_weight",
    "bimodality_score",
    "half_68ci",
    "compute_summary",
    "compute_summary_maps",
]

# ---------------------------------------------------------------------------
# Legacy API (kept for backward compatibility)
# ---------------------------------------------------------------------------


def compute_moments(pdf_samples, grid_centers):
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
            ``(posterior_mean, posterior_std)`` of the normalised third
            central moment.
        ``'kurtosis'`` : (float, float)
            ``(posterior_mean, posterior_std)`` of the excess kurtosis
            (fourth central moment / σ⁴ − 3).
    """
    pdf_samples = np.asarray(pdf_samples)  # (n_samples, n_bins)
    grid_centers = np.asarray(grid_centers)  # (n_bins,)

    # Mean velocity for each posterior sample: shape (n_samples,)
    means = pdf_samples @ grid_centers

    # Centred residuals: (n_samples, n_bins) - (n_samples, 1)
    delta = grid_centers[np.newaxis, :] - means[:, np.newaxis]

    # Variance and standard deviation via row-wise weighted dot product
    variance = np.einsum("ij,ij->i", pdf_samples, delta**2)
    stds = np.sqrt(variance)

    # Skewness and excess kurtosis; guard against zero-dispersion samples
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


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def cdf_percentile(pdf_samples, grid_centers, p):
    """
    Interpolate velocity values at cumulative probability level(s) *p*.

    Builds the empirical CDF for each MCMC sample by cumulative summation of
    the probability mass, then uses linear interpolation to find the velocity
    at which the CDF equals *p*.  This is the building block for
    :func:`compute_summary`'s median and IQR calculations, but is also useful
    on its own for constructing credible bands on cumulative profiles.

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
    Fraction of probability mass outside ±1σ of the mean, per MCMC sample.

    This is a direct, model-free measure of tail heaviness that can be
    interpreted without any expansion assumption.  It is the non-parametric
    analogue of the Gauss-Hermite *h₄* coefficient, but more intuitive: for a
    Gaussian the value is exactly ``1 − erf(1/√2) ≈ 0.3173``.

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
        gaussian_tail_weight = 1 - erf(1 / sqrt(2))   # ≈ 0.3173

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
            Bimodal — could indicate counter-rotation, two kinematic
            components, or a contaminating population.
        ``≥ 3``
            Highly irregular; inspect visually before interpreting any
            scalar summaries.

    Notes
    -----
    A more principled treatment would count peaks on every individual MCMC
    sample and return a distribution over the count.  This is deferred as a
    future improvement; the posterior-mean approach is sufficient for
    identifying bins that require visual inspection.

    Examples
    --------
    >>> score = bimodality_score(solver.samples["intrinsic_pdf"])
    >>> if score >= 2:
    ...     print("Multimodal — inspect full histogram before trusting moments")
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
    n_peaks = int(
        np.sum((interior > left) & (interior > right) & (interior > min_height))
    )
    return n_peaks


def half_68ci(samples):
    """
    Half-width of the 68% posterior credible interval.

    Computes ``(p84 − p16) / 2`` — the symmetric ± uncertainty reported
    throughout veldist for both LOSVD values and scalar summary metrics.
    This matches the convention used by BayesLOSVD and is the natural
    non-parametric analogue of a 1-σ Gaussian error bar.

    Parameters
    ----------
    samples : array-like, shape (n_samples,)
        Posterior samples of a scalar quantity (e.g., one metric evaluated
        over all MCMC draws).

    Returns
    -------
    float
        ``(p84 − p16) / 2``, in the same units as *samples*.

    Notes
    -----
    For a Gaussian posterior this equals the posterior standard deviation.
    For skewed or heavy-tailed posteriors it can differ substantially, but
    it always has the interpretation "the true value lies within ±half_68ci
    of the median with approximately 68% posterior probability."

    Examples
    --------
    >>> v_mean_samples = pdf_samples @ grid_centers
    >>> uncertainty = half_68ci(v_mean_samples)
    >>> median = float(np.median(v_mean_samples))
    >>> print(f"v_mean = {median:.1f} ± {uncertainty:.1f} km/s")
    """
    samples = np.asarray(samples, dtype=float)
    p16, p84 = np.percentile(samples, [16, 84])
    return float((p84 - p16) / 2.0)


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------


def compute_summary(pdf_samples, grid_centers):
    """
    Compute spatially-mappable scalar summaries from posterior LOSVD samples.

    This is the primary analysis function for extracting kinematic maps from
    :class:`~veldist.KinematicSolver` output.  All metrics except
    ``bimodality_score`` are evaluated independently on every MCMC sample,
    so the full posterior uncertainty — measurement noise, finite star count,
    and prior regularisation — is propagated automatically with no separate
    bootstrap or error-propagation step.

    Each metric is summarised as ``(posterior_median, half_68ci)`` using the
    same convention as the LOSVD itself: the reported uncertainty is
    ``(p84 − p16) / 2``.

    Parameters
    ----------
    pdf_samples : array-like, shape (n_samples, n_bins)
        MCMC samples of the probability mass function.  Each row must sum to 1.
    grid_centers : array-like, shape (n_bins,)
        Centres of the velocity bins (km/s or consistent velocity unit).

    Returns
    -------
    dict
        Each key maps to a ``(median, half_68ci)`` tuple of floats — both in
        the same units as *grid_centers* for velocity quantities, and
        dimensionless for shape metrics — **except** ``'bimodality_score'``,
        which is a plain ``int``.

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
            to *h₃*, but does not require computing higher-order moments.

        **Dispersion**

        ``'sigma'``
            Standard deviation of the LOSVD.  GH analogue: *σ*.
        ``'iqr'``
            Interquartile range Q75 − Q25.  Robust dispersion estimate
            insensitive to tail contamination.
        ``'sigma_iqr'``
            IQR / 1.3490 — the Gaussian-equivalent dispersion derived from
            the IQR.  For a Gaussian, ``sigma_iqr ≈ sigma``.
            ``sigma_iqr < sigma`` → heavy tails (radial anisotropy);
            ``sigma_iqr > sigma`` → flat top (tangential anisotropy).

        **Shape**

        ``'skewness'``
            Normalised third central moment *γ₁*.  Zero for symmetric
            distributions.  GH analogue: *h₃* ≈ −*γ₁* / √6.  Note sign:
            *γ₁* > 0 (right-skewed, trailing tail) → *h₃* < 0, the
            expected signal on the receding side of a rotating system.
        ``'kurtosis'``
            Excess kurtosis *κ* = fourth central moment / σ⁴ − 3.  Zero for
            a Gaussian.  GH analogue: *h₄* ≈ *κ* / √24.  Positive
            (leptokurtic) → radially anisotropic; negative (platykurtic) →
            tangentially anisotropic / flat-topped.
        ``'tail_weight'``
            Fraction of probability mass outside ±1*σ* of the mean.
            Gaussian reference: 0.3173.  A more direct anisotropy diagnostic
            than *h₄*: no expansion assumption, always interpretable even for
            non-Gaussian shapes.  See also :func:`tail_weight`.

        **Diagnostic**

        ``'bimodality_score'``
            Integer number of peaks in the smoothed posterior-mean LOSVD
            (see :func:`bimodality_score`).  Score 1 = unimodal; ≥ 2 =
            inspect visually.  No uncertainty is returned for this metric.

    Notes
    -----
    The approximate Gauss-Hermite conversions (valid for |h₃|, |h₄| ≲ 0.2):

    .. code-block:: text

        h3 ≈ -skewness / sqrt(6)
        h4 ≈  kurtosis / sqrt(24)

    These allow cross-validation against GH-based models and literature maps.

    For spatial bins where ``bimodality_score ≥ 2``, the mean, sigma,
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
    >>> print(f"V = {v:.1f} ± {dv:.1f}  σ = {s:.1f} ± {ds:.1f}  km/s")
    """
    pdf_samples = np.asarray(pdf_samples, dtype=float)  # (n_samples, n_bins)
    grid_centers = np.asarray(grid_centers, dtype=float)  # (n_bins,)

    # ------------------------------------------------------------------
    # Moment-based quantities (fully vectorised)
    # ------------------------------------------------------------------
    means = pdf_samples @ grid_centers  # (n_samples,)
    delta = grid_centers[np.newaxis, :] - means[:, np.newaxis]  # (n_s, n_bins)

    variance = np.einsum("ij,ij->i", pdf_samples, delta**2)  # (n_samples,)
    stds = np.sqrt(variance)  # (n_samples,)
    safe_stds = np.where(stds > 0, stds, 1.0)

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
    sigma_iqr = iqr / 1.3490  # Gaussian-equivalent σ
    v_asym = means - medians  # (n_samples,)

    # ------------------------------------------------------------------
    # Bimodality score (scalar — from posterior mean, not per-sample)
    # ------------------------------------------------------------------
    bscore = bimodality_score(pdf_samples)

    # ------------------------------------------------------------------
    # Summarise each per-sample array as (median, half_68ci)
    # ------------------------------------------------------------------
    def _summarise(arr):
        p16, p50, p84 = np.percentile(arr, [16, 50, 84])
        return (float(p50), float((p84 - p16) / 2.0))

    return {
        # Location
        "v_mean": _summarise(means),
        "v_median": _summarise(medians),
        "v_asymmetry": _summarise(v_asym),
        # Dispersion
        "sigma": _summarise(stds),
        "iqr": _summarise(iqr),
        "sigma_iqr": _summarise(sigma_iqr),
        # Shape
        "skewness": _summarise(skews),
        "kurtosis": _summarise(kurts),
        "tail_weight": _summarise(tw),
        # Diagnostic (no uncertainty)
        "bimodality_score": bscore,
    }


def compute_summary_maps(solvers):
    """
    Compute summary statistics for all solved bins from :func:`~veldist.fit_all_bins`.

    Iterates over a list of :class:`~veldist.KinematicSolver` instances,
    calls :func:`compute_summary` on each one, and assembles the results into
    arrays shaped ``(n_bins,)`` — one entry per spatial bin — ready to pass
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

    Examples
    --------
    Build velocity and dispersion maps and plot them side by side:

    >>> maps = compute_summary_maps(solvers)
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    >>> for ax, key, label in zip(axes,
    ...                           ["v_mean", "sigma"],
    ...                           ["Mean velocity (km/s)", "Dispersion (km/s)"]):
    ...     sc = ax.scatter(xbin, ybin, c=maps[key]["median"], cmap="RdBu_r")
    ...     plt.colorbar(sc, ax=ax, label=label)

    Flag multimodal bins before plotting moments:

    >>> bscore = maps["bimodality_score"]["median"]
    >>> multimodal = bscore >= 2
    >>> print(f"{multimodal.sum():.0f} bins have bimodality_score ≥ 2")

    Notes
    -----
    The ``bimodality_score`` sub-dict has ``'median'`` filled with the
    integer score cast to float and ``'uncertainty'`` filled with ``NaN``.
    This keeps the return structure uniform so callers can iterate over all
    keys without special-casing the diagnostic metric.
    """
    n_bins = len(solvers)

    # Infer the full set of metric keys from the first solved bin.
    first = next((s for s in solvers if s is not None), None)
    if first is None:
        raise ValueError("All solvers are None — no bins to summarise.")

    ref_summary = compute_summary(first.samples["intrinsic_pdf"], first.grid["centers"])
    metric_keys = list(ref_summary.keys())

    # Initialise all arrays to NaN.
    maps = {
        k: {"median": np.full(n_bins, np.nan), "uncertainty": np.full(n_bins, np.nan)}
        for k in metric_keys
    }

    for i, solver in enumerate(solvers):
        if solver is None:
            continue

        summary = compute_summary(
            solver.samples["intrinsic_pdf"], solver.grid["centers"]
        )

        for k, v in summary.items():
            if k == "bimodality_score":
                maps[k]["median"][i] = float(v)
                # uncertainty stays NaN — bimodality_score has no posterior CI
            else:
                maps[k]["median"][i] = v[0]
                maps[k]["uncertainty"][i] = v[1]

    return maps
