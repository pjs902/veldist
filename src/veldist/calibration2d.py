"""Observing profiles for the 2D (proper-motion) solver.

As in 1D (``calibration.py``), the velocity grid is derived from the
observing regime rather than chosen by hand. Before this module existed,
the 2D tests had no profile: the grid and errors came from the SBC harness
(whose own comment calls the grid an "arbitrary physical span"), and the
star count was copied from the *line-of-sight* profile. Proper motions
reach about 6 magnitudes deeper than the spectroscopy, so both star counts
and errors are very different.

Calibration source: the oMEGaCat proper-motion error-versus-magnitude
figures. Units are converted with 1 mas/yr = 4.740470 * distance[kpc] km/s
at the adopted cluster distance of 5494 pc (set 2026-08-06), so
1 mas/yr = 26.04 km/s. The quality cut is 0.3 mas/yr.
"""

from dataclasses import dataclass, field

import numpy as np

from veldist.calibration import coverage_floor

__all__ = [
    "ObservingProfile2D",
    "RecoveryCurve2D",
    "HST_BRIGHT",
    "HST_FAINT",
    "GAIA_OUTER",
    "PROFILES_2D",
    "CLUSTER_DISTANCE_PC",
    "KMS_PER_MASYR",
    "PM_QUALITY_CUT_KMS",
    "truths_for",
    "coverage_floor",
    "recovery_curve_2d",
    "recommend_grid_2d",
    "cell_per_sigma_for",
]

#: Adopted cluster distance (Peter, 2026-08-06).
CLUSTER_DISTANCE_PC = 5494.0

#: 1 mas/yr = 4.740470 * distance[kpc] km/s (standard proper-motion relation).
KMS_PER_MASYR = 4.740470 * CLUSTER_DISTANCE_PC / 1000.0

#: HST's 0.3 mas/yr proper-motion quality cut, in km/s. Applied upstream of
#: the HST catalogue, whose largest error (7.741 km/s) sits just under it.
#: **This is HST's cut, not a project-wide one** -- passing it as Gaia's
#: ``err_cut`` puts the cut below Gaia's own median error, which
#: :meth:`ObservingProfile2D.__post_init__` now rejects.
PM_QUALITY_CUT_KMS = 0.30 * KMS_PER_MASYR

#: Gaia's proper-motion error ceiling, in km/s. The Gaia selection applies no
#: PM-error quality cut, so this sits above every measured error rather than
#: truncating the distribution: the spread is set by the magnitude
#: distribution (measured ``err_log_sigma`` 0.727), not by a truncation.
#: Named because "Gaia's cut" is 34x HST's and confusing the two is silent
#: everywhere except the ``err_cut > err_median`` guard.
GAIA_PM_ERR_CEILING_KMS = 10.0 * KMS_PER_MASYR


#: Measured (err_median/sigma_lo, cell_per_sigma) anchors for
#: :func:`cell_per_sigma_for`. **The ratio is against ``sigma_lo``, matching
#: how the function is called** -- an earlier revision anchored on
#: ``sigma_ref`` while calling with ``sigma_lo``, which silently returned the
#: wrong cell width for HST (0.463 instead of its measured 0.58) and was
#: right for Gaia only because the value clipped at the top of the range.
#:
#: Gaia (err/sigma_lo = 8.60/7.03 = 1.22): swept {0.85, 1.10, 1.40, 1.80} at
#: 435 stars; 0.85 chosen -- 2.8% bias on the narrow axis, both rms_z near 1,
#: and 1.40+ degrades fast.
#: HST (err/sigma_lo = 1.51/11.50 = 0.13): swept {0.42, 0.58, 0.70} at 426
#: stars; 0.58 chosen -- 1.0% bias, sigma_y rms_z 0.87. 0.42 is cleaner still
#: (0.0%, 0.81) but costs 1.7x the cells for no measured gain; 0.70 is
#: borderline (2.2%, rms_z 1.11); 0.85 (Gaia's value) fails outright at
#: rms_z 1.36.
_CPS_ANCHORS = ((0.13, 0.58), (1.22, 0.85))


def cell_per_sigma_for(err_over_sigma):
    """Target cell width in units of sigma for a given measurement-error regime.

    One global constant does not work: HST and Gaia need values about 1.5x
    apart. The reason is that ``rms_z`` is bias divided by interval width.
    HST's err/sigma_lo is 0.13 against Gaia's 1.22, so HST's posterior is
    sharp and leaves nothing to hide a leftover discretisation bias behind,
    while Gaia's large errors widen the intervals enough to absorb the same
    absolute bias. **Precise data need FINER grids**, the opposite of the
    usual intuition.

    This is an **empirical power law through two points**, not a derived
    result. The obvious model, keeping the discretisation bias below the
    statistical error (which scales as ``(1 + (err/sigma)^2)^(1/4)``),
    predicts a ratio of 1.20 between the two regimes, against 1.47 measured.
    Something else contributes, most likely that a weak likelihood lets the
    roughness prior smooth the recovered pdf, so coarse cells cost less than
    the error budget alone suggests. Until that is understood, do not
    extrapolate far beyond the two anchor points.

    Parameters
    ----------
    err_over_sigma : float
        Median per-star error divided by the dispersion the grid must resolve
        (``profile.err_median / profile.sigma_lo``).

    Returns
    -------
    float
        Cell width in units of sigma, clipped to the measured range so a wild
        input cannot silently give an absurd grid.
    """
    (e_lo, c_lo), (e_hi, c_hi) = _CPS_ANCHORS
    p = np.log(c_hi / c_lo) / np.log(e_hi / e_lo)
    e = float(np.clip(err_over_sigma, e_lo, e_hi))
    return float(c_hi * (e / e_hi) ** p)


def _log_sigma_from_p95(err):
    """Log-normal width that reproduces an error sample's p95/median ratio."""
    err = np.asarray(err, dtype=float)
    return float(np.log(np.percentile(err, 95) / np.median(err)) / 1.645)


@dataclass(frozen=True)
class ObservingProfile2D:
    """A proper-motion observing regime and the velocity grid it implies.

    Parameters
    ----------
    name : str
        Registry key, also used in reports.
    sigma_ref : float
        Representative intrinsic velocity dispersion, km/s.
    err_median : float
        Median per-star measurement error, km/s.
    err_cut : float
        Upper limit of the error distribution (the quality cut), km/s. Must
        exceed ``err_median``: a cut below the median of the distribution it
        truncates would collapse it (see :meth:`draw_errors`).
    err_log_sigma : float or None
        Log-normal width of the per-star error distribution. ``None``
        (default) derives it from ``err_cut``, assuming the cut sits at the
        95th percentile. Set it whenever it has been measured, as ``from_data``
        always does. It is a separate field for the same reason as
        :attr:`ObservingProfile.err_log_sigma` in 1D.
    n_stars : int
        Stars per spatial (Voronoi) bin, the science target. Fewer stars per
        bin means more bins and better spatial coverage, so this is a
        resolution choice, not a convenience value.
    n_sigma_grid : float
        Half-width of the velocity grid in units of ``sigma_ref``.
    cell_per_sigma : float or None
        Target cell width in units of ``sigma_lo``. ``None`` (default) derives
        it from this profile's error regime with :func:`cell_per_sigma_for`,
        which is what you want, since the right value depends on the regime
        and a shared constant fails one of the two measured datasets. Set it
        only to pin a grid.

        **Re-measured on 2026-09-01** against the real Gaia profile (435
        stars, the measured dispersion range and rotation span), after the
        h^2/12 fix, sweeping {0.85, 1.10, 1.40, 1.80} on both truths with 40
        realisations each:

            cps    K   stars/cell   sigma_y bias   sigma_y rms_z   rho rms_z
            0.85  25      0.70      +0.150 (2.8%)      0.97          1.18
            1.10  19      1.20      +0.238 (4.5%)      1.01          1.16
            1.40  15      1.93      +0.439 (8.2%)      1.17          1.05
            1.80  13      2.57      +0.670 (12.5%)     1.53          1.50

        The bias is on the narrow axis of the ANISOTROPIC truth; percentages
        are of that axis's own sy = 5.34. The isotropic truth is flat across
        the whole range, so a check on an isotropic truth alone would miss
        the problem.

        0.85 is chosen for a bias under 3% with both rms_z near 1. 1.10 is
        defensible if compute requires it; 1.40 and above are not. An rms_z of
        1.5 at 1.80 means the reported intervals are about two-thirds of the
        width they should be, on both sigma_y and rho.

        This also exposes a scale mismatch. cell_per_sigma is defined relative
        to ``sigma_lo``, but the narrow axis of an anisotropic ellipsoid is
        smaller still (0.65x here), so even at 0.85 the cells are about 1.3x
        that axis's sigma. Defining resolution against the narrowest *axis*
        rather than the narrowest *bin* would be cleaner, and has not been
        done yet.

        The previous value, 0.47, came from a sweep run BEFORE the h^2/12
        correction. Refining the grid then shrank a bias that the estimator
        itself was creating, so the sweep measured how the bug scaled with
        resolution and read it as a resolution requirement. (That sweep ran
        cell_per_sigma from 0.78 to 0.37 at N=400 with the gaussian_core
        prior; its conclusion that "K=19 breaks on anisotropic truths" no
        longer holds: the failure was the estimator's, not the grid's.)
    """

    name: str
    sigma_ref: float
    err_median: float
    err_cut: float
    n_stars: int
    n_sigma_grid: float = 3.5
    cell_per_sigma: float | None = None
    sigma_min: float | None = None
    sigma_max: float | None = None
    rotation_span: float = 0.0
    bins_per_error: float = 2.0
    err_log_sigma: float | None = None

    def __post_init__(self):
        # Trust boundary: err_cut below err_median silently turns draw_errors
        # into a spike at the cut (65% of draws pinned there, for the Gaia
        # profile this caught on 2026-09-01) while every reported summary
        # keeps quoting the declared median. Types are identical either way,
        # so nothing downstream can notice.
        if self.err_cut <= self.err_median:
            msg = (
                f"{self.name}: err_cut ({self.err_cut:g}) must exceed err_median "
                f"({self.err_median:g}) -- a cut below the median collapses the "
                f"drawn error distribution onto the cut"
            )
            raise ValueError(msg)

    @property
    def sigma_lo(self):
        """Narrowest per-bin dispersion; sets the resolution requirement."""
        return self.sigma_ref if self.sigma_min is None else self.sigma_min

    @property
    def sigma_hi(self):
        """Widest per-bin dispersion; sets the extent requirement."""
        return self.sigma_ref if self.sigma_max is None else self.sigma_max

    @property
    def grid_width(self):
        """Total width of the (square) velocity grid, km/s.

        Dynamite takes one ``vxrange``/``vyrange`` per map, so a single grid has
        to serve every spatial bin: it must hold the widest distribution in the
        field plus the mean-velocity offset of the bins furthest from systemic.
        Same as :attr:`ObservingProfile.grid_width` in 1D.
        """
        return 2.0 * self.n_sigma_grid * self.sigma_hi + self.rotation_span

    @property
    def cells_per_sigma_target(self):
        """Cell width in units of sigma: the explicit value if set, otherwise
        derived from this profile's error regime with :func:`cell_per_sigma_for`.
        """
        if self.cell_per_sigma is not None:
            return self.cell_per_sigma
        return cell_per_sigma_for(self.err_median / self.sigma_lo)

    @property
    def cell_width(self):
        """Target cell width, km/s, chosen to resolve the narrowest distribution
        in the field.

        Uses ``sigma_lo``, not ``sigma_ref``: one shared grid has to resolve
        every bin, and the narrowest one is the limiting case.
        """
        return self.cells_per_sigma_target * self.sigma_lo

    @property
    def error_floor_width(self):
        """Cell width below which refining the grid gains nothing, km/s.

        In 1D the bin width is set to exactly this (``bins_per_error *
        err_median``), since the measurement errors have already blurred the
        signal on that scale. In 2D it is only reported, because the number of
        cells grows quadratically, and because when it is LARGER than
        :attr:`cell_width` the two requirements conflict: the errors dominate
        and no grid can resolve the narrowest bins. That is the Gaia regime
        (err/sigma near 1), and it is a property of the data, not a tuning
        choice. :func:`recommend_grid_2d` flags it.
        """
        return self.bins_per_error * self.err_median

    @property
    def n_bins(self):
        """Cells per axis. Always odd, because Dynamite's ProperMotions reader
        (``set_default_hist_bins``) rejects even counts.
        """
        n = int(round(self.grid_width / self.cell_width))
        n = max(n, 5)
        return n if n % 2 == 1 else n + 1

    @property
    def err_over_sigma(self):
        """The ratio that drives deconvolution difficulty."""
        return self.err_median / self.sigma_ref

    def draw_errors(self, n, rng):
        """Draw ``n`` per-star measurement errors, km/s.

        Log-normal around ``err_median``, truncated at ``err_cut``. The spread is
        :attr:`err_log_sigma` when known. Otherwise it is derived from
        ``err_cut`` assuming the cut sits near the 95th percentile, which is
        roughly what a magnitude-dependent error distribution looks like after a
        quality cut.

        That fallback is only as good as its assumption. Gaia's real cut is
        10 mas/yr = 260 km/s, effectively no cut, so its spread comes from the
        magnitude distribution (measured ``err_log_sigma`` 0.727), not from any
        truncation. Measure it where possible.
        """
        sigma_log = (
            self.err_log_sigma
            if self.err_log_sigma is not None
            else max(0.25, np.log(self.err_cut / self.err_median) / 1.645)
        )
        e = rng.lognormal(np.log(self.err_median), sigma_log, n)
        return np.clip(e, 1e-3, self.err_cut)

    @classmethod
    def from_data(cls, pm1, pm2, err1, err2, bin_ids, err_cut, name="measured", min_stars=10):
        """Measure a profile from a real proper-motion catalogue.

        The 2D version of :meth:`ObservingProfile.from_data`. The HST and Gaia
        data-preparation notebooks had each written this same per-bin estimator,
        so it lives here instead.

        ``err_cut`` is a quality cut applied upstream (Gaia's is about 4x HST's)
        and cannot be measured from the data after the cut, so it is a required
        argument.

        Parameters
        ----------
        pm1, pm2 : array-like, shape (n_stars,)
            Proper-motion components in km/s, in the frame of the grid.
        err1, err2 : array-like, shape (n_stars,)
            Per-star errors on ``pm1`` and ``pm2``, km/s.
        bin_ids : array-like, shape (n_stars,)
            Spatial bin index of each star; need not be contiguous.
        err_cut : float
            Upper limit of the error distribution, km/s.
        name : str
            Label for the returned profile.
        min_stars : int
            Bins with fewer stars are left out of every measured statistic
            (``sigma_ref``, ``err_median``, ``n_stars``).

        Returns
        -------
        ObservingProfile2D

        Raises
        ------
        ValueError
            If fewer than 2 bins pass the *min_stars* cut.
        """
        pm1 = np.asarray(pm1, dtype=float)
        pm2 = np.asarray(pm2, dtype=float)
        err1 = np.asarray(err1, dtype=float)
        err2 = np.asarray(err2, dtype=float)
        bin_ids = np.asarray(bin_ids)

        per_bin_n, per_bin_sigma, per_bin_err_med = [], [], []
        per_bin_mean1, per_bin_mean2 = [], []
        kept_err_mag = []
        for b in np.unique(bin_ids):
            sel = bin_ids == b
            n = int(np.sum(sel))
            if n < min_stars:
                continue
            e1, e2 = err1[sel], err2[sel]
            err_mag = np.hypot(e1, e2) / np.sqrt(2)
            err_med = np.median(err_mag)
            var1 = np.var(pm1[sel]) - np.mean(e1**2)
            var2 = np.var(pm2[sel]) - np.mean(e2**2)
            sigma = np.sqrt(max(0.5 * (var1 + var2), err_med**2))
            per_bin_n.append(n)
            per_bin_sigma.append(sigma)
            per_bin_err_med.append(err_med)
            per_bin_mean1.append(np.mean(pm1[sel]))
            per_bin_mean2.append(np.mean(pm2[sel]))
            kept_err_mag.append(err_mag)

        if len(per_bin_sigma) < 2:
            msg = f"at least 2 bins with >= {min_stars} stars are required, got {len(per_bin_sigma)}"
            raise ValueError(msg)

        return cls(
            name=name,
            sigma_ref=float(np.median(per_bin_sigma)),
            err_median=float(np.median(per_bin_err_med)),
            err_cut=float(err_cut),
            n_stars=int(round(float(np.median(per_bin_n)))),
            sigma_min=float(np.min(per_bin_sigma)),
            sigma_max=float(np.max(per_bin_sigma)),
            rotation_span=float(
                max(np.ptp(per_bin_mean1), np.ptp(per_bin_mean2))
            ),
            # Anchored on p95/median, NOT std(log): the real error
            # distributions are heavier-tailed than log-normal, and std(log)
            # fits the body while understating the tail (HST: 0.287 vs 0.434,
            # a p95 of 2.42 against a measured 3.14 km/s). The tail is the
            # part that matters -- it sets how much the errors inflate the
            # posterior intervals, which is the denominator of rms_z.
            err_log_sigma=_log_sigma_from_p95(np.concatenate(kept_err_mag)),
        )

    def cells_per_sigma(self, axis_sigma):
        """Cell width in units of one axis's own intrinsic dispersion.

        ``cell_per_sigma`` and ``n_bins`` are defined relative to the single
        scalar ``sigma_ref``, which describes the real per-axis resolution only
        for an isotropic velocity ellipsoid. For an anisotropic truth, a narrow
        axis (smaller ``axis_sigma``) gets a LARGER, coarser value here and a wide
        axis a smaller, finer one. The grid is the same; only its resolution
        relative to each axis's spread differs.
        """
        return (self.grid_width / self.n_bins) / axis_sigma

    def extent_in_sigma(self, axis_sigma):
        """Grid half-extent in units of an axis's own intrinsic dispersion."""
        return (self.grid_width / 2.0) / axis_sigma

    def report(self):
        """One-line-per-fact summary, for printing in test output."""
        aniso = truths_for(self.sigma_ref)["anisotropic"]
        sx, sy = aniso["sx"], aniso["sy"]
        return (
            f"{self.name}: sigma_ref={self.sigma_ref:g} km/s, "
            f"{self.n_stars} stars/bin, err median={self.err_median:g} km/s "
            f"(cut {self.err_cut:g}), err/sigma={self.err_over_sigma:.3f}\n"
            f"  grid {self.grid_width:.0f} km/s, n_bins={self.n_bins} per axis "
            f"({self.n_bins**2} cells), cell "
            f"{self.grid_width / self.n_bins:.1f} km/s "
            f"({self.grid_width / self.n_bins / self.sigma_ref:.2f} sigma)\n"
            f"  anisotropic truth: x-axis (sx={sx:.2f}) "
            f"{self.cells_per_sigma(sx):.2f} sigma/cell, "
            f"+/-{self.extent_in_sigma(sx):.2f} sigma extent\n"
            f"  anisotropic truth: y-axis (sy={sy:.2f}) "
            f"{self.cells_per_sigma(sy):.2f} sigma/cell, "
            f"+/-{self.extent_in_sigma(sy):.2f} sigma extent"
        )


def truths_for(sigma):
    """Scale the two test truths (isotropic and anisotropic) to a profile's
    ``sigma_ref`` instead of hard-coding km/s values.

    Shared by ``test_coverage_2d.py`` and :func:`recovery_curve_2d` so the two
    cannot drift apart.
    """
    return {
        "isotropic": dict(mux=0.0, muy=0.0, sx=sigma, sy=sigma, rho=0.0),
        "anisotropic": dict(
            mux=0.18 * sigma, muy=-0.12 * sigma,
            sx=1.18 * sigma, sy=0.76 * sigma, rho=0.4,
        ),
    }


def _draw_stars(rng, truth, n_stars, profile):
    """Draw one mock bin: observed (x, y) proper motions and per-star
    diagonal covariances, for a given truth, star count and profile error
    distribution.
    """
    mean = [truth["mux"], truth["muy"]]
    cov_true = [
        [truth["sx"] ** 2, truth["rho"] * truth["sx"] * truth["sy"]],
        [truth["rho"] * truth["sx"] * truth["sy"], truth["sy"] ** 2],
    ]
    true_xy = rng.multivariate_normal(mean, cov_true, size=n_stars)

    err_x = profile.draw_errors(n_stars, rng)
    err_y = profile.draw_errors(n_stars, rng)
    obs_x = true_xy[:, 0] + rng.normal(0.0, err_x)
    obs_y = true_xy[:, 1] + rng.normal(0.0, err_y)

    cov = np.zeros((n_stars, 2, 2))
    cov[:, 0, 0] = err_x**2
    cov[:, 1, 1] = err_y**2
    return obs_x, obs_y, cov


def _discretised_truth_moments(t, edges_x, edges_y, centers_2d):
    """Moments to score the fit against, plus the exact per-cell mass of the
    truth (for per-cell coverage).

    The returned mean, sigma and rho are those of the CONTINUOUS truth, i.e.
    exactly ``t``'s own ``mux``, ``muy``, ``sx``, ``sy`` and ``rho``,
    independent of the grid. The exact per-cell mass on ``edges_x`` /
    ``edges_y`` is also returned, for callers that check per-cell coverage
    (e.g. ``test_per_cell_losvd_coverage_2d``).

    This function used to return point-mass moments of the cell masses at the
    cell centres (``V + h^2/12`` per axis, Sheppard-inflated), on the grounds
    that comparing against the continuous truth would "charge the model for
    grid discretisation". That was consistent while
    ``_moments_from_pdf_samples_2d`` also computed point-mass moments. It now
    adds ``h^2/12`` so it estimates the continuous variance that the
    likelihood fits (see its docstring). Scoring that against the old
    Sheppard-inflated target would count the ``h^2/12`` term twice and leave
    an ``h^2/6`` gap the other way. The correct target is the continuous
    truth, which is also independent of the grid, as it should be.

    ``edges_x`` and ``edges_y`` may differ in length (a rectangular grid,
    ``kx != ky``). The flat index follows ``setup_grid_2d``'s row-major
    convention, ``m = ix * ky + iy``.
    """
    from scipy.stats import multivariate_normal

    cov = [[t["sx"] ** 2, t["rho"] * t["sx"] * t["sy"]],
           [t["rho"] * t["sx"] * t["sy"], t["sy"] ** 2]]
    mvn = multivariate_normal(mean=[t["mux"], t["muy"]], cov=cov)
    kx = len(edges_x) - 1
    ky = len(edges_y) - 1
    mass = np.empty(kx * ky)
    for ix in range(kx):
        for iy in range(ky):
            mass[ix * ky + iy] = (
                mvn.cdf([edges_x[ix + 1], edges_y[iy + 1]])
                - mvn.cdf([edges_x[ix], edges_y[iy + 1]])
                - mvn.cdf([edges_x[ix + 1], edges_y[iy]])
                + mvn.cdf([edges_x[ix], edges_y[iy]])
            )
    mass /= mass.sum()
    moments = dict(
        mean_x=t["mux"], mean_y=t["muy"],
        sigma_x=t["sx"], sigma_y=t["sy"], rho=t["rho"],
    )
    return moments, mass


def _moments_from_pdf_samples_2d(pdf_samples, centers_2d, grid):
    """Per-sample mean_x, mean_y, sigma_x, sigma_y and rho from 2D pdf draws.

    ``grid`` must provide the per-axis cell widths ``grid["width_x"]`` and
    ``grid["width_y"]``. Despite the names, these are CELL widths
    (``edges_x[1] - edges_x[0]``), not the total span of the grid; see
    ``setup_grid_2d``. Pass the grid dict used to build ``centers_2d``
    (e.g. ``solver.grid``).

    Why the cell width is needed
    ----------------------------
    A cell value ``p_m`` was interpreted three inconsistent ways in this
    code:

    1. THE LIKELIHOOD (``precompute_design_matrix`` and its 2D counterpart)
       treats ``p_m`` as mass spread UNIFORMLY over cell ``m``: taking
       ``p(v) ~= p_m/h`` out of the per-cell integral assumes a
       piecewise-constant density. The fitted density ``q(v)`` therefore has
       ``Var(q) = sum_m p_m (v_m - mu)^2 + h^2/12``, where ``h^2/12`` is the
       variance of a uniform distribution over one cell.
    2. THIS FUNCTION, before the fix, treated ``p_m`` as a POINT MASS at the
       cell centre, ``Var = sum_m p_m (v_m - mu)^2``, with no within-cell
       term, and so reported a smaller quantity than the one fitted.
    3. ``_discretised_truth_moments`` took point-mass moments of the exact
       cell masses of the truth, giving ``V + h^2/12`` (Sheppard's
       correction), where ``V`` is the continuous variance.

    The data push the likelihood's ``Var(q)`` toward the true ``V``, so (2)
    reported ``V - h^2/12`` while (3) expected ``V + h^2/12``: a gap of
    ``h^2/6`` in variance, about ``h^2/(12*sigma)`` in sigma, that depends on
    resolution. Adding ``h^2/12`` here makes (2) estimate the same continuous
    quantity the likelihood fits and that ``_discretised_truth_moments`` now
    targets (see its docstring for the other half of the fix).

    The x/y covariance needs no correction: cells are axis-aligned
    rectangles, so within a cell x and y are independent and add no
    covariance. ``rho`` is recomputed from the corrected variances, which
    reduces ``|rho|`` slightly; that is the intended effect of the
    correction.
    """
    pdf_samples = np.asarray(pdf_samples, dtype=float)
    cx = centers_2d[:, 0]
    cy = centers_2d[:, 1]
    h_x = grid["width_x"]
    h_y = grid["width_y"]

    mean_x = pdf_samples @ cx
    mean_y = pdf_samples @ cy
    dx = cx[None, :] - mean_x[:, None]
    dy = cy[None, :] - mean_y[:, None]

    var_x = np.einsum("ij,ij->i", pdf_samples, dx**2) + h_x**2 / 12.0
    var_y = np.einsum("ij,ij->i", pdf_samples, dy**2) + h_y**2 / 12.0
    cov_xy = np.einsum("ij,ij->i", pdf_samples, dx * dy)

    sigma_x = np.sqrt(var_x)
    sigma_y = np.sqrt(var_y)
    safe_denom = np.where((sigma_x > 0) & (sigma_y > 0), sigma_x * sigma_y, 1.0)
    rho = np.where((sigma_x > 0) & (sigma_y > 0), cov_xy / safe_denom, 0.0)

    return mean_x, mean_y, sigma_x, sigma_y, rho


@dataclass
class RecoveryCurve2D:
    """How well each 2D moment, including the tilt ``rho``, is recovered as a
    function of star count.

    The 2D counterpart of :class:`RecoveryCurve`. It sweeps ``n_stars``
    directly, because there is no 2D equivalent of ``ivar`` yet (see
    :func:`recovery_curve_2d`).

    Notes
    -----
    ``cr_bound`` for ``rho`` uses the bivariate-normal approximation
    ``Var(rho_hat) ~= (1 - rho**2)**2 / n``, i.e. ``(1 - rho**2) / sqrt(n)``
    as an interval-width scale. Like the 1D bounds for skewness and kurtosis,
    this is exact only when all stars have the same error, and it is not
    reliable with the unequal errors this package handles. So
    :meth:`threshold` gates ``rho`` on coverage only, not on the CI/CR
    ratio. :meth:`report` still prints the ratio for every metric, but for
    ``rho`` it is advisory.

    ``rms_z`` (see :func:`recovery_curve_2d`) avoids this problem. It
    measures interval calibration directly from the standardised residuals
    ``(median - truth) / half68``, with no Cramér-Rao bound involved, so it
    is reliable for ``rho`` as well.

    It also has better resolution than ``coverage``. Coverage turns each
    residual into hit or miss at 1.0, so missing by 1.01 half-widths counts
    the same as missing by 3.0. On an isotropic truth, where ``sigma_x`` and
    ``sigma_y`` are identical in expectation (an A/A test with a true
    difference of zero), their coverage differed by 0.14 at n_real=100: that
    is the end-to-end noise floor of coverage. The bias differed by only
    0.052 in the same run, about three times better. Several earlier
    conclusions rested on coverage differences of 0.04-0.09, below the noise
    floor. ``rms_z`` keeps the continuous residual and is the better
    statistic for small differences between runs; use :meth:`aa_noise` to
    measure the noise floor of any of these statistics on your own setup.
    """

    profile: object
    truth_name: str
    rows: list = field(default_factory=list)
    n_real: int = None

    def threshold(self, metric, min_coverage=None, max_ci_ratio=1.5, band=0.99):
        """Smallest ``n_stars`` at which *metric* can be trusted, or ``None``.

        Uses the same two conditions and top-down walk as
        :meth:`RecoveryCurve.threshold`, over ``n_stars`` instead of ``ivar``;
        see that method for the reasoning.

        For ``metric == "rho"`` only the coverage floor applies, not the
        ``ci_width <= max_ci_ratio * cr_bound`` check, because ``rho``'s
        ``cr_bound`` is not reliable with unequal errors (see the class
        docstring). Every other metric uses both checks.
        """
        sel = [r for r in self.rows if r["metric"] == metric]
        if not sel:
            msg = f"no rows for metric {metric!r}"
            raise ValueError(msg)

        floor, _floor_desc = self._resolve_coverage_floor(min_coverage, band)

        by_n = {}
        for r in sel:
            by_n.setdefault(r["n_stars"], []).append(r)

        def ok(rows):
            for r in rows:
                if r["coverage"] < floor:
                    return False
                if metric != "rho" and r["ci_width"] > max_ci_ratio * r["cr_bound"]:
                    return False
            return True

        n_values = sorted(by_n)
        best = None
        for n in reversed(n_values):
            if not ok(by_n[n]):
                break
            best = n
        return best

    def _resolve_coverage_floor(self, min_coverage, band):
        if min_coverage is not None:
            return min_coverage, "explicit"
        if self.n_real is not None:
            floor = coverage_floor(self.n_real, band=band)
            return floor, f"{band:.0%} binomial band at n_real={self.n_real}"
        return 0.60, "historical default, n_real unknown"

    def report(self, min_coverage=None, max_ci_ratio=1.5, band=0.99):
        """Human-readable table, one block per metric."""
        floor, floor_desc = self._resolve_coverage_floor(min_coverage, band)
        metrics = sorted({r["metric"] for r in self.rows})
        lines = [
            f"RecoveryCurve2D: {self.profile.name} / {self.truth_name}",
            f"  {len({r['n_stars'] for r in self.rows})} n_stars value(s)",
            f"  coverage floor {floor:.3f} ({floor_desc})",
        ]
        for metric in metrics:
            t = self.threshold(metric, min_coverage=min_coverage, max_ci_ratio=max_ci_ratio, band=band)
            n_values = sorted({r["n_stars"] for r in self.rows if r["metric"] == metric})
            note = ""
            if t is not None and n_values:
                if t == n_values[0]:
                    note = " (at the bottom of the swept range, true threshold may be lower)"
                elif t == n_values[-1]:
                    note = " (at the top of the swept range, may not be bracketed)"
            lines.append(f"  {metric}: threshold n_stars = " + ("not reached" if t is None else f"{t:.4g}{note}"))
            lines.append("    n_stars  cover  CI/CR  bias    rms_z  mean_z")
            for r in sorted([x for x in self.rows if x["metric"] == metric], key=lambda x: x["n_stars"]):
                ratio = r["ci_width"] / r["cr_bound"] if r["cr_bound"] > 0 else float("nan")
                rms_z = r.get("rms_z", float("nan"))
                mean_z = r.get("mean_z", float("nan"))
                lines.append(
                    f"    {r['n_stars']:<8.4g} {r['coverage']:5.2f}  {ratio:5.2f}  {r['bias']:+.3f}  "
                    f"{rms_z:5.2f}  {mean_z:+.3f}"
                )
        return "\n".join(lines)

    def aa_noise(self, metric_a, metric_b, n_stars):
        """Observed difference between two metrics that should be identical under
        the truth used: an A/A test whose true difference is zero.

        Returns the differences in coverage, bias and rms_z, which estimate the
        end-to-end noise floor of each statistic, including mock-draw and NUTS
        sampling noise, not just the binomial term the coverage floor assumes.

        This only makes sense if *metric_a* and *metric_b* really are
        interchangeable, which means ``("sigma_x", "sigma_y")`` or
        ``("mean_x", "mean_y")`` on the ``"isotropic"`` truth, where the square
        grid and symmetric prior make x and y statistically identical. On an
        anisotropic truth the result is meaningless. The method can only check
        the truth, not the pair, so it raises unless ``self.truth_name`` is
        ``"isotropic"``; choosing an interchangeable pair is up to the caller.

        Parameters
        ----------
        metric_a, metric_b : str
            The two metrics to compare.
        n_stars : float or int
            Star count at which to compare them.

        Returns
        -------
        dict
            ``{"d_coverage": ..., "d_bias": ..., "d_rms_z": ...}``, each the
            absolute difference between the two metrics' rows.

        Raises
        ------
        ValueError
            If ``self.truth_name != "isotropic"``, or if either metric has no row
            at ``n_stars``.
        """
        if self.truth_name != "isotropic":
            msg = (
                f"aa_noise is only meaningful on the 'isotropic' truth, where "
                f"sigma_x/sigma_y and mean_x/mean_y are exchangeable by "
                f"symmetry; this curve's truth_name is {self.truth_name!r}"
            )
            raise ValueError(msg)

        def _row(metric):
            matches = [
                r for r in self.rows
                if r["metric"] == metric and r["n_stars"] == float(n_stars)
            ]
            if not matches:
                msg = f"no row for metric={metric!r}, n_stars={n_stars!r}"
                raise ValueError(msg)
            return matches[0]

        row_a = _row(metric_a)
        row_b = _row(metric_b)
        return {
            "d_coverage": abs(row_a["coverage"] - row_b["coverage"]),
            "d_bias": abs(row_a["bias"] - row_b["bias"]),
            "d_rms_z": abs(row_a["rms_z"] - row_b["rms_z"]),
        }

    def mcnemar(self, other, metric, n_stars):
        """Paired comparison with *other* at one metric and star count.

        Only valid if both curves were built with the same seed and ``n_stars``.
        ``recovery_curve_2d`` reseeds from the base ``seed`` at every ``n_stars``
        point, so two such curves see identical mock datasets, realisation by
        realisation. That pairing is what makes a McNemar test meaningful: it
        isolates whatever differs between the curves (grid settings, say) from
        realisation noise. If the seeds differ, the discordant counts mean
        nothing, and this method cannot detect that; it is up to the caller.

        Parameters
        ----------
        other : RecoveryCurve2D
            The curve to compare with.
        metric : str
            One of the five 2D moments.
        n_stars : float or int
            Star count at which to compare.

        Returns
        -------
        b, c : int
            Discordant counts: ``b`` where this curve hit and ``other`` missed,
            ``c`` the reverse.
        pvalue : float
            Two-sided exact binomial p-value for ``b`` out of ``b + c`` at
            p = 0.5 (``scipy.stats.binomtest``).

        Raises
        ------
        ValueError
            If either curve has no row for ``metric`` at ``n_stars``, or the two
            hit vectors differ in length.
        """
        from scipy.stats import binomtest

        def _row(curve, label):
            matches = [
                r for r in curve.rows
                if r["metric"] == metric and r["n_stars"] == float(n_stars)
            ]
            if not matches:
                msg = (
                    f"{label} curve has no row for metric={metric!r}, "
                    f"n_stars={n_stars!r}"
                )
                raise ValueError(msg)
            return matches[0]

        row_self = _row(self, "self")
        row_other = _row(other, "other")

        hits_self = row_self.get("hits")
        hits_other = row_other.get("hits")
        if hits_self is None or hits_other is None:
            msg = "both rows must carry a 'hits' vector (produced by recovery_curve_2d)"
            raise ValueError(msg)
        if len(hits_self) != len(hits_other):
            msg = (
                f"hit vector length mismatch: self has {len(hits_self)}, "
                f"other has {len(hits_other)} -- curves are not paired"
            )
            raise ValueError(msg)

        b = sum(1 for hs, ho in zip(hits_self, hits_other) if hs and not ho)
        c = sum(1 for hs, ho in zip(hits_self, hits_other) if ho and not hs)
        result = binomtest(b, b + c, 0.5) if (b + c) > 0 else None
        pvalue = 1.0 if result is None else result.pvalue
        return b, c, pvalue


def _validate_grid_override(grid):
    """Check a ``grid`` override for ``recovery_curve_2d`` for values that
    would silently break the grid or the Dynamite output.

    Dynamite needs an odd bin count on each axis (so there is a centre bin at
    zero), and widths must be positive. Raises ``ValueError`` naming the
    offending axis and value.
    """
    width = grid["width"]
    n_bins = grid["n_bins"]
    wx, wy = (width, width) if np.isscalar(width) else tuple(width)
    kx, ky = (n_bins, n_bins) if np.isscalar(n_bins) else tuple(n_bins)

    for axis, k in (("x", kx), ("y", ky)):
        if int(k) % 2 == 0:
            msg = f"grid override n_bins[{axis}] = {k} is even; DYNAMITE requires an odd bin count per axis"
            raise ValueError(msg)
    for axis, w in (("x", wx), ("y", wy)):
        if not (w > 0):
            msg = f"grid override width[{axis}] = {w} is not positive"
            raise ValueError(msg)


def _resolve_grid(profile, grid):
    """The one place that decides the (center, width, n_bins) used by
    ``recovery_curve_2d``'s truth-moment solver, its per-realisation solver,
    and ``_discretised_truth_moments``, which must all agree.

    ``grid`` is ``None`` (a square grid from the profile, the original
    behaviour) or an override dict with keys ``width`` and ``n_bins``, each a
    scalar or a 2-tuple.
    """
    center = (0.0, 0.0)
    if grid is None:
        return center, (profile.grid_width, profile.grid_width), profile.n_bins

    _validate_grid_override(grid)
    return center, grid["width"], grid["n_bins"]


def square_cell_grid(sigma_ref, half_extent_x_sigma, half_extent_y_sigma, cell_sigma):
    """Rectangular grid with SQUARE cells, sized per axis in units of
    ``sigma_ref``.

    ``sigma_ref`` cancels out of the bin counts and only scales the widths;
    it is a parameter so callers can pass a profile's ``sigma_ref`` directly.

    The cell width is ``h = cell_sigma * sigma_ref``. Each axis gets
    ``2 * half_extent_*_sigma * sigma_ref / h`` bins, rounded UP to the next
    ODD integer (Dynamite needs odd counts), and its width is then set to
    exactly ``n_bins * h`` so the cells stay square. Each axis is therefore
    at least as wide as requested, never narrower.

    Square cells matter because the GMRF prior's diagonal weighting
    (``diag_weight=1/sqrt(2)`` in ``build_gmrf_precision``) assumes a square
    lattice. Non-square cells would change what the smoothness prior means,
    and supporting them is out of scope.

    Returns
    -------
    dict
        ``{"width": (wx, wy), "n_bins": (kx, ky)}``, suitable for the
        ``grid`` argument of ``recovery_curve_2d``.
    """
    h = cell_sigma * sigma_ref

    def _axis(half_extent_sigma):
        full_width = 2.0 * half_extent_sigma * sigma_ref
        k = int(np.ceil(full_width / h))
        if k % 2 == 0:
            k += 1
        return k, k * h

    kx, wx = _axis(half_extent_x_sigma)
    ky, wy = _axis(half_extent_y_sigma)
    return {"width": (wx, wy), "n_bins": (kx, ky)}


def recovery_curve_2d(
    profile,
    truth_name,
    n_stars_values,
    n_real=25,
    prior="gaussian_core",
    num_warmup=300,
    num_samples=600,
    seed=20260805,
    grid=None,
):
    """Sweep star count and measure bias, coverage and efficiency for all
    five 2D moments, including the tilt ``rho``.

    The 2D counterpart of :func:`veldist.calibration.recovery_curve`. It
    sweeps ``n_stars`` directly, because there is no 2D ``ivar`` yet: the
    information in a correlation coefficient is not a simple sum over stars
    the way it is for a mean. So it answers "is it calibrated at this star
    count", not "does that carry over to data with different errors". Build
    the generalisation only if a regime beyond Gaia needs it.

    ``profile.n_stars`` is ignored. Only the profile's error distribution
    (``profile.draw_errors``) and grid (``profile.grid_width``,
    ``profile.n_bins``) are used, with each swept ``n_stars`` value
    substituted.

    The cost is ``len(n_stars_values) * n_real`` NUTS runs. Lower ``n_real``
    for a smoke test, but not for a result meant to set a threshold.

    Each row also holds the standardised residuals ``z``, one per
    realisation, ``(median - truth) / half68`` (``nan`` where ``half68`` is
    zero or not finite; see ``n_z_excluded``), and their summaries:
    ``rms_z`` (1.0 if calibrated; above 1 means intervals too narrow, below 1
    too wide), ``mean_abs_z`` (target sqrt(2/pi) ~= 0.7979) and ``mean_z``
    (standardised bias, target 0). These keep the information that
    ``coverage`` discards by thresholding at 1.0; :class:`RecoveryCurve2D`
    explains why that matters.

    Parameters
    ----------
    profile : ObservingProfile2D
        Source of the error distribution and grid; ``n_stars`` is replaced at
        each sweep point.
    truth_name : str
        ``"isotropic"`` or ``"anisotropic"`` (see :func:`truths_for`).
    n_stars_values : sequence of int
        Star counts to sweep.
    n_real : int
        Mock realisations per ``n_stars`` value.
    prior, num_warmup, num_samples
        Passed to ``KinematicSolver2D.run``.
    seed : int
        Base seed; realisation ``i`` at each ``n_stars`` uses ``seed + i``,
        as in :func:`veldist.calibration.recovery_curve`.
    grid : dict, optional
        Replaces the grid derived from the profile:
        ``{"width": w, "n_bins": k}``, each a scalar (square grid) or an
        ``(x, y)`` tuple (rectangular grid). ``center`` is always
        ``(0.0, 0.0)``. With ``None`` (default) the grid comes from
        ``profile`` as before. :func:`square_cell_grid` builds a rectangular,
        square-celled override from a target resolution and extent. Checked by
        :func:`_validate_grid_override`: per-axis counts must be odd (a
        Dynamite requirement) and widths positive.

    Returns
    -------
    RecoveryCurve2D
    """
    from veldist.veldist2d import KinematicSolver2D

    truth = truths_for(profile.sigma_ref)[truth_name]
    rows = []
    metrics = ["mean_x", "mean_y", "sigma_x", "sigma_y", "rho"]

    # Single source of truth for (center, width, n_bins): read by solver0
    # (truth-moment edges), _discretised_truth_moments, and the per-
    # realisation solver in the loop below. Keep it that way -- if these
    # ever diverge the coverage numbers are silently meaningless.
    grid_center, grid_width, n_bins = _resolve_grid(profile, grid)

    solver0 = KinematicSolver2D()
    solver0.setup_grid(center=grid_center, width=grid_width, n_bins=n_bins)
    centers_2d = solver0.grid["centers_2d"]
    edges_x = solver0.grid["edges_x"]
    edges_y = solver0.grid["edges_y"]

    true_moments, _ = _discretised_truth_moments(truth, edges_x, edges_y, centers_2d)

    for n_stars in n_stars_values:
        hits = {m: 0 for m in metrics}
        meds = {m: [] for m in metrics}
        widths = {m: [] for m in metrics}
        rng = np.random.default_rng(seed)

        hit_vectors = {m: [] for m in metrics}
        z_vectors = {m: [] for m in metrics}

        for i in range(n_real):
            obs_x, obs_y, cov = _draw_stars(rng, truth, n_stars, profile)

            solver = KinematicSolver2D()
            solver.setup_grid(center=grid_center, width=grid_width, n_bins=n_bins)
            solver.add_data(obs_x, obs_y, cov)
            samples = solver.run(
                num_warmup=num_warmup, num_samples=num_samples, seed=seed + i, prior=prior
            )
            pdf_samples = np.asarray(samples["intrinsic_pdf"])
            mean_x, mean_y, sigma_x, sigma_y, rho = _moments_from_pdf_samples_2d(
                pdf_samples, centers_2d, solver.grid
            )
            draws = {"mean_x": mean_x, "mean_y": mean_y, "sigma_x": sigma_x, "sigma_y": sigma_y, "rho": rho}

            for m in metrics:
                median = float(np.median(draws[m]))
                half68 = 0.5 * (np.percentile(draws[m], 84) - np.percentile(draws[m], 16))
                meds[m].append(median)
                widths[m].append(half68)
                hit = abs(median - true_moments[m]) <= half68
                hit_vectors[m].append(bool(hit))
                if hit:
                    hits[m] += 1
                if half68 > 0 and np.isfinite(half68):
                    z_vectors[m].append((median - true_moments[m]) / half68)
                else:
                    z_vectors[m].append(float("nan"))

        # Cramer-Rao-style bounds at this n_stars. See RecoveryCurve2D's
        # docstring for the rho approximation's caveat.
        sx, sy, rho_t = truth["sx"], truth["sy"], truth["rho"]
        cr = {
            "mean_x": sx / np.sqrt(n_stars),
            "mean_y": sy / np.sqrt(n_stars),
            "sigma_x": sx / np.sqrt(2 * n_stars),
            "sigma_y": sy / np.sqrt(2 * n_stars),
            "rho": (1.0 - rho_t**2) / np.sqrt(n_stars),
        }

        for m in metrics:
            z = np.asarray(z_vectors[m], dtype=float)
            finite = np.isfinite(z)
            n_excluded = int(np.sum(~finite))
            z_ok = z[finite]
            if z_ok.size > 0:
                rms_z = float(np.sqrt(np.mean(z_ok**2)))
                mean_abs_z = float(np.mean(np.abs(z_ok)))
                mean_z = float(np.mean(z_ok))
            else:
                rms_z = mean_abs_z = mean_z = float("nan")

            rows.append(
                {
                    "n_stars": float(n_stars),
                    "truth": truth_name,
                    "metric": m,
                    "bias": float(np.median(meds[m]) - true_moments[m]),
                    "coverage": hits[m] / n_real,
                    "ci_width": float(np.mean(widths[m])),
                    "cr_bound": float(cr[m]),
                    "hits": list(hit_vectors[m]),
                    "z": list(z_vectors[m]),
                    "rms_z": rms_z,
                    "mean_abs_z": mean_abs_z,
                    "mean_z": mean_z,
                    "n_z_excluded": n_excluded,
                }
            )

    return RecoveryCurve2D(profile=profile, truth_name=truth_name, rows=rows, n_real=n_real)


def recommend_grid_2d(profile, v_systemic=(0.0, 0.0)):
    """Grid arguments for ``KinematicSolver2D.setup_grid`` or
    ``fit_all_bins_2d(grid_kwargs=...)`` from a measured
    :class:`ObservingProfile2D`, instead of choosing a grid by hand. The 2D
    counterpart of :func:`veldist.calibration.recommend_grid`.

    Read the returned ``warnings``: they name the cases where the profile's
    own numbers say no grid will work, instead of quietly returning one that
    looks fine.

    Parameters
    ----------
    profile : ObservingProfile2D
        Usually ``ObservingProfile2D.from_data(...)`` on the real catalogue.
    v_systemic : tuple of float
        Grid centre ``(v1, v2)``, km/s. Default ``(0, 0)``. Dynamite requires
        a proper-motion grid centred on zero, so other values are for
        diagnostics only.

    Returns
    -------
    dict
        ``center``, ``width`` and ``n_bins`` (as ``setup_grid`` takes them),
        plus ``cell_width``, ``stars_per_cell`` and ``warnings``.
    """
    n = profile.n_bins
    width = profile.grid_width
    cell = width / n
    warnings = []

    if profile.error_floor_width > cell:
        warnings.append(
            f"errors dominate: cells are {cell:.2f} km/s but the measurement "
            f"errors only support {profile.error_floor_width:.2f} km/s "
            f"(err/sigma_min = {profile.err_median / profile.sigma_lo:.2f}). "
            "The narrowest bins are smeared beyond what any grid recovers."
        )

    # Lowest occupancy any sweep has actually validated. The Gaia re-anchor
    # (2026-09-01 pm, cps 0.70-1.15 at 435 stars) passed at 0.52 stars/cell,
    # superseding the earlier 0.70 floor -- which was tripping on Gaia's own
    # adopted grid at 0.696, i.e. by 0.004, and reading as if the ADOPTED
    # setting were unvalidated when it sits between two measured points.
    MEASURED_OCCUPANCY_FLOOR = 0.52
    stars_per_cell = profile.n_stars / n**2
    if stars_per_cell < MEASURED_OCCUPANCY_FLOOR:
        warnings.append(
            f"{stars_per_cell:.2f} stars/cell at {profile.n_stars} stars/bin; "
            f"below the {MEASURED_OCCUPANCY_FLOOR:.2f} measured on the "
            "2026-09-01 sweeps, so this is extrapolation."
        )

    return {
        "center": tuple(v_systemic),
        "width": width,
        "n_bins": n,
        "cell_width": cell,
        "stars_per_cell": stars_per_cell,
        "warnings": warnings,
    }


#: HST, inner region, bright stars (m_F625W ~ 18). Median 1D PM error
#: 0.011 mas/yr = 0.24 km/s. err/sigma ~ 0.014: the errors are about 1% of the
#: signal, so there is very little left to deconvolve.
HST_BRIGHT = ObservingProfile2D(
    name="hst_bright",
    sigma_ref=17.0,
    err_median=0.24,
    err_cut=PM_QUALITY_CUT_KMS,
    n_stars=400,
)

#: HST, faint end, near the quality cut. Comparable to the 1D LOS regime,
#: whose err/sigma is 0.11.
HST_FAINT = ObservingProfile2D(
    name="hst_faint",
    sigma_ref=17.0,
    err_median=2.5,
    err_cut=PM_QUALITY_CUT_KMS,
    n_stars=400,
)

#: Gaia DR3, outer region. Gaia is ~30x worse than HST at the same magnitude
#: (median ~0.35 mas/yr = 7.5 km/s at G~18), and the outer dispersion is
#: SMALLER, so err/sigma is far worse. For reference, 1D classifies its own
#: err/sigma = 0.36 regime as a *structural* failure -- this is harder still,
#: with 13x the stars as the only compensation. Larger errors also demand a
#: wider grid, hence the raised n_sigma_grid.
GAIA_OUTER = ObservingProfile2D(
    name="gaia_outer",
    sigma_ref=8.0,
    err_median=5.0,
    err_cut=4.0 * PM_QUALITY_CUT_KMS,
    n_stars=2000,
    n_sigma_grid=4.0,
)

#: The three profiles above are hand-picked regimes, kept because the whole
#: calibration campaign was run against them. The two below are what the
#: production dataprep notebooks actually produce, measured 2026-08-31 by
#: replaying each notebook's own binning call on the real catalogues
#: (``omegaCen/dynamite_dataprep/{gaia,hst}_veldist.ipynb``). Where they
#: disagree with the hand-picked versions, these are right.
#:
#: sigma_min/sigma_max are taken across BOTH axes (the narrowest bin on
#: either component, and the widest), not from the isotropic-equivalent
#: sigma: the grid must resolve and contain each axis separately. The two
#: axes' medians differ by only ~5%, so the axis-to-axis asymmetry is minor;
#: what matters is the ~2x min-to-max spread WITHIN an axis across bins.
#:
#: Gaia: ``do_powerbin(target_capacity=400)``, 300-1500 arcsec, 148 bins.
#: n_stars=2000 in ``GAIA_OUTER`` was a guess and is 4.6x the truth.
#:
#: **err_cut/err_log_sigma corrected 2026-09-01.** This profile was declared
#: with ``err_cut=PM_QUALITY_CUT_KMS`` (7.81 km/s), which is BELOW its own
#: ``err_median`` of 8.60: ``draw_errors`` clamped the spread to its 0.25
#: floor and then clipped, so 65% of every mock's per-star errors came out
#: pinned at exactly 7.81 km/s and the p95 was 7.81 against a real 27.5.
#: Gaia's notebook applies no meaningful error cut at all -- its filter is
#: ``pmrae/pmdece < 10 mas/yr`` = 260 km/s, and ``PM_QUALITY_CUT_KMS``
#: appears there only as a reference line on a plot. The spread is therefore
#: set by the magnitude distribution, measured over the 64537 stars in
#: [300, 1500) arcsec: median 8.33, p95 27.54, p99 40.35 km/s, giving
#: ``err_log_sigma`` 0.727. The Gaia entry in ``_CPS_ANCHORS`` was measured
#: with the collapsed errors and needs re-measuring; its direction is the
#: safe one (real errors are larger, which inflates intervals and makes
#: coarse cells easier to justify), so 0.85 is more likely conservative than
#: optimistic, but it is not yet earned.
GAIA_OUTER_MEASURED = ObservingProfile2D(
    name="gaia_outer_measured",
    sigma_ref=11.1,
    err_median=8.60,
    err_cut=GAIA_PM_ERR_CEILING_KMS,
    err_log_sigma=0.727,
    n_stars=435,
    n_sigma_grid=4.0,
    sigma_min=7.03,
    sigma_max=16.02,
    rotation_span=12.8,
)

#: HST: ``do_powerbin(target_capacity=400)``, cell_width=5, 1415 bins. The
#: err_median measured here is 6x the 0.24 km/s ``HST_BRIGHT`` assumes.
#: n_stars is the MEDIAN; the minimum bin holds 174, which is the case the
#: calibration has never been run at.
#:
#: **err_log_sigma added 2026-09-01**, same audit that caught Gaia's collapse
#: but failing the other way. HST's ``err_cut`` is real -- the 0.3 mas/yr
#: quality cut is applied upstream and the catalogue's largest error is
#: 7.741 km/s against the 7.81 cut -- but it sits at ~p100, not the p95 the
#: back-derivation assumes, so ``max(0.25, log(7.81/1.51)/1.645)`` returned
#: 0.999 against a measured 0.434. Mocks drew p95 = 7.78 km/s where the
#: 610846 selected stars give 3.14 (p99 4.44, max 7.74).
#:
#: This is the DANGEROUS direction: over-broad errors inflate the posterior
#: intervals, which is what ``rms_z`` divides by, so coarse cells looked more
#: acceptable than they are. HST's ``_CPS_ANCHORS`` entry (0.58) may
#: therefore be too coarse and must be re-measured -- unlike Gaia's, its
#: error does not point somewhere safe.
HST_MEASURED = ObservingProfile2D(
    name="hst_measured",
    sigma_ref=16.08,
    err_median=1.51,
    err_cut=PM_QUALITY_CUT_KMS,
    err_log_sigma=0.434,
    n_stars=426,
    sigma_min=11.50,
    sigma_max=21.54,
    rotation_span=16.7,
)

PROFILES_2D = {
    p.name: p
    for p in (HST_BRIGHT, HST_FAINT, GAIA_OUTER, GAIA_OUTER_MEASURED, HST_MEASURED)
}
