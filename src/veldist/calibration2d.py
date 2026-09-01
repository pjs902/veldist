"""
Observing profiles for the 2D (proper-motion) solver.

The 1D harness derives its velocity grid from the observing regime rather than
having it chosen by hand (see ``calibration.py``); this is the 2D counterpart.
It exists because the 2D test harness previously had no profile at all: its
grid and errors were inherited from the SBC harness, whose own comment calls
the grid an "arbitrary physical span", and its star count was copied from the
*line-of-sight* profile. Proper motions reach ~6 magnitudes deeper than the
spectroscopy, so both the star counts and the errors are quite different.

Calibration source: the oMEGaCat proper-motion uncertainty-vs-magnitude
figures. The unit conversion uses the standard 1 mas/yr = 4.740470 *
distance[kpc] km/s relation at the adopted cluster distance of 5494 pc
(Peter, 2026-08-06), giving 1 mas/yr = 26.04 km/s. The quality cut is
0.3 mas/yr.
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

#: The 0.3 mas/yr proper-motion quality cut, in km/s.
PM_QUALITY_CUT_KMS = 0.30 * KMS_PER_MASYR


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
    """Target cell width in units of sigma, for a given measurement-error
    regime.

    A single global constant is the wrong shape. HST and Gaia disagree by
    ~1.5x, and the reason is that ``rms_z`` is bias over interval width:
    HST's err/sigma_lo is 0.13 against Gaia's 1.22, so its posterior is sharp
    and there is nothing to hide a residual discretisation bias behind. Gaia
    tolerates coarse cells because its large errors inflate the intervals
    enough to swallow the same absolute error. **Precise data needs FINER
    grids** -- the opposite of the usual intuition.

    This is an **empirical two-point power law**, not a derived result. The
    obvious physical model -- keep the discretisation bias below the
    statistical error, which scales as ``(1 + (err/sigma)^2)^(1/4)`` --
    predicts a ratio of 1.20 between these two regimes, against 1.47
    measured. Something else contributes, most plausibly that a weak
    likelihood lets the roughness prior smooth the recovered pdf, so coarse
    cells cost less than the error budget alone suggests. Until that is
    understood, do not extrapolate this far outside the anchors; the
    exponent is fitted to two points and carries no theory.

    Parameters
    ----------
    err_over_sigma : float
        Median per-star measurement error divided by the dispersion the grid
        has to resolve (``profile.err_median / profile.sigma_lo``).

    Returns
    -------
    float
        Cell width in units of sigma, clipped to the measured range so a
        wild input cannot silently produce an absurd grid.
    """
    (e_lo, c_lo), (e_hi, c_hi) = _CPS_ANCHORS
    p = np.log(c_hi / c_lo) / np.log(e_hi / e_lo)
    e = float(np.clip(err_over_sigma, e_lo, e_hi))
    return float(c_hi * (e / e_hi) ** p)


@dataclass(frozen=True)
class ObservingProfile2D:
    """A proper-motion observing regime, and the velocity grid it implies.

    Parameters
    ----------
    name : str
        Registry key, also used in reports.
    sigma_ref : float
        Representative intrinsic velocity dispersion, km/s.
    err_median : float
        Sample-median per-star measurement error, km/s.
    err_cut : float
        Upper truncation on the error distribution, km/s (the quality cut).
    n_stars : int
        Stars per spatial (Voronoi) bin -- the science target. Fewer stars per
        bin means more bins, hence better spatial coverage, so this is a
        resolution choice and not a convenience value.
    n_sigma_grid : float
        Half-width of the velocity grid in units of ``sigma_ref``.
    cell_per_sigma : float or None
        Target cell width in units of ``sigma_lo``. ``None`` (the default)
        derives it from this profile's own error regime via
        :func:`cell_per_sigma_for` -- the right behaviour, since the
        requirement is regime-dependent and a shared constant fails one of
        the two measured datasets. Set it explicitly only to pin a grid.

        **Re-measured 2026-09-01** against the real Gaia profile (435 stars,
        the measured dispersion range and rotation span), post-h^2/12-fix,
        sweeping {0.85, 1.10, 1.40, 1.80} on both truths, 40 realisations
        each:

            cps    K   stars/cell   sigma_y bias   sigma_y rms_z   rho rms_z
            0.85  25      0.70      +0.150 (2.8%)      0.97          1.18
            1.10  19      1.20      +0.238 (4.5%)      1.01          1.16
            1.40  15      1.93      +0.439 (8.2%)      1.17          1.05
            1.80  13      2.57      +0.670 (12.5%)     1.53          1.50

        (bias on the ANISOTROPIC truth's narrow axis; percentages are of that
        axis's own sy=5.34. The isotropic truth stays flat across the whole
        range -- the failure only appears on an anisotropic velocity
        ellipsoid, so an isotropic-only check does not see it.)

        0.85 is chosen for <3% bias with both rms_z near 1. 1.10 is
        defensible if compute forces it; 1.40 and beyond are not -- rms_z 1.5
        at 1.80 means the reported intervals are half the width they should
        be, on both sigma_y and rho.

        Note the scale mismatch this exposes: cell_per_sigma is defined
        against ``sigma_lo``, but the narrow axis of an anisotropic ellipsoid
        is smaller still (0.65x here), so the cells are ~1.3x that axis's own
        sigma even at 0.85. Resolution has the same per-axis character extent
        does; defining it against the narrowest AXIS rather than the
        narrowest BIN would be the cleaner fix, and is not done yet.

        The previous value (0.47) came from a sweep run BEFORE the h^2/12
        correction, when refining the grid shrank a bias the estimator itself
        was manufacturing -- that sweep measured the bug's
        resolution-dependence and read it as a resolution requirement. (That
        sweep ran cell_per_sigma in {0.78 ... 0.37} at N=400 under the
        gaussian_core prior; its conclusion that "K=19 breaks on anisotropic
        truths" is superseded -- the breakage was the estimator's, not the
        grid's.)
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

        DYNAMITE takes one scalar ``vxrange``/``vyrange`` per map, so a single
        grid must serve every spatial bin: it has to hold the widest LOSVD in
        the field plus the mean-velocity offset of the bins furthest from
        systemic. Mirrors :attr:`ObservingProfile.grid_width` in 1D.
        """
        return 2.0 * self.n_sigma_grid * self.sigma_hi + self.rotation_span

    @property
    def cells_per_sigma_target(self):
        """Cell width in units of sigma: explicit override, else derived from
        this profile's own error regime via :func:`cell_per_sigma_for`."""
        if self.cell_per_sigma is not None:
            return self.cell_per_sigma
        return cell_per_sigma_for(self.err_median / self.sigma_lo)

    @property
    def cell_width(self):
        """Target cell width, km/s: resolve the narrowest LOSVD in the field.

        Uses ``sigma_lo``, not ``sigma_ref`` -- one shared grid must resolve
        every bin, and the narrowest one is the binding case.
        """
        return self.cells_per_sigma_target * self.sigma_lo

    @property
    def error_floor_width(self):
        """Cell width below which refining buys nothing, km/s.

        1D sets its bin width to exactly this (``bins_per_error *
        err_median``): the measurement errors have already smeared the signal
        at that scale. It is reported rather than imposed here because in 2D
        cells cost quadratically, and because when it EXCEEDS
        :attr:`cell_width` the two rules genuinely conflict -- the errors
        dominate and no grid resolves the narrowest bins. That is the Gaia
        regime (err/sigma approaches 1) and it is a fact about the data, not
        a tuning choice. :func:`recommend_grid_2d` flags it.
        """
        return self.bins_per_error * self.err_median

    @property
    def n_bins(self):
        """Cells per axis. Always odd: DYNAMITE's ProperMotions reader
        (``set_default_hist_bins``) raises ValueError on even counts."""
        n = int(round(self.grid_width / self.cell_width))
        n = max(n, 5)
        return n if n % 2 == 1 else n + 1

    @property
    def err_over_sigma(self):
        """The ratio that drives deconvolution difficulty."""
        return self.err_median / self.sigma_ref

    def draw_errors(self, n, rng):
        """Draw ``n`` per-star measurement errors, km/s.

        Log-normal about ``err_median``, truncated at ``err_cut``. The spread
        is set so the cut sits at roughly the 95th percentile, which is what
        the magnitude-dependent error distribution looks like once a quality
        cut is applied.
        """
        sigma_log = max(0.25, np.log(self.err_cut / self.err_median) / 1.645)
        e = rng.lognormal(np.log(self.err_median), sigma_log, n)
        return np.clip(e, 1e-3, self.err_cut)

    @classmethod
    def from_data(cls, pm1, pm2, err1, err2, bin_ids, err_cut, name="measured", min_stars=10):
        """Measure a profile from a real proper-motion catalogue.

        Mirrors :meth:`ObservingProfile.from_data`: both HST and Gaia
        dataprep notebooks independently hand-rolled this exact per-bin
        estimator, so it belongs here instead of duplicated per notebook.

        ``err_cut`` is a quality cut applied upstream (Gaia's is ~4x HST's),
        not something measurable from post-cut data, so it is a required
        argument rather than derived.

        Parameters
        ----------
        pm1, pm2 : array-like, shape (n_stars,)
            Proper-motion components, km/s, in the frame the grid is defined
            in.
        err1, err2 : array-like, shape (n_stars,)
            Per-star measurement errors on ``pm1``/``pm2``, km/s.
        bin_ids : array-like, shape (n_stars,)
            Spatial bin index for each star. Values need not be contiguous.
        err_cut : float
            Upper truncation on the error distribution, km/s.
        name : str
            Label carried into the returned profile.
        min_stars : int
            Bins with fewer stars are excluded from every measured
            statistic (``sigma_ref``, ``err_median``, ``n_stars``), applied
            consistently rather than per-statistic.

        Returns
        -------
        ObservingProfile2D

        Raises
        ------
        ValueError
            If fewer than 2 bins survive the *min_stars* cut.
        """
        pm1 = np.asarray(pm1, dtype=float)
        pm2 = np.asarray(pm2, dtype=float)
        err1 = np.asarray(err1, dtype=float)
        err2 = np.asarray(err2, dtype=float)
        bin_ids = np.asarray(bin_ids)

        per_bin_n, per_bin_sigma, per_bin_err_med = [], [], []
        per_bin_mean1, per_bin_mean2 = [], []
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
        )

    def cells_per_sigma(self, axis_sigma):
        """Cell width in units of an axis's own intrinsic dispersion.

        ``cell_per_sigma`` and ``n_bins`` are defined relative to the single
        scalar ``sigma_ref``, which only describes the actual per-axis
        resolution when the velocity ellipsoid is isotropic. For an
        anisotropic truth, a narrow axis (smaller ``axis_sigma``) gets a
        LARGER (coarser) number here than ``sigma_ref`` does, and a wide axis
        gets a smaller (finer) one -- the grid itself doesn't change, only
        how fine it is relative to that axis's own spread.
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
    """Scale the two test truths (isotropic, anisotropic) to a profile's
    sigma_ref rather than hardcoding absolute km/s values.

    Shared by ``test_coverage_2d.py`` and :func:`recovery_curve_2d` so the
    two never drift apart -- previously duplicated in the test module only.
    """
    return {
        "isotropic": dict(mux=0.0, muy=0.0, sx=sigma, sy=sigma, rho=0.0),
        "anisotropic": dict(
            mux=0.18 * sigma, muy=-0.12 * sigma,
            sx=1.18 * sigma, sy=0.76 * sigma, rho=0.4,
        ),
    }


def _draw_stars(rng, truth, n_stars, profile):
    """Draw one mock bin's observed (x, y) proper motions and per-star
    diagonal covariance for a given truth, star count, and profile's error
    distribution."""
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
    """Moments to compare the recovered fit against, plus the TRUE exact
    per-cell probability mass (used for per-cell coverage, not for the
    returned moments).

    Returns the CONTINUOUS analytic truth's mean/sigma/rho -- i.e. exactly
    ``t``'s own ``mux``/``muy``/``sx``/``sy``/``rho``, independent of the
    grid -- alongside the exact per-cell mass array on ``edges_x``/
    ``edges_y`` (still needed by callers that check per-cell coverage
    against ``mass``, e.g. ``test_per_cell_losvd_coverage_2d``).

    This function used to return cell-centre POINT-MASS moments of ``mass``
    instead (i.e. Sheppard-inflated: ``V + h^2/12`` per axis), on the
    argument that this was "the fair comparison" because comparing a
    cell-centre moment against the continuous truth "charges the model for
    grid discretisation". That argument was correct for the OLD
    ``_moments_from_pdf_samples_2d``, which computed the same kind of
    point-mass moment on the recovered posterior (``V_hat - h^2/12``, since
    the likelihood's forward model spreads each cell's mass uniformly
    across the cell -- see that function's docstring for the full
    three-way derivation). Comparing two point-mass estimators of the same
    biased quantity was consistent.

    ``_moments_from_pdf_samples_2d`` now adds ``h^2/12`` back so it
    estimates the CONTINUOUS variance instead (matching what the
    likelihood actually fits). Once the recovered side targets the
    continuous quantity, comparing it against the OLD Sheppard-inflated
    target here would double-count the ``h^2/12`` term (add it once on
    "truth", once again implicitly via the recovered side, netting a
    ``h^2/6`` gap in the wrong direction). The correct target now is the
    quantity `_moments_from_pdf_samples_2d` actually estimates: the
    continuous truth, full stop -- which is also grid-independent, as it
    should be.

    ``edges_x``/``edges_y`` need not have the same length -- a rectangular
    grid (``kx != ky``) is fine. The flat index follows the same row-major
    convention as ``setup_grid_2d``: ``m = ix * ky + iy``.
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
    """Per-sample mean_x, mean_y, sigma_x, sigma_y, rho from 2D pdf draws.

    ``grid`` must supply the per-axis cell widths (``grid["width_x"]``,
    ``grid["width_y"]`` -- despite the name these are the CELL width, i.e.
    ``edges_x[1] - edges_x[0]``, not the grid's total span; see
    ``setup_grid_2d``'s docstring). Pass the same grid dict the caller used
    to build ``centers_2d`` (e.g. ``solver.grid``).

    Why this needs the cell width at all
    --------------------------------------
    ``p_m`` (one entry of ``pdf_samples``) is interpreted THREE different,
    mutually inconsistent ways across this codebase:

    1. THE LIKELIHOOD (``precompute_design_matrix`` / its 2D counterpart)
       treats ``p_m`` as mass spread UNIFORMLY across cell ``m`` -- that is
       the piecewise-constant assumption implicit in pulling ``p(v) ~= p_m/h``
       out of the per-cell integral when deriving the forward model. The
       density the likelihood actually fits, ``q(v)``, therefore has
       ``Var(q) = sum_m p_m (v_m - mu)^2 + h^2/12`` -- the ``h^2/12`` is the
       exact variance of a Uniform(cell) distribution.
    2. THIS FUNCTION, before this fix, treated ``p_m`` as a POINT MASS at
       the cell centre: ``Var = sum_m p_m (v_m - mu)^2``, with no
       within-cell term -- i.e. it was reporting a different, smaller
       quantity than the one the likelihood fits.
    3. ``_discretised_truth_moments`` computes exact cell masses of the
       analytic truth and (before this fix) ALSO took point-mass moments at
       cell centres, giving ``V + h^2/12`` (Sheppard's correction) where
       ``V`` is the continuous truth variance.

    Since the data drive the likelihood's ``Var(q)`` toward the true
    continuous ``V``, the old (2) reported ``V - h^2/12`` while the old (3)
    target was ``V + h^2/12`` -- a resolution-dependent gap of ``h^2/6`` in
    variance (about ``h^2/(12*sigma)`` in sigma) between what this function
    reported and what it was compared against. Adding ``h^2/12`` here makes
    (2) estimate the same continuous quantity the likelihood fits and
    ``_discretised_truth_moments`` now targets (see that function's
    docstring for the other half of the fix).

    The x/y COVARIANCE term gets no correction: cells are axis-aligned
    rectangles, so the within-cell distribution is a uniform product over
    the cell and x/y are independent within a cell -- zero cross-covariance
    contribution. ``rho`` is recomputed from the corrected variances, which
    slightly reduces ``|rho|`` -- that is the correct consequence of the
    correction, not a separate bug.
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
    """How well each 2D moment (including tilt, ``rho``) is recovered as a
    function of raw star count.

    2D counterpart of :class:`RecoveryCurve`, swept over ``n_stars`` directly
    rather than an information-content proxy (no 2D ``ivar`` equivalent
    exists yet -- see :func:`recovery_curve_2d`'s docstring).

    Notes
    -----
    ``cr_bound`` for ``rho`` uses the standard bivariate-normal MLE
    approximation ``Var(rho_hat) ~= (1 - rho**2)**2 / n``, i.e.
    ``(1 - rho**2) / sqrt(n)`` as a CI-width-like quantity. Like 1D's
    skewness/kurtosis Cramer-Rao bounds, this is only exact for homogeneous
    per-star errors; with the heterogeneous errors this package actually
    fits, this approximation is not reliable for ``rho``. Because of that,
    :meth:`threshold` does not gate ``rho`` on the CI/CR efficiency check --
    only on coverage. The ratio is still computed and printed by
    :meth:`report` for every metric, ``rho`` included, but it is advisory
    only there.

    ``rms_z`` (see :func:`recovery_curve_2d`) sidesteps this entirely: it
    measures interval calibration directly from the standardised residuals
    ``(median - truth) / half68``, with no analytic Cramer-Rao bound
    involved at all. That makes it a Cramer-Rao-FREE efficiency measure --
    unlike the ``ci_width``/``cr_bound`` ratio, it is reliable for ``rho``
    too, and is not exempted from anything.

    It also fixes a resolution problem with ``coverage`` itself: coverage
    thresholds the continuous residual at 1.0 (hit or miss), so a fit that
    misses by 1.01 half-widths counts identically to one that misses by
    3.0. Measured on an isotropic truth, where ``sigma_x`` and ``sigma_y``
    are provably identical in expectation (an A/A test whose true
    difference is zero), the observed coverage difference between them at
    n_real=100 was 0.14 -- the end-to-end noise floor of the coverage
    statistic. The bias statistic's A/A spread over the same run was only
    0.052, about 3x better resolution. Several conclusions had been drawn
    from coverage differences of 0.04-0.09, i.e. below its noise floor.
    ``rms_z`` keeps the continuous residual instead of thresholding it, and
    is the preferred statistic for reading off small differences between
    runs; use :meth:`aa_noise` to measure the noise floor of any of these
    statistics for your own data.
    """

    profile: object
    truth_name: str
    rows: list = field(default_factory=list)
    n_real: int = None

    def threshold(self, metric, min_coverage=None, max_ci_ratio=1.5, band=0.99):
        """Smallest ``n_stars`` at which *metric* is trustworthy, or ``None``.

        Same two-condition, walk-down-from-the-top logic as
        :meth:`RecoveryCurve.threshold`, keyed on ``n_stars`` instead of
        ``ivar``. See that method's docstring for the full rationale.

        The ``ci_width <= max_ci_ratio * cr_bound`` efficiency check does
        NOT gate ``metric == "rho"`` -- only the coverage floor does. This is
        because ``rho``'s ``cr_bound`` uses the bivariate-normal MLE
        approximation, which (see the class docstring) is only exact for
        homogeneous per-star errors; under this package's heterogeneous
        errors it is not a reliable efficiency yardstick, so ``max_ci_ratio``
        is advisory-only for ``rho``. Every other metric keeps both checks.
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
        """Observed difference between two metrics that are exchangeable
        under the truth used, i.e. an A/A test whose true difference is
        zero.

        Returns the observed difference in coverage, bias, and rms_z, which
        together estimate the end-to-end noise floor of each statistic --
        including mock-draw noise and NUTS sampling noise, not just the
        binomial term the coverage floor assumes.

        This is ONLY meaningful when *metric_a* and *metric_b* really are
        exchangeable under the truth this curve was built with --
        specifically ``("sigma_x", "sigma_y")`` or ``("mean_x", "mean_y")``
        on the ``"isotropic"`` truth, where the square grid and symmetric
        prior make x and y statistically identical. On an anisotropic truth
        these metrics are NOT exchangeable and the result is meaningless.
        This method cannot verify the pair itself, only the truth, so it
        raises if ``self.truth_name`` is not ``"isotropic"`` -- callers are
        responsible for only passing an exchangeable pair.

        Parameters
        ----------
        metric_a, metric_b : str
            The two metric names to compare.
        n_stars : float or int
            The swept star count to compare at.

        Returns
        -------
        dict
            ``{"d_coverage": ..., "d_bias": ..., "d_rms_z": ...}``, each the
            absolute difference between the two metrics' rows.

        Raises
        ------
        ValueError
            If ``self.truth_name != "isotropic"``, or if either metric lacks
            a row at ``n_stars``.
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
        """Paired comparison of this curve against *other* at one metric and
        star count, valid only when both were produced at the same seed and
        n_stars (so the mock datasets are identical).

        ``recovery_curve_2d`` reseeds its RNG from the same base ``seed`` at
        the start of every ``n_stars`` sweep point, so two curves built with
        the same ``seed`` and the same ``n_stars`` see byte-identical mock
        datasets realisation-for-realisation. That pairing is what makes a
        McNemar test meaningful here: it isolates the effect of whatever
        differs between the two curves (e.g. grid settings) from
        realisation-to-realisation noise. If the two curves were built with
        different seeds -- or if one of them just happens to share a coverage
        number with the other by coincidence -- the discordant counts
        returned here mean nothing, and this method has no way to detect
        that mismatch on its own; the caller is responsible for only
        comparing curves that share a seed and n_stars.

        Parameters
        ----------
        other : RecoveryCurve2D
            The curve to compare against.
        metric : str
            One of the five 2D moment names.
        n_stars : float or int
            The swept star count to compare at.

        Returns
        -------
        b, c : int
            Discordant counts: ``b`` is the count where this curve hit and
            ``other`` missed; ``c`` is the reverse.
        pvalue : float
            Two-sided exact binomial p-value for ``b`` vs ``b + c`` trials
            at p=0.5 (``scipy.stats.binomtest``).

        Raises
        ------
        ValueError
            If either curve lacks a row for ``metric``/``n_stars``, or if
            the two rows' hit vectors differ in length.
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
    """Check a ``recovery_curve_2d`` ``grid`` override dict for values that
    would silently corrupt DYNAMITE output or the grid itself.

    DYNAMITE requires an odd bin count per axis (a centre bin at zero);
    widths must be positive. Raises ``ValueError`` with a message naming the
    offending axis/value on failure.
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
    """Single source of truth for the (center, width, n_bins) used by
    ``recovery_curve_2d``'s truth-moment solver, the per-realisation solver,
    and ``_discretised_truth_moments`` -- see the module docstring warning
    about those three needing to agree.

    ``grid`` is ``None`` (profile-derived square grid, original behaviour)
    or an override dict with keys ``width`` and ``n_bins``, each a scalar or
    a 2-tuple.
    """
    center = (0.0, 0.0)
    if grid is None:
        return center, (profile.grid_width, profile.grid_width), profile.n_bins

    _validate_grid_override(grid)
    return center, grid["width"], grid["n_bins"]


def square_cell_grid(sigma_ref, half_extent_x_sigma, half_extent_y_sigma, cell_sigma):
    """Rectangular grid with SQUARE cells, sized per axis in units of
    ``sigma_ref``.

    ``sigma_ref`` cancels out of the returned bin counts (only the widths
    scale with it) -- it is here so callers can pass a profile's own
    ``sigma_ref`` without doing the multiplication themselves.

    The cell width is ``h = cell_sigma * sigma_ref``. Per-axis bin count is
    ``2 * half_extent_*_sigma * sigma_ref / h`` rounded UP to the nearest ODD
    integer (DYNAMITE requires odd per-axis counts), then that axis's width
    is set to exactly ``n_bins * h`` so the cells stay exactly square. This
    means the returned width on each axis is always >= the requested extent,
    never smaller -- rounding up preserves the requested minimum extent, it
    just doesn't hit it exactly.

    Square cells matter here because the GMRF prior's diagonal-neighbour
    weighting (``diag_weight=1/sqrt(2)`` in ``build_gmrf_precision``) assumes
    a square lattice; non-square cells would change what the smoothness
    prior actually means, and supporting that is deliberately out of scope.

    Returns
    -------
    dict
        ``{"width": (wx, wy), "n_bins": (kx, ky)}``, suitable for
        ``recovery_curve_2d``'s ``grid`` parameter.
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
    """Sweep raw star count and measure bias, coverage, and efficiency for
    all five 2D moments, including tilt (``rho``).

    2D counterpart of :func:`veldist.calibration.recovery_curve`. Swept over
    ``n_stars`` directly rather than an information-content proxy -- there is
    no 2D ``ivar`` equivalent yet (a correlation coefficient's information
    content is not a simple sum of per-star terms the way a mean's is), so
    this only answers "does it calibrate at this star count", not "does it
    transfer to a dataset with different errors". Build that generalisation
    only if a second regime beyond Gaia's actually needs it.

    ``profile.n_stars`` is ignored; only its error-drawing behaviour
    (``profile.draw_errors``) and grid (``profile.grid_width``/``n_bins``)
    are used, with the swept ``n_stars_values`` substituted per cell.

    Cost is ``len(n_stars_values) * n_real`` NUTS runs. Reduce ``n_real`` for
    a smoke test; do not reduce it for a result meant to set a threshold.

    Parameters
    ----------
    profile : ObservingProfile2D
        Error distribution and grid come from here; ``n_stars`` is
        overridden per sweep point.
    truth_name : str
        ``"isotropic"`` or ``"anisotropic"`` (see :func:`truths_for`).
    n_stars_values : sequence of int
        Star counts to sweep.
    n_real : int
        Mock realisations per ``n_stars`` value.
    prior, num_warmup, num_samples
        Passed through to ``KinematicSolver2D.run``.
    seed : int
        Base RNG seed; realisation ``i`` at a given ``n_stars`` uses
        ``seed + i``, matching :func:`veldist.calibration.recovery_curve`'s
        convention.
    grid : dict, optional
        Overrides the profile-derived grid. ``{"width": w, "n_bins": k}``
        with each of ``w``/``k`` a scalar (square grid, original behaviour)
        or a ``(x, y)`` 2-tuple (rectangular grid). ``center`` is always
        ``(0.0, 0.0)`` and is not configurable here. When ``None`` (default),
        the grid is derived from ``profile`` exactly as before -- this
        parameter changes nothing about the default numerical behaviour.
        See :func:`square_cell_grid` for a helper that builds a rectangular,
        square-celled override from a target resolution and extent.
        Validated by :func:`_validate_grid_override`: per-axis bin counts
        must be odd (DYNAMITE requirement) and widths must be positive.

    Each row also carries the standardised residual vector ``z`` (one entry
    per realisation, ``(median - truth) / half68``, ``nan`` where ``half68``
    is zero or non-finite -- see ``n_z_excluded``) and its aggregates
    ``rms_z`` (target 1.0 under correct calibration; >1 means intervals too
    narrow, <1 too wide), ``mean_abs_z`` (target sqrt(2/pi) ~= 0.7979), and
    ``mean_z`` (standardised bias, target 0). These keep the continuous
    information ``coverage`` throws away by thresholding at 1.0 -- see
    :class:`RecoveryCurve2D`'s docstring for why that matters.

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
    """``KinematicSolver2D.setup_grid`` / ``fit_all_bins_2d(grid_kwargs=...)``
    from a measured :class:`ObservingProfile2D`, instead of hand-picking a
    grid. The 2D counterpart of :func:`veldist.calibration.recommend_grid`.

    The returned ``warnings`` list is the part worth reading: it names the
    cases where the profile's own numbers say no grid will do, rather than
    silently returning one that looks fine.

    Parameters
    ----------
    profile : ObservingProfile2D
        Typically ``ObservingProfile2D.from_data(...)`` on the real catalogue.
    v_systemic : tuple of float
        Grid centre ``(v1, v2)``, km/s. Default ``(0, 0)``: DYNAMITE requires
        a zero-centred PM grid, so this is here for diagnostics, not for
        production use.

    Returns
    -------
    dict
        ``center``, ``width``, ``n_bins`` (as ``setup_grid`` wants them), plus
        ``cell_width``, ``stars_per_cell`` and ``warnings``.
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

    stars_per_cell = profile.n_stars / n**2
    if stars_per_cell < 0.70:
        warnings.append(
            f"{stars_per_cell:.2f} stars/cell at {profile.n_stars} stars/bin; "
            "below the 0.70 measured on the 2026-09-01 sweep, so this is "
            "extrapolation."
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
GAIA_OUTER_MEASURED = ObservingProfile2D(
    name="gaia_outer_measured",
    sigma_ref=11.1,
    err_median=8.60,
    err_cut=PM_QUALITY_CUT_KMS,
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
HST_MEASURED = ObservingProfile2D(
    name="hst_measured",
    sigma_ref=16.08,
    err_median=1.51,
    err_cut=PM_QUALITY_CUT_KMS,
    n_stars=426,
    sigma_min=11.50,
    sigma_max=21.54,
    rotation_span=16.7,
)

PROFILES_2D = {
    p.name: p
    for p in (HST_BRIGHT, HST_FAINT, GAIA_OUTER, GAIA_OUTER_MEASURED, HST_MEASURED)
}
