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
]

#: Adopted cluster distance (Peter, 2026-08-06).
CLUSTER_DISTANCE_PC = 5494.0

#: 1 mas/yr = 4.740470 * distance[kpc] km/s (standard proper-motion relation).
KMS_PER_MASYR = 4.740470 * CLUSTER_DISTANCE_PC / 1000.0

#: The 0.3 mas/yr proper-motion quality cut, in km/s.
PM_QUALITY_CUT_KMS = 0.30 * KMS_PER_MASYR


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
    cell_per_sigma : float
        Target cell width in units of ``sigma_ref``. **Set by measurement:**
        sweeping cell_per_sigma ∈ {0.78, 0.64, 0.54, 0.47, 0.41, 0.37} under
        the ``gaussian_core`` prior at N=400, cell_per_sigma=0.47 (→K=15, 225 cells, 1.8
        stars/cell) was the effective limit for the HST regimes. Finer cells
        (K=17) buys marginal improvement at +28% compute; K=19 (1.1 stars/cell)
        breaks on anisotropic truths. The GMRF-era numbers (+2.34 bias at K=9)
        are obsolete: the null-space fix removed them, so this was re-chosen
        post-gaussian_core as the spec required.
    """

    name: str
    sigma_ref: float
    err_median: float
    err_cut: float
    n_stars: int
    n_sigma_grid: float = 3.5
    cell_per_sigma: float = 0.47

    @property
    def grid_width(self):
        """Total width of the (square) velocity grid, km/s."""
        return 2.0 * self.n_sigma_grid * self.sigma_ref

    @property
    def n_bins(self):
        """Cells per axis. Always odd: DYNAMITE's ProperMotions reader
        (``set_default_hist_bins``) raises ValueError on even counts."""
        n = int(round(self.grid_width / (self.cell_per_sigma * self.sigma_ref)))
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

        if len(per_bin_sigma) < 2:
            msg = f"at least 2 bins with >= {min_stars} stars are required, got {len(per_bin_sigma)}"
            raise ValueError(msg)

        return cls(
            name=name,
            sigma_ref=float(np.median(per_bin_sigma)),
            err_median=float(np.median(per_bin_err_med)),
            err_cut=float(err_cut),
            n_stars=int(round(float(np.median(per_bin_n)))),
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
    """Moments of the TRUE per-cell probability mass, taken at cell centres.

    This is the fair comparison and it is also exactly what DYNAMITE
    chi-squares. Comparing a cell-centre moment against a continuous truth
    charges the model for grid discretisation: binning inflates a variance by
    ~h^2/12 (Sheppard), which at cell_per_sigma=0.78 is +0.42 km/s on
    sigma=17 -- over half the posterior interval at N=250, enough on its own
    to destroy coverage.

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
    cx, cy = centers_2d[:, 0], centers_2d[:, 1]
    mx, my = mass @ cx, mass @ cy
    vx = mass @ (cx - mx) ** 2
    vy = mass @ (cy - my) ** 2
    cxy = mass @ ((cx - mx) * (cy - my))
    sx, sy = np.sqrt(vx), np.sqrt(vy)
    return (dict(mean_x=mx, mean_y=my, sigma_x=sx, sigma_y=sy,
                 rho=cxy / (sx * sy)), mass)


def _moments_from_pdf_samples_2d(pdf_samples, centers_2d):
    """Per-sample mean_x, mean_y, sigma_x, sigma_y, rho from 2D pdf draws."""
    pdf_samples = np.asarray(pdf_samples, dtype=float)
    cx = centers_2d[:, 0]
    cy = centers_2d[:, 1]

    mean_x = pdf_samples @ cx
    mean_y = pdf_samples @ cy
    dx = cx[None, :] - mean_x[:, None]
    dy = cy[None, :] - mean_y[:, None]

    var_x = np.einsum("ij,ij->i", pdf_samples, dx**2)
    var_y = np.einsum("ij,ij->i", pdf_samples, dy**2)
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
            lines.append("    n_stars  cover  CI/CR  bias")
            for r in sorted([x for x in self.rows if x["metric"] == metric], key=lambda x: x["n_stars"]):
                ratio = r["ci_width"] / r["cr_bound"] if r["cr_bound"] > 0 else float("nan")
                lines.append(
                    f"    {r['n_stars']:<8.4g} {r['coverage']:5.2f}  {ratio:5.2f}  {r['bias']:+.3f}"
                )
        return "\n".join(lines)

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


def recovery_curve_2d(
    profile,
    truth_name,
    n_stars_values,
    n_real=25,
    prior="gaussian_core",
    num_warmup=300,
    num_samples=600,
    seed=20260805,
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

    Returns
    -------
    RecoveryCurve2D
    """
    from veldist.veldist2d import KinematicSolver2D

    truth = truths_for(profile.sigma_ref)[truth_name]
    rows = []
    metrics = ["mean_x", "mean_y", "sigma_x", "sigma_y", "rho"]

    grid_center = (0.0, 0.0)
    grid_width = (profile.grid_width, profile.grid_width)
    n_bins = profile.n_bins

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

        for i in range(n_real):
            obs_x, obs_y, cov = _draw_stars(rng, truth, n_stars, profile)

            solver = KinematicSolver2D()
            solver.setup_grid(center=grid_center, width=grid_width, n_bins=n_bins)
            solver.add_data(obs_x, obs_y, cov)
            samples = solver.run(
                num_warmup=num_warmup, num_samples=num_samples, seed=seed + i, prior=prior
            )
            pdf_samples = np.asarray(samples["intrinsic_pdf"])
            mean_x, mean_y, sigma_x, sigma_y, rho = _moments_from_pdf_samples_2d(pdf_samples, centers_2d)
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
                }
            )

    return RecoveryCurve2D(profile=profile, truth_name=truth_name, rows=rows, n_real=n_real)


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

PROFILES_2D = {p.name: p for p in (HST_BRIGHT, HST_FAINT, GAIA_OUTER)}
