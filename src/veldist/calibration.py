"""Calibrating veldist to a dataset.

The methodology has to be re-tuned whenever the observational regime changes:
the velocity grid, the regularisation strength, and the achievable precision
all depend on how many stars there are per spatial bin, how precise their
velocities are, and how broad the LOSVD is. This module makes those
assumptions declarative and checkable rather than scattered constants.

Typical use::

    from veldist.calibration import OMEGACAT, make_truths, calibrate

    print(OMEGACAT.report())          # is the grid sane for this dataset?
    result = calibrate(OMEGACAT, make_truths(OMEGACAT.sigma_ref), n_real=25)
    print(result.summary())
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from collections.abc import Callable

import numpy as np
from scipy import stats
from scipy.stats import binom

from veldist.analysis import half_68ci

__all__ = [
    "ObservingProfile",
    "OMEGACAT",
    "recommend_grid",
    "recommend_cuts",
    "Truth",
    "make_truths",
    "true_moments",
    "METRICS",
    "CalibrationResult",
    "calibrate",
    "PROXY_TO_GH",
    "measure_proxy_to_gh",
    "RecoveryCurve",
    "recovery_curve",
    "coverage_floor",
]

# Gauss-Hermite conversions, lowest order (van der Marel & Franx 1993).
# Only valid for weak non-Gaussianity; Sanders & Evans (2020) note the h4
# relation is good to 10% only for |h4| < 0.01, so treat these as indicative
# amplitude conversions, not as a route to DYNAMITE inputs.
SKEW_PER_H3 = 4.0 * np.sqrt(3.0)
EXKURT_PER_H4 = 8.0 * np.sqrt(6.0)

#: Measured mapping from the robust proxies in ``compute_percentile_summary``
#: to Gauss-Hermite coefficients, over ``make_truths()`` at OMEGACAT's grid,
#: restricted to the amplitude envelope ``measure_proxy_to_gh``'s defaults
#: describe (|h3| <= 0.15, |h4| <= 0.10). Each entry is the dict
#: ``measure_proxy_to_gh`` returns: ``slope``, ``median_ratio``,
#: ``ratio_std``, ``n_truths``, ``outliers``. ``median_ratio`` is the number
#: to apply in practice: it is robust to ``cold_disk_component``, whose
#: proxy and GH coefficient have opposite sign and which both mappings flag
#: as an ``outliers`` entry. Regenerate with ``measure_proxy_to_gh`` if the
#: truth library or the grid changes. Do not apply this conversion outside
#: the envelope it was measured over. The two mappings are not equally
#: trustworthy: ``kurtosis_pct_to_h4`` rests on 5 ratio-eligible truths and
#: its flagged outlier, ``cold_disk_component``, has an unambiguous sign
#: flip. ``skew_pct_to_h3`` rests on only 3 ratio-eligible truths, one of
#: which is that same flagged outlier, and a MAD outlier test on 3 points
#: has very little power, so the h3 ``median_ratio`` and its ``outliers``
#: entry should be read with substantially less confidence than the h4
#: ones.
PROXY_TO_GH = {
    "skew_pct_to_h3": {
        "slope": 1.1913269380278009,
        "median_ratio": 1.17612262772906,
        "ratio_std": 0.20728990309116752,
        "n_truths": 3,
        "outliers": ["cold_disk_component"],
    },
    "kurtosis_pct_to_h4": {
        "slope": 0.625306989336929,
        "median_ratio": 0.6330000772983341,
        "ratio_std": 1.628837233074273,
        "n_truths": 5,
        "outliers": ["cold_disk_component"],
    },
}


@dataclass(frozen=True)
class ObservingProfile:
    """Everything about a dataset that the methodology must be tuned to.

    The velocity grid is *derived* from these rather than chosen. Two ratios
    govern whether it is sane, and both were wrong in this repo before being
    made explicit:

    - ``bin_width / median_error``: the grid cannot resolve structure finer
      than the measurement error, because the errors convolve it away. Much
      below 1 wastes latent dimensions; much above 2-3 discards real
      resolution.
    - ``informative_fraction``: bins with no mass are pure prior-driven
      dimensions with no data to anchor them. They are what post-hoc tail
      truncation exists to suppress.
    """

    name: str
    n_stars: int  # stars per spatial bin (the science target)
    err_median: float  # median measurement error, km/s
    err_log_sigma: float  # log-normal width of the error distribution
    sigma_max: float  # largest LOSVD dispersion in the field
    sigma_min: float  # smallest; sets the worst-case informative fraction
    rotation_span: float  # full spread of mean velocity across spatial bins
    n_sigma_grid: float = 4.0  # grid half-width, in sigma_max
    bins_per_error: float = 2.0  # bin width, in median measurement error

    @property
    def median_error(self) -> float:
        return self.err_median

    def draw_errors(self, n, rng):
        """Per-star measurement errors. Log-normal rather than uniform: real
        errors are magnitude-dependent with a tail, and Sanders & Evans (2020)
        find the *floor* matters more than the spread for kurtosis sign
        determination."""
        return np.exp(rng.normal(np.log(self.err_median), self.err_log_sigma, size=n))

    @staticmethod
    def ivar(sigma, err):
        """Total Fisher information on the mean velocity, for these stars.

        Deliberately ``1/(sigma^2 + err_i^2)`` and not ``1/err_i^2``: a star's
        velocity is informative about the LOSVD centroid only up to the
        intrinsic spread it is drawn from, so the relevant variance is that of
        the *observed* velocity. Using ``1/err_i^2`` would claim unbounded
        information from perfectly measured stars, which is wrong, and would
        make the resulting bins far too small.

        The reciprocal square root of this quantity is the Cramer-Rao bound on
        ``v_mean``, which is what makes it the natural target for spatial
        binning: ``ivar = 1`` means ``v_mean`` good to 1 km/s.
        """
        err = np.asarray(err, dtype=float)
        return float(np.sum(1.0 / (sigma**2 + err**2)))

    def draw_sample(self, target_ivar, sigma, rng):
        """Draw per-star errors until the bin reaches *target_ivar*.

        The recovery curve varies information content while holding the error
        distribution fixed, so the star count is an output here, not an input.
        That inversion is the point: it is what lets the sweep report a
        threshold in units that transfer between datasets with different
        error properties.

        Parameters
        ----------
        target_ivar : float
            Required total ``ivar``. Must be positive.
        sigma : float
            LOSVD dispersion, km/s.
        rng : numpy.random.Generator

        Returns
        -------
        ndarray
            Per-star measurement errors, km/s. The smallest number of stars
            whose total ``ivar`` reaches the target, so the total slightly
            overshoots by at most one star's contribution.

        Raises
        ------
        ValueError
            If *target_ivar* is not positive.
        """
        if target_ivar <= 0:
            msg = "target_ivar must be positive"
            raise ValueError(msg)

        # Each star contributes at most 1/sigma^2 (in the zero-error limit),
        # so this many stars is a guaranteed lower bound on what is needed.
        chunk = max(16, int(np.ceil(target_ivar * sigma**2)))
        err = np.empty(0)
        while True:
            err = np.concatenate([err, self.draw_errors(chunk, rng)])
            contrib = 1.0 / (sigma**2 + err**2)
            reached = np.searchsorted(np.cumsum(contrib), target_ivar)
            if reached < len(err):
                return err[: reached + 1]

    @property
    def sigma_ref(self) -> float:
        """Reference dispersion for scaling mock truths."""
        return self.sigma_max

    @property
    def grid_width(self) -> float:
        """Shared velocity grid width. DYNAMITE requires one grid for all
        spatial bins, so it must hold the widest LOSVD plus the rotation
        offset of the bins furthest from the systemic velocity."""
        return 2.0 * self.n_sigma_grid * self.sigma_max + self.rotation_span

    @property
    def bin_width(self) -> float:
        return self.bins_per_error * self.median_error

    @property
    def n_bins(self) -> int:
        return int(round(self.grid_width / self.bin_width))

    @property
    def err_over_sigma(self) -> tuple:
        """Deconvolution difficulty. Amorisco & Evans (2012) show the
        attenuation of non-Gaussian signal depends on this ratio alone."""
        return (self.median_error / self.sigma_max, self.median_error / self.sigma_min)

    def informative_fraction(self, sigma: float) -> float:
        """Fraction of grid bins carrying mass for a LOSVD of this width."""
        return min(1.0, 2.0 * self.n_sigma_grid * sigma / self.grid_width)

    def moment_precision(self) -> dict:
        """Best achievable per-bin precision, as sanity limits on any claim.

        mean and sigma are the Gaussian Cramer-Rao bounds; h3/h4 use the
        (2N)^-1/2 approximation Sanders & Evans (2020) recommend for small
        samples.
        """
        n = self.n_stars
        return {
            "v_mean": self.sigma_max / np.sqrt(n),
            "sigma": self.sigma_max / np.sqrt(2 * n),
            "h3": 1.0 / np.sqrt(2 * n),
            "h4": 1.0 / np.sqrt(2 * n),
        }

    def matched_grid(self, sigma):
        """Grid width and bin count matched to a single dispersion.

        DYNAMITE requires one shared grid across all spatial bins, but that
        constraint is on the *output*. Nothing stops veldist fitting each bin
        on a grid matched to its own dispersion and aggregating the posterior
        samples onto the shared output grid afterwards. If the fitted grid is
        at least as fine as the output grid and their edges align, that
        aggregation is exact, being just a sum of mass within output bins
        taken per sample, so uncertainties propagate correctly.
        """
        width = 2.0 * self.n_sigma_grid * sigma
        return width, int(round(width / self.bin_width))

    def report(self) -> str:
        p = self.moment_precision()
        lo, hi = self.err_over_sigma
        lines = [
            f"ObservingProfile: {self.name}",
            f"  {self.n_stars} stars/bin, log-normal errors " f"(median {self.err_median:.1f}, s={self.err_log_sigma})",
            f"  LOSVD sigma {self.sigma_min}-{self.sigma_max} km/s, " f"rotation span {self.rotation_span} km/s",
            f"  grid: {self.grid_width:.0f} km/s / {self.n_bins} bins "
            f"= {self.bin_width:.1f} km/s "
            f"({self.bins_per_error:.1f}x median error)",
            f"  informative bins: {self.informative_fraction(self.sigma_max):.0%} "
            f"at sigma_max, {self.informative_fraction(self.sigma_min):.0%} at sigma_min",
            f"  err/sigma: {lo:.2f} (widest) to {hi:.2f} (narrowest)",
            "  achievable per-bin precision:",
            f"    v_mean {p['v_mean']:.2f} km/s   sigma {p['sigma']:.2f} km/s",
            f"    h3 {p['h3']:.3f}          h4 {p['h4']:.3f}",
        ]
        return "\n".join(lines)

    @classmethod
    def from_data(cls, vel, err, bin_ids, name="measured", min_stars=10):
        """Measure a profile from a real catalogue.

        Every parameter of this class was originally hand-typed (see
        ``OMEGACAT``), which meant the mock suite validated the method against
        a guess about the data rather than the data. This measures them
        instead. Only scalars come out, so the result is safe to commit as a
        test fixture even when the catalogue itself is not redistributable.

        Per-bin dispersions come from ``gaussian_mle`` rather than
        ``numpy.std``: the latter returns ``sqrt(sigma^2 + err^2)``, which
        would inflate ``sigma_min`` most in exactly the low-dispersion bins
        that set the hardest deconvolution regime.

        Parameters
        ----------
        vel, err : array-like, shape (n_stars,)
            Velocities and per-star uncertainties for the whole field, km/s.
        bin_ids : array-like, shape (n_stars,)
            Spatial bin index for each star. Values need not be contiguous.
        name : str
            Label carried into the returned profile.
        min_stars : int
            Bins with fewer stars are excluded from the dispersion and
            rotation estimates.

        Returns
        -------
        ObservingProfile

        Raises
        ------
        ValueError
            If fewer than 2 bins survive the *min_stars* cut.
        """
        from veldist.baseline import gaussian_mle

        vel = np.asarray(vel, dtype=float)
        err = np.asarray(err, dtype=float)
        bin_ids = np.asarray(bin_ids)

        sigmas, means, counts = [], [], []
        for b in np.unique(bin_ids):
            sel = bin_ids == b
            if int(np.sum(sel)) < min_stars:
                continue
            fit = gaussian_mle(vel[sel], err[sel])
            sigmas.append(fit["sigma"])
            means.append(fit["v_mean"])
            counts.append(int(np.sum(sel)))

        if len(sigmas) < 2:
            msg = f"at least 2 bins with >= {min_stars} stars are required, got {len(sigmas)}"
            raise ValueError(msg)

        sigmas = np.asarray(sigmas)
        means = np.asarray(means)

        log_err = np.log(err)
        # Percentile-based sigma_min/max rather than the extremes: one badly
        # fit bin should not set the grid width for the entire campaign.
        return cls(
            name=name,
            n_stars=int(round(float(np.median(counts)))),
            err_median=float(np.median(err)),
            err_log_sigma=float(np.std(log_err)),
            sigma_max=float(np.percentile(sigmas, 95)),
            sigma_min=float(np.percentile(sigmas, 5)),
            rotation_span=float(np.ptp(means)),
        )

    def to_json(self, path):
        """Write this profile to an indented JSON file."""
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path):
        """Read a profile written by :meth:`to_json`."""
        return cls(**json.loads(Path(path).read_text()))


#: The oMEGACat dataset: HST+MUSE within omega Cen's half-light radius.
#: 30k LOS spectra after quality cuts over r_h ~ 4.65', giving ~200 Voronoi
#: bins at 150 stars each. Dispersions from van de Ven et al. (2006).
OMEGACAT = ObservingProfile(
    name="oMEGACat",
    n_stars=150,
    err_median=2.5,
    err_log_sigma=0.4,
    sigma_max=22.0,
    sigma_min=7.0,
    rotation_span=10.0,
)


def recommend_grid(profile: ObservingProfile, v_systemic: float = 0.0) -> dict:
    """``KinematicSolver.setup_grid`` / ``fit_all_bins(grid_kwargs=...)`` from
    a measured :class:`ObservingProfile`, instead of hand-picking ``n_bins``.

    Just exposes ``profile.grid_width``/``n_bins`` (already derived from the
    error and dispersion scale, see the class docstring) in the shape those
    callers expect.

    Parameters
    ----------
    profile : ObservingProfile
        Typically ``ObservingProfile.from_data(...)`` on the real catalogue.
    v_systemic : float
        Grid centre, km/s. Default 0.0 (velocities already systemic-subtracted).
    """
    return {"center": v_systemic, "width": profile.grid_width, "n_bins": profile.n_bins}


def recommend_cuts(profile: ObservingProfile, recovery=None, metric="sigma", min_stars=10, **threshold_kwargs) -> dict:
    """``fit_all_bins(min_stars=, min_ivar=, sigma_ref=)`` from a measured
    :class:`ObservingProfile`.

    ``min_ivar`` cannot be derived from the profile alone: it is a coverage
    threshold, which requires actually running recovery sims (see
    ``RecoveryCurve.threshold``, the mechanism ``fit_all_bins``'s own
    ``min_ivar`` docstring points to). That sweep is expensive (~hours), so
    it is opt-in here rather than run implicitly.

    Parameters
    ----------
    profile : ObservingProfile
    recovery : RecoveryCurve, optional
        Output of ``recovery_curve(profile, ..., sigma=profile.sigma_min)`` --
        must be built at ``sigma_min``, the hardest case in the field, so the
        threshold is conservative rather than optimistic; a curve built at a
        higher dispersion would pass bins that actually fail at low
        dispersion. Checked: raises if ``recovery.sigma != profile.sigma_min``.
        If omitted, the returned ``min_ivar`` is ``None`` and only the
        ``min_stars`` cut applies -- the same default ``fit_all_bins`` already
        uses on its own.
    metric : str
        Passed to ``recovery.threshold``. Default ``"sigma"``, since ``v_mean``
        is not the binding constraint for a dispersion-map data product.
    min_stars : int
        Floor applied regardless of ``recovery`` (avoids degenerate fits
        below this even where ``min_ivar`` alone would pass). Default 10,
        matching ``fit_all_bins``'s own default.
    **threshold_kwargs
        Forwarded to ``recovery.threshold`` (``min_coverage``, ``max_ci_ratio``,
        ``band``).

    Returns
    -------
    dict
        ``min_stars``, ``min_ivar`` (``None`` if ``recovery`` was not given),
        ``sigma_ref`` -- unpack straight into ``fit_all_bins(**recommend_cuts(...))``.

    Raises
    ------
    ValueError
        If ``recovery`` is given but was built at a ``sigma`` other than
        ``profile.sigma_min``.
    """
    min_ivar = None
    if recovery is not None:
        if recovery.sigma != profile.sigma_min:
            msg = (
                f"recovery was built at sigma={recovery.sigma}, but recommend_cuts "
                f"requires sigma_min={profile.sigma_min} (the hardest case in the "
                "field) so the threshold is conservative rather than optimistic."
            )
            raise ValueError(msg)
        min_ivar = recovery.threshold(metric, **threshold_kwargs)
    return {"min_stars": min_stars, "min_ivar": min_ivar, "sigma_ref": profile.sigma_ref}


@dataclass
class Truth:
    """A mock LOSVD shape, defined at unit dispersion and scaled on demand.

    Shapes are specified in dimensionless form so the same library rescales
    to any dataset. ``scaled(sigma)`` returns (pdf, rvs) with zero mean and
    the requested dispersion.
    """

    name: str
    note: str
    _pdf: Callable  # unit-ish pdf, arbitrary location/scale
    _rvs: Callable  # matching sampler
    _cache: dict = field(default_factory=dict, repr=False)

    def _standardise(self):
        if "loc" not in self._cache:
            v = np.linspace(-500, 500, 400001)
            p = np.asarray(self._pdf(v), dtype=float)
            p = p / np.trapezoid(p, v)
            mu = float(np.trapezoid(v * p, v))
            sd = float(np.sqrt(np.trapezoid((v - mu) ** 2 * p, v)))
            self._cache["loc"], self._cache["scale"] = mu, sd
        return self._cache["loc"], self._cache["scale"]

    def scaled(self, sigma: float):
        mu, sd = self._standardise()
        k = sigma / sd

        def pdf(x, _mu=mu, _k=k):
            return np.asarray(self._pdf(np.asarray(x) / _k + _mu)) / _k

        def rvs(n, rng, _mu=mu, _k=k):
            return (self._rvs(n, rng) - _mu) * _k

        return pdf, rvs


def _uniform_gauss(a, s):
    """Uniform(-a, a) convolved with a Gaussian, the Sanders & Evans (2020)
    negative-excess-kurtosis kernel. Excess kurtosis is -1.2 r^2 where r is the
    fraction of variance carried by the uniform part; -1.2 is the hard floor."""

    def pdf(x):
        x = np.asarray(x, dtype=float)
        return (stats.norm.cdf((x + a) / s) - stats.norm.cdf((x - a) / s)) / (2 * a)

    def rvs(n, rng):
        return rng.uniform(-a, a, size=n) + rng.normal(0.0, s, size=n)

    return pdf, rvs


def _split_uniform_gauss(a1, a2, s):
    """Two-piece uniform kernel, equal weight but different widths either side
    of zero, convolved with a Gaussian: the SE20 skewness option. Note a
    *shifted* uniform is still symmetric about its own midpoint and gives no
    skewness at all; the widths must differ."""

    def pdf(x):
        x = np.asarray(x, dtype=float)
        left = (stats.norm.cdf((x + a1) / s) - stats.norm.cdf(x / s)) / a1
        right = (stats.norm.cdf(x / s) - stats.norm.cdf((x - a2) / s)) / a2
        return 0.5 * (left + right)

    def rvs(n, rng):
        left = rng.random(n) < 0.5
        d = rng.uniform(0.0, a2, size=n)
        d[left] = rng.uniform(-a1, 0.0, size=left.sum())
        return d + rng.normal(0.0, s, size=n)

    return pdf, rvs


def _mixture(locs, scales, weights):
    def pdf(x):
        out = 0.0
        for lo, sc, w in zip(locs, scales, weights):
            out = out + w * stats.norm(loc=lo, scale=sc).pdf(x)
        return out

    def rvs(n, rng):
        comp = rng.choice(len(locs), size=n, p=list(weights))
        return rng.normal(np.array(locs)[comp], np.array(scales)[comp])

    return pdf, rvs


def make_truths():
    """The mock LOSVD shapes, dimensionless. Scale with ``Truth.scaled(sigma)``.

    Chosen to span the non-Gaussianity expected in a rotating,
    anisotropic globular cluster, at realistic h3/h4 amplitude (``|h3|`` <~ 0.15,
    ``|h4|`` <~ 0.05-0.1). Physical motivation for each is in its ``note``.
    """
    t = []
    t.append(
        Truth(
            "gaussian",
            "isotropic, no rotation: the null case",
            stats.norm(0, 1).pdf,
            lambda n, rng: rng.normal(0, 1, size=n),
        )
    )
    t.append(
        Truth(
            "student_t_h4",
            "radial anisotropy, h4 > 0; excess kurtosis 1.0",
            stats.t(df=10).pdf,
            lambda n, rng: stats.t(df=10).rvs(size=n, random_state=rng),
        )
    )
    t.append(
        Truth(
            "mild_radial_h4",
            "weak radial anisotropy, the inner-region case",
            stats.t(df=19).pdf,
            lambda n, rng: stats.t(df=19).rvs(size=n, random_state=rng),
        )
    )
    t.append(
        Truth(
            "skew_normal_h3",
            "rotation, h3 != 0",
            stats.skewnorm(a=2).pdf,
            lambda n, rng: stats.skewnorm(a=2).rvs(size=n, random_state=rng),
        )
    )
    t.append(Truth("flat_top_tangential", "tangential anisotropy, h4 < 0", *_uniform_gauss(28.1, 5.0)))
    t.append(Truth("rotating_tangential", "rotation AND tangential anisotropy", *_split_uniform_gauss(34.0, 13.0, 6.0)))
    t.append(
        Truth(
            "cold_disk_component",
            "van de Ven+06 disk-like component, 4% of mass, kinematically cold",
            *_mixture((0.0, 26.0), (17.0, 5.0), (0.96, 0.04)),
        )
    )
    t.append(
        Truth(
            "two_population",
            "Norris+97 metal-poor hot/rotating + metal-rich cool/static",
            *_mixture((4.0, -2.0), (19.0, 12.0), (0.65, 0.35)),
        )
    )
    t.append(
        Truth(
            "bimodal_counter_rotation",
            "counter-rotating populations",
            *_mixture((-18.0, 18.0), (10.0, 10.0), (0.5, 0.5)),
        )
    )
    return t


METRICS = ["v_mean", "sigma", "skewness", "kurtosis", "tail_weight"]
NOMINAL_BAND = (0.440, 0.920)  # binom(25, 0.68) 99% band
CATASTROPHIC = 0.30


def coverage_floor(n_real, band=0.99, nominal=0.68):
    """Lower edge of a binomial coverage band, as a fraction of ``n_real``.

    Empirical coverage from ``n_real`` mock realisations is a binomial
    proportion, not a free-floating number: its acceptable range depends on
    ``n_real``. This is the same convention as :data:`NOMINAL_BAND` (the
    ``binom(25, 0.68)`` 99% band), generalised to any ``n_real`` instead of
    being hardcoded for 25. ``coverage_floor(25)`` reproduces
    ``NOMINAL_BAND[0]`` exactly.

    Parameters
    ----------
    n_real : int
        Number of mock realisations the coverage fraction was estimated
        from.
    band : float
        Confidence level of the two-sided binomial interval, e.g. 0.99 for
        a 99% band.
    nominal : float
        Nominal coverage of the credible interval being checked, e.g. 0.68
        for a 68% interval.

    Returns
    -------
    float
        The lower edge of the band, as a fraction of ``n_real``.
    """
    return float(binom.ppf((1 - band) / 2, n_real, nominal)) / n_real


def true_moments(pdf, lo=-500.0, hi=500.0, n_grid=400001):
    """Moments of a truth, on a dense grid.

    Deliberately not ``scipy.integrate.quad``: its adaptive subdivision
    silently under-samples a narrow component sitting on a broad base. For the
    cold-disk truth quad returned skewness +0.0000 against a true -0.0320, and
    a silently wrong truth invalidates a coverage test rather than failing it.
    """
    v = np.linspace(lo, hi, n_grid)
    p = np.asarray(pdf(v), dtype=float)
    p = p / np.trapezoid(p, v)
    mean = float(np.trapezoid(v * p, v))
    var = float(np.trapezoid((v - mean) ** 2 * p, v))
    sd = np.sqrt(var)
    inside = (v >= mean - sd) & (v <= mean + sd)
    return {
        "v_mean": mean,
        "sigma": sd,
        "skewness": float(np.trapezoid(((v - mean) / sd) ** 3 * p, v)),
        "kurtosis": float(np.trapezoid(((v - mean) / sd) ** 4 * p, v)) - 3.0,
        "tail_weight": 1.0 - float(np.trapezoid(p[inside], v[inside])),
    }


@dataclass
class CalibrationResult:
    """Coverage and efficiency for one (profile, sigma, model) combination."""

    profile: ObservingProfile
    sigma: float
    coverage: dict  # {truth: {metric: fraction}}
    medians: dict  # {truth: {metric: [per-realisation posterior medians]}}
    truth_values: dict

    @staticmethod
    def _robust_scatter(x):
        return half_68ci(x)

    def efficiency(self):
        """Actual estimator scatter divided by the statistical optimum.

        ~1 means the estimator extracts what the data contain. >1 means
        information is being lost. **<1 is not better than optimal**: it
        means the prior is shrinking estimates, and must be read alongside the
        bias.

        This is the check coverage cannot do. A posterior can reach nominal
        coverage by reporting large error bars on a poor estimator; efficiency
        is what distinguishes that from a good estimator with honest ones.

        Uses a robust scatter, because with a few dozen realisations a single
        failed fit dominates a standard deviation.
        """
        n = self.profile.n_stars
        g = self.medians["gaussian"]
        return {
            "v_mean": self._robust_scatter(g["v_mean"]) / (self.sigma / np.sqrt(n)),
            "sigma": self._robust_scatter(g["sigma"]) / (self.sigma / np.sqrt(2 * n)),
        }

    def score(self):
        flat = [c for d in self.coverage.values() for c in d.values()]
        return {
            "in_band": sum(NOMINAL_BAND[0] <= c <= NOMINAL_BAND[1] for c in flat),
            "n_entries": len(flat),
            "catastrophic": sum(c < CATASTROPHIC for c in flat),
        }

    def summary(self):
        s, e = self.score(), self.efficiency()
        out = [
            f"{self.profile.name} @ sigma={self.sigma:.0f} km/s, "
            f"N={self.profile.n_stars}, {self.profile.n_bins} bins",
            f"  in-band {s['in_band']}/{s['n_entries']}, " f"catastrophic {s['catastrophic']}",
            f"  efficiency: v_mean {e['v_mean']:.2f}x, sigma {e['sigma']:.2f}x",
        ]
        for name, d in self.coverage.items():
            out.append(f"  {name:<26} " + " ".join(f"{m}={d[m]:.2f}" for m in METRICS))
        return "\n".join(out)


def calibrate(
    profile,
    truths,
    sigma=None,
    *,
    n_real=25,
    prior="gaussian_core",
    n_bins=None,
    seed=20260803,
    num_warmup=300,
    num_samples=600,
    n_sigma_truncate=None,
):
    """Fit mock realisations of each truth; measure coverage and efficiency.

    ``sigma`` defaults to the profile's widest LOSVD. **Run it at
    ``profile.sigma_min`` as well.** Across omega Cen's 7-22 km/s range
    err/sigma goes from 0.11 to 0.36 and the informative bin fraction from 95%
    to 30%, so a regularisation tuned at one end is not necessarily calibrated
    at the other, since the narrow bins sit in the dwarf-spheroidal difficulty
    regime (Amorisco & Evans 2012 quote Sculptor at 0.33).
    """
    from veldist.veldist import KinematicSolver
    from veldist.analysis import compute_summary

    sigma = profile.sigma_max if sigma is None else sigma
    n_bins = profile.n_bins if n_bins is None else n_bins
    coverage, medians, tvals = {}, {}, {}

    for t in truths:
        pdf, rvs = t.scaled(sigma)
        tv = true_moments(pdf)
        tvals[t.name] = tv
        hits = {m: 0 for m in METRICS}
        meds = {m: [] for m in METRICS}
        rng = np.random.default_rng(seed)
        for i in range(n_real):
            true_v = rvs(profile.n_stars, rng)
            err = profile.draw_errors(profile.n_stars, rng)
            obs = true_v + rng.normal(0.0, err)
            solver = KinematicSolver()
            solver.setup_grid(center=0.0, width=profile.grid_width, n_bins=n_bins)
            solver.add_data(obs, err)
            solver.run(num_warmup=num_warmup, num_samples=num_samples, seed=seed + i, prior=prior)
            summ = compute_summary(
                solver.samples["intrinsic_pdf"], solver.grid["centers"], n_sigma_truncate=n_sigma_truncate
            )
            for m in METRICS:
                med, h68 = summ[m]
                meds[m].append(med)
                if abs(med - tv[m]) <= h68:
                    hits[m] += 1
        coverage[t.name] = {m: hits[m] / n_real for m in METRICS}
        medians[t.name] = {m: list(meds[m]) for m in METRICS}

    return CalibrationResult(profile, sigma, coverage, medians, tvals)


#: Metrics tracked by the recovery curve. ``v_mean`` and ``sigma`` are the
#: gating pair per ``TASKS.md``; the shape metrics are reported for interest
#: and are expected to need far more information before they calibrate.
RECOVERY_METRICS = ["v_mean", "sigma", "skewness", "kurtosis"]


@dataclass
class RecoveryCurve:
    """How well each statistic is recovered as a function of information content.

    The question this answers is the one behind both ``min_stars=10`` and any
    spatial binning target: how much information does a bin need before the
    posterior can be believed? Answering it empirically, in units of Fisher
    information rather than star count, gives a threshold that transfers to
    datasets with different measurement errors.

    Notes
    -----
    The ``cr_bound`` column is exact for ``v_mean`` by construction (the
    swept ``ivar`` IS the Fisher information of the mean). For ``sigma``,
    ``skewness`` and ``kurtosis`` it uses an equal-error Gaussian
    approximation built from the same effective sample size, which is exact
    only for homogeneous per-star errors. With the heterogeneous errors this
    package actually fits, that approximation is indicative rather than
    exact, so treat the CI/CR ratio for those three metrics as a rough
    efficiency check, not a precise one.
    """

    profile: object
    sigma: float
    rows: list
    n_real: int = None

    def threshold(self, metric, min_coverage=None, max_ci_ratio=1.5, band=0.99):
        """Smallest ``ivar`` at which *metric* is trustworthy, or ``None``.

        Trustworthy means two things at once, because either alone is
        gameable: coverage at least the floor (an interval that contains
        the truth often enough) **and** a credible interval no wider than
        *max_ci_ratio* times the Cramer-Rao bound (an interval that is not
        merely wide enough to contain everything).

        A point qualifies only if every higher-``ivar`` point also qualifies,
        so a single lucky low-information point cannot set the threshold.

        Parameters
        ----------
        metric : str
            One of :data:`RECOVERY_METRICS`.
        min_coverage : float, optional
            Required empirical coverage of the nominal 68% interval. Coverage
            is a binomial proportion estimated from ``n_real`` mock draws, so
            a fixed floor is only correct at the ``n_real`` it was tuned for:
            at ``n_real=40`` the standard error is ``sqrt(0.68*0.32/40) =
            0.074``, and a fixed 0.60 sits one standard error below nominal,
            rejecting perfectly calibrated methods roughly one sweep cell in
            ten. This module already treats coverage this way in
            :data:`NOMINAL_BAND`, the ``binom(25, 0.68)`` 99% band used by
            :meth:`CalibrationResult.score`; this parameter generalises that
            convention to whatever ``n_real`` the curve was actually built
            with instead of hardcoding it for 25.

            Resolution order: if given explicitly, used as-is (this keeps
            every existing caller and test working unchanged). Otherwise, if
            ``self.n_real`` is set, the floor is
            :func:`coverage_floor` ``(self.n_real, band)``. Otherwise it falls
            back to the historical constant 0.60.
        max_ci_ratio : float
            Required efficiency, as a multiple of the Cramer-Rao bound.
        band : float
            Confidence level of the binomial coverage band used to derive
            the floor when ``min_coverage`` is not given and ``self.n_real``
            is set. Ignored otherwise.

        Returns
        -------
        float or None
            ``None`` if no swept ``ivar`` value qualifies, which means the
            sweep did not reach high enough information content.

        Raises
        ------
        ValueError
            If *metric* appears in no row.
        """
        sel = [r for r in self.rows if r["metric"] == metric]
        if not sel:
            msg = f"no rows for metric {metric!r}"
            raise ValueError(msg)

        floor, _floor_desc = self._resolve_coverage_floor(min_coverage, band)

        by_ivar = {}
        for r in sel:
            by_ivar.setdefault(r["ivar"], []).append(r)

        def ok(rows):
            return all(r["coverage"] >= floor and r["ci_width"] <= max_ci_ratio * r["cr_bound"] for r in rows)

        ivars = sorted(by_ivar)
        # Walk down from the top while the run of passes stays unbroken. The
        # answer is None unless that walk actually advances past the top
        # point, so a failing top ivar with a lower, non-adjacent pass never
        # gets returned.
        best = None
        for iv in reversed(ivars):
            if not ok(by_ivar[iv]):
                break
            best = iv
        return best

    def _resolve_coverage_floor(self, min_coverage, band):
        """Return ``(floor, description)`` per the resolution order in ``threshold``."""
        if min_coverage is not None:
            return min_coverage, "explicit"
        if self.n_real is not None:
            floor = coverage_floor(self.n_real, band=band)
            return floor, f"{band:.0%} binomial band at n_real={self.n_real}"
        return 0.60, "historical default, n_real unknown"

    def report(self, min_coverage=None, max_ci_ratio=1.5, band=0.99):
        """Human-readable table, one block per metric.

        Parameters
        ----------
        min_coverage, max_ci_ratio, band
            Passed straight through to :meth:`threshold`, so the printed
            table can never disagree with a threshold computed directly.
        """
        floor, floor_desc = self._resolve_coverage_floor(min_coverage, band)
        lines = [
            f"RecoveryCurve: {self.profile.name} @ sigma={self.sigma:.0f} km/s",
            f"  {len({r['truth'] for r in self.rows})} truth shape(s), "
            f"{len({r['ivar'] for r in self.rows})} ivar value(s)",
            f"  coverage floor {floor:.3f} ({floor_desc})",
        ]
        for metric in [m for m in RECOVERY_METRICS if any(r["metric"] == m for r in self.rows)]:
            t = self.threshold(metric, min_coverage=min_coverage, max_ci_ratio=max_ci_ratio, band=band)
            metric_ivars = sorted({r["ivar"] for r in self.rows if r["metric"] == metric})
            note = ""
            if t is not None and metric_ivars:
                if t == metric_ivars[0]:
                    note = " (at the bottom of the swept range, true threshold may be lower)"
                elif t == metric_ivars[-1]:
                    note = " (at the top of the swept range, may not be bracketed)"
            lines.append(f"  {metric}: threshold ivar = " + ("not reached" if t is None else f"{t:.3g}{note}"))
            lines.append("    ivar    truth              cover  CI/CR  CI/base  bias")
            for r in sorted([x for x in self.rows if x["metric"] == metric], key=lambda x: (x["ivar"], x["truth"])):
                ratio = r["ci_width"] / r["cr_bound"] if r["cr_bound"] > 0 else float("nan")
                base = r["ci_width"] / r["baseline_ci_width"] if r["baseline_ci_width"] > 0 else float("nan")
                lines.append(
                    f"    {r['ivar']:<7.3g} {r['truth']:<18s} {r['coverage']:5.2f}  "
                    f"{ratio:5.2f}  {base:7.2f}  {r['bias']:+.3f}"
                )
        return "\n".join(lines)


def recovery_curve(
    profile,
    truths,
    ivar_values,
    sigma=None,
    *,
    n_real=50,
    seed=20260811,
    num_warmup=300,
    num_samples=600,
    prior="gaussian_core",
):
    """Sweep information content and measure bias, coverage, and efficiency.

    For each ``ivar`` in *ivar_values* and each truth, draws *n_real* mock
    realisations sized to hit that information content, fits each with
    ``KinematicSolver`` and with the ``gaussian_mle`` baseline, and records
    per metric: the median bias, the empirical coverage of the nominal 68%
    interval, the mean credible-interval width, the Cramer-Rao bound, and the
    baseline's interval width.

    The Cramer-Rao column is what makes the result actionable. Coverage alone
    can be bought by inflating uncertainties; the ratio of interval width to
    the bound says whether the method is actually extracting the information
    present.

    ``sigma`` defaults to ``profile.sigma_max``. **Run it at
    ``profile.sigma_min`` too** for the same reason ``calibrate`` says so:
    across omega Cen's range ``err/sigma`` spans 0.11 to 0.36 and a
    regularisation calibrated at one end need not hold at the other.

    Cost is ``len(ivar_values) * len(truths) * n_real`` NUTS runs. At the
    defaults with 6 ivar values and 3 truths that is 900 runs, several hours.
    Reduce *n_real* for a smoke test; do not reduce it for a result.

    ``cr_bound`` is exact for ``v_mean`` only, since *target_ivar* IS its
    Fisher information. For ``sigma``, ``skewness`` and ``kurtosis`` it is an
    equal-error Gaussian approximation via ``n_eff = target_ivar * sigma**2``.
    With heteroscedastic errors that ``n_eff`` is a centroid-weighted
    effective sample size and is not the correct effective N for a second or
    fourth moment, which weight the per-star variances differently, so it is
    optimistic for those three and the reported CI over CR ratio
    UNDERSTATES their inefficiency. See the ``RecoveryCurve`` Notes.

    Parameters
    ----------
    profile : ObservingProfile
    truths : list of Truth
    ivar_values : sequence of float
        Information contents to sweep, as returned by ``ObservingProfile.ivar``.
    sigma : float, optional
        LOSVD dispersion for the mocks. Defaults to ``profile.sigma_max``.
    n_real : int
        Mock realisations per (ivar, truth) cell.
    seed : int
    num_warmup, num_samples, prior
        Passed through to ``KinematicSolver.run``.

    Returns
    -------
    RecoveryCurve
    """
    from veldist.analysis import compute_summary
    from veldist.baseline import gaussian_mle
    from veldist.veldist import KinematicSolver

    sigma = profile.sigma_max if sigma is None else sigma
    rows = []

    for target_ivar in ivar_values:
        for t in truths:
            pdf, rvs = t.scaled(sigma)
            tv = true_moments(pdf)
            hits = {m: 0 for m in RECOVERY_METRICS}
            meds = {m: [] for m in RECOVERY_METRICS}
            widths = {m: [] for m in RECOVERY_METRICS}
            base_widths = {"v_mean": [], "sigma": []}
            rng = np.random.default_rng(seed)

            for i in range(n_real):
                err = profile.draw_sample(target_ivar, sigma, rng)
                n = len(err)
                obs = rvs(n, rng) + rng.normal(0.0, err)

                solver = KinematicSolver()
                solver.setup_grid(center=0.0, width=profile.grid_width, n_bins=profile.n_bins)
                solver.add_data(obs, err)
                solver.run(num_warmup=num_warmup, num_samples=num_samples, seed=seed + i, prior=prior)
                summ = compute_summary(solver.samples["intrinsic_pdf"], solver.grid["centers"])

                for m in RECOVERY_METRICS:
                    med, h68 = summ[m]
                    meds[m].append(med)
                    widths[m].append(h68)
                    if abs(med - tv[m]) <= h68:
                        hits[m] += 1

                base = gaussian_mle(obs, err)
                base_widths["v_mean"].append(base["v_mean_err"])
                base_widths["sigma"].append(base["sigma_err"])

            # Cramer-Rao bounds at this information content. v_mean is exact
            # by construction (ivar IS its Fisher information); the others use
            # the equal-error Gaussian approximation with an effective N.
            n_eff = target_ivar * sigma**2
            cr = {
                "v_mean": 1.0 / np.sqrt(target_ivar),
                "sigma": sigma / np.sqrt(2 * n_eff),
                "skewness": np.sqrt(6.0 / n_eff),
                "kurtosis": np.sqrt(24.0 / n_eff),
            }

            for m in RECOVERY_METRICS:
                rows.append(
                    {
                        "ivar": float(target_ivar),
                        "truth": t.name,
                        "metric": m,
                        "bias": float(np.median(meds[m]) - tv[m]),
                        "coverage": hits[m] / n_real,
                        "ci_width": float(np.mean(widths[m])),
                        "cr_bound": float(cr[m]),
                        "baseline_ci_width": float(np.mean(base_widths[m])) if m in base_widths else float("nan"),
                    }
                )

    return RecoveryCurve(profile=profile, sigma=sigma, rows=rows, n_real=n_real)


def measure_proxy_to_gh(truths, sigma, n_bins, grid_width, max_h3=0.15, max_h4=0.10):
    """Measure the mapping from robust shape proxies to Gauss-Hermite h3/h4.

    ``SKEW_PER_H3`` and ``EXKURT_PER_H4`` at the top of this module are
    *analytic* small-amplitude conversions between ordinary moments and GH
    coefficients. They say nothing about the percentile-based proxies in
    ``compute_percentile_summary``, which are the statistics actually worth
    reporting for noisy discrete data. This measures that relation directly,
    by evaluating both on the same set of analytic truths discretised onto the
    working grid.

    No sampling and no MCMC: truths are evaluated exactly, so the result is
    a property of the statistics and the grid, not of any dataset.

    The conversion is calibrated only within the amplitude envelope given by
    ``max_h3``/``max_h4``, which defaults to the range ``make_truths()``'s own
    docstring claims to span (``|h3| <~ 0.15``, ``|h4| <~ 0.05-0.1``). Truths
    outside that envelope, such as ``bimodal_counter_rotation`` and
    ``flat_top_tangential``, are strongly non-Gaussian, a low-order
    Gauss-Hermite series is a poor description of them in the first place,
    and their large amplitude would otherwise dominate an origin-fit slope
    through ``sum(x * x)`` weighting. They are excluded from the fit, not
    just down-weighted. Do not apply this conversion to curves with larger
    non-Gaussianity than the envelope; ``bimodality_score`` is the right
    diagnostic for those instead. Even within the envelope, at least one
    truth in the library (``cold_disk_component``) has a proxy and a GH
    coefficient of opposite sign, so the conversion is not reliable for
    individual low-amplitude curves. It is a population-level guide only.

    Parameters
    ----------
    truths : list of Truth
        At least 2. More, and more varied, gives a better-constrained slope.
    sigma : float
        Dispersion to scale each truth to, km/s.
    n_bins : int
        Number of velocity bins.
    grid_width : float
        Full grid width, km/s.
    max_h3 : float
        Truths with ``abs(h3) > max_h3`` are excluded from the ``h3`` fit.
        Filtering is per mapping: a truth can be in-range for ``h3`` and out
        of range for ``h4``, or vice versa.
    max_h4 : float
        Truths with ``abs(h4) > max_h4`` are excluded from the ``h4`` fit.

    Returns
    -------
    dict
        ``'skew_pct_to_h3'`` and ``'kurtosis_pct_to_h4'``, each a dict with:

        ``slope``
            Least-squares slope of GH coefficient against proxy through the
            origin, fit only over truths that survive the amplitude envelope
            for that mapping.
        ``median_ratio``
            Median of the per-truth ratios ``y / x`` (GH coefficient over
            proxy), over included truths with ``abs(x) > 3e-3`` (below that
            the ratio is numerically meaningless: several truths have a
            proxy of exactly 0 by symmetry, and the ``3e-3`` cut also
            excludes ``gaussian``'s grid-discretisation residual while
            keeping ``cold_disk_component`` in). This is the number to apply
            in practice, because unlike ``slope`` it is robust to a single
            sign-flipped truth.
        ``ratio_std``
            Standard deviation of those same ratios. Kept alongside
            ``median_ratio`` so no information is lost, but do not read it
            as "the mapping is uncertain by this much": a single outlier
            truth can dominate it while the rest of the population is tight
            (see ``outliers`` below).
        ``n_truths``
            Number of truths whose ratio entered the ``median_ratio`` and
            ``ratio_std`` statistics. The two mappings are not equally
            trustworthy on this count: ``kurtosis_pct_to_h4`` rests on 5
            ratio-eligible truths, while ``skew_pct_to_h3`` rests on only 3,
            one of which is the flagged ``cold_disk_component`` outlier. A
            MAD outlier test on 3 points has very little power, so the h3
            ``median_ratio`` and its ``outliers`` entry should be read with
            substantially less confidence than the h4 ones.
        ``outliers``
            Names of included truths whose ratio is more than 3 scaled
            median-absolute-deviations (MAD * 1.4826) from ``median_ratio``.
            Computed, not hardcoded, so it stays correct if the truth
            library changes. Empty if the MAD is zero (all ratios equal) or
            if there are too few truths to define an outlier.

        A large ``ratio_std`` relative to ``median_ratio`` (as for
        ``kurtosis_pct_to_h4``, where ``cold_disk_component`` disagrees in
        sign with the rest) means the mapping is genuinely shape-dependent
        for that one truth, not that it is universally loose. This directly
        reports how shape-dependent the conversion is, unlike an RMS
        residual about the fit, which can look small even when the fit is
        dominated by a few extreme points. If fewer than 2 truths clear the
        ``3e-3`` threshold, ``median_ratio`` and ``ratio_std`` are
        ``float("nan")`` rather than fabricated numbers.

    Raises
    ------
    ValueError
        If fewer than 2 truths are given.
    """
    from veldist.analysis import compute_percentile_summary, gauss_hermite_fit

    if len(truths) < 2:
        msg = "at least 2 truths are required to fit a slope"
        raise ValueError(msg)

    edges = np.linspace(-grid_width / 2.0, grid_width / 2.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    names_skew, proxies_skew, gh3 = [], [], []
    names_kurt, proxies_kurt, gh4 = [], [], []
    for t in truths:
        pdf, _ = t.scaled(sigma)
        mass = np.asarray(pdf(centers), dtype=float)
        mass = mass / mass.sum()
        # gauss_hermite_fit needs at least 2 successful fits to report a
        # median rather than nan, and it only ever fits n_samples rows
        # regardless of n_draws. The curve is identical on both rows, so
        # this stays deterministic (no sampling), it just satisfies that
        # minimum.
        row = np.tile(mass[None, :], (2, 1))

        pct = compute_percentile_summary(row, centers)
        gh = gauss_hermite_fit(row, centers, n_draws=2)
        h3, h4 = gh["h3"][0], gh["h4"][0]
        if not np.isfinite(h3):
            continue
        if abs(h3) <= max_h3:
            names_skew.append(t.name)
            proxies_skew.append(pct["skew_pct"][0])
            gh3.append(h3)
        if abs(h4) <= max_h4:
            names_kurt.append(t.name)
            proxies_kurt.append(pct["kurtosis_pct"][0])
            gh4.append(h4)

    def _mapping(names, x, y):
        names = np.asarray(names)
        x, y = np.asarray(x), np.asarray(y)

        denom = float(np.sum(x * x))
        slope = float(np.sum(x * y) / denom) if denom != 0 else float("nan")

        big = np.abs(x) > 3e-3
        ratios = y[big] / x[big]
        ratio_names = names[big]
        n_truths = int(ratios.size)

        if n_truths < 2:
            return {
                "slope": slope,
                "median_ratio": float("nan"),
                "ratio_std": float("nan"),
                "n_truths": n_truths,
                "outliers": [],
            }

        median_ratio = float(np.median(ratios))
        ratio_std = float(np.std(ratios))
        mad = float(np.median(np.abs(ratios - median_ratio)))
        scaled_mad = 1.4826 * mad
        if scaled_mad == 0:
            outliers = []
        else:
            outlier_mask = np.abs(ratios - median_ratio) > 3 * scaled_mad
            outliers = sorted(ratio_names[outlier_mask].tolist())

        return {
            "slope": slope,
            "median_ratio": median_ratio,
            "ratio_std": ratio_std,
            "n_truths": n_truths,
            "outliers": outliers,
        }

    return {
        "skew_pct_to_h3": _mapping(names_skew, proxies_skew, gh3),
        "kurtosis_pct_to_h4": _mapping(names_kurt, proxies_kurt, gh4),
    }
