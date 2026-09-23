"""Calibrating veldist to a dataset.

The velocity grid, the regularisation strength and the achievable
precision all depend on the observing regime: how many stars each spatial
bin holds, how precise their velocities are, and how broad the LOSVD is.
When the regime changes, the method has to be re-tuned. This module
states those assumptions in one place, where they can be checked, instead
of leaving them as scattered constants.

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
    """Everything about a dataset that the method has to be tuned to.

    The velocity grid is derived from these values rather than chosen by hand.
    Two ratios decide whether it is sensible, and both were wrong in this
    repository before they were made explicit:

    - ``bin_width / median_error``: the grid cannot resolve structure finer
      than the measurement errors, which blur it out. Well below 1 wastes
      latent dimensions; well above 2-3 throws away real resolution.
    - ``informative_fraction``: bins with no mass are dimensions driven only
      by the prior, with no data to anchor them. They are what post-hoc tail
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
        """Draw per-star measurement errors.

        Log-normal rather than uniform, because real errors depend on magnitude
        and have a tail, and Sanders & Evans (2020) find that the error *floor*
        matters more than the spread for getting the sign of the kurtosis right.
        """
        return np.exp(rng.normal(np.log(self.err_median), self.err_log_sigma, size=n))

    @staticmethod
    def ivar(sigma, err):
        """Total Fisher information on the mean velocity for these stars.

        This is ``sum 1/(sigma^2 + err_i^2)``, not ``sum 1/err_i^2``. A star's
        velocity tells you about the LOSVD centre only to within the intrinsic
        spread it was drawn from, so the relevant variance is that of the observed
        velocity. Using ``1/err_i^2`` would claim unlimited information from
        perfectly measured stars and make bins far too small.

        Its inverse square root is the Cramér-Rao bound on ``v_mean``, which makes
        it a natural binning target: ``ivar = 1`` means ``v_mean`` to 1 km/s.
        """
        err = np.asarray(err, dtype=float)
        return float(np.sum(1.0 / (sigma**2 + err**2)))

    def draw_sample(self, target_ivar, sigma, rng):
        """Draw per-star errors until the bin reaches *target_ivar*.

        The recovery curve varies the information content while keeping the error
        distribution fixed, so the number of stars is an output here, not an
        input. That is what lets the sweep report thresholds in units that carry
        over between datasets with different errors.

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
            Per-star errors in km/s, for the smallest number of stars whose total
            ``ivar`` reaches the target (so it overshoots by at most one star).

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
        """Width of the shared velocity grid.

        Dynamite needs one grid for all spatial bins, so it must hold the widest
        LOSVD plus the rotation offset of the bins furthest from the systemic
        velocity.
        """
        return 2.0 * self.n_sigma_grid * self.sigma_max + self.rotation_span

    @property
    def bin_width(self) -> float:
        return self.bins_per_error * self.median_error

    @property
    def n_bins(self) -> int:
        return int(round(self.grid_width / self.bin_width))

    @property
    def err_over_sigma(self) -> tuple:
        """How hard the deconvolution is.

        Amorisco & Evans (2012) show that the loss of non-Gaussian signal depends
        on this ratio alone.
        """
        return (self.median_error / self.sigma_max, self.median_error / self.sigma_min)

    def informative_fraction(self, sigma: float) -> float:
        """Fraction of grid bins carrying mass for a LOSVD of this width."""
        return min(1.0, 2.0 * self.n_sigma_grid * sigma / self.grid_width)

    def moment_precision(self) -> dict:
        """Best achievable per-bin precision, as a sanity limit on any claim.

        The mean and sigma use the Gaussian Cramér-Rao bounds; h3 and h4 use the
        (2N)^-1/2 approximation that Sanders & Evans (2020) recommend for small
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

        Dynamite needs one shared grid across all spatial bins, but only for its
        input. veldist can fit each bin on a grid matched to its own dispersion
        and then sum the posterior samples onto the shared output grid. If the
        fitted grid is at least as fine as the output grid and the edges line up,
        this is exact (a per-sample sum of mass within output bins), so the
        uncertainties carry over correctly.
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

        Every parameter of this class was once typed in by hand (see
        ``OMEGACAT``), so the mock suite was testing the method against a guess
        about the data. This measures them instead. The result contains only
        scalars, so it can be committed as a test fixture even when the catalogue
        itself cannot be shared.

        Per-bin dispersions come from ``gaussian_mle`` rather than ``numpy.std``.
        ``numpy.std`` returns ``sqrt(sigma^2 + err^2)``, which would inflate
        ``sigma_min`` most in the low-dispersion bins that define the hardest
        regime.

        Parameters
        ----------
        vel, err : array-like, shape (n_stars,)
            Velocities and per-star errors for the whole field, km/s.
        bin_ids : array-like, shape (n_stars,)
            Spatial bin index of each star; need not be contiguous.
        name : str
            Label for the returned profile.
        min_stars : int
            Bins with fewer stars are left out of the dispersion and rotation
            estimates.

        Returns
        -------
        ObservingProfile

        Raises
        ------
        ValueError
            If fewer than 2 bins pass the *min_stars* cut.
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

#: What the production MUSE notebook actually produces, measured 2026-09-01
#: by replaying ``muse_veldist.ipynb``'s own binning call
#: (``do_powerbin(target_capacity=140)``, cell_width=15, 163 bins) on the real
#: catalogue. ``OMEGACAT`` above is the hand-picked stand-in it replaces.
#:
#: Every scalar that differs, differs in the same direction -- the hand-picked
#: profile understates the real spread:
#:   rotation_span  10.0  -> 17.9  (1.8x; this one enters grid_width directly)
#:   err_median      2.5  ->  4.0  (1.6x)
#:   err_log_sigma   0.4  ->  0.62 (1.55x; the error tail is much heavier)
#: The dispersion range is the exception: measured 10.4-20.0 is NARROWER than
#: the assumed 7.0-22.0, so the assumed grid was wide enough there by luck.
#:
#: rotation_span is the full ptp of the per-bin mean velocity, not the 5-95
#: span (13.8): the grid has to hold the worst bin, and a percentile range
#: discards exactly the tail that defines it.
OMEGACAT_MEASURED = ObservingProfile(
    name="oMEGACat_measured",
    n_stars=152,
    err_median=4.0,
    err_log_sigma=0.62,
    sigma_max=20.01,
    sigma_min=10.37,
    rotation_span=17.9,
)


def recommend_grid(profile: ObservingProfile, v_systemic: float = 0.0) -> dict:
    """Grid arguments for ``KinematicSolver.setup_grid`` or
    ``fit_all_bins(grid_kwargs=...)`` from a measured :class:`ObservingProfile`,
    instead of choosing ``n_bins`` by hand.

    It returns ``profile.grid_width`` and ``profile.n_bins`` (already derived
    from the error and dispersion scales; see the class docstring) in the form
    those functions take.

    Parameters
    ----------
    profile : ObservingProfile
        Usually ``ObservingProfile.from_data(...)`` on the real catalogue.
    v_systemic : float
        Grid centre, km/s. Default 0.0, for velocities with the systemic
        velocity already removed.
    """
    return {"center": v_systemic, "width": profile.grid_width, "n_bins": profile.n_bins}


def recommend_cuts(profile: ObservingProfile, recovery=None, metric="sigma", min_stars=10, **threshold_kwargs) -> dict:
    """Arguments ``min_stars``, ``min_ivar`` and ``sigma_ref`` for
    ``fit_all_bins`` from a measured :class:`ObservingProfile`.

    ``min_ivar`` cannot come from the profile alone. It is a coverage
    threshold, which needs recovery simulations (``RecoveryCurve.threshold``,
    as referenced by ``fit_all_bins``). That sweep takes hours, so it is only
    used if you pass one in.

    Parameters
    ----------
    profile : ObservingProfile
    recovery : RecoveryCurve, optional
        Output of ``recovery_curve(profile, ..., sigma=profile.sigma_min)``.
        It must be built at ``sigma_min``, the hardest case in the field, so
        the threshold is conservative: a curve built at a larger dispersion
        would accept bins that fail at low dispersion. An error is raised if
        ``recovery.sigma != profile.sigma_min``. Without it, ``min_ivar`` is
        ``None`` and only ``min_stars`` applies, as in ``fit_all_bins`` by
        default.
    metric : str
        Passed to ``recovery.threshold``. Default ``"sigma"``; for a
        dispersion map, ``v_mean`` is not the limiting quantity.
    min_stars : int
        Floor applied whatever ``recovery`` says, to avoid degenerate fits.
        Default 10, as in ``fit_all_bins``.
    **threshold_kwargs
        Passed to ``recovery.threshold`` (``min_coverage``, ``max_ci_ratio``,
        ``band``).

    Returns
    -------
    dict
        ``min_stars``, ``min_ivar`` (``None`` without ``recovery``) and
        ``sigma_ref``, ready for ``fit_all_bins(**recommend_cuts(...))``.

    Raises
    ------
    ValueError
        If ``recovery`` was built at a ``sigma`` other than
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

    Shapes are dimensionless so the same library works for any dataset.
    ``scaled(sigma)`` returns ``(pdf, rvs)`` with zero mean and the requested
    dispersion.
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
    """Uniform(-a, a) convolved with a Gaussian: the Sanders & Evans (2020)
    kernel for negative excess kurtosis.

    The excess kurtosis is -1.2 r^2, where r is the fraction of the variance
    in the uniform part, so -1.2 is the lower limit.
    """

    def pdf(x):
        x = np.asarray(x, dtype=float)
        return (stats.norm.cdf((x + a) / s) - stats.norm.cdf((x - a) / s)) / (2 * a)

    def rvs(n, rng):
        return rng.uniform(-a, a, size=n) + rng.normal(0.0, s, size=n)

    return pdf, rvs


def _split_uniform_gauss(a1, a2, s):
    """Two-piece uniform kernel convolved with a Gaussian: the Sanders & Evans
    (2020) option for skewness.

    The two halves have equal weight but different widths. Simply shifting a
    uniform would not work, since it stays symmetric about its own midpoint.
    """

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
    """The library of mock LOSVD shapes, dimensionless; scale them with
    ``Truth.scaled(sigma)``.

    They cover the non-Gaussianity expected in a rotating, anisotropic
    globular cluster at realistic amplitudes (``|h3|`` <~ 0.15, ``|h4|`` <~
    0.05-0.1). Each truth's ``note`` gives its physical motivation.
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

    Coverage measured from ``n_real`` mocks is a binomial proportion, so the
    acceptable range depends on ``n_real``. This follows the convention of
    :data:`NOMINAL_BAND` (the 99% band of ``binom(25, 0.68)``) for any
    ``n_real``; ``coverage_floor(25)`` reproduces ``NOMINAL_BAND[0]``
    exactly.

    Parameters
    ----------
    n_real : int
        Number of mock realisations behind the coverage fraction.
    band : float
        Confidence level of the two-sided binomial interval, e.g. 0.99.
    nominal : float
        Nominal coverage of the interval being tested, e.g. 0.68.

    Returns
    -------
    float
        Lower edge of the band, as a fraction of ``n_real``.
    """
    return float(binom.ppf((1 - band) / 2, n_real, nominal)) / n_real


def true_moments(pdf, lo=-500.0, hi=500.0, n_grid=400001):
    """Moments of a truth, computed on a dense grid.

    Not ``scipy.integrate.quad``: its adaptive subdivision can miss a narrow
    component on a broad base. For the cold-disk truth, quad gave a skewness of
    +0.0000 against a true -0.0320, and a wrong truth silently invalidates a
    coverage test instead of failing it.
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

        About 1 means the estimator extracts what the data contain; above 1 means
        information is lost. **Below 1 is not better than optimal**: it means the
        prior is shrinking the estimates, so read it together with the bias.

        Coverage cannot check this. A posterior can reach nominal coverage by
        putting large error bars on a poor estimator; efficiency separates that
        from a good estimator with honest error bars.

        Uses a robust scatter, because with a few dozen realisations one failed
        fit would dominate a standard deviation.
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
    """Fit mock realisations of each truth and measure coverage and
    efficiency.

    ``sigma`` defaults to the widest LOSVD in the profile. **Run it at
    ``profile.sigma_min`` as well.** With the original hand-typed omega Cen
    profile (7-22 km/s), err/sigma ran from 0.11 to 0.36 and the informative
    fraction of bins from 95% to 30%, so a regularisation tuned at one end
    need not be calibrated at the other; the narrowest bins sat in the
    dwarf-spheroidal regime (Amorisco & Evans 2012 give 0.33 for Sculptor).
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
    """How well each statistic is recovered as a function of information
    content.

    This answers the question behind both ``min_stars=10`` and any spatial
    binning target: how much information does a bin need before its posterior
    can be trusted? Measuring it in units of Fisher information rather than
    star count gives a threshold that carries over to datasets with different
    errors.

    Notes
    -----
    The ``cr_bound`` column is exact for ``v_mean``, since the swept ``ivar``
    is the Fisher information of the mean. For ``sigma``, ``skewness`` and
    ``kurtosis`` it uses an equal-error Gaussian approximation with the same
    effective sample size, which is exact only when all stars have the same
    error. With unequal errors it is indicative only, so treat the CI/CR
    ratio of those three as a rough efficiency check.
    """

    profile: object
    sigma: float
    rows: list
    n_real: int = None

    def threshold(self, metric, min_coverage=None, max_ci_ratio=1.5, band=0.99):
        """Smallest ``ivar`` at which *metric* can be trusted, or ``None``.

        Trusted means two things together, since either alone can be gamed: the
        coverage is at least the floor (the interval contains the truth often
        enough) **and** the credible interval is no wider than *max_ci_ratio*
        times the Cramér-Rao bound (the interval is not simply wide enough to
        contain everything).

        A point counts only if every point at higher ``ivar`` also passes, so one
        lucky low-information point cannot set the threshold.

        Parameters
        ----------
        metric : str
            One of :data:`RECOVERY_METRICS`.
        min_coverage : float, optional
            Required coverage of the nominal 68% interval. Coverage from
            ``n_real`` mocks is a binomial proportion, so a fixed floor is right
            only for one ``n_real``. At ``n_real=40`` the standard error is
            ``sqrt(0.68*0.32/40) = 0.074``, so a fixed 0.60 is one standard error
            below nominal and rejects a well-calibrated method in about one cell in
            ten. :data:`NOMINAL_BAND` already handles coverage this way for
            ``n_real=25``; this generalises it to the ``n_real`` the curve was
            built with.

            If given, it is used as-is. Otherwise, if ``self.n_real`` is set, the
            floor is :func:`coverage_floor` ``(self.n_real, band)``. Otherwise it
            falls back to the old constant 0.60.
        max_ci_ratio : float
            Required efficiency, as a multiple of the Cramér-Rao bound.
        band : float
            Confidence level of the binomial band used when ``min_coverage`` is
            not given and ``self.n_real`` is set; ignored otherwise.

        Returns
        -------
        float or None
            ``None`` if no swept ``ivar`` passes, meaning the sweep did not reach
            enough information.

        Raises
        ------
        ValueError
            If *metric* does not appear in any row.
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
            Passed to :meth:`threshold`, so the table always agrees with a
            threshold computed directly.
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
    """Sweep information content and measure bias, coverage and efficiency.

    For each ``ivar`` in *ivar_values* and each truth, this draws *n_real*
    mock bins with that information content, fits each with
    ``KinematicSolver`` and with the ``gaussian_mle`` baseline, and records for
    every metric the median bias, the coverage of the nominal 68% interval,
    the mean interval width, the Cramér-Rao bound, and the baseline's interval
    width.

    The Cramér-Rao column is what makes the result useful. Coverage alone can
    be bought by inflating the uncertainties; the ratio of interval width to
    the bound shows whether the method actually extracts the information in
    the data.

    ``sigma`` defaults to ``profile.sigma_max``. **Run it at
    ``profile.sigma_min`` too**, for the reason given in ``calibrate``: a
    regularisation calibrated at one end of the dispersion range need not hold
    at the other.

    The cost is ``len(ivar_values) * len(truths) * n_real`` NUTS runs: 900 for
    6 ivar values and 3 truths at the default ``n_real``, which takes several
    hours. Lower *n_real* for a smoke test, but not for a result.

    ``cr_bound`` is exact only for ``v_mean``, whose Fisher information is
    *target_ivar*. For ``sigma``, ``skewness`` and ``kurtosis`` it is an
    equal-error Gaussian approximation with ``n_eff = target_ivar * sigma**2``.
    With unequal errors that ``n_eff`` is the right effective sample size for
    the mean but not for the second or fourth moments, so the bound is
    optimistic and the reported CI/CR ratio UNDERSTATES their inefficiency
    (see the ``RecoveryCurve`` notes).

    Parameters
    ----------
    profile : ObservingProfile
    truths : list of Truth
    ivar_values : sequence of float
        Information contents to sweep, as from ``ObservingProfile.ivar``.
    sigma : float, optional
        LOSVD dispersion of the mocks. Defaults to ``profile.sigma_max``.
    n_real : int
        Mock realisations per (ivar, truth) cell.
    seed : int
    num_warmup, num_samples, prior
        Passed to ``KinematicSolver.run``.

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
    """Measure how the robust shape statistics map onto Gauss-Hermite h3/h4.

    ``SKEW_PER_H3`` and ``EXKURT_PER_H4`` at the top of this module are
    analytic small-amplitude conversions between ordinary moments and GH
    coefficients. They say nothing about the percentile statistics from
    ``compute_percentile_summary``, which are the ones worth reporting for
    noisy discrete data. This function measures that relation directly by
    evaluating both kinds of statistic on the same analytic truths,
    discretised onto the working grid.

    There is no sampling or MCMC: the truths are evaluated exactly, so the
    result depends only on the statistics and the grid, not on any dataset.

    The mapping is calibrated only inside the amplitude limits
    ``max_h3``/``max_h4``, which default to the range ``make_truths()``
    covers (``|h3|`` <~ 0.15, ``|h4|`` <~ 0.05-0.1). Truths outside it, such as
    ``bimodal_counter_rotation`` and ``flat_top_tangential``, are strongly
    non-Gaussian and poorly described by a low-order GH series anyway, and
    their large amplitudes would dominate a slope fitted through the origin.
    They are excluded from the fit, not just down-weighted. Do not apply the
    mapping to more strongly non-Gaussian curves; use ``bimodality_score``
    for those. Even inside the limits, ``cold_disk_component`` has a proxy and
    a GH coefficient of opposite sign, so the mapping is unreliable for an
    individual low-amplitude curve and should be used only as a guide across
    a population.

    Parameters
    ----------
    truths : list of Truth
        At least 2. More, and more varied, truths constrain the slope better.
    sigma : float
        Dispersion to scale each truth to, km/s.
    n_bins : int
        Number of velocity bins.
    grid_width : float
        Full grid width, km/s.
    max_h3 : float
        Truths with ``abs(h3) > max_h3`` are left out of the ``h3`` fit. The
        cut is per mapping, so a truth can be used for ``h3`` but not ``h4``,
        or the other way round.
    max_h4 : float
        Truths with ``abs(h4) > max_h4`` are left out of the ``h4`` fit.

    Returns
    -------
    dict
        ``'skew_pct_to_h3'`` and ``'kurtosis_pct_to_h4'``, each a dict with:

        ``slope``
            Least-squares slope of the GH coefficient against the proxy,
            through the origin, over the truths inside the amplitude limits.
        ``median_ratio``
            Median of the per-truth ratios ``y / x`` (GH coefficient over
            proxy) for included truths with ``abs(x) > 3e-3``. Below that the
            ratio means nothing: several truths have a proxy of exactly 0 by
            symmetry. The cut also removes the ``gaussian`` truth's
            discretisation residual while keeping ``cold_disk_component``.
            This is the value to use in practice, because unlike ``slope`` it
            is robust to one truth with the opposite sign.
        ``ratio_std``
            Standard deviation of the same ratios. It is kept for
            completeness, but it is not the uncertainty of the mapping: one
            outlying truth can dominate it while the rest agree closely (see
            ``outliers``).
        ``n_truths``
            Number of truths behind ``median_ratio`` and ``ratio_std``. The
            two mappings differ here: ``kurtosis_pct_to_h4`` uses 5 truths,
            ``skew_pct_to_h3`` only 3, one of them the outlier
            ``cold_disk_component``. An outlier test on 3 points has very
            little power, so trust the h3 ``median_ratio`` and its
            ``outliers`` much less than the h4 ones.
        ``outliers``
            Included truths whose ratio is more than 3 scaled MADs
            (MAD * 1.4826) from ``median_ratio``. Computed rather than
            hard-coded, so it stays correct if the library changes. Empty if
            the MAD is zero or there are too few truths.

        A ``ratio_std`` that is large relative to ``median_ratio`` (as for
        ``kurtosis_pct_to_h4``, where ``cold_disk_component`` has the opposite
        sign) means the mapping depends on shape for that truth, not that it
        is loose for all of them. It shows the shape dependence directly, which
        an RMS residual about the fit would not, since that can look small when
        a few extreme points dominate the fit. If fewer than 2 truths pass the
        ``3e-3`` cut, ``median_ratio`` and ``ratio_std`` are ``float("nan")``.

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
