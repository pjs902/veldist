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

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats

from veldist.analysis import half_68ci

__all__ = [
    "ObservingProfile",
    "OMEGACAT",
    "Truth",
    "make_truths",
    "true_moments",
    "METRICS",
    "CalibrationResult",
    "calibrate",
]

# Gauss-Hermite conversions, lowest order (van der Marel & Franx 1993).
# Only valid for weak non-Gaussianity; Sanders & Evans (2020) note the h4
# relation is good to 10% only for |h4| < 0.01, so treat these as indicative
# amplitude conversions, not as a route to DYNAMITE inputs.
SKEW_PER_H3 = 4.0 * np.sqrt(3.0)
EXKURT_PER_H4 = 8.0 * np.sqrt(6.0)


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
    anisotropic globular cluster, at realistic h3/h4 amplitude (|h3| <~ 0.15,
    |h4| <~ 0.05-0.1). Physical motivation for each is in its ``note``.
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
