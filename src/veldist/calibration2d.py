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

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ObservingProfile2D",
    "HST_BRIGHT",
    "HST_FAINT",
    "GAIA_OUTER",
    "PROFILES_2D",
    "CLUSTER_DISTANCE_PC",
    "KMS_PER_MASYR",
    "PM_QUALITY_CUT_KMS",
    "truths_for",
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

    def report(self):
        """One-line-per-fact summary, for printing in test output."""
        return (
            f"{self.name}: sigma_ref={self.sigma_ref:g} km/s, "
            f"{self.n_stars} stars/bin, err median={self.err_median:g} km/s "
            f"(cut {self.err_cut:g}), err/sigma={self.err_over_sigma:.3f}\n"
            f"  grid {self.grid_width:.0f} km/s, n_bins={self.n_bins} per axis "
            f"({self.n_bins**2} cells), cell "
            f"{self.grid_width / self.n_bins:.1f} km/s "
            f"({self.grid_width / self.n_bins / self.sigma_ref:.2f} sigma)"
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
    """
    from scipy.stats import multivariate_normal

    cov = [[t["sx"] ** 2, t["rho"] * t["sx"] * t["sy"]],
           [t["rho"] * t["sx"] * t["sy"], t["sy"] ** 2]]
    mvn = multivariate_normal(mean=[t["mux"], t["muy"]], cov=cov)
    k = len(edges_x) - 1
    mass = np.empty(k * k)
    for ix in range(k):
        for iy in range(k):
            mass[ix * k + iy] = (
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
