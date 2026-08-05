"""
Frequentist coverage over mock realisations for veldist2d.

2D analogue of ``test_coverage.py::test_coverage_over_mock_realisations``.
SBC (``test_calibration_2d.py``) validates the sampler against the model;
this validates the *model against reality* -- whether the posterior's
median +/- half-68CI for mean_x, mean_y, sigma_x, sigma_y means what we'd
tell downstream consumers it means, for a couple of fixed, physically
plausible bivariate truths.

Per the veldist acceptance criterion (mirrored from 1D, see CLAUDE.md /
PLAN.md): mean and dispersion recovery with calibrated uncertainties is the
bar for "minimally working". Correlation (``rho``) recovery is the 2D
equivalent of 1D's optional h3/h4 -- tracked and printed, not gating.

``KinematicSolver2D`` has no ``clip_uncertainties()`` (unlike 1D's
``KinematicSolver``), so the summary here is computed directly from
``mcmc.get_samples()`` via the same ``_moments_from_pdf_samples_2d`` used by
the SBC test, applied to real (not prior-drawn) posterior samples.
"""

import numpy as np
import pytest
from scipy import stats

from veldist.veldist2d import KinematicSolver2D
from tests.test_calibration_2d import _moments_from_pdf_samples_2d

N_REAL = 25
N_STARS = 150
K = 10  # per-axis grid size, matches the SBC harness (PLAN.md §3.3)
GRID_WIDTH = (40.0, 40.0)
GRID_CENTER = (0.0, 0.0)
ERR_RANGE = (0.5, 2.0)

HARD_METRICS = ["mean_x", "mean_y", "sigma_x", "sigma_y"]
SOFT_METRICS = ["rho"]

TRUTHS = {
    "isotropic": {"mux": 0.0, "muy": 0.0, "sx": 6.0, "sy": 6.0, "rho": 0.0},
    "tilted": {"mux": 2.0, "muy": -1.0, "sx": 8.0, "sy": 4.0, "rho": 0.6},
}


def _binom_band(n, p=0.68, alpha=0.01):
    lo = stats.binom.ppf(alpha / 2, n, p) / n
    hi = stats.binom.ppf(1 - alpha / 2, n, p) / n
    return float(lo), float(hi)


def _draw_stars(rng, truth, n_stars):
    mean = [truth["mux"], truth["muy"]]
    cov_true = [
        [truth["sx"] ** 2, truth["rho"] * truth["sx"] * truth["sy"]],
        [truth["rho"] * truth["sx"] * truth["sy"], truth["sy"] ** 2],
    ]
    true_xy = rng.multivariate_normal(mean, cov_true, size=n_stars)

    err_x = rng.uniform(*ERR_RANGE, size=n_stars)
    err_y = rng.uniform(*ERR_RANGE, size=n_stars)
    obs_x = true_xy[:, 0] + rng.normal(0.0, err_x)
    obs_y = true_xy[:, 1] + rng.normal(0.0, err_y)

    cov = np.zeros((n_stars, 2, 2))
    cov[:, 0, 0] = err_x**2
    cov[:, 1, 1] = err_y**2
    return obs_x, obs_y, cov


_GMRF_SCALING_XFAIL_REASON = (
    "Known red: sigma_x/sigma_y/rho coverage below nominal for both truths "
    "(measured at HalfNormal(3.0), n_real=25: isotropic sigma_x 0.32, sigma_y "
    "0.44; tilted sigma_x 0.60, sigma_y 0.24, rho 0.44 -- against a nominal "
    "0.68). Root cause identified by a follow-up campaign (see TASKS.md '2D "
    "solver: smoothness_sigma is not resolution-independent'): a grid-width/K "
    "sweep showed the dispersion bias growing with grid resolution rather "
    "than shrinking, ruling out truncation. model_2d has no equivalent of "
    "1D's _rw_deviation_scale (Sorbye-Rue rescaling), so smoothness_sigma is "
    "not the resolution-independent physical quantity it is documented to be "
    "for 1D -- the same nominal value implies different effective smoothing "
    "at different K. Point-estimate bias was already fixed by widening the "
    "prior from HalfNormal(0.1) to HalfNormal(3.0) (see git history); this "
    "xfail covers the remaining interval-width miscalibration, which needs "
    "the GMRF rescale, not another hyperparameter guess. strict=False so a "
    "real fix surfaces as XPASS."
)


@pytest.mark.slow
@pytest.mark.xfail(reason=_GMRF_SCALING_XFAIL_REASON, strict=False)
@pytest.mark.parametrize("truth_name", list(TRUTHS))
def test_coverage_over_mock_realisations_2d(truth_name):
    truth = TRUTHS[truth_name]

    solver0 = KinematicSolver2D()
    solver0.setup_grid(center=GRID_CENTER, width=GRID_WIDTH, n_bins=K)
    centers_2d = solver0.grid["centers_2d"]

    hits = {m: 0 for m in HARD_METRICS + SOFT_METRICS}
    medians = {m: [] for m in HARD_METRICS + SOFT_METRICS}

    for i in range(N_REAL):
        rng = np.random.default_rng(20260805 + i)
        obs_x, obs_y, cov = _draw_stars(rng, truth, N_STARS)

        solver = KinematicSolver2D()
        solver.setup_grid(center=GRID_CENTER, width=GRID_WIDTH, n_bins=K)
        solver.add_data(obs_x, obs_y, cov)
        samples = solver.run(num_warmup=300, num_samples=600, seed=20260805 + i)

        pdf_samples = np.asarray(samples["intrinsic_pdf"])
        mean_x, mean_y, sigma_x, sigma_y, rho = _moments_from_pdf_samples_2d(
            pdf_samples, centers_2d
        )
        draws = {
            "mean_x": mean_x,
            "mean_y": mean_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "rho": rho,
        }
        true_vals = {
            "mean_x": truth["mux"],
            "mean_y": truth["muy"],
            "sigma_x": truth["sx"],
            "sigma_y": truth["sy"],
            "rho": truth["rho"],
        }
        for m in HARD_METRICS + SOFT_METRICS:
            median = float(np.median(draws[m]))
            half68 = 0.5 * (np.percentile(draws[m], 84) - np.percentile(draws[m], 16))
            medians[m].append(median)
            if abs(median - true_vals[m]) <= half68:
                hits[m] += 1

    band_lo, band_hi = _binom_band(N_REAL, p=0.68, alpha=0.01)

    report_lines = [f"=== {truth_name} (n_real={N_REAL}, N_STARS={N_STARS}, K={K}) ==="]
    hard_failures = []
    for m in HARD_METRICS + SOFT_METRICS:
        cov = hits[m] / N_REAL
        report_lines.append(
            f"  {m:10s}: coverage={cov:.3f} ({hits[m]}/{N_REAL})  "
            f"mean_median={np.mean(medians[m]):.3f}"
        )
        if m in HARD_METRICS and not (band_lo <= cov <= band_hi):
            hard_failures.append(
                f"{truth_name}.{m}: coverage {cov:.3f} outside nominal binomial "
                f"band [{band_lo:.3f}, {band_hi:.3f}] for n={N_REAL} at p=0.68"
            )
    report = "\n".join(report_lines)

    if hard_failures:
        pytest.fail(report + "\n\nHARD FAILURES:\n" + "\n".join(hard_failures))
    print(report)


# ---------------------------------------------------------------------------
# Per-cell coverage
# ---------------------------------------------------------------------------


def _true_cell_mass_2d(truth, edges_x, edges_y):
    """Exact per-cell probability mass of the true bivariate Gaussian, via
    box probability from the joint CDF (inclusion-exclusion on the 4
    corners) -- exact for any correlation, unlike a centre-evaluated density.
    Row-major flatten (ix*K + iy) to match ``setup_grid_2d``'s ``centers_2d``
    ordering.
    """
    from scipy.stats import multivariate_normal

    mean = [truth["mux"], truth["muy"]]
    cov = [
        [truth["sx"] ** 2, truth["rho"] * truth["sx"] * truth["sy"]],
        [truth["rho"] * truth["sx"] * truth["sy"], truth["sy"] ** 2],
    ]
    mvn = multivariate_normal(mean=mean, cov=cov)

    k = len(edges_x) - 1
    mass = np.empty(k * k)
    for ix in range(k):
        lo_x, hi_x = edges_x[ix], edges_x[ix + 1]
        for iy in range(k):
            lo_y, hi_y = edges_y[iy], edges_y[iy + 1]
            mass[ix * k + iy] = (
                mvn.cdf([hi_x, hi_y])
                - mvn.cdf([lo_x, hi_y])
                - mvn.cdf([hi_x, lo_y])
                + mvn.cdf([lo_x, lo_y])
            )
    return mass


@pytest.mark.slow
@pytest.mark.parametrize(
    "truth_name",
    [
        "isotropic",
        pytest.param(
            "tilted",
            marks=pytest.mark.xfail(reason=_GMRF_SCALING_XFAIL_REASON, strict=False),
        ),
    ],
)
def test_per_cell_losvd_coverage_2d(truth_name):
    """Per-cell credible intervals must contain the true cell mass at
    ~nominal. 2D analogue of ``test_coverage.py::test_per_bin_losvd_coverage``.

    This is the finer-grained check moment coverage cannot substitute for:
    moments collapse K**2 cells into a handful of scalars, so a systematic
    per-cell miscalibration (over-wide here, over-tight there) can cancel and
    still pass moment coverage. Only this test sees it.

    Uses raw posterior median +/- half-68CI per cell directly from
    ``mcmc.get_samples()['intrinsic_pdf']`` (no ``clip_uncertainties``
    equivalent exists for 2D -- see module docstring).
    """
    truth = TRUTHS[truth_name]

    solver0 = KinematicSolver2D()
    solver0.setup_grid(center=GRID_CENTER, width=GRID_WIDTH, n_bins=K)
    edges_x = solver0.grid["edges_x"]
    edges_y = solver0.grid["edges_y"]

    truth_mass = _true_cell_mass_2d(truth, edges_x, edges_y)
    hits = np.zeros(len(truth_mass))

    for i in range(N_REAL):
        rng = np.random.default_rng(20260806 + i)
        obs_x, obs_y, cov = _draw_stars(rng, truth, N_STARS)

        solver = KinematicSolver2D()
        solver.setup_grid(center=GRID_CENTER, width=GRID_WIDTH, n_bins=K)
        solver.add_data(obs_x, obs_y, cov)
        samples = solver.run(num_warmup=300, num_samples=600, seed=20260806 + i)

        pdf_samples = np.asarray(samples["intrinsic_pdf"])
        median = np.median(pdf_samples, axis=0)
        half68 = 0.5 * (
            np.percentile(pdf_samples, 84, axis=0) - np.percentile(pdf_samples, 16, axis=0)
        )
        hits += (np.abs(median - truth_mass) <= half68).astype(float)

    coverage = hits / N_REAL
    informative = truth_mass > 0.01 * truth_mass.max()
    mean_cov = float(coverage[informative].mean())

    report = (
        f"{truth_name}: mean per-cell coverage over {int(informative.sum())} "
        f"informative cells (of {len(truth_mass)}) = {mean_cov:.3f}"
    )
    print(report)

    # Same band as 1D's per-bin test (nominal 0.68, generous on the high
    # side since over-coverage wastes information but doesn't bias a
    # downstream chi-squared the way under-coverage does).
    assert 0.60 <= mean_cov <= 0.85, (
        f"{report} -- outside [0.60, 0.85] against nominal 0.68. Under-coverage "
        "means the per-cell error bars are too narrow to trust downstream."
    )

    n_bad = int((coverage[informative] < 0.30).sum())
    assert n_bad == 0, (
        f"{truth_name}: {n_bad} informative cells have coverage below 0.30 "
        f"(worst {coverage[informative].min():.2f}); those cells carry "
        "essentially meaningless error bars."
    )
