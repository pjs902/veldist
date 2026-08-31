"""
Frequentist coverage over mock realisations for veldist2d.

2D analogue of ``test_coverage.py::test_coverage_over_mock_realisations``.
SBC (``test_calibration_2d.py``) validates the sampler against the model;
this validates the *model against reality* -- whether the posterior's
median +/- half-68CI for mean_x, mean_y, sigma_x, sigma_y means what we'd
tell downstream consumers it means, for physically plausible bivariate
truths drawn against real proper-motion observing regimes.

Per the veldist acceptance criterion (mirrored from 1D, see CLAUDE.md /
PLAN.md): mean and dispersion recovery with calibrated uncertainties is the
bar for "minimally working". Correlation (``rho``) recovery is the 2D
equivalent of 1D's optional h3/h4 -- tracked and printed, not gating.

``KinematicSolver2D`` has no ``clip_uncertainties()`` (unlike 1D's
``KinematicSolver``), so the summary here is computed directly from
``mcmc.get_samples()`` via the same ``_moments_from_pdf_samples_2d`` used by
the SBC test, applied to real (not prior-drawn) posterior samples.

Grids, error distributions, and star counts are drawn from
``veldist.calibration2d.PROFILES_2D`` -- calibrated proper-motion observing
regimes -- rather than the arbitrary values this module used to hardcode.
Truths are scaled to each profile's ``sigma_ref`` rather than fixed absolute
numbers, so the same relative geometry (isotropic / anisotropic) is tested
at each regime's native scale.
"""

import numpy as np
import pytest
from scipy import stats

from veldist.calibration2d import (
    PROFILES_2D,
    truths_for,
    _draw_stars,
    _discretised_truth_moments,
    _moments_from_pdf_samples_2d,
)
from veldist.veldist2d import KinematicSolver2D

N_REAL = 25

HARD_METRICS = ["mean_x", "mean_y", "sigma_x", "sigma_y", "rho"]
SOFT_METRICS = []

TRUTHS = truths_for(1.0)  # names only ("isotropic", "anisotropic"); values built per-profile


def _binom_band(n, p=0.68, alpha=0.01):
    lo = stats.binom.ppf(alpha / 2, n, p) / n
    hi = stats.binom.ppf(1 - alpha / 2, n, p) / n
    return float(lo), float(hi)


_GMRF_SCALING_XFAIL_REASON = (
    "Known red: sigma_x/sigma_y/rho coverage below nominal under the legacy "
    "pure-GMRF prior (measured at HalfNormal(3.0), K=10, n_real=25, arbitrary "
    "grid: isotropic sigma_x 0.32, sigma_y 0.44; tilted sigma_x 0.60, sigma_y "
    "0.24, rho 0.44 -- against a nominal 0.68). Root cause: model_2d has no "
    "equivalent of 1D's _rw_deviation_scale (Sorbye-Rue rescaling), so "
    "smoothness_sigma is not the resolution-independent physical quantity it "
    "is documented to be for 1D -- the same nominal value implies different "
    "effective smoothing at different K. The gaussian_core prior "
    "(prior='gaussian_core') is the fix under test; this xfail is retained on "
    "the gmrf parametrisation, which is now the known-worse legacy path kept "
    "for comparison. strict=False so a real fix (or an unexpectedly-passing "
    "gmrf run) surfaces as XPASS rather than silently disappearing."
)

_GAIA_OUTER_XFAIL_REASON = (
    "gaia_outer is not required to pass: err/sigma = 0.63 exceeds the "
    "err/sigma = 0.36 regime the 1D solver already classifies as a "
    "structural failure (too little signal left after convolution with the "
    "measurement error to deconvolve reliably), regardless of prior. "
    "strict=False so an unexpected pass is visible rather than silently lost."
)


def _maybe_xfail(prior, profile_name):
    """Broad, intentionally coarse xfail markers per the task-6 brief: keep
    them broad (strict=False) so XPASS is the signal that a fix landed,
    rather than narrowing them ourselves before the real measurement runs."""
    marks = []
    if profile_name == "gaia_outer":
        marks.append(pytest.mark.xfail(reason=_GAIA_OUTER_XFAIL_REASON, strict=False))
    if prior == "gmrf":
        marks.append(pytest.mark.xfail(reason=_GMRF_SCALING_XFAIL_REASON, strict=False))
    return marks


@pytest.mark.slow
@pytest.mark.parametrize("prior", ["gmrf", "gaussian_core"])
@pytest.mark.parametrize("profile_name", ["hst_bright", "hst_faint", "gaia_outer"])
@pytest.mark.parametrize("truth_name", list(TRUTHS))
def test_coverage_over_mock_realisations_2d(truth_name, profile_name, prior, request):
    for mark in _maybe_xfail(prior, profile_name):
        request.node.add_marker(mark)

    profile = PROFILES_2D[profile_name]
    truth = truths_for(profile.sigma_ref)[truth_name]
    grid_width = (profile.grid_width, profile.grid_width)
    grid_center = (0.0, 0.0)
    n_bins = profile.n_bins

    solver0 = KinematicSolver2D()
    solver0.setup_grid(center=grid_center, width=grid_width, n_bins=n_bins)
    centers_2d = solver0.grid["centers_2d"]
    edges_x = solver0.grid["edges_x"]
    edges_y = solver0.grid["edges_y"]

    true_moments, _ = _discretised_truth_moments(truth, edges_x, edges_y, centers_2d)

    hits = {m: 0 for m in HARD_METRICS + SOFT_METRICS}
    medians = {m: [] for m in HARD_METRICS + SOFT_METRICS}

    for i in range(N_REAL):
        rng = np.random.default_rng(20260805 + i)
        obs_x, obs_y, cov = _draw_stars(rng, truth, profile.n_stars, profile)

        solver = KinematicSolver2D()
        solver.setup_grid(center=grid_center, width=grid_width, n_bins=n_bins)
        solver.add_data(obs_x, obs_y, cov)
        samples = solver.run(
            num_warmup=300, num_samples=600, seed=20260805 + i, prior=prior
        )

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
        for m in HARD_METRICS + SOFT_METRICS:
            median = float(np.median(draws[m]))
            half68 = 0.5 * (np.percentile(draws[m], 84) - np.percentile(draws[m], 16))
            medians[m].append(median)
            if abs(median - true_moments[m]) <= half68:
                hits[m] += 1

    band_lo, band_hi = _binom_band(N_REAL, p=0.68, alpha=0.01)

    report_lines = [
        f"=== {truth_name}/{profile_name}/{prior} (n_real={N_REAL}, "
        f"N_STARS={profile.n_stars}, n_bins={n_bins}) ==="
    ]
    hard_failures = []
    for m in HARD_METRICS + SOFT_METRICS:
        cov = hits[m] / N_REAL
        report_lines.append(
            f"  {m:10s}: coverage={cov:.3f} ({hits[m]}/{N_REAL})  "
            f"mean_median={np.mean(medians[m]):.3f}  truth={true_moments[m]:.3f}"
        )
        if m in HARD_METRICS and not (band_lo <= cov <= band_hi):
            hard_failures.append(
                f"{truth_name}/{profile_name}/{prior}.{m}: coverage {cov:.3f} "
                f"outside nominal binomial band [{band_lo:.3f}, {band_hi:.3f}] "
                f"for n={N_REAL} at p=0.68"
            )
    report = "\n".join(report_lines)

    if hard_failures:
        pytest.fail(report + "\n\nHARD FAILURES:\n" + "\n".join(hard_failures))
    print(report)


# ---------------------------------------------------------------------------
# Per-cell coverage
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("prior", ["gmrf", "gaussian_core"])
@pytest.mark.parametrize("profile_name", ["hst_bright", "hst_faint", "gaia_outer"])
@pytest.mark.parametrize("truth_name", list(TRUTHS))
def test_per_cell_losvd_coverage_2d(truth_name, profile_name, prior, request):
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
    for mark in _maybe_xfail(prior, profile_name):
        request.node.add_marker(mark)

    profile = PROFILES_2D[profile_name]
    truth = truths_for(profile.sigma_ref)[truth_name]
    grid_width = (profile.grid_width, profile.grid_width)
    grid_center = (0.0, 0.0)
    n_bins = profile.n_bins

    solver0 = KinematicSolver2D()
    solver0.setup_grid(center=grid_center, width=grid_width, n_bins=n_bins)
    centers_2d = solver0.grid["centers_2d"]
    edges_x = solver0.grid["edges_x"]
    edges_y = solver0.grid["edges_y"]

    _, truth_mass = _discretised_truth_moments(truth, edges_x, edges_y, centers_2d)
    hits = np.zeros(len(truth_mass))

    for i in range(N_REAL):
        rng = np.random.default_rng(20260806 + i)
        obs_x, obs_y, cov = _draw_stars(rng, truth, profile.n_stars, profile)

        solver = KinematicSolver2D()
        solver.setup_grid(center=grid_center, width=grid_width, n_bins=n_bins)
        solver.add_data(obs_x, obs_y, cov)
        samples = solver.run(
            num_warmup=300, num_samples=600, seed=20260806 + i, prior=prior
        )

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
        f"{truth_name}/{profile_name}/{prior}: mean per-cell coverage over "
        f"{int(informative.sum())} informative cells (of {len(truth_mass)}) = "
        f"{mean_cov:.3f}"
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
        f"{truth_name}/{profile_name}/{prior}: {n_bad} informative cells have "
        f"coverage below 0.30 (worst {coverage[informative].min():.2f}); those "
        "cells carry essentially meaningless error bars."
    )
