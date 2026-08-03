"""Point-estimate bias in recovered sigma and kurtosis.

Distinct from tests/test_coverage.py: that file asks whether the reported
68% credible interval contains the truth (calibration). This file asks
whether the posterior *median* is systematically offset (bias). Both matter
downstream -- Schwarzschild modelling reads sigma and h4 as point estimates,
so a +4% sigma bias propagates straight into enclosed mass and a spurious
positive h4 is read as radial anisotropy.

The headline symptom being guarded against is that the RW1 prior's sigma
bias GROWS WITH BIN COUNT, because more bins with the same number of stars
means more parameters over which the flat-null-space prior can spread mass
(the sparse-data bias of Cole, Chu & Greenland 2014, AJE 179, 252).

Measured baseline for the RW1 prior (Gaussian truth, sigma=40, 400 km/s
grid, N=150 stars, 5 realisations):

    n_bins   kurtosis bias   sigma bias
        10          +0.92         +0.8%
        20          +1.13         +2.0%
        40          +1.19         +3.6%
        80          +1.16         +4.1%
"""

import numpy as np
import pytest

from veldist.analysis import compute_summary
from veldist.veldist import KinematicSolver

GRID_CENTER = 200.0
GRID_WIDTH = 400.0
TRUE_SIGMA = 40.0
TRUE_EXCESS_KURTOSIS = 0.0  # Gaussian truth
N_STARS = 150
N_REAL = 10
NUM_WARMUP = 500
NUM_SAMPLES = 800
MEASUREMENT_ERROR = 8.0

# Thresholds. The Gaussian-core prior measured +0.00 kurtosis bias in
# prototyping; 0.35 leaves room for realisation noise at N_REAL=5 while
# still failing hard on the RW1 prior's +1.1.
MAX_ABS_KURTOSIS_BIAS = 0.35
# 3% on sigma. RW1 reaches 4.1% at n_bins=80 and keeps climbing.
MAX_ABS_SIGMA_FRAC_BIAS = 0.03


def _fit_moment_medians(prior, n_bins, n_real=N_REAL, seed=100):
    """Fit `n_real` Gaussian-truth mocks and return the per-realisation
    posterior-median sigma and excess kurtosis.

    Note `compute_summary` is called WITHOUT n_sigma_truncate: this suite
    measures the bias of the model itself, not of the post-hoc truncation
    repair. Truncation is a separate, lossy mitigation and using it here
    would mask exactly what we are trying to measure.
    """
    sigmas = np.empty(n_real)
    kurtoses = np.empty(n_real)

    for i in range(n_real):
        rng = np.random.default_rng(seed + i)
        true_v = rng.normal(GRID_CENTER, TRUE_SIGMA, N_STARS)
        err = np.full(N_STARS, MEASUREMENT_ERROR)
        obs_v = true_v + rng.normal(0.0, err)

        solver = KinematicSolver()
        solver.setup_grid(center=GRID_CENTER, width=GRID_WIDTH, n_bins=n_bins)
        solver.add_data(obs_v, err)
        solver.run(
            num_warmup=NUM_WARMUP,
            num_samples=NUM_SAMPLES,
            seed=seed + i,
            prior=prior,
        )
        summary = compute_summary(solver.samples["intrinsic_pdf"], solver.grid["centers"])
        sigmas[i] = summary["sigma"][0]
        kurtoses[i] = summary["kurtosis"][0]

    return sigmas, kurtoses


@pytest.mark.slow
@pytest.mark.parametrize("n_bins", [20, 80])
def test_gaussian_core_kurtosis_is_unbiased_on_a_gaussian_truth(n_bins):
    """A Gaussian truth must not produce spurious positive excess kurtosis.

    This is the single most important assertion in the suite: a spurious
    positive h4 is read downstream as radial anisotropy that is not there.
    """
    _, kurtoses = _fit_moment_medians("gaussian_core", n_bins)
    bias = float(np.median(kurtoses) - TRUE_EXCESS_KURTOSIS)
    assert abs(bias) < MAX_ABS_KURTOSIS_BIAS, (
        f"n_bins={n_bins}: median excess kurtosis bias {bias:+.2f} exceeds "
        f"{MAX_ABS_KURTOSIS_BIAS}. Per-realisation values: "
        f"{np.round(kurtoses, 2).tolist()}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("n_bins", [20, 80])
def test_gaussian_core_sigma_is_unbiased_on_a_gaussian_truth(n_bins):
    """Velocity dispersion must be recovered to within 3%."""
    sigmas, _ = _fit_moment_medians("gaussian_core", n_bins)
    frac_bias = float((np.median(sigmas) - TRUE_SIGMA) / TRUE_SIGMA)
    assert abs(frac_bias) < MAX_ABS_SIGMA_FRAC_BIAS, (
        f"n_bins={n_bins}: median sigma fractional bias {frac_bias:+.1%} "
        f"exceeds {MAX_ABS_SIGMA_FRAC_BIAS:.0%}. Per-realisation values: "
        f"{np.round(sigmas, 2).tolist()}"
    )


@pytest.mark.slow
def test_gaussian_core_sigma_bias_does_not_grow_with_bin_count():
    """The defining symptom of the flat-null-space bug.

    With the RW1 prior, sigma bias climbs monotonically with n_bins
    (+0.8% -> +4.1% from 10 to 80 bins) because more bins means more
    parameters for the uniform-limit prior to spread mass across, with no
    additional data to constrain them. A Gaussian-null-space prior has no
    such default and should stay flat.
    """
    biases = {}
    for n_bins in (20, 80):
        sigmas, _ = _fit_moment_medians("gaussian_core", n_bins)
        biases[n_bins] = float((np.median(sigmas) - TRUE_SIGMA) / TRUE_SIGMA)

    growth = biases[80] - biases[20]
    assert growth < 0.02, (
        f"sigma bias grew by {growth:+.1%} going from 20 to 80 bins "
        f"({biases[20]:+.1%} -> {biases[80]:+.1%}); the prior is still "
        "spreading mass into the extra bins"
    )


@pytest.mark.slow
def test_rw1_bias_is_reproduced_as_a_negative_control():
    """Documents the OLD prior's bias so the new tests are known to have
    teeth.

    If this ever stops failing to be biased -- i.e. if RW1 suddenly looks
    unbiased -- then either the harness broke or the RW1 prior changed, and
    the thresholds above need re-deriving rather than trusting.
    """
    sigmas, kurtoses = _fit_moment_medians("rw1", n_bins=80)
    kurt_bias = float(np.median(kurtoses))
    sigma_frac_bias = float((np.median(sigmas) - TRUE_SIGMA) / TRUE_SIGMA)

    assert kurt_bias > 0.5, (
        f"expected the RW1 prior to show a large positive kurtosis bias, "
        f"got {kurt_bias:+.2f} -- the harness may be broken"
    )
    assert sigma_frac_bias > 0.015, (
        f"expected the RW1 prior to show a positive sigma bias at n_bins=80, "
        f"got {sigma_frac_bias:+.1%} -- the harness may be broken"
    )
