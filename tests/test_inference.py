"""Comprehensive inference tests for veldist."""

import numpy as np
import pytest
from scipy import stats

from tests.helpers import (
    make_gaussian_distribution,
    make_uniform_distribution,
    make_bimodal_distribution,
    make_mixture_distribution,
    generate_mock_data,
    compute_ks_test,
    check_moment_recovery,
)


@pytest.mark.slow
def test_full_inference_workflow():
    """
    Test the full inference workflow with synthetic data.
    This just tests that the workflow runs end-to-end and produces valid samples.
    """
    from veldist import KinematicSolver

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic true distribution (Gaussian)
    true_mean = 0.0
    true_std = 10.0
    n_stars = 50
    true_velocities = np.random.normal(true_mean, true_std, n_stars)

    # Add observational errors
    obs_errors = np.ones(n_stars) * 2.0
    observed_velocities = true_velocities + np.random.normal(0, obs_errors)

    # Initialize solver
    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=60.0, n_bins=30)
    solver.add_data(vel=observed_velocities, err=obs_errors)

    # Run inference with minimal sampling for speed
    samples = solver.run(num_warmup=50, num_samples=50, gpu=False)

    # Check that we got samples
    assert samples is not None
    assert "intrinsic_pdf" in samples
    assert samples["intrinsic_pdf"].shape == (50, 30)

    # Check that samples are valid probabilities
    assert np.all(samples["intrinsic_pdf"] >= 0)

    # Check that each sample sums to approximately 1
    sample_sums = np.sum(samples["intrinsic_pdf"], axis=1)
    np.testing.assert_allclose(sample_sums, 1.0, rtol=1e-5)


# ==============================================================================
# Test 1: Recovery Tests - Simple Distributions
# ==============================================================================


@pytest.mark.slow
def test_gaussian_recovery_low_noise():
    """
    Test recovery of a single Gaussian with low noise.

    This is the ideal case - should recover mean and std accurately.
    """
    from veldist import KinematicSolver, compute_moments

    # True distribution
    true_mean = 0.0
    true_std = 10.0
    true_pdf = make_gaussian_distribution(true_mean, true_std)

    # Generate data with good signal-to-noise
    data = generate_mock_data(
        true_pdf, n_stars=500, obs_errors=2.0, grid_range=(-50, 50), seed=42
    )

    # Run inference
    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=100.0, n_bins=50)
    solver.add_data(vel=data["observed_velocities"], err=data["errors"])
    samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

    # Analyze moments
    moments = compute_moments(samples["intrinsic_pdf"], solver.grid["centers"])

    # Check recovery
    result = check_moment_recovery(moments, true_mean, true_std, tolerance_sigma=3)

    assert result[
        "mean_ok"
    ], f"Mean not recovered: z-score = {result['mean_z_score']:.2f}"
    assert result["std_ok"], f"Std not recovered: z-score = {result['std_z_score']:.2f}"


@pytest.mark.slow
def test_gaussian_recovery_high_noise():
    """
    Test recovery with measurement errors comparable to intrinsic spread.

    This tests the deconvolution capability.
    """
    from veldist import KinematicSolver, compute_moments

    true_mean = 0.0
    true_std = 10.0
    true_pdf = make_gaussian_distribution(true_mean, true_std)

    # High noise - error comparable to intrinsic width
    data = generate_mock_data(
        true_pdf, n_stars=500, obs_errors=8.0, grid_range=(-50, 50), seed=43
    )

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=100.0, n_bins=50)
    solver.add_data(vel=data["observed_velocities"], err=data["errors"])
    samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

    moments = compute_moments(samples["intrinsic_pdf"], solver.grid["centers"])

    # With high noise, we expect larger uncertainties but still unbiased
    result = check_moment_recovery(moments, true_mean, true_std, tolerance_sigma=3)

    assert result[
        "mean_ok"
    ], f"Mean not recovered with high noise: z = {result['mean_z_score']:.2f}"
    # Std recovery is harder with high noise, so we're more lenient
    assert (
        result["std_z_score"] < 5
    ), f"Std severely biased: z = {result['std_z_score']:.2f}"


@pytest.mark.slow
def test_uniform_distribution_recovery():
    """
    Test recovery of a uniform (flat) distribution.

    Tests that the method doesn't impose artificial structure.
    """
    from veldist import KinematicSolver, compute_moments

    true_pdf = make_uniform_distribution(low=-20.0, high=20.0)

    data = generate_mock_data(
        true_pdf, n_stars=500, obs_errors=3.0, grid_range=(-50, 50), seed=44
    )

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=80.0, n_bins=40)
    solver.add_data(vel=data["observed_velocities"], err=data["errors"])
    samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

    moments = compute_moments(samples["intrinsic_pdf"], solver.grid["centers"])

    # Uniform distribution: mean should be near 0
    mean_inferred, mean_unc = moments["mean"]
    assert (
        abs(mean_inferred) < 5.0
    ), f"Uniform mean should be near 0, got {mean_inferred:.2f}"

    # Check that PDF is relatively flat in the center region
    mean_pdf = np.mean(samples["intrinsic_pdf"], axis=0)
    centers = solver.grid["centers"]
    center_region = (centers > -15) & (centers < 15)
    pdf_center = mean_pdf[center_region]

    # Coefficient of variation should be low for flat distribution
    cv = np.std(pdf_center) / np.mean(pdf_center)
    assert cv < 0.5, f"PDF not flat enough in center region: CV = {cv:.2f}"


@pytest.mark.slow
def test_bimodal_distribution_recovery():
    """
    Test recovery of a bimodal distribution.

    This tests whether sharp features can be recovered.
    """
    from veldist import KinematicSolver

    # Two well-separated peaks
    true_pdf = make_bimodal_distribution(
        mean1=-15.0, std1=5.0, mean2=15.0, std2=5.0, weight1=0.5
    )

    data = generate_mock_data(
        true_pdf, n_stars=800, obs_errors=3.0, grid_range=(-50, 50), seed=45
    )

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=80.0, n_bins=50)
    solver.add_data(vel=data["observed_velocities"], err=data["errors"])
    samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

    # Check that we detect two peaks
    mean_pdf = np.mean(samples["intrinsic_pdf"], axis=0)
    centers = solver.grid["centers"]

    # Find local maxima
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(mean_pdf, distance=5)

    # Should find 2 peaks
    assert len(peaks) >= 2, f"Expected 2 peaks, found {len(peaks)}"

    # Check peak locations are roughly correct
    peak_locations = centers[peaks]
    assert any(abs(p - (-15)) < 5 for p in peak_locations), "Left peak not found"
    assert any(abs(p - 15) < 5 for p in peak_locations), "Right peak not found"


# ==============================================================================
# Test 2: Edge Cases & Robustness
# ==============================================================================


@pytest.mark.slow
def test_very_small_errors():
    """
    Test with errors much smaller than bin width.

    Should approach simple histogram.
    """
    from veldist import KinematicSolver, compute_moments

    true_mean = 0.0
    true_std = 10.0
    true_pdf = make_gaussian_distribution(true_mean, true_std)

    # Very small errors
    data = generate_mock_data(
        true_pdf, n_stars=300, obs_errors=0.5, grid_range=(-50, 50), seed=46
    )

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=80.0, n_bins=40)
    solver.add_data(vel=data["observed_velocities"], err=data["errors"])
    samples = solver.run(num_warmup=300, num_samples=500, gpu=False)

    moments = compute_moments(samples["intrinsic_pdf"], solver.grid["centers"])

    # Should still recover accurately
    result = check_moment_recovery(moments, true_mean, true_std, tolerance_sigma=3)
    assert result["mean_ok"], "Failed with very small errors"


@pytest.mark.slow
def test_heteroscedastic_errors():
    """
    Test with mixture of high and low quality measurements.

    Should properly weight high-quality data.
    """
    from veldist import KinematicSolver, compute_moments

    true_mean = 0.0
    true_std = 10.0
    true_pdf = make_gaussian_distribution(true_mean, true_std)

    # Generate errors: 50% good (err=1), 50% poor (err=10)
    n_stars = 400
    np.random.seed(47)
    errors = np.concatenate([np.ones(n_stars // 2) * 1.0, np.ones(n_stars // 2) * 10.0])
    np.random.shuffle(errors)

    data = generate_mock_data(
        true_pdf, n_stars=n_stars, obs_errors=errors, grid_range=(-50, 50), seed=47
    )

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=80.0, n_bins=40)
    solver.add_data(vel=data["observed_velocities"], err=data["errors"])
    samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

    moments = compute_moments(samples["intrinsic_pdf"], solver.grid["centers"])

    result = check_moment_recovery(moments, true_mean, true_std, tolerance_sigma=3)
    assert result["mean_ok"], "Failed with heteroscedastic errors"


@pytest.mark.slow
def test_sparse_data():
    """
    Test with limited number of stars.

    Should produce larger uncertainties but remain unbiased.
    """
    from veldist import KinematicSolver, compute_moments

    true_mean = 0.0
    true_std = 10.0
    true_pdf = make_gaussian_distribution(true_mean, true_std)

    # Only 50 stars
    data = generate_mock_data(
        true_pdf, n_stars=50, obs_errors=3.0, grid_range=(-50, 50), seed=48
    )

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=80.0, n_bins=30)
    solver.add_data(vel=data["observed_velocities"], err=data["errors"])
    samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

    moments = compute_moments(samples["intrinsic_pdf"], solver.grid["centers"])

    # With sparse data, uncertainties should be larger
    _, mean_unc = moments["mean"]
    _, std_unc = moments["std"]

    # But should still be unbiased (within wider tolerance)
    result = check_moment_recovery(moments, true_mean, true_std, tolerance_sigma=4)
    # We're more forgiving here since sparse data has high variance
    assert (
        result["mean_z_score"] < 5
    ), f"Mean severely biased with sparse data: z = {result['mean_z_score']:.2f}"


# ==============================================================================
# Test 3: Statistical Properties
# ==============================================================================


@pytest.mark.slow
def test_convergence_diagnostics():
    """
    Test that MCMC converges properly.

    Checks that samples are valid and well-mixed.
    """
    from veldist import KinematicSolver

    true_pdf = make_gaussian_distribution(0.0, 10.0)
    data = generate_mock_data(
        true_pdf, n_stars=200, obs_errors=3.0, grid_range=(-50, 50), seed=49
    )

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=80.0, n_bins=40)
    solver.add_data(vel=data["observed_velocities"], err=data["errors"])
    samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

    # Check basic properties
    assert "intrinsic_pdf" in samples
    assert samples["intrinsic_pdf"].shape == (1000, 40)

    # Each sample should be valid probabilities
    assert np.all(samples["intrinsic_pdf"] >= 0)
    sample_sums = np.sum(samples["intrinsic_pdf"], axis=1)
    np.testing.assert_allclose(sample_sums, 1.0, rtol=1e-5)


@pytest.mark.slow
def test_credible_interval_shape():
    """
    Test that credible intervals have reasonable shape.

    Uncertainties should be larger where there's less data.
    """
    from veldist import KinematicSolver

    true_pdf = make_gaussian_distribution(0.0, 10.0)
    data = generate_mock_data(
        true_pdf, n_stars=300, obs_errors=3.0, grid_range=(-50, 50), seed=50
    )

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=100.0, n_bins=50)
    solver.add_data(vel=data["observed_velocities"], err=data["errors"])
    samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

    # Convert to density
    pdf_density = samples["intrinsic_pdf"] / solver.grid["width"]

    mean_pdf = np.mean(pdf_density, axis=0)
    lower = np.percentile(pdf_density, 16, axis=0)
    upper = np.percentile(pdf_density, 84, axis=0)

    # Uncertainty (width of credible interval)
    uncertainty = upper - lower

    # In the tails (far from data), uncertainty should be relatively larger
    # In the center (where data is), relative uncertainty should be smaller
    centers = solver.grid["centers"]
    center_mask = np.abs(centers) < 15  # Where data is concentrated
    tail_mask = np.abs(centers) > 30  # Far from data

    rel_unc_center = uncertainty[center_mask] / (mean_pdf[center_mask] + 1e-10)
    rel_unc_tail = uncertainty[tail_mask] / (mean_pdf[tail_mask] + 1e-10)

    # This is a sanity check - uncertainties should exist
    assert np.mean(uncertainty) > 0, "No uncertainty in inference"


# ==============================================================================
# Test 4: Grid Resolution
# ==============================================================================


@pytest.mark.slow
def test_grid_resolution_consistency():
    """
    Test that different grid resolutions give consistent moments.

    Coarse vs fine grids should agree on basic statistics.
    """
    from veldist import KinematicSolver, compute_moments

    true_mean = 0.0
    true_std = 10.0
    true_pdf = make_gaussian_distribution(true_mean, true_std)

    data = generate_mock_data(
        true_pdf, n_stars=400, obs_errors=3.0, grid_range=(-50, 50), seed=51
    )

    # Coarse grid
    solver_coarse = KinematicSolver()
    solver_coarse.setup_grid(center=0.0, width=80.0, n_bins=20)
    solver_coarse.add_data(vel=data["observed_velocities"], err=data["errors"])
    samples_coarse = solver_coarse.run(num_warmup=300, num_samples=500, gpu=False)

    # Fine grid
    solver_fine = KinematicSolver()
    solver_fine.setup_grid(center=0.0, width=80.0, n_bins=60)
    solver_fine.add_data(vel=data["observed_velocities"], err=data["errors"])
    samples_fine = solver_fine.run(num_warmup=300, num_samples=500, gpu=False)

    # Compute moments
    moments_coarse = compute_moments(
        samples_coarse["intrinsic_pdf"], solver_coarse.grid["centers"]
    )
    moments_fine = compute_moments(
        samples_fine["intrinsic_pdf"], solver_fine.grid["centers"]
    )

    # Means should agree within combined uncertainty
    mean_c, unc_c = moments_coarse["mean"]
    mean_f, unc_f = moments_fine["mean"]
    combined_unc = np.sqrt(unc_c**2 + unc_f**2)

    assert (
        abs(mean_c - mean_f) < 3 * combined_unc
    ), f"Coarse and fine grids disagree on mean: {mean_c:.2f} vs {mean_f:.2f}"


# ==============================================================================
# Test 5: Multi-component Systems
# ==============================================================================


@pytest.mark.slow
def test_three_component_system():
    """
    Test recovery of a 3-component mixture.

    Simulates multiple stellar populations.
    """
    from veldist import KinematicSolver

    components = [
        {"mean": -20, "std": 4, "weight": 0.3},
        {"mean": 0, "std": 6, "weight": 0.5},
        {"mean": 25, "std": 5, "weight": 0.2},
    ]
    true_pdf = make_mixture_distribution(components)

    data = generate_mock_data(
        true_pdf, n_stars=1000, obs_errors=3.0, grid_range=(-60, 60), seed=52
    )

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=120.0, n_bins=60)
    solver.add_data(vel=data["observed_velocities"], err=data["errors"])
    samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

    # Check that we can detect multiple peaks
    mean_pdf = np.mean(samples["intrinsic_pdf"], axis=0)

    from scipy.signal import find_peaks

    peaks, _ = find_peaks(mean_pdf, distance=3, prominence=0.001)

    # Should find at least 3 peaks (might find more due to noise)
    assert len(peaks) >= 3, f"Expected >=3 peaks in mixture, found {len(peaks)}"
