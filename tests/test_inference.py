"""Integration test for full inference workflow."""

import numpy as np
import pytest


@pytest.mark.slow
def test_full_inference_workflow():
    """Test the full inference workflow with synthetic data."""
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
