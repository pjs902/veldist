"""Basic tests for veldist package."""

import numpy as np
import pytest


def test_import():
    """Test that the package can be imported."""
    import veldist

    assert veldist is not None


def test_kinematic_solver_init():
    """Test that KinematicSolver can be instantiated."""
    from veldist import KinematicSolver

    solver = KinematicSolver()
    assert solver is not None
    assert solver.matrix is None
    assert solver.samples is None


def test_setup_grid():
    """Test grid setup."""
    from veldist import KinematicSolver

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=100.0, n_bins=50)

    assert solver.grid is not None
    assert solver.grid["n_bins"] == 50
    assert len(solver.grid["centers"]) == 50
    assert len(solver.grid["edges"]) == 51


def test_precompute_design_matrix():
    """Test design matrix computation."""
    from veldist import precompute_design_matrix

    obs_val = np.array([0.0, 1.0, 2.0])
    obs_err = np.array([0.5, 0.5, 0.5])
    bin_centers = np.linspace(-5, 5, 20)

    matrix = precompute_design_matrix(obs_val, obs_err, bin_centers)

    assert matrix.shape == (3, 20)
    # Each row should sum to approximately 1 (probability distribution)
    row_sums = np.sum(matrix, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-2)


def test_add_data():
    """Test adding data to solver."""
    from veldist import KinematicSolver

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=100.0, n_bins=50)

    vel = np.random.randn(100)
    err = np.abs(np.random.randn(100)) * 0.5 + 0.1

    solver.add_data(vel, err)

    assert solver.matrix is not None
    assert solver.matrix.shape == (100, 50)


def test_add_data_without_grid():
    """Test that adding data without grid setup raises error."""
    from veldist import KinematicSolver

    solver = KinematicSolver()
    vel = np.random.randn(10)
    err = np.ones(10) * 0.5

    with pytest.raises(ValueError, match="Run setup_grid"):
        solver.add_data(vel, err)


def test_run_without_data():
    """Test that running without data raises error."""
    from veldist import KinematicSolver

    solver = KinematicSolver()

    with pytest.raises(ValueError, match="No data added"):
        solver.run(num_warmup=10, num_samples=10)
