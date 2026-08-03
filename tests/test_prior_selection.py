"""The prior= selector on KinematicSolver.run."""

import numpy as np
import pytest

from veldist.veldist import KinematicSolver


def _fitted_solver(prior, n_bins=20, seed=3):
    rng = np.random.default_rng(0)
    vel = rng.normal(200.0, 40.0, 120)
    err = np.full(120, 8.0)
    solver = KinematicSolver()
    solver.setup_grid(center=200.0, width=400.0, n_bins=n_bins)
    solver.add_data(vel, err)
    solver.run(num_warmup=100, num_samples=100, seed=seed, prior=prior)
    return solver


@pytest.mark.slow
@pytest.mark.parametrize("prior", ["rw1", "gaussian_core"])
def test_run_accepts_both_priors(prior):
    solver = _fitted_solver(prior)
    pdf = np.asarray(solver.samples["intrinsic_pdf"])
    assert pdf.shape == (100, 20)
    np.testing.assert_allclose(pdf.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.slow
def test_gaussian_core_run_exposes_its_own_sites():
    solver = _fitted_solver("gaussian_core")
    for site in ("v0", "s0", "sigma3", "d3"):
        assert site in solver.samples, f"missing sample site {site!r}"
    assert "steps" not in solver.samples, "RW1 site leaked into gaussian_core run"


def test_run_rejects_unknown_prior():
    rng = np.random.default_rng(0)
    solver = KinematicSolver()
    solver.setup_grid(center=200.0, width=400.0, n_bins=20)
    solver.add_data(rng.normal(200.0, 40.0, 50), np.full(50, 8.0))
    with pytest.raises(ValueError, match="prior"):
        solver.run(num_warmup=10, num_samples=10, prior="nonsense")
