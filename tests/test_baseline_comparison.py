"""veldist against the simplest thing that could work instead.

The comparison is deliberately asymmetric. On a Gaussian truth the two-
parameter MLE is the exact optimum, so veldist *cannot* beat it; matching it
is the pass condition, and losing badly would mean the flexibility is being
paid for in precision with nothing to show. On a non-Gaussian truth the MLE is
misspecified and veldist should win.

Per TASKS.md, only v_mean and sigma gate. Shape metrics are printed, not
asserted.
"""

import numpy as np
import pytest

from veldist.analysis import compute_summary
from veldist.baseline import gaussian_mle
from veldist.calibration import OMEGACAT, make_truths, true_moments
from veldist.veldist import KinematicSolver

N_REAL = 20
SEED = 20260811


def _run_pair(truth, sigma, n_stars, seed):
    """One realisation: veldist summary and the MLE fit on identical data."""
    pdf, rvs = truth.scaled(sigma)
    rng = np.random.default_rng(seed)
    err = OMEGACAT.draw_errors(n_stars, rng)
    obs = rvs(n_stars, rng) + rng.normal(0.0, err)

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=OMEGACAT.grid_width, n_bins=OMEGACAT.n_bins)
    solver.add_data(obs, err)
    solver.run(num_warmup=300, num_samples=600, seed=seed)
    summ = compute_summary(solver.samples["intrinsic_pdf"], solver.grid["centers"])

    return summ, gaussian_mle(obs, err)


@pytest.mark.slow
def test_veldist_matches_mle_on_gaussian_truth():
    truth = next(t for t in make_truths() if t.name == "gaussian")
    sigma = OMEGACAT.sigma_max
    tv = true_moments(truth.scaled(sigma)[0])

    v_ratio, s_ratio, v_err, s_err = [], [], [], []
    for i in range(N_REAL):
        summ, base = _run_pair(truth, sigma, OMEGACAT.n_stars, SEED + i)
        v_ratio.append(summ["v_mean"][1] / base["v_mean_err"])
        s_ratio.append(summ["sigma"][1] / base["sigma_err"])
        v_err.append(summ["v_mean"][0] - tv["v_mean"])
        s_err.append(summ["sigma"][0] - tv["sigma"])

    print(f"v_mean CI/MLE = {np.mean(v_ratio):.2f}, sigma CI/MLE = {np.mean(s_ratio):.2f}")

    # Not wildly less precise than the exact optimum.
    assert np.mean(v_ratio) < 1.30
    assert np.mean(s_ratio) < 1.40
    # And not biased: the mean residual is small against its own scatter.
    assert abs(np.mean(v_err)) < 2.0 * np.std(v_err) / np.sqrt(N_REAL) + 0.5
    assert abs(np.mean(s_err)) < 2.0 * np.std(s_err) / np.sqrt(N_REAL) + 0.5


@pytest.mark.slow
def test_veldist_beats_mle_on_non_gaussian_truth():
    """The MLE reports a Gaussian sigma for a heavy-tailed LOSVD, so it is
    biased where veldist is not. If veldist is not better here, the model
    complexity is not earning anything."""
    truth = next(t for t in make_truths() if t.name == "student_t_h4")
    sigma = OMEGACAT.sigma_max
    tv = true_moments(truth.scaled(sigma)[0])

    veldist_err, mle_err = [], []
    for i in range(N_REAL):
        summ, base = _run_pair(truth, sigma, OMEGACAT.n_stars, SEED + i)
        veldist_err.append(abs(summ["sigma"][0] - tv["sigma"]))
        mle_err.append(abs(base["sigma"] - tv["sigma"]))

    print(f"sigma |error|: veldist {np.mean(veldist_err):.3f}, MLE {np.mean(mle_err):.3f}")

    assert np.mean(veldist_err) <= np.mean(mle_err)
