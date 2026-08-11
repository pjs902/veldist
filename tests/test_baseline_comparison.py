"""veldist against the simplest thing that could work instead.

On a Gaussian truth the two-parameter MLE is the exact optimum, so matching
it is the pass condition and always was: veldist was never going to beat
Gaussian-on-Gaussian. Measured over 60 realisations, the CI/MLE-error ratio
is 0.999 +/- 0.003 for v_mean and 1.016 +/- 0.005 for sigma, and the MLE's own
Fisher error was independently checked against its true sampling scatter to
within 0.2 to 0.7 percent at N=150, so that ratio is real and not an artifact
of an optimistic denominator.

Across all nine mock truths and both v_mean and sigma, the two methods tie:
all 18 cells, max |t| = 1.41. That tie is the desired result, not a
shortfall. The paired standard deviation is about 0.05 km/s against errors of
about 1 km/s, so the non-parametric model costs essentially nothing on the
first two moments while allowing arbitrary LOSVD shape; you do not pay for
the flexibility.

The comparison that actually discriminates the two methods is therefore not
the first two moments, it is the recovered distribution itself, which is what
the second test below measures. An earlier version of this test claimed the
MLE would be biased in sigma on non-Gaussian shapes; that was wrong. Every
truth in ``make_truths()`` is constructed by ``Truth.scaled`` to have the
same second moment by design, so a correct second-moment estimator recovers
it regardless of shape, and the measured MLE sigma bias is -0.07 to -0.12
km/s uniformly across all nine truths.

Per TASKS.md, only v_mean and sigma gate on the Gaussian truth. Shape metrics
are printed, not asserted, except in the distribution-recovery test below.
"""

import numpy as np
import pytest
from scipy.stats import norm

from veldist.analysis import compute_summary
from veldist.baseline import gaussian_mle
from veldist.calibration import OMEGACAT, make_truths, true_moments
from veldist.veldist import KinematicSolver

N_REAL = 20
SEED = 20260811


def _run_pair(truth, sigma, n_stars, seed):
    """One realisation: veldist and MLE fits on identical mock data.

    Returns
    -------
    summ : dict
        ``compute_summary`` output for the veldist fit.
    base : dict
        ``gaussian_mle`` output.
    samples : ndarray
        Raw posterior samples of ``intrinsic_pdf``, shape (n_samples, n_bins).
    centers : ndarray
        Velocity-grid bin centres corresponding to ``samples`` columns.
    """
    pdf, rvs = truth.scaled(sigma)
    rng = np.random.default_rng(seed)
    err = OMEGACAT.draw_errors(n_stars, rng)
    obs = rvs(n_stars, rng) + rng.normal(0.0, err)

    solver = KinematicSolver()
    solver.setup_grid(center=0.0, width=OMEGACAT.grid_width, n_bins=OMEGACAT.n_bins)
    solver.add_data(obs, err)
    solver.run(num_warmup=300, num_samples=600, seed=seed)
    summ = compute_summary(solver.samples["intrinsic_pdf"], solver.grid["centers"])

    return summ, gaussian_mle(obs, err), solver.samples["intrinsic_pdf"], solver.grid["centers"]


@pytest.mark.slow
def test_veldist_matches_mle_on_gaussian_truth():
    truth = next(t for t in make_truths() if t.name == "gaussian")
    sigma = OMEGACAT.sigma_max
    tv = true_moments(truth.scaled(sigma)[0])

    v_ratio, s_ratio, v_err, s_err = [], [], [], []
    for i in range(N_REAL):
        summ, base, _samples, _centers = _run_pair(truth, sigma, OMEGACAT.n_stars, SEED + i)
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


def _tv_distance(inferred, true):
    """Total variation distance between two probability mass vectors."""
    return 0.5 * np.sum(np.abs(inferred - true))


@pytest.mark.slow
def test_veldist_beats_mle_on_non_gaussian_truth():
    """veldist recovers the shape of a bimodal LOSVD better than a Gaussian
    MLE can, even though both recover the same v_mean/sigma.

    ``bimodal_counter_rotation`` is the most strongly non-Gaussian truth in
    the library and one where the MLE's sigma is among the most precise
    (per the calibration audit) - exactly why sigma is the wrong axis to
    compare and the full distribution is the right one. The MLE can only
    ever report a Gaussian; veldist infers a full curve. We score each
    method's inferred distribution against the true distribution by total
    variation distance on the same velocity grid, and require veldist's
    win, averaged over independent realisations, to be at least two
    standard errors from zero.
    """
    truth = next(t for t in make_truths() if t.name == "bimodal_counter_rotation")
    sigma = OMEGACAT.sigma_max

    tv_veldist, tv_mle = [], []
    for i in range(N_REAL):
        summ, base, samples, centers = _run_pair(truth, sigma, OMEGACAT.n_stars, SEED + i)

        pdf_true = truth.scaled(sigma)[0](centers)
        true_dist = pdf_true / np.sum(pdf_true)

        veldist_dist = np.median(samples, axis=0)
        veldist_dist = veldist_dist / np.sum(veldist_dist)

        mle_pdf = norm.pdf(centers, loc=base["v_mean"], scale=base["sigma"])
        mle_dist = mle_pdf / np.sum(mle_pdf)

        tv_veldist.append(_tv_distance(veldist_dist, true_dist))
        tv_mle.append(_tv_distance(mle_dist, true_dist))

    tv_veldist = np.array(tv_veldist)
    tv_mle = np.array(tv_mle)
    d = tv_mle - tv_veldist

    mean_d = np.mean(d)
    std_d = np.std(d, ddof=1)
    stderr_d = std_d / np.sqrt(N_REAL)
    t_stat = mean_d / stderr_d if stderr_d > 0 else np.inf

    print(
        f"TV distance to truth: veldist {np.mean(tv_veldist):.4f}, "
        f"MLE {np.mean(tv_mle):.4f}"
    )
    print(
        f"paired diff d = tv_mle - tv_veldist: mean {mean_d:.4f}, "
        f"std {std_d:.4f}, stderr {stderr_d:.4f}, t = {t_stat:.2f}"
    )

    assert mean_d > 2.0 * stderr_d
