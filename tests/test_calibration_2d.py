"""
Simulation-Based Calibration (SBC) for the veldist 2D (bivariate) NUTS model.

Direct 2D analogue of ``test_calibration.py`` -- see that module's docstring
for the statistical methodology (rank-of-truth-among-posterior-draws test,
Bonferroni-corrected KS uniformity test, ESS-based thinning, counting rather
than dropping failed simulations). The mechanics carry over unchanged; only
the plumbing (2D model, 2D quantities, GMRF Cholesky factor) differs.

Per PLAN.md §3.3, uses K=10 (100 cells) for the SBC configuration -- small
enough to be affordable, large enough to exercise the same code paths as
production use.

Fallback ladder taken: rung 1, same as 1D (plain Python loop over n_sims,
not vectorised). n_sims=30, marked `slow`.

Gotchas actively guarded against here (mirroring the 1D module):
- The prior draw comes from `Predictive(model_2d, num_samples=1)` with no
  conditioning. `model_2d`'s latent parameterisation is already non-centred
  and fully generative (a real `z ~ Normal(0, I)` sample site followed by a
  deterministic `solve_triangular` transform), so -- unlike 1D's original
  broken factor-based version -- no fix was expected to be needed here. This
  was verified empirically (see `test_prior_predictive_is_smooth_2d` below)
  rather than assumed.
- Posterior draws are thinned to ~independent draws using ESS on
  `smoothness_sigma`, not just truncated to a fixed L.
- Failed/degenerate simulations are counted, not silently dropped.
"""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import pytest
from numpyro.infer import MCMC, NUTS, Predictive
from scipy import stats

from veldist.calibration2d import _moments_from_pdf_samples_2d
from veldist.veldist2d import (
    build_gmrf_precision,
    model_2d,
    model_gaussian_core_2d,
    precompute_design_matrix_2d,
    setup_grid_2d,
)

# ---------------------------------------------------------------------------
# Configuration (small, per PLAN.md §3.3 explicit K=10 guidance)
# ---------------------------------------------------------------------------
K = 10  # per-axis grid size -> 100 cells, per PLAN.md §3.3
N_STARS = 250
NUM_WARMUP = 500
NUM_SAMPLES = 1200
N_SIMS = 30
GRID_WIDTH = (40.0, 40.0)  # arbitrary physical span for the mock grid
GRID_CENTER = (0.0, 0.0)

GRID = setup_grid_2d(center=GRID_CENTER, width=GRID_WIDTH, n_bins=K)
N_CELLS = GRID["n_cells"]
CENTERS_2D = GRID["centers_2d"]
CENTERS_2D_JAX = jnp.asarray(CENTERS_2D)
EDGES_X = GRID["edges_x"]
EDGES_Y = GRID["edges_y"]

# GMRF Cholesky factor: a fixed structural input, built once (never drawn
# from the prior, never recomputed inside the loop) -- see model_2d's
# docstring and PLAN.md §3.2.
_Q, _Q_REG = build_gmrf_precision(K)
_L_NP = np.linalg.cholesky(_Q_REG)
if not np.all(np.isfinite(_L_NP)):
    msg = "Cholesky of ridge-regularised GMRF precision produced non-finite values."
    raise ValueError(msg)
L_JAX = jnp.asarray(_L_NP)

QUANTITY_NAMES = [
    "mean_x",
    "mean_y",
    "sigma_x",
    "sigma_y",
    "rho",
]

MAX_FAILURE_FRACTION = 0.02


def _draw_prior(rng_key, n_stars_dummy, model_fn=model_2d, extra_kwargs=None):
    """Draw a single prior sample of theta_tilde via Predictive, no conditioning.

    Mirrors `test_calibration.py::_draw_prior`: the `matrix` argument is
    structurally required by `model_2d` but its *values* only enter the
    likelihood factor, which Predictive does not evaluate when generating
    prior (ancestral) samples. A dummy zeros matrix of the right shape is
    passed for shape purposes only.
    """
    extra_kwargs = extra_kwargs or {}
    dummy_matrix = jnp.zeros((n_stars_dummy, N_CELLS))
    predictive = Predictive(model_fn, num_samples=1)
    prior_sample = predictive(
        rng_key, matrix=dummy_matrix, n_cells=N_CELLS, L=L_JAX, **extra_kwargs
    )
    return prior_sample


def test_prior_predictive_is_smooth_2d():
    """Sanity check (fast, not marked slow): Predictive draws of
    intrinsic_pdf from model_2d should look like smooth 2D GMRF draws (mass
    spread over many cells, not a near-one-hot spike on a single cell). This
    is the empirical verification, per the task, that the already-generative
    (z -> deterministic transform) parameterisation is in fact
    Predictive-compatible for 2D, exactly analogous to what the 1D fix was
    checked against -- rather than simply assuming it based on the code
    structure.
    """
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 20)
    max_masses = []
    for k in keys:
        prior_sample = _draw_prior(k, n_stars_dummy=5)
        pdf = np.asarray(prior_sample["intrinsic_pdf"][0])
        assert pdf.shape == (N_CELLS,)
        assert np.isfinite(pdf).all()
        np.testing.assert_allclose(pdf.sum(), 1.0, atol=1e-5)
        max_masses.append(pdf.max())

    max_masses = np.array(max_masses)
    # A near-one-hot spike (the broken-factor failure mode) would put
    # essentially all mass (~1.0) in a single cell. A smooth GMRF draw over
    # 100 cells should have much less concentrated mass than that.
    assert np.median(max_masses) < 0.2, (
        f"Prior predictive draws of intrinsic_pdf look spiky (median max cell "
        f"mass {np.median(max_masses):.3f}), suggesting Predictive is not "
        "correctly forward-sampling through the generative z -> x transform."
    )


def _simulate_observations(rng, intrinsic_pdf, n_stars):
    """Draw n_stars true (pm1, pm2) pairs from the flattened intrinsic_pdf
    (cell index -> uniform position within cell), add Gaussian noise with
    per-star diagonal covariance."""
    cell_idx = rng.choice(N_CELLS, size=n_stars, p=intrinsic_pdf)
    ix, iy = np.unravel_index(cell_idx, GRID["shape"])

    lo_x, hi_x = EDGES_X[ix], EDGES_X[ix + 1]
    lo_y, hi_y = EDGES_Y[iy], EDGES_Y[iy + 1]
    true_x = rng.uniform(lo_x, hi_x)
    true_y = rng.uniform(lo_y, hi_y)

    # Plausible per-star errors, diagonal covariance only (keeps the harness
    # simple -- correlated per-star errors aren't the point of this test).
    err_x = rng.uniform(0.5, 2.0, size=n_stars)
    err_y = rng.uniform(0.5, 2.0, size=n_stars)
    obs_x = true_x + rng.normal(0.0, err_x)
    obs_y = true_y + rng.normal(0.0, err_y)

    cov = np.zeros((n_stars, 2, 2))
    cov[:, 0, 0] = err_x**2
    cov[:, 1, 1] = err_y**2
    return obs_x, obs_y, cov


def _run_one_sbc_iteration(
    sim_idx,
    base_seed=20260803,
    model_fn=model_2d,
    extra_kwargs=None,
    hyper_name="smoothness_sigma",
):
    """Run a single 2D SBC simulation. Returns dict of ranks, or None on failure."""
    extra_kwargs = extra_kwargs or {}
    key = jax.random.PRNGKey(base_seed + sim_idx)
    prior_key, mcmc_key = jax.random.split(key)

    prior_sample = _draw_prior(
        prior_key, N_STARS, model_fn=model_fn, extra_kwargs=extra_kwargs
    )
    true_intrinsic_pdf = np.asarray(prior_sample["intrinsic_pdf"][0])  # (N_CELLS,)
    true_hyper = float(np.asarray(prior_sample[hyper_name][0]))

    if not np.all(np.isfinite(true_intrinsic_pdf)) or not np.isfinite(true_hyper):
        return None

    rng = np.random.default_rng(base_seed + 1000 + sim_idx)
    obs_x, obs_y, cov = _simulate_observations(rng, true_intrinsic_pdf, N_STARS)

    matrix = precompute_design_matrix_2d(obs_x, obs_y, cov, GRID)
    if not np.all(np.isfinite(np.asarray(matrix))):
        return None

    nuts_kernel = NUTS(model_fn)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        progress_bar=False,
    )
    try:
        mcmc.run(
            mcmc_key,
            matrix=jnp.asarray(matrix),
            n_cells=N_CELLS,
            L=L_JAX,
            **extra_kwargs,
        )
    except Exception:
        return None

    samples = mcmc.get_samples()
    pdf_samples = np.asarray(samples["intrinsic_pdf"])
    hyper_samples = np.asarray(samples[hyper_name])

    if not np.all(np.isfinite(pdf_samples)) or not np.all(np.isfinite(hyper_samples)):
        return None

    # --- ESS check + thinning, exactly mirroring test_calibration.py ---
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    ess = float(
        np.asarray(
            numpyro.diagnostics.effective_sample_size(
                np.asarray(samples_by_chain[hyper_name])
            )
        )
    )
    n_draws = pdf_samples.shape[0]
    l_draws = int(min(n_draws, max(ess, 1)))
    if l_draws < 20:
        return None

    thin = max(1, n_draws // l_draws)
    idx = np.arange(0, n_draws, thin)[:l_draws]
    l_draws = len(idx)

    pdf_thin = pdf_samples[idx]
    hyper_thin = hyper_samples[idx]

    mean_x, mean_y, sigma_x, sigma_y, rho = _moments_from_pdf_samples_2d(
        pdf_thin, CENTERS_2D
    )
    (
        true_mean_x,
        true_mean_y,
        true_sigma_x,
        true_sigma_y,
        true_rho,
    ) = _moments_from_pdf_samples_2d(true_intrinsic_pdf[None, :], CENTERS_2D)
    true_mean_x = float(true_mean_x[0])
    true_mean_y = float(true_mean_y[0])
    true_sigma_x = float(true_sigma_x[0])
    true_sigma_y = float(true_sigma_y[0])
    true_rho = float(true_rho[0])

    def _rank(true_val, draws):
        return int(np.sum(draws < true_val))

    ranks = {
        "mean_x": (_rank(true_mean_x, mean_x), l_draws),
        "mean_y": (_rank(true_mean_y, mean_y), l_draws),
        "sigma_x": (_rank(true_sigma_x, sigma_x), l_draws),
        "sigma_y": (_rank(true_sigma_y, sigma_y), l_draws),
        "rho": (_rank(true_rho, rho), l_draws),
        hyper_name: (
            _rank(true_hyper, hyper_thin),
            l_draws,
        ),
    }
    return ranks


@pytest.mark.slow
@pytest.mark.parametrize("prior", ["gmrf", "gaussian_core"])
def test_sbc_calibration_2d(prior):
    """
    Simulation-Based Calibration of the veldist 2D NUTS model.

    Runs N_SIMS independent prior-draw -> simulate -> refit -> rank cycles
    and checks each test quantity's rank distribution against Uniform{0..L}
    via a Bonferroni-adjusted KS test, per PLAN.md §3.3.

    `log p(theta_tilde | y)` is skipped for the same reason given in
    `test_calibration.py::test_sbc_calibration` -- extracting a single joint
    log-density comparable across posterior draws under NUTS's internal
    unconstrained parameterisation is non-essential harness complexity given
    the other six quantities are ranked.
    """
    if prior == "gaussian_core":
        model_fn = model_gaussian_core_2d
        extra_kwargs = {"centers_2d": CENTERS_2D_JAX}
        hyper_name = "sigma3"
        quantity_names = [*QUANTITY_NAMES, hyper_name]
    else:
        model_fn = model_2d
        extra_kwargs = {}
        hyper_name = "smoothness_sigma"
        quantity_names = [*QUANTITY_NAMES, hyper_name]

    rank_records = {q: [] for q in quantity_names}
    n_failed = 0

    for sim_idx in range(N_SIMS):
        result = _run_one_sbc_iteration(
            sim_idx,
            model_fn=model_fn,
            extra_kwargs=extra_kwargs,
            hyper_name=hyper_name,
        )
        if result is None:
            n_failed += 1
            continue
        for q in quantity_names:
            rank, l_draws = result[q]
            rank_records[q].append((rank, l_draws))

    failure_fraction = n_failed / N_SIMS
    assert failure_fraction <= MAX_FAILURE_FRACTION, (
        f"{n_failed}/{N_SIMS} ({failure_fraction:.1%}) SBC simulations failed "
        "(NaN posterior, sampler exception, or inadequate ESS). This exceeds "
        "the 2% budget in PLAN.md; investigate the failure mode (e.g. bump "
        "num_warmup/num_samples if it is a low-ESS issue, per the 1D "
        "precedent) rather than dropping these simulations, as silently "
        "dropping them biases the test toward passing."
    )

    n_ok = N_SIMS - n_failed
    assert n_ok >= 10, f"Too few successful simulations ({n_ok}) to test calibration."

    alpha = 0.005 / len(quantity_names)

    results = {}
    failures = []
    for q in quantity_names:
        records = rank_records[q]
        normalized = np.array([r / n_draws for r, n_draws in records])
        ks_stat, p_value = stats.kstest(normalized, "uniform")
        results[q] = (ks_stat, p_value, len(records))
        if p_value <= alpha:
            failures.append(q)

    report_lines = [
        f"2D SBC results (prior={prior}, K={K}, n_cells={N_CELLS}, n_sims={N_SIMS}, "
        f"n_ok={n_ok}, n_failed={n_failed}, alpha={alpha:.5f}):"
    ]
    for q in quantity_names:
        ks_stat, p_value, n = results[q]
        flag = "FAIL" if q in failures else "pass"
        report_lines.append(
            f"  [{flag}] {q}: KS={ks_stat:.4f}  p={p_value:.4f}  n={n}"
        )
    report = "\n".join(report_lines)

    assert not failures, (
        "2D SBC uniformity test failed for quantities: "
        f"{failures}.\n{report}\n"
        "This indicates a genuine miscalibration of the 2D posterior (or a "
        "harness bug) -- do not loosen this tolerance to force a pass; "
        "diagnose the rank-histogram shape (U = overconfident, "
        "inverted-U = underconfident, ramp = biased) instead."
    )

    print(report)
