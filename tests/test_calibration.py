"""
Simulation-Based Calibration (SBC) for the veldist NUTS model.

Implements the correctness test described in PLAN.md §1.2. SBC exploits the
identity: if theta_tilde ~ p(theta) and y ~ p(y|theta_tilde) and
theta^(1..L) ~ p(theta|y), then the rank of theta_tilde among the L posterior
draws is Uniform{0..L}. Any systematic deviation from uniformity indicates a
bug in the model or sampler (not merely "the fit looks a bit off").

Fallback ladder taken: rung 1 (plain Python loop over n_sims, run once
locally, not vectorised via chain_method="vectorized"). Vectorising the
`matrix` argument across simulations was judged not worth the implementation
risk/effort for this harness -- see PLAN.md's explicit permission to use this
fallback. n_sims=30, marked `slow`.

Gotchas from the plan actively guarded against here:
- The prior draw MUST come from `Predictive(model, num_samples=1)` with no
  conditioning -- never hand-roll dist.HalfNormal(...).sample() etc.
- Posterior draws must be thinned to ~independent draws using ESS, not just
  truncated to some fixed L.
- `bimodality_score` is excluded from ranking (integer, heavy ties).
- Failed/degenerate simulations are counted, not silently dropped; if more
  than ~2% fail we stop and report rather than quietly discard them.
"""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import pytest
from numpyro.infer import MCMC, NUTS, Predictive
from scipy import stats

from veldist.veldist import model, precompute_design_matrix
from veldist.analysis import tail_weight as _tail_weight

# ---------------------------------------------------------------------------
# Configuration (small, per PLAN.md cost-mitigation guidance)
# ---------------------------------------------------------------------------
N_BINS = 15
N_STARS = 200
NUM_WARMUP = 500
NUM_SAMPLES = 1200
N_SIMS = 30
GRID_WIDTH = 300.0  # arbitrary physical velocity span for the mock grid
GRID_CENTER = 0.0

BIN_WIDTH = GRID_WIDTH / N_BINS
BIN_EDGES = np.linspace(
    GRID_CENTER - GRID_WIDTH / 2, GRID_CENTER + GRID_WIDTH / 2, N_BINS + 1
)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

# Quantities to rank. bimodality_score is deliberately excluded (see module
# docstring / plan gotcha #4).
QUANTITY_NAMES = [
    "v_mean",
    "sigma",
    "skewness",
    "kurtosis",
    "tail_weight",
    "smoothness_sigma",
]

MAX_FAILURE_FRACTION = 0.02


def _moments_from_pdf_samples(pdf_samples, grid_centers):
    """Per-sample v_mean, sigma, skewness, kurtosis, tail_weight.

    Mirrors the vectorised computation in veldist.analysis.compute_summary,
    but returns raw per-sample arrays (not median/CI-summarised) since SBC
    needs individual posterior draws to rank against, not a summary.
    """
    pdf_samples = np.asarray(pdf_samples, dtype=float)
    grid_centers = np.asarray(grid_centers, dtype=float)

    means = pdf_samples @ grid_centers
    delta = grid_centers[None, :] - means[:, None]
    variance = np.einsum("ij,ij->i", pdf_samples, delta**2)
    stds = np.sqrt(variance)
    safe_stds = np.where(stds > 0, stds, 1.0)

    skews = np.einsum("ij,ij->i", pdf_samples, delta**3) / safe_stds**3
    skews = np.where(stds > 0, skews, 0.0)

    kurts = np.einsum("ij,ij->i", pdf_samples, delta**4) / safe_stds**4 - 3.0
    kurts = np.where(stds > 0, kurts, 0.0)

    tw = _tail_weight(pdf_samples, grid_centers, means, stds)

    return means, stds, skews, kurts, tw


def _draw_prior(rng_key, n_stars_dummy):
    """Draw a single prior sample of theta_tilde from the model's own prior.

    Uses numpyro.infer.Predictive with NO conditioning -- this is the
    mechanism PLAN.md insists on (gotcha #1) to guarantee the prior draw
    matches exactly what inference assumes, rather than a hand-rolled mirror
    that can silently drift out of sync with the model.

    The `matrix` argument is structurally required by `model()` but its
    *values* only enter the likelihood factor, which Predictive does not
    evaluate when generating prior samples (no `obs=` conditioning site
    exists in this model; the likelihood is a `numpyro.factor`, which
    Predictive does not need real data to trace through -- it only needs the
    shape of `matrix` for `jnp.dot(matrix, intrinsic_pdf)` inside the
    (untouched) log_prob computation, which is irrelevant to *sampling* `x`
    and `smoothness_sigma`). We pass a dummy zeros matrix of the right shape.
    """
    dummy_matrix = jnp.zeros((n_stars_dummy, N_BINS))
    predictive = Predictive(model, num_samples=1)
    prior_sample = predictive(rng_key, matrix=dummy_matrix, n_bins=N_BINS, bin_width=BIN_WIDTH)
    return prior_sample


def _simulate_observations(rng, intrinsic_pdf, n_stars):
    """Sample n_stars true velocities from intrinsic_pdf, add Gaussian noise."""
    bin_idx = rng.choice(N_BINS, size=n_stars, p=intrinsic_pdf)
    lo = BIN_EDGES[bin_idx]
    hi = BIN_EDGES[bin_idx + 1]
    true_vel = rng.uniform(lo, hi)

    # Plausible heteroscedastic errors: e.g. 3-15 km/s per star.
    errs = rng.uniform(3.0, 15.0, size=n_stars)
    obs_vel = true_vel + rng.normal(0.0, errs)
    return obs_vel, errs


def _run_one_sbc_iteration(sim_idx, base_seed=20260802):
    """Run a single SBC simulation. Returns dict of ranks, or None on failure."""
    key = jax.random.PRNGKey(base_seed + sim_idx)
    prior_key, mcmc_key = jax.random.split(key)

    prior_sample = _draw_prior(prior_key, N_STARS)
    true_intrinsic_pdf = np.asarray(prior_sample["intrinsic_pdf"][0])  # (N_BINS,)
    true_smoothness_sigma = float(np.asarray(prior_sample["smoothness_sigma"][0]))

    if not np.all(np.isfinite(true_intrinsic_pdf)) or not np.isfinite(
        true_smoothness_sigma
    ):
        return None

    rng = np.random.default_rng(base_seed + 1000 + sim_idx)
    obs_vel, obs_err = _simulate_observations(rng, true_intrinsic_pdf, N_STARS)

    matrix = precompute_design_matrix(obs_vel, obs_err, BIN_CENTERS, bin_width=BIN_WIDTH)
    if not np.all(np.isfinite(np.asarray(matrix))):
        return None

    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        progress_bar=False,
    )
    try:
        mcmc.run(mcmc_key, matrix=matrix, n_bins=N_BINS, bin_width=BIN_WIDTH)
    except Exception:
        return None

    samples = mcmc.get_samples()
    pdf_samples = np.asarray(samples["intrinsic_pdf"])
    smoothness_samples = np.asarray(samples["smoothness_sigma"])

    if not np.all(np.isfinite(pdf_samples)) or not np.all(
        np.isfinite(smoothness_samples)
    ):
        return None

    # --- ESS check + thinning (gotcha #2: autocorrelated draws -> spurious
    # U-shape). Compute ESS on smoothness_sigma (scalar site; cheap and a
    # decent proxy for overall chain mixing) and thin to at most ESS.
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    ess = float(
        np.asarray(
            numpyro.diagnostics.effective_sample_size(
                np.asarray(samples_by_chain["smoothness_sigma"])
            )
        )
    )
    n_draws = pdf_samples.shape[0]
    l_draws = int(min(n_draws, max(ess, 1)))
    if l_draws < 20:
        # Too few independent draws to say anything meaningful about ranks;
        # treat as a failed simulation rather than ranking noise.
        return None

    thin = max(1, n_draws // l_draws)
    idx = np.arange(0, n_draws, thin)[:l_draws]
    l_draws = len(idx)

    pdf_thin = pdf_samples[idx]
    smoothness_thin = smoothness_samples[idx]

    means, stds, skews, kurts, tw = _moments_from_pdf_samples(pdf_thin, BIN_CENTERS)

    true_means, true_stds, true_skews, true_kurts, true_tw = _moments_from_pdf_samples(
        true_intrinsic_pdf[None, :], BIN_CENTERS
    )
    true_v_mean = float(true_means[0])
    true_sigma = float(true_stds[0])
    true_skew = float(true_skews[0])
    true_kurt = float(true_kurts[0])
    true_tail_weight = float(true_tw[0])

    def _rank(true_val, draws):
        return int(np.sum(draws < true_val))

    ranks = {
        "v_mean": (_rank(true_v_mean, means), l_draws),
        "sigma": (_rank(true_sigma, stds), l_draws),
        "skewness": (_rank(true_skew, skews), l_draws),
        "kurtosis": (_rank(true_kurt, kurts), l_draws),
        "tail_weight": (_rank(true_tail_weight, tw), l_draws),
        "smoothness_sigma": (
            _rank(true_smoothness_sigma, smoothness_thin),
            l_draws,
        ),
    }
    return ranks


@pytest.mark.slow
def test_sbc_calibration():
    """
    Simulation-Based Calibration of the veldist NUTS model.

    Runs N_SIMS independent prior-draw -> simulate -> refit -> rank cycles
    and checks each test quantity's rank distribution against Uniform{0..L}
    via a Bonferroni-adjusted KS test, per PLAN.md §1.2.

    Note on `log p(theta_tilde | y)`: the plan lists this as an optional
    catch-all quantity, explicitly the hardest to extract cleanly and
    explicitly permitted to be skipped. It is skipped here: extracting a
    single joint log-density comparable across posterior draws with
    numpyro's factor-based model (as opposed to a `log_prob` on a fixed
    parameter transform) would require re-deriving the potential energy at
    each thinned sample under the NUTS-internal unconstrained parameterisation,
    which adds meaningful harness complexity for a quantity the plan itself
    treats as non-essential given the other six are ranked.
    """
    rank_records = {q: [] for q in QUANTITY_NAMES}
    n_failed = 0

    for sim_idx in range(N_SIMS):
        result = _run_one_sbc_iteration(sim_idx)
        if result is None:
            n_failed += 1
            continue
        for q in QUANTITY_NAMES:
            rank, l_draws = result[q]
            rank_records[q].append((rank, l_draws))

    failure_fraction = n_failed / N_SIMS
    assert failure_fraction <= MAX_FAILURE_FRACTION, (
        f"{n_failed}/{N_SIMS} ({failure_fraction:.1%}) SBC simulations failed "
        "(NaN posterior, sampler exception, or inadequate ESS). This exceeds "
        "the 2% budget in PLAN.md; investigate the failure mode rather than "
        "dropping these simulations, as silently dropping them biases the "
        "test toward passing."
    )

    n_ok = N_SIMS - n_failed
    assert n_ok >= 10, f"Too few successful simulations ({n_ok}) to test calibration."

    alpha = 0.005 / len(QUANTITY_NAMES)

    results = {}
    failures = []
    for q in QUANTITY_NAMES:
        records = rank_records[q]
        # All successful sims should share the same L (ESS-driven, but stable
        # across sims of the same size/config); normalise rank/L per-record
        # in case L varies slightly sim-to-sim.
        normalized = np.array([r / n_draws for r, n_draws in records])
        ks_stat, p_value = stats.kstest(normalized, "uniform")
        results[q] = (ks_stat, p_value, len(records))
        if p_value <= alpha:
            failures.append(q)

    report_lines = [
        f"SBC results (n_sims={N_SIMS}, n_ok={n_ok}, n_failed={n_failed}, "
        f"alpha={alpha:.5f}):"
    ]
    for q in QUANTITY_NAMES:
        ks_stat, p_value, n = results[q]
        flag = "FAIL" if q in failures else "pass"
        report_lines.append(
            f"  [{flag}] {q}: KS={ks_stat:.4f}  p={p_value:.4f}  n={n}"
        )
    report = "\n".join(report_lines)

    assert not failures, (
        "SBC uniformity test failed for quantities: "
        f"{failures}.\n{report}\n"
        "This indicates a genuine miscalibration of the posterior (or a "
        "harness bug) -- do not loosen this tolerance to force a pass; "
        "diagnose the rank-histogram shape (U = overconfident, "
        "inverted-U = underconfident, ramp = biased) instead."
    )

    print(report)
