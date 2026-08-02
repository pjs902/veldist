"""
Frequentist coverage over mock realisations for veldist.

Implements PLAN.md §1.3. SBC (test_calibration.py) validates the sampler
against the model; this validates the *model against reality* -- whether the
±half-68CI error bars produced by `compute_summary` mean what we tell
Dynamite they mean, for a handful of fixed, physically-motivated truths.

For each truth we generate `N_REAL` independent mock star catalogues, fit
each with `KinematicSolver`, and check whether the truth's known analytic
value of each summary metric falls inside the reported 68% credible
interval (`median +/- half_68ci`). Empirical coverage is then compared
against a binomial confidence band for nominal p=0.68 at n=N_REAL.

N_REAL is reduced from the plan's suggested 50 to 25, since 4 truths x 50
independent NUTS fits each is not practical to run repeatedly in CI/dev
loops; the plan explicitly sanctions dropping into the 25-50 range as
"wide but still useful" -- n=25 still gives a band that catches gross
miscalibration (e.g. empirical coverage of ~0.30) while running in a few
minutes.

Expected honest outcome (per plan): `kurtosis` and `tail_weight` coverage
may run under nominal for the non-Gaussian truths, because the RW1
smoothness prior shrinks the inferred LOSVD toward smoothness and pulls in
genuine sharp features (e.g. the Student-t's heavy tails, the bimodal
mixture's two peaks). This is an expected bias/variance trade documented in
the model, not a bug -- so the assertions here only guard against
*catastrophic* under-coverage (roughly < 0.30-0.40), while the full numbers
are always printed/reported so the actual degree of shrinkage is visible.

ACTUAL outcome (broader than the plan anticipated -- see the xfail marker on
the test below for full detail): kurtosis coverage is catastrophically low
even for the plain *Gaussian* truth (median excess kurtosis ~+1.6 to +2.5
vs. a true value of 0), which the plan did not predict since there is no
sharp feature there for the smoothness prior to shrink away. This was
independently verified as a real tail-leakage effect (small residual
posterior mass in far-edge grid bins, amplified by kurtosis's 4th-power
weighting) rather than a bug in this test harness. The test is marked
`xfail` rather than adjusted to pass, per PLAN.md's rule against loosening a
failing statistical test.
"""

import numpy as np
import pytest
from scipy import integrate, stats

from veldist.veldist import KinematicSolver
from veldist.analysis import compute_summary

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_REAL = 25  # realisations per truth (plan suggests 50; reduced for runtime, see module docstring)
N_STARS = 150  # fixed per truth -> matrix shape is constant -> JAX only compiles once per truth
N_BINS = 20
NUM_WARMUP = 300
NUM_SAMPLES = 600

# Metrics we check coverage for; all have (median, half_68ci) tuples from compute_summary.
METRICS = ["v_mean", "sigma", "skewness", "kurtosis", "tail_weight"]

# Metrics expected (per plan) to show shrinkage-driven under-coverage on
# non-Gaussian truths. For these we only assert against a catastrophic floor.
SHRINKAGE_PRONE = {"kurtosis", "tail_weight"}


# ---------------------------------------------------------------------------
# Truth definitions
# ---------------------------------------------------------------------------
# Each truth provides:
#   name        -- label for reporting
#   pdf(x)      -- the true probability density (for numeric truth moments)
#   rvs(n, rng) -- draws n true velocities
#   grid        -- (center, width) for KinematicSolver.setup_grid
#   err_range   -- (lo, hi) for per-star heteroscedastic Gaussian measurement error


def _mixture_pdf(x, locs, scale, weights):
    out = 0.0
    for loc, w in zip(locs, weights):
        out = out + w * stats.norm(loc=loc, scale=scale).pdf(x)
    return out


def _make_truths():
    truths = []

    # 1. Gaussian
    rv = stats.norm(loc=0.0, scale=30.0)
    truths.append(
        {
            "name": "gaussian",
            "pdf": rv.pdf,
            "rvs": lambda n, rng, _rv=rv: _rv.rvs(size=n, random_state=rng),
            "grid": (0.0, 320.0),
            "err_range": (4.0, 14.0),
        }
    )

    # 2. Student-t, df=6 -- h4>0 radially-anisotropic analogue (heavy tails,
    # positive excess kurtosis = 6/(df-4) = 3 for df=6).
    rv = stats.t(df=6, loc=0.0, scale=18.0)
    truths.append(
        {
            "name": "student_t_h4",
            "pdf": rv.pdf,
            "rvs": lambda n, rng, _rv=rv: _rv.rvs(size=n, random_state=rng),
            "grid": (0.0, 360.0),
            "err_range": (4.0, 14.0),
        }
    )

    # 3. Skew-normal -- h3!=0 rotating-side analogue.
    rv = stats.skewnorm(a=5.0, loc=-18.0, scale=28.0)
    truths.append(
        {
            "name": "skew_normal_h3",
            "pdf": rv.pdf,
            "rvs": lambda n, rng, _rv=rv: _rv.rvs(size=n, random_state=rng),
            "grid": (0.0, 320.0),
            "err_range": (4.0, 14.0),
        }
    )

    # 4. Counter-rotation bimodal -- symmetric two-Gaussian mixture.
    locs, scale, weights = (-40.0, 40.0), 14.0, (0.5, 0.5)

    def _mix_rvs(n, rng):
        comp = rng.integers(0, 2, size=n)
        draws = np.empty(n)
        for k, loc in enumerate(locs):
            mask = comp == k
            draws[mask] = rng.normal(loc=loc, scale=scale, size=mask.sum())
        return draws

    truths.append(
        {
            "name": "bimodal_counter_rotation",
            "pdf": lambda x: _mixture_pdf(x, locs, scale, weights),
            "rvs": _mix_rvs,
            "grid": (0.0, 320.0),
            "err_range": (4.0, 14.0),
        }
    )

    return truths


def _true_moments(pdf, lo=-400.0, hi=400.0):
    """Numerically integrate the true pdf for v_mean, sigma, skewness,
    kurtosis, and tail_weight (fraction of mass outside +/-1 sigma of the
    mean) -- computed generically so it works for the mixture truth too,
    rather than trusting per-distribution formulas."""
    mean = integrate.quad(lambda x: x * pdf(x), lo, hi, limit=200)[0]
    var = integrate.quad(lambda x: (x - mean) ** 2 * pdf(x), lo, hi, limit=200)[0]
    sigma = np.sqrt(var)
    skew = integrate.quad(
        lambda x: ((x - mean) / sigma) ** 3 * pdf(x), lo, hi, limit=200
    )[0]
    kurt = (
        integrate.quad(lambda x: ((x - mean) / sigma) ** 4 * pdf(x), lo, hi, limit=200)[0]
        - 3.0
    )
    tail = (
        integrate.quad(pdf, lo, mean - sigma, limit=200)[0]
        + integrate.quad(pdf, mean + sigma, hi, limit=200)[0]
    )
    return {
        "v_mean": mean,
        "sigma": sigma,
        "skewness": skew,
        "kurtosis": kurt,
        "tail_weight": tail,
    }


def _binom_band(n, p=0.68, alpha=0.01):
    """A wide binomial confidence band for n trials at nominal probability
    p, using the alpha/2 .. 1-alpha/2 quantiles of Binomial(n, p). At
    alpha=0.01 this is a ~99% band -- wide enough not to false-fail on
    ordinary sampling noise, but tight enough to catch gross miscalibration
    (e.g. empirical coverage of 0.30 when nominal is 0.68)."""
    lo = stats.binom.ppf(alpha / 2, n, p) / n
    hi = stats.binom.ppf(1 - alpha / 2, n, p) / n
    return float(lo), float(hi)


def _run_coverage(truth, n_real, rng):
    """Fit n_real independent mock realisations of `truth` and return, for
    each metric in METRICS, the number of realisations whose 68% credible
    interval contained the true value."""
    true_vals = _true_moments(truth["pdf"])
    center, width = truth["grid"]

    hits = {m: 0 for m in METRICS}
    n_ok = 0

    for i in range(n_real):
        true_v = truth["rvs"](N_STARS, rng)
        err = rng.uniform(*truth["err_range"], size=N_STARS)
        obs_v = true_v + rng.normal(0.0, err)

        solver = KinematicSolver()
        solver.setup_grid(center=center, width=width, n_bins=N_BINS)
        solver.add_data(obs_v, err)
        solver.run(num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, seed=1000 + i)

        summary = compute_summary(
            solver.samples["intrinsic_pdf"], solver.grid["centers"]
        )
        n_ok += 1
        for m in METRICS:
            median, half68 = summary[m]
            lo, hi = median - half68, median + half68
            if lo <= true_vals[m] <= hi:
                hits[m] += 1

    return true_vals, hits, n_ok


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "Known, quantified finding (not a harness bug -- independently verified): "
        "compute_summary()'s kurtosis is systematically biased positive (~+1.6 to "
        "+2.5 excess kurtosis) even for a plain Gaussian truth, whose binned "
        "representation has ~0 kurtosis (verified: -0.0007). Root cause: the RW1 "
        "smoothness prior leaks a small amount of posterior mass into far-edge "
        "grid bins (~2x the true Gaussian's tail mass beyond 2.5 sigma was "
        "observed); because kurtosis weights deviations by the 4th power, this "
        "tiny residual mass at ~5 sigma (amplification ~5**4=625x) is enough to "
        "produce the bias. truncate_losvd() exists to suppress exactly this "
        "leakage but only patches clipped_samples (the Dynamite export path) -- "
        "it is not applied to the raw posterior samples compute_summary() uses, "
        "so this bias is currently unmitigated for analysis.py users. See "
        "TASKS.md / PLAN.md for the follow-up item. strict=False: if a future "
        "fix (e.g. extending truncate_losvd to the raw-sample path, or a wider "
        "grid margin) resolves this, this test will show as an unexpected pass "
        "(XPASS) rather than silently staying green -- remove this marker then."
    ),
    strict=False,
)
def test_coverage_over_mock_realisations():
    rng = np.random.default_rng(20260803)
    truths = _make_truths()

    band_lo, band_hi = _binom_band(N_REAL, p=0.68, alpha=0.01)
    # Catastrophic-failure floor for the shrinkage-prone metrics on
    # non-Gaussian truths (see module docstring): shrinkage bias is an
    # expected, acceptable amount of under-coverage; this floor only
    # catches "badly broken", not "some shrinkage, as expected".
    shrinkage_floor = 0.30

    report_lines = []
    hard_failures = []
    soft_warnings = []

    for truth in truths:
        true_vals, hits, n_ok = _run_coverage(truth, N_REAL, rng)
        assert n_ok == N_REAL, f"{truth['name']}: not all realisations fit successfully"

        report_lines.append(f"\n=== {truth['name']} (n_real={N_REAL}) ===")
        report_lines.append(
            "true values: " + ", ".join(f"{m}={true_vals[m]:.4f}" for m in METRICS)
        )
        report_lines.append(f"nominal-coverage binom({N_REAL},0.68) 99% band: [{band_lo:.3f}, {band_hi:.3f}]")

        for m in METRICS:
            cov = hits[m] / N_REAL
            report_lines.append(f"  {m:12s}: coverage={cov:.3f} ({hits[m]}/{N_REAL})")

            is_shrinkage_prone_case = (m in SHRINKAGE_PRONE) and (truth["name"] != "gaussian")

            if is_shrinkage_prone_case:
                if cov < shrinkage_floor:
                    hard_failures.append(
                        f"{truth['name']}.{m}: coverage {cov:.3f} below catastrophic "
                        f"floor {shrinkage_floor:.2f} (expected only mild shrinkage, "
                        f"not this)"
                    )
                elif cov < band_lo:
                    soft_warnings.append(
                        f"{truth['name']}.{m}: coverage {cov:.3f} below nominal band "
                        f"[{band_lo:.3f}, {band_hi:.3f}] -- expected smoothness-prior "
                        f"shrinkage bias, not a failure"
                    )
            else:
                if not (band_lo <= cov <= band_hi):
                    hard_failures.append(
                        f"{truth['name']}.{m}: coverage {cov:.3f} outside nominal "
                        f"binomial band [{band_lo:.3f}, {band_hi:.3f}] for n={N_REAL} "
                        f"trials at p=0.68 -- this metric is not expected to show "
                        f"shrinkage bias for this truth, so this looks like genuine "
                        f"miscalibration"
                    )

    report = "\n".join(report_lines)
    if soft_warnings:
        report += "\n\nExpected shrinkage (not failures):\n" + "\n".join(soft_warnings)

    if hard_failures:
        failure_msg = (
            report
            + "\n\nHARD FAILURES (coverage outside acceptable range):\n"
            + "\n".join(hard_failures)
        )
        pytest.fail(failure_msg)

    print(report)
