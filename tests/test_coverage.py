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

Observational calibration
-------------------------
`N_STARS`, `N_BINS`, `ERR_RANGE` and the truth amplitudes were recalibrated
against the actual target dataset (oMEGACat: HST+MUSE within omega Cen's
half-light radius) rather than being chosen for convenience. The previous
values were testing a substantially harder problem than the science requires,
in three separate ways:

- **Measurement errors.** Was `err_range = (4, 14)` km/s against sigma 17-42,
  i.e. err/sigma ~ 0.5. oMEGACat MUSE gives 2-3 km/s for bright RGB stars,
  and van de Ven et al. (2006, A&A 445, 513) selected LOS velocities at
  <2.0 km/s against sigma_z' of 8.8-20.9. Deconvolution difficulty is driven
  by err/sigma, so this was a ~3-4x overstatement.
- **Sample size.** Kept at `N_STARS = 150`, which is the science target rather
  than a convenience value. After quality cuts there are ~30k LOS spectra
  within r_h (~4.65 arcmin, ~68 sq arcmin), so 150 stars per bin yields ~200
  Voronoi bins. Raising it to 250 would make the fits measurably easier (~120
  bins) but coarsens the map exactly where it matters, in the inner MUSE
  region. The harness therefore holds the sample size fixed at the value the
  science needs and requires the *model* to meet it, rather than relaxing the
  target until the current model passes.
- **Truth amplitudes.** Converting via the standard Gauss-Hermite relations
  `gamma1 = 4*sqrt(3)*h3` and `gamma2 = 8*sqrt(6)*h4`, the old skew-normal was
  h3 ~ 0.123 and the old Student-t was h4 ~ 0.153. Real systems run
  |h3| <~ 0.15 and |h4| <~ 0.05-0.1, so both sat at or past the top of the
  range. They are now h3 ~ 0.066 and h4 ~ 0.051.

Note that the recalibration does not uniformly make the test easier. The
milder h3 truth is *harder* to recover than the extreme one, because
shrinkage bias toward a Gaussian is roughly fixed in absolute skewness units
and therefore eats a larger fraction of a smaller signal. That is the point:
these settings test the regime the science actually operates in.

Expected honest outcome (per plan): `kurtosis` and `tail_weight` coverage
may run under nominal for the non-Gaussian truths, because the RW1
smoothness prior shrinks the inferred LOSVD toward smoothness and pulls in
genuine sharp features (e.g. the Student-t's heavy tails, the bimodal
mixture's two peaks). This is an expected bias/variance trade documented in
the model, not a bug -- so the assertions here only guard against
*catastrophic* under-coverage (roughly < 0.30-0.40), while the full numbers
are always printed/reported so the actual degree of shrinkage is visible.

ACTUAL outcome (broader than the plan anticipated -- see the xfail marker on
the test below for full detail): kurtosis coverage was catastrophically low
even for the plain *Gaussian* truth (median excess kurtosis ~+1.6 to +2.5
vs. a true value of 0), which the plan did not predict since there is no
sharp feature there for the smoothness prior to shrink away. This was
independently verified as a real tail-leakage effect (small residual
posterior mass in far-edge grid bins, amplified by kurtosis's 4th-power
weighting) rather than a bug in this test harness.

Partial fix applied: `compute_summary` now accepts `n_sigma_truncate`
(see `analysis.truncate_pdf_samples`), used here at 3.0. This resolves the
Gaussian-truth catastrophe (kurtosis coverage 0.000 -> 0.840) and also fixes
skew_normal_h3 (0.000 -> 0.800), but does not resolve kurtosis coverage for
the genuinely heavy-tailed/multimodal truths (student_t_h4 stays at 0.000,
bimodal_counter_rotation stays at 0.080) -- truncation trades away some of
their real tail mass along with the leaked mass. The test therefore remains
marked `xfail` (updated reason, not the original one) rather than adjusted
to pass, per PLAN.md's rule against loosening a failing statistical test.
"""

import numpy as np
import pytest
from scipy import stats

from veldist.veldist import KinematicSolver
from veldist.analysis import compute_summary

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_REAL = 25  # realisations per truth (plan suggests 50; reduced for runtime, see module docstring)
# Calibrated against the oMEGACat target dataset -- see "Observational
# calibration" in the module docstring. 150 stars per bin is the science
# target: it gives ~200 Voronoi bins across r_h, and the finer binning
# matters specifically in the inner regions of the MUSE map, where spatial
# resolution is the point. 250 would be easier to fit but coarser on sky.
N_STARS = 150  # fixed per truth -> matrix shape is constant -> JAX only compiles once per truth

# Velocity grid sized to the data rather than chosen for convenience.
#
# DYNAMITE requires one shared velocity grid across all spatial bins, so it
# must hold the largest dispersion in the field. omega Cen's central LOS
# dispersion is ~20 km/s, falling to ~9 km/s in the outskirts, plus a ~10 km/s
# span in the mean velocity from rotation. GRID_WIDTH = 200 gives +-4.8 sigma
# for the central dispersion after allowing for that offset (Gaussian mass
# outside the grid ~2e-6).
#
# The bin width is 2x the 2-3 km/s measurement error. Finer is unrecoverable:
# the errors convolve away structure below their own scale. The previous
# setting -- 320 km/s over 20 bins -- gave 16 km/s bins, i.e. 6.4x *coarser*
# than the data resolves, while leaving ~71% of bins carrying no mass at all.
# Those empty bins are pure prior-driven latent dimensions and are the likely
# reason `n_sigma_truncate` was needed to suppress far-edge tail leakage.
GRID_WIDTH = 200.0
N_BINS = 40  # -> 5.0 km/s bins
NUM_WARMUP = 300
NUM_SAMPLES = 600

# MUSE line-of-sight velocity uncertainties for bright RGB stars in oMEGACat
# are 2-3 km/s. Against the truths below (sigma 15-42 km/s) this gives
# err/sigma ~ 0.05-0.20, matching van de Ven et al. (2006), who selected LOS
# velocities at <2.0 km/s against sigma_z' of 8.8-20.9 km/s.
ERR_RANGE = (2.0, 3.0)

# Truncation applied to raw posterior samples before compute_summary's
# moments are evaluated, to mitigate RW1 tail-leakage bias in kurtosis (see
# analysis.truncate_pdf_samples and PLAN.md sec 1.3). n_sigma=3.0 was found
# empirically (scratch verification, not committed) to remove essentially
# all of the median kurtosis bias for a Gaussian truth on this grid setup
# (+1.78 -> +0.05 median excess kurtosis over 12 realisations).
N_SIGMA_TRUNCATE = 3.0

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
    rv = stats.norm(loc=0.0, scale=20.0)  # omega Cen central LOS dispersion
    truths.append(
        {
            "name": "gaussian",
            "pdf": rv.pdf,
            "rvs": lambda n, rng, _rv=rv: _rv.rvs(size=n, random_state=rng),
            "grid": (0.0, GRID_WIDTH),
            "err_range": ERR_RANGE,
        }
    )

    # 2. Student-t, df=10 -- h4>0 radially-anisotropic analogue (heavy tails,
    # positive excess kurtosis = 6/(df-4) = 1.0 for df=10, i.e. h4 ~ +0.051
    # via gamma2 = 8*sqrt(6)*h4). df=6 was used previously, giving excess
    # kurtosis 3.0 and h4 ~ 0.153 -- roughly twice the largest h4 seen in real
    # systems (|h4| <~ 0.05-0.1), so it was testing an unrepresentative
    # extreme.
    rv = stats.t(df=10, loc=0.0, scale=18.0)
    truths.append(
        {
            "name": "student_t_h4",
            "pdf": rv.pdf,
            "rvs": lambda n, rng, _rv=rv: _rv.rvs(size=n, random_state=rng),
            "grid": (0.0, GRID_WIDTH),
            "err_range": ERR_RANGE,
        }
    )

    # 3. Skew-normal -- h3!=0 rotating-side analogue. a=2 gives skewness
    # ~+0.45, i.e. h3 ~ +0.066 via gamma1 = 4*sqrt(3)*h3, representative of a
    # strongly rotating system. a=5 was used previously, giving skewness 0.85
    # and h3 ~ 0.123, at the very top of the observed range (|h3| <~ 0.15).
    # Note the milder truth is the *harder* test: shrinkage bias toward the
    # Gaussian is roughly fixed in absolute skewness units, so it consumes a
    # larger fraction of a smaller signal.
    rv = stats.skewnorm(a=2.0, loc=-12.0, scale=22.0)
    truths.append(
        {
            "name": "skew_normal_h3",
            "pdf": rv.pdf,
            "rvs": lambda n, rng, _rv=rv: _rv.rvs(size=n, random_state=rng),
            "grid": (0.0, GRID_WIDTH),
            "err_range": ERR_RANGE,
        }
    )

    # 4. Flat-topped -- tangential anisotropy analogue, h4 < 0. This is the
    # shape a tangentially anisotropic LOSVD actually has: a *unimodal*
    # flat/box-topped profile, which is physically distinct from the
    # counter-rotating bimodal truth below even though both have h4 < 0.
    # van de Ven et al. (2006) find omega Cen increasingly tangentially
    # anisotropic outside ~10 arcmin, so this is the dominant expected
    # non-Gaussianity, and it is the regime Sanders & Evans (2020) report as
    # detectable at ~200 stars (positive excess kurtosis needs >~2000).
    #
    # Constructed as uniform(-A, A) convolved with a Gaussian, i.e. exactly
    # the Sanders & Evans uniform-kernel family. Excess kurtosis is
    # -1.2 * r**2 where r is the fraction of variance carried by the uniform
    # component; r=0.913 gives kappa = -1.0. Note -1.2 (pure uniform) is the
    # hard floor for this family, and for any unimodal flat-top.
    #
    # Caveat on h4: eq (5) of Sanders & Evans, h4 = kappa / (8*sqrt(6)),
    # would map kappa=-1.0 to h4 ~ -0.051, but they warn it is only good to
    # 10% for |h4| < 0.01. Measured directly, a GH series with h4 = -0.05 has
    # actual excess kurtosis -2.15, and h4 = -0.10 gives -12.7 -- the latter
    # is an artefact of the series going negative, not a physical LOSVD. So
    # "h4 = -0.1" from a GH-fitting code should be read as "strongly
    # flat-topped or bimodal", not as a literal kurtosis.
    FLAT_A, FLAT_S = 28.1, 5.0

    def _flat_pdf(x, _mu=0.0, _a=FLAT_A, _s=FLAT_S):
        return (
            stats.norm.cdf((x - _mu + _a) / _s) - stats.norm.cdf((x - _mu - _a) / _s)
        ) / (2.0 * _a)

    def _flat_rvs(n, rng, _a=FLAT_A, _s=FLAT_S):
        return rng.uniform(-_a, _a, size=n) + rng.normal(0.0, _s, size=n)

    truths.append(
        {
            "name": "flat_top_tangential",
            "pdf": _flat_pdf,
            "rvs": _flat_rvs,
            "grid": (0.0, GRID_WIDTH),
            "err_range": ERR_RANGE,
        }
    )

    # 5. Rotating + tangentially anisotropic -- h3 and h4 together. Real bins
    # in the outer region have both at once: rotation gives h3, tangential
    # anisotropy gives h4 < 0. Every other truth here isolates a single
    # moment, which is not how the data arrives. Built as an *asymmetric*
    # uniform kernel convolved with a Gaussian, which is the Sanders & Evans
    # (2020) skewness option for their uniform family.
    # Note a *shifted* uniform kernel is still symmetric about its own
    # midpoint and produces no skewness at all -- only a mean offset. The
    # skewed version needs a two-piece kernel with different widths either
    # side of zero, carrying equal weight, so the density steps at the
    # centre. Here the left piece is wider and lower, giving negative h3.
    ROT_A1, ROT_A2, ROT_S = 34.0, 13.0, 6.0

    def _rot_tan_pdf(x, _a1=ROT_A1, _a2=ROT_A2, _s=ROT_S):
        x = np.asarray(x, dtype=float)
        left = (stats.norm.cdf((x + _a1) / _s) - stats.norm.cdf(x / _s)) / _a1
        right = (stats.norm.cdf(x / _s) - stats.norm.cdf((x - _a2) / _s)) / _a2
        return 0.5 * (left + right)

    def _rot_tan_rvs(n, rng, _a1=ROT_A1, _a2=ROT_A2, _s=ROT_S):
        left = rng.random(n) < 0.5
        draws = rng.uniform(0.0, _a2, size=n)
        draws[left] = rng.uniform(-_a1, 0.0, size=left.sum())
        return draws + rng.normal(0.0, _s, size=n)

    truths.append(
        {
            "name": "rotating_tangential",
            "pdf": _rot_tan_pdf,
            "rvs": _rot_tan_rvs,
            "grid": (0.0, GRID_WIDTH),
            "err_range": ERR_RANGE,
        }
    )

    # 6. Cold disk component -- omega Cen specific. van de Ven et al. (2006)
    # find a separate disk-like component between 1 and 3 arcmin contributing
    # ~4% of the total mass, which is "the kinematically coldest component"
    # and maximally rotating. A small, cold, offset component on a warm base
    # is a distinctive shape: sharply peaked core, and neither a pure scale
    # mixture nor a clean bimodal. This is also the feature a hierarchical
    # spatial model would be at risk of over-smoothing, so it needs to be in
    # the suite before that work starts.
    disk_locs, disk_scales, disk_w = (0.0, 26.0), (17.0, 5.0), (0.96, 0.04)

    def _disk_pdf(x, _l=disk_locs, _s=disk_scales, _w=disk_w):
        out = 0.0
        for loc, sc, w in zip(_l, _s, _w):
            out = out + w * stats.norm(loc=loc, scale=sc).pdf(x)
        return out

    def _disk_rvs(n, rng, _l=disk_locs, _s=disk_scales, _w=disk_w):
        comp = rng.random(n) < _w[1]
        draws = rng.normal(_l[0], _s[0], size=n)
        draws[comp] = rng.normal(_l[1], _s[1], size=comp.sum())
        return draws

    truths.append(
        {
            "name": "cold_disk_component",
            "pdf": _disk_pdf,
            "rvs": _disk_rvs,
            "grid": (0.0, GRID_WIDTH),
            "err_range": ERR_RANGE,
        }
    )

    # 7. Two stellar populations -- omega Cen specific. Norris et al. (1997)
    # and van de Ven et al. sec 9.6: the metal-poor population is
    # kinematically hotter and rotates, the metal-rich population is cooler
    # and nearly non-rotating. Any Voronoi bin mixes them, giving a scale
    # mixture with a small relative offset. Distinct from student_t: the
    # heavy tail here is a genuine two-component superposition, which is what
    # the data actually contains, rather than an idealised t distribution.
    pop_locs, pop_scales, pop_w = (4.0, -2.0), (19.0, 12.0), (0.65, 0.35)

    def _pop_pdf(x, _l=pop_locs, _s=pop_scales, _w=pop_w):
        out = 0.0
        for loc, sc, w in zip(_l, _s, _w):
            out = out + w * stats.norm(loc=loc, scale=sc).pdf(x)
        return out

    def _pop_rvs(n, rng, _l=pop_locs, _s=pop_scales, _w=pop_w):
        comp = rng.random(n) < _w[1]
        draws = rng.normal(_l[0], _s[0], size=n)
        draws[comp] = rng.normal(_l[1], _s[1], size=comp.sum())
        return draws

    truths.append(
        {
            "name": "two_population",
            "pdf": _pop_pdf,
            "rvs": _pop_rvs,
            "grid": (0.0, GRID_WIDTH),
            "err_range": ERR_RANGE,
        }
    )

    # 8. Mild radial anisotropy -- the inner region. van de Ven et al. find
    # omega Cen only slightly radially anisotropic between about 3 and 5
    # arcmin, so the realistic positive-h4 case is much weaker than the
    # student_t_h4 truth above. df=19 gives excess kurtosis 6/(df-4) = 0.4,
    # i.e. h4 ~ +0.020. Retained alongside the stronger truth because
    # Sanders & Evans report positive excess kurtosis as the hard sign to
    # detect (>~2000 stars), so this is expected to be a non-detection and
    # the test is whether the interval is honest about that.
    rv = stats.t(df=19, loc=0.0, scale=17.0)
    truths.append(
        {
            "name": "mild_radial_h4",
            "pdf": rv.pdf,
            "rvs": lambda n, rng, _rv=rv: _rv.rvs(size=n, random_state=rng),
            "grid": (0.0, GRID_WIDTH),
            "err_range": ERR_RANGE,
        }
    )

    # 9. Counter-rotation bimodal -- symmetric two-Gaussian mixture. Excess
    # kurtosis ~-1.59, i.e. h4 ~ -0.081, which is within the observed range
    # and the right sign for the tangential anisotropy van de Ven et al.
    # (2006) find in omega Cen's outer region. Kept unchanged: it is the one
    # truth here that is both realistic in amplitude and a genuine stress
    # test of multimodality.
    locs, scale, weights = (-18.0, 18.0), 10.0, (0.5, 0.5)  # sigma ~20.6, omega Cen-like

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
            "grid": (0.0, GRID_WIDTH),
            "err_range": ERR_RANGE,
        }
    )

    return truths


def _true_moments(pdf, lo=-500.0, hi=500.0, n_grid=2_000_001):
    """Numerically integrate the true pdf for v_mean, sigma, skewness,
    kurtosis, and tail_weight (fraction of mass outside +/-1 sigma of the
    mean) -- computed generically so it works for the mixture truths too,
    rather than trusting per-distribution formulas.

    Uses a dense uniform grid rather than ``scipy.integrate.quad``. quad's
    adaptive subdivision silently under-samples a narrow component sitting on
    a broad base: for the ``cold_disk_component`` truth (a 4%-weight, 5 km/s
    component on a 17 km/s base) it returned skewness +0.0000 against a true
    -0.0320. Since these values are what coverage is measured against, a
    silently wrong truth would invalidate the test rather than fail it. The
    grid spacing here is 5e-4 km/s, and results agree with quad to 4 decimal
    places on every truth where quad is reliable.
    """
    v = np.linspace(lo, hi, n_grid)
    p = np.asarray(pdf(v), dtype=float)
    norm = np.trapezoid(p, v)
    p = p / norm

    mean = float(np.trapezoid(v * p, v))
    var = float(np.trapezoid((v - mean) ** 2 * p, v))
    sigma = np.sqrt(var)
    skew = float(np.trapezoid(((v - mean) / sigma) ** 3 * p, v))
    kurt = float(np.trapezoid(((v - mean) / sigma) ** 4 * p, v)) - 3.0
    inside = (v >= mean - sigma) & (v <= mean + sigma)
    tail = 1.0 - float(np.trapezoid(p[inside], v[inside]))
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


def _run_coverage(truth, n_real, rng, prior="rw1"):
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
        solver.run(
            num_warmup=NUM_WARMUP,
            num_samples=NUM_SAMPLES,
            seed=1000 + i,
            prior=prior,
        )

        # The Gaussian-core prior should not need the post-hoc truncation
        # repair -- needing it would mean the root-cause fix did not work.
        # rw1 keeps 3.0 so its numbers stay comparable to the values already
        # recorded in docs/validation.md.
        truncate = None if prior == "gaussian_core" else N_SIGMA_TRUNCATE

        summary = compute_summary(
            solver.samples["intrinsic_pdf"],
            solver.grid["centers"],
            n_sigma_truncate=truncate,
        )
        n_ok += 1
        for m in METRICS:
            median, half68 = summary[m]
            lo, hi = median - half68, median + half68
            if lo <= true_vals[m] <= hi:
                hits[m] += 1

    return true_vals, hits, n_ok


_RW1_XFAIL_REASON = (
    "Partially fixed, still red. compute_summary() now accepts "
    "n_sigma_truncate (analysis.truncate_pdf_samples), applied here with "
    "n_sigma_truncate=3.0, to suppress RW1 tail-leakage mass in the raw "
    "posterior samples before moments are computed. This resolves the "
    "originally catastrophic Gaussian-truth kurtosis bug (coverage "
    "0.000 -> 0.840 over n_real=25, well inside the nominal band) and "
    "also brings skew_normal_h3.kurtosis into band (0.000 -> 0.800). "
    "BUT it does not fix -- and does not clearly improve -- kurtosis "
    "coverage for the two genuinely heavy-tailed/multimodal truths: "
    "student_t_h4.kurtosis stayed at 0.000 (0/25, previously 0.120 "
    "pre-truncation) and bimodal_counter_rotation.kurtosis stayed at "
    "0.080 (2/25, previously 0.040). This is the expected trade-off of "
    "a fixed n_sigma cut: n_sigma=3.0 aggressively removes far-edge "
    "leaked mass (good for the Gaussian truth, which has no real mass "
    "there) but also removes some of the *genuine* heavy tail these "
    "truths have (student_t_h4 has true kurtosis=2.82; the truncation "
    "itself flattens the fitted LOSVD's tail, which is exactly what "
    "produces under-coverage). So this is a real, only-partial fix: "
    "it eliminates the 'even a Gaussian is biased' finding but leaves "
    "the harder shrinkage-vs-heavy-tail problem open for kurtosis on "
    "non-Gaussian truths. See analysis.py's compute_summary docstring "
    "and PLAN.md sec 1.3 for the full numbers. strict=False: if a "
    "future improvement (e.g. an n_sigma schedule, adaptive per-truth "
    "cut, or a different tail-suppression approach) resolves the "
    "remaining student_t_h4/bimodal failures too, this test will show "
    "as an unexpected pass (XPASS) rather than silently staying green "
    "-- remove this marker then. This xfail is scoped to prior='rw1' only "
    "-- see _GAUSSIAN_CORE_XFAIL_REASON for the other parametrisation."
)

_GAUSSIAN_CORE_XFAIL_REASON = (
    "Improved by the RW3 deviation scaling fix, still red. The Gaussian-core "
    "prior was originally expected to pass outright, since fixing the flat "
    "prior null space was the point of that plan. It does not, and it did not "
    "before the scaling fix either -- this parametrisation was already failing "
    "on main, so the marker records a standing known-red result rather than "
    "pardoning a regression. "
    "Numbers below are on the observationally recalibrated harness (see "
    "'Observational calibration' above): N_STARS=150, N_BINS=40, "
    "ERR_RANGE=(2,3) km/s, h3~0.066 and h4~0.051 truths. "
    "What now passes: v_mean and sigma are inside the nominal band for every "
    "truth; student_t_h4.tail_weight came out of catastrophic failure "
    "(0.160 -> 0.640) purely from using realistic measurement errors; "
    "bimodal skewness 0.800 and tail_weight 1.000. "
    "What still fails, and it is now a single coherent defect rather than a "
    "scatter of symptoms: the model cannot recover h3/h4 at realistic "
    "amplitude. skew_normal_h3 skewness 0.040 (1/25) and kurtosis 0.040, "
    "student_t_h4 kurtosis 0.000 (0/25). The finite-sample achievable "
    "skewness at N=150 is 0.428 against a true 0.454, so this is not an "
    "estimator artefact -- the information is in the data and the posterior "
    "is not extracting it. Root cause is the sigma3 prior's mode at zero "
    "(= 'the LOSVD is exactly Gaussian'); see TASKS.md and the measurements "
    "doc for the prior sweep. "
    "Over-coverage (1.000) on gaussian kurtosis/tail_weight and student_t "
    "skewness is the same defect seen from the other side: a posterior pinned "
    "near Gaussian trivially covers a Gaussian-valued truth. "
    "strict=False so that a real fix surfaces as XPASS -- remove this marker "
    "then. Full tables in "
    "docs/superpowers/plans/2026-08-03-rw3-measurements.md."
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "prior",
    [
        pytest.param(
            "rw1", marks=pytest.mark.xfail(reason=_RW1_XFAIL_REASON, strict=False)
        ),
        pytest.param(
            "gaussian_core",
            marks=pytest.mark.xfail(
                reason=_GAUSSIAN_CORE_XFAIL_REASON, strict=False
            ),
        ),
    ],
)
def test_coverage_over_mock_realisations(prior):
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
        true_vals, hits, n_ok = _run_coverage(truth, N_REAL, rng, prior=prior)
        assert n_ok == N_REAL, f"{truth['name']}: not all realisations fit successfully"

        report_lines.append(f"\n=== {truth['name']} [{prior}] (n_real={N_REAL}) ===")
        report_lines.append(
            "true values: " + ", ".join(f"{m}={true_vals[m]:.4f}" for m in METRICS)
        )
        achievable = _achievable_moments(truth["rvs"], N_STARS)
        report_lines.append(
            "  finite-sample achievable (median of sample estimator at "
            f"N={N_STARS}): "
            + ", ".join(f"{m}={achievable[m]:.3f}" for m in METRICS)
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


def _achievable_moments(rvs, n_stars, n_trials=2000, seed=7):
    """Median of the *sample* moment estimators over `n_trials` draws of
    `n_stars` stars.

    Population moments are not always reachable from a finite sample. The
    sample excess-kurtosis estimator in particular is severely biased low
    for heavy-tailed distributions: 150 draws from Student-t(df=6) have a
    median sample excess kurtosis of ~1.16 against a population value of
    3.0, with a 16-84% range of [0.33, 3.01]. Comparing a fit from 150 stars
    against 3.0 therefore demands something the data cannot deliver, and
    attributes an estimator property to the model as if it were a bias.

    Returns the same keys as `_true_moments` so the two can be compared
    directly.
    """
    rng = np.random.default_rng(seed)
    acc = {m: np.empty(n_trials) for m in METRICS}
    for t in range(n_trials):
        v = rvs(n_stars, rng)
        mean = v.mean()
        d = v - mean
        var = (d ** 2).mean()
        sigma = np.sqrt(var)
        acc["v_mean"][t] = mean
        acc["sigma"][t] = sigma
        acc["skewness"][t] = (d ** 3).mean() / sigma ** 3
        acc["kurtosis"][t] = (d ** 4).mean() / var ** 2 - 3.0
        acc["tail_weight"][t] = np.mean(np.abs(d) > sigma)
    return {m: float(np.median(acc[m])) for m in METRICS}


def test_population_kurtosis_is_unreachable_for_heavy_tailed_truths():
    """Guards the reasoning behind `_achievable_moments`.

    If this ever fails, the finite-sample argument recorded in PLAN.md
    section 1.3 no longer holds and the coverage expectations for
    student_t_h4 need re-deriving.
    """
    truths = {t["name"]: t for t in _make_truths()}

    gaussian = truths["gaussian"]
    achievable = _achievable_moments(gaussian["rvs"], N_STARS)
    population = _true_moments(gaussian["pdf"])
    assert abs(achievable["kurtosis"] - population["kurtosis"]) < 0.25, (
        "for a Gaussian truth the sample kurtosis estimator should be nearly "
        f"unbiased at N={N_STARS}, got achievable={achievable['kurtosis']:.2f} "
        f"vs population={population['kurtosis']:.2f}"
    )

    student = truths["student_t_h4"]
    achievable = _achievable_moments(student["rvs"], N_STARS)
    population = _true_moments(student["pdf"])
    assert achievable["kurtosis"] < 0.6 * population["kurtosis"], (
        "expected the sample kurtosis estimator to badly under-report the "
        f"heavy tail at N={N_STARS}: achievable={achievable['kurtosis']:.2f}, "
        f"population={population['kurtosis']:.2f}"
    )
