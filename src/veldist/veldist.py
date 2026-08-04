# file src/veldist/veldist.py
"""
Bayesian Matrix-Based Kinematic Deconvolution
=============================================

This module infers the intrinsic Line-of-Sight Velocity Distribution (LOSVD)
from discrete, heteroscedastic stellar observations using a pre-computed
linear design matrix and a hierarchical smoothness prior.
"""

import warnings
from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt


__all__ = [
    "KinematicSolver",
    "precompute_design_matrix",
    "generate_smooth_curve",
    "model",
    "fit_all_bins",
    "write_dynamite_kinematics",
]

# Penalised-complexity prior rate for the Gaussian-core deviation scale.
# Defined by P(sigma3 > 1) = 0.1, i.e. rate = -ln(0.1). After Sorbye-Rue
# standardisation sigma3 is the typical log-density departure from a Gaussian
# LOSVD, so sigma3 = 1 means a departure of a factor ~e at typical velocities.
# Pinned by test_gaussian_core_prior_spans_nongaussian_shapes (p90 |kurtosis| = 1.361).
SIGMA3_RATE = 0.35  # Exp(0.35), adopted 2026-08-03 — see docs/superpowers/specs/2026-08-03-regularisation-decision.md

# ==============================================================================
# Design Matrix
# ==============================================================================


def precompute_design_matrix(obs_val, obs_err, bin_centers, bin_width=None):
    """
    Computes the Probability Design Matrix (M) for 1D LOSVD inference.

    This function bakes the observations and uncertainties into a static matrix, converting the
    deconvolution problem into a single matrix multiplication during inference.

    Parameters
    ----------
    obs_val : array-like (N,)
        Observed velocities of individual stars.
    obs_err : array-like (N,)
        Standard deviation (measurement error) for each star.
    bin_centers : array-like (K,)
        The velocity grid centers (the intrinsic histogram bins).
    bin_width : float, optional
        Width of bins. If None, inferred from centers.

    Returns
    -------
    M : jnp.ndarray (N, K)
        Probability matrix. M[i, j] = P(Star i | True Velocity is in Bin j)
    """
    # Reshape for broadcasting: (N, 1)
    y = jnp.array(obs_val)[:, None]
    sig = jnp.array(obs_err)[:, None]

    if bin_width is None:
        bin_width = bin_centers[1] - bin_centers[0]

    # Define bin edges (K+1 edges for K bins)
    edges = jnp.concatenate([bin_centers - bin_width / 2, jnp.array([bin_centers[-1] + bin_width / 2])])

    # --- Box Integration ---
    # Instead of evaluating a Gaussian PDF at the bin center, we integrate
    # the Gaussian over the full width of the bin.
    # This prevents aliasing when error < bin_width.

    # Calculate Z-scores at every bin edge
    # Broadcasting: (1, K+1) - (N, 1) -> (N, K+1)
    z = (edges[None, :] - y) / (sig * jnp.sqrt(2))

    # Error Function (CDF of Gaussian)
    cdf_values = 0.5 * (1 + jsp.erf(z))

    # The probability mass in Bin j is CDF(Right Edge) - CDF(Left Edge)
    prob_matrix = jnp.diff(cdf_values, axis=1)

    # Numerical stability: Add epsilon to prevent log(0)
    return prob_matrix + 1e-30


# ==============================================================================
# Smoothness Prior
# ==============================================================================


def generate_smooth_curve(N_bins, smoothness_sigma, bin_width=1.0):
    """
    Generates a 1D smooth curve from an intrinsic RW1 (random-walk) GMRF prior.

    Unlike a cumulative-sum construction that pins ``curve[0] = 0`` and
    leaves it there, this is translation-invariant in bin index: every bin
    is regularised identically by its neighbours, with no special edge bin.
    This matters because the output is later passed through softmax, which
    is itself shift-invariant — an asymmetric prior on the un-normalised
    curve would otherwise bias the *shape* of the inferred LOSVD toward one
    edge of the velocity grid.

    The construction is a genuinely *generative* one — every random quantity
    is drawn via ``numpyro.sample``, and the returned curve is a purely
    deterministic function of those draws::

        sigma_step = smoothness_sigma * sqrt(bin_width)
        step[i] ~ Normal(0, sigma_step),  i = 1 .. N_bins - 1
        raw = concat([0], cumsum(step))              # pinned at bin 0
        curve = raw - mean(raw)                       # remove the pin

    This is deliberately **not** implemented as a ``numpyro.factor`` penalty
    on an unconditioned base measure. A factor only reweights the *posterior*
    density used by NUTS — it has no effect on prior-predictive / ancestral
    sampling (e.g. ``numpyro.infer.Predictive`` with no data conditioning),
    which forward-simulates only through ``sample`` sites. A factor-based
    version of this prior would therefore look correct under real inference
    (NUTS integrates the full unnormalised density regardless of site
    structure) while silently drawing the wrong thing under
    ``Predictive`` — exactly the kind of prior/posterior mismatch that
    simulation-based calibration (see ``tests/test_calibration.py``) exists
    to catch, and did catch during development of this function. The
    increments-then-center construction above sidesteps the issue entirely:
    ``step`` is what NUTS samples, and its distribution *is* the actual
    prior, with no separate normalising-constant bookkeeping required
    (``dist.Normal`` supplies it automatically) and no discrepancy between
    what NUTS conditions on and what ``Predictive`` draws.

    The final ``curve - mean(curve)`` centering is what makes this
    translation-invariant rather than reproducing the old pinned-at-bin-0
    asymmetry: subtracting the sample mean of the pinned random walk
    produces a bowl-shaped, bin-index-symmetric variance profile (verified
    numerically — ``Var(curve[k])`` is symmetric under ``k -> N_bins-1-k``),
    rather than the monotonically increasing ``Var(curve[k]) = k * sigma^2``
    of the raw pinned walk.

    ``smoothness_sigma`` is a *physical* smoothness scale, independent of the
    velocity-grid resolution: the actual per-bin random-walk step scale is
    ``smoothness_sigma * sqrt(bin_width)`` (the natural Brownian-motion
    scaling — a random walk that traverses a fixed velocity range with twice
    as many, half-width steps has each step scaled by ``sqrt(0.5)``, not
    ``0.5``). Without this scaling, refining the velocity grid silently
    changes what the prior means, since ``smoothness_sigma`` would then be a
    prior on the *per-bin* step regardless of how much velocity each bin
    spans.

    Parameters
    ----------
    N_bins : int
        Number of velocity bins.
    smoothness_sigma : float
        Physical smoothness scale, independent of grid resolution.
        - Low sigma: Stiff, very smooth.
        - High sigma: Flexible, jagged.
    bin_width : float
        Width of one velocity bin, used to rescale ``smoothness_sigma`` into
        the per-bin step scale actually used by the random walk. Default 1.0
        (i.e. no rescaling — matches the original per-bin-step convention if
        the caller does not know or care about physical units).

    Returns
    -------
    curve : jnp.ndarray (N_bins,)
        Latent log-density curve.
    """
    sigma_step = smoothness_sigma * jnp.sqrt(bin_width)

    steps = numpyro.sample("steps", dist.Normal(0.0, sigma_step).expand([N_bins - 1]).to_event(1))
    raw = jnp.concatenate([jnp.zeros(1), jnp.cumsum(steps)])
    curve = raw - jnp.mean(raw)

    return curve


@lru_cache(maxsize=None)
def _rw_deviation_scale(n_bins, order=3):
    """Sorbye-Rue scaling constant for the constrained RW-k deviation.

    Returns the factor that makes the generalised variance -- the geometric
    mean of the per-bin marginal variances -- of the projected k-fold
    integrated random walk equal to 1, so that ``sigma3`` means "typical
    log-density departure from the null space", independent of grid
    resolution (Sorbye & Rue 2014, Spatial Statistics 8, 39; this is what
    ``scale.model=TRUE`` does in R-INLA).

    ``order`` sets the null space, and the null space is what the prior does
    NOT shrink:

    - order 3: null space {1, u, u^2}, i.e. quadratic log-densities.
      Gaussians are unpenalised, so v and sigma are free, but h3 and h4 are
      shrunk -- they are the first things inside the penalised space.
    - order 4: adds u^3. Skewness becomes free.
    - order 5: adds u^4. Skewness and kurtosis are both free, and the penalty
      suppresses only structure beyond fourth order.

    Depends only on ``n_bins`` and ``order``. Cached because it costs an
    O(n^3) QR and matrix product and is evaluated at JAX trace time, where
    the result is constant-folded into the compiled model.
    """
    idx = np.arange(n_bins, dtype=float)
    u = (idx - idx.mean()) / (n_bins - 1)
    basis = np.stack([u**k for k in range(order)], axis=1)
    q, _ = np.linalg.qr(basis)
    proj = np.eye(n_bins) - q @ q.T
    lo = np.tril(np.ones((n_bins, n_bins)))
    a = np.linalg.matrix_power(lo, order)  # k-fold cumulative-sum operator
    var = np.clip(np.diag(proj @ (a @ a.T) @ proj.T), 1e-300, None)
    return float(1.0 / np.sqrt(np.exp(np.mean(np.log(var)))))


def _legendre_basis(u, order):
    x = 2.0 * u / jnp.max(jnp.abs(u))
    cols = [jnp.ones_like(x)]
    if order > 1:
        cols.append(x)
    for k in range(1, order - 1):
        cols.append(((2 * k + 1) * x * cols[k] - k * cols[k - 1]) / (k + 1))
    return jnp.stack(cols[:order], axis=1)


def generate_gaussian_core_curve(N_bins, centers, bin_width=1.0, rw_order=3):
    """
    Generate a latent log-density curve whose smoothness prior has a
    *Gaussian* null space rather than a flat one.

    This is the discrete, generative analogue of the Silverman (1982)
    roughness penalty used by Merritt (1997, AJ, 114, 228)::

        P(N) = integral [ d^3/dV^3 log N(V) ]^2 dV

    A Gaussian's log-density is exactly quadratic in velocity, so its third
    derivative vanishes identically and the penalty is exactly zero for any
    Gaussian. The infinite-smoothing limit is therefore a Gaussian with the
    data's own mean and dispersion -- not the uniform-over-the-grid limit of
    the first-difference prior in :func:`generate_smooth_curve`.

    That difference is the whole point. A flat null space means that
    wherever the data is uninformative the posterior relaxes toward putting
    probability mass everywhere out to the grid edge. Because kurtosis
    weights deviations by the fourth power, a bin at 5 sigma carries ~625x
    the weight of a bin at 1 sigma, so even a small amount of misplaced edge
    mass produces a large positive kurtosis bias -- and, because the grid is
    wider than the true distribution, a positive velocity-dispersion bias
    that grows with the number of bins.

    The construction splits the curve into a free core and a penalised
    deviation::

        core  = -0.5 * ((v - v0) / s0)^2                  # unpenalised
        w     = cumsum(cumsum(cumsum(d3 * sigma3)))       # triple-integrated RW
        dev   = w - Q (Q^T w),  Q = orth basis of {1, u, u^2}
        curve = core + dev

    ``v0`` and ``s0`` are the Gaussian null-space parameters and carry no
    smoothness penalty at all. ``dev`` is projected orthogonal to the
    quadratic subspace so it cannot mimic the core; without that projection
    the two terms trade off freely, ``v0``/``s0`` become unidentifiable, and
    NUTS diverges. The projection is performed in the normalised coordinate
    ``u = (v - mean(v)) / (max(v) - min(v))`` because a QR factorisation of
    the raw velocity Vandermonde ``[1, v, v^2]`` is catastrophically
    ill-conditioned when velocities are of order hundreds.

    Like :func:`generate_smooth_curve` this is a genuinely *generative*
    construction -- every random quantity is drawn via ``numpyro.sample``,
    never ``numpyro.factor``. A factor-based penalty is invisible to
    ``numpyro.infer.Predictive``, so the model would behave correctly under
    NUTS while silently drawing the wrong thing under prior-predictive
    sampling and SBC. That exact bug has already been hit and fixed once in
    this codebase; do not reintroduce it.

    ``d3`` is drawn standard-normal and scaled by ``sigma3`` afterwards
    (non-centred parameterisation) rather than drawn from
    ``Normal(0, sigma3)`` directly, which would produce Neal's funnel and
    divergences.

    ``sigma3`` is standardised so that the *generalised variance* of the
    projected deviation -- the geometric mean of its per-bin marginal
    variances -- is exactly 1 (see :func:`_rw3_deviation_scale`). This is the
    standard Sorbye & Rue (2014) treatment for intrinsic GMRFs, and it makes
    ``sigma3`` directly interpretable as the typical log-density departure
    from a Gaussian LOSVD, independent of grid resolution.

    An earlier version instead multiplied by ``(bin_width / span) ** 2.5``.
    That exponent correctly cancels the resolution dependence -- the measured
    deviation scale drifts only ~11% between 20 and 120 bins -- but it lands
    on a constant of ~0.0036 rather than 1. A deviation that small is
    invisible to the likelihood, so the posterior collapsed onto the pure
    Gaussian null space regardless of the data. Do not reintroduce a
    dimensional or hand-tuned constant here; the standardisation is exact.

    The prior on ``sigma3`` is a penalised-complexity prior (Simpson et al.):
    an ``Exponential`` whose base model, ``sigma3 = 0``, is exactly a Gaussian
    LOSVD. The rate is set by ``SIGMA3_RATE``. This shrinks toward
    Gaussianity, which is the physically right default, while leaving
    strongly non-Gaussian shapes reachable when the data demand them.

    Parameters
    ----------
    N_bins : int
        Number of velocity bins.
    centers : array-like, shape (N_bins,)
        Physical centres of the velocity bins. Required because the Gaussian
        core is quadratic in *velocity*, not in bin index.
    bin_width : float
        Unused by this prior -- the Sorbye-Rue standardisation is already
        resolution-invariant, and the latent curve is in dimensionless
        log-mass units, so no physical scale enters the deviation. Retained
        for signature compatibility with ``generate_smooth_curve`` and
        ``model_gaussian_core``.

    Returns
    -------
    curve : jnp.ndarray, shape (N_bins,)
        Latent log-density curve, to be passed through ``softmax``.
    """
    centers = jnp.asarray(centers)
    span = jnp.max(centers) - jnp.min(centers)
    mid = jnp.mean(centers)

    # --- Gaussian null space: free, unpenalised ---
    # LogNormal rather than HalfNormal/HalfCauchy on the core width. Both
    # half-distributions put substantial mass near zero, and an s0 near zero
    # collapses the core onto a single bin -- measured prior-predictive
    # median sigma of exactly 0.00, i.e. a delta function, for >99% of
    # draws. LogNormal is bounded away from both 0 and infinity: median
    # span/8 with ~1 dex of spread either side.
    v0 = numpyro.sample("v0", dist.Normal(mid, span / 4.0))
    s0 = numpyro.sample("s0", dist.LogNormal(jnp.log(span / 8.0), 1.0))
    core = -0.5 * ((centers - v0) / jnp.clip(s0, 1e-3)) ** 2

    # --- penalised non-Gaussian deviation ---
    sigma3 = numpyro.sample("sigma3", dist.Exponential(SIGMA3_RATE)) * _rw_deviation_scale(N_bins, rw_order)
    d3 = numpyro.sample("d3", dist.Normal(0.0, 1.0).expand([N_bins]).to_event(1))
    w = d3 * sigma3
    for _ in range(rw_order):
        w = jnp.cumsum(w)

    # Project out the null space: polynomials of degree < rw_order.
    u = (centers - mid) / span
    q, _ = jnp.linalg.qr(_legendre_basis(u, rw_order))
    deviation = w - q @ (q.T @ w)

    return core + deviation


# ==============================================================================
# Model Inference
# ==============================================================================


def model(matrix, n_bins, bin_width=1.0):
    """
    The Model definition for NumPyro.

    Parameters
    ----------
    matrix : jnp.ndarray (N_stars, N_bins)
        The pre-computed Design Matrix M.
    n_bins : int
        Number of velocity bins.
    bin_width : float
        Width of one velocity bin. Used to make ``smoothness_sigma`` a
        physical smoothness scale independent of grid resolution (see
        ``generate_smooth_curve``). Default 1.0 preserves the original
        per-bin-step convention when the caller does not pass a physical
        bin width.

    Returns
    -------
    None
        This function defines the probabilistic graph and has no return value.
    """
    # --- Hyperparameters ---

    # We infer the smoothness sigma from the data.
    # HalfNormal(0.1) is a relatively conservative prior, favoring smooth curves.
    smoothness_sigma = numpyro.sample("smoothness_sigma", dist.HalfNormal(0.1))

    # --- Latent Density ---
    # Generate the log-density curve using the Random Walk prior. This is
    # already translation-invariant (see generate_smooth_curve), so no
    # separate centering step is needed here — softmax is shift-invariant
    # regardless.
    latent_curve = generate_smooth_curve(n_bins, smoothness_sigma, bin_width)

    # Convert latent values to valid Probabilities (Sum = 1)
    intrinsic_pdf = jax.nn.softmax(latent_curve)

    # Save the PDF for plotting and later use
    numpyro.deterministic("intrinsic_pdf", intrinsic_pdf)

    # --- Likelihood ---

    # The probability of observing Star i is the weighted sum of the
    # probabilities of it coming from each bin.
    # P(Star_i) = DotProduct( Row_i, Intrinsic_Weights )
    # This is a normalised-shape likelihood: intrinsic_pdf sums to 1 by
    # construction (softmax), so there is no free "total flux" parameter to
    # infer — the number of stars observed does not carry information about
    # the *shape* of the LOSVD beyond what per_star_prob already captures.
    per_star_prob = jnp.dot(matrix, intrinsic_pdf)
    log_prob = jnp.sum(jnp.log(per_star_prob))

    # Register the likelihood with NumPyro
    numpyro.factor("obs_log_lik", log_prob)


def model_gaussian_core(matrix, n_bins, centers, bin_width=1.0, rw_order=3):
    """
    NumPyro model using the Gaussian-null-space prior.

    Identical to :func:`model` except that the latent log-density curve comes
    from :func:`generate_gaussian_core_curve` instead of
    :func:`generate_smooth_curve`. See that function's docstring for why the
    null space matters.

    Parameters
    ----------
    matrix : jnp.ndarray, shape (N_stars, N_bins)
        Pre-computed design matrix M.
    n_bins : int
        Number of velocity bins.
    centers : array-like, shape (N_bins,)
        Physical centres of the velocity bins.
    bin_width : float
        Width of one velocity bin.

    Returns
    -------
    None
        Defines the probabilistic graph; has no return value.
    """
    latent_curve = generate_gaussian_core_curve(n_bins, centers, bin_width, rw_order)
    intrinsic_pdf = jax.nn.softmax(latent_curve)
    numpyro.deterministic("intrinsic_pdf", intrinsic_pdf)

    per_star_prob = jnp.dot(matrix, intrinsic_pdf)
    numpyro.factor("obs_log_lik", jnp.sum(jnp.log(per_star_prob)))


# ==============================================================================
# Solver Class
# ==============================================================================


class KinematicSolver:
    """
    A high-level interface for performing Bayesian kinematic deconvolution.

    This class manages the full inference workflow:
    1.  Defining the velocity grid (``setup_grid``).
    2.  Ingesting data and building the design matrix (``add_data``).
    3.  Running the MCMC sampler (``run``).
    4.  Visualizing the results (``plot_result``).

    Attributes
    ----------
    matrix : jnp.ndarray or None
        The pre-computed design matrix of shape (N_stars, N_bins).
    grid : dict
        Metadata defining the velocity grid (centers, edges, width).
    n_stars : int or None
        Number of stars loaded via ``add_data``.  Used as the ``bin_flux``
        analog when writing Dynamite output (see ``write_dynamite_kinematics``).
    samples : dict or None
        Posterior samples from the MCMC run.
    clipped_samples : dict or None
        Per-bin summary statistics (median LOSVD and clipped uncertainties)
        populated by ``clip_uncertainties``.
    """

    def __init__(self):
        """
        Initializes the KinematicSolver instance.
        """
        self.matrix = None
        self.grid = {}
        self.n_stars = None
        self.samples = None
        self.clipped_samples = None

    def setup_grid(self, center, width, n_bins):
        """
        Define the velocity grid (histogram bins).

        Parameters
        ----------
        center : float
            Center of the velocity grid.
        width : float
            Total width of the velocity grid.
        n_bins : int
            Number of bins in the grid.

        Returns
        -------
        None
            Sets ``self.grid``.
        """
        edges = np.linspace(center - width / 2, center + width / 2, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.grid = {
            "centers": centers,
            "edges": edges,
            "width": edges[1] - edges[0],
            "n_bins": n_bins,
        }

    def add_data(self, vel, err):
        """
        Load observations and pre-compute the design matrix.

        Parameters
        ----------
        vel : array-like
            Observed velocities of stars.
        err : array-like
            Measurement errors associated with the velocities.

        Returns
        -------
        None
            Sets ``self.matrix``.
        """
        if not self.grid:
            msg = "Run setup_grid() first."
            raise ValueError(msg)

        self.n_stars = len(vel)
        print(f"Computing Design Matrix for {self.n_stars} stars...")

        # This one-time computation replaces the convolution loop
        self.matrix = precompute_design_matrix(vel, err, self.grid["centers"], bin_width=self.grid["width"])
        print(f"Matrix ready. Shape: {self.matrix.shape}")

    def run(self, num_warmup=500, num_samples=1000, gpu=None, seed=5567, prior="gaussian_core", rw_order=3):
        """
        Run the NUTS sampler.

        Parameters
        ----------
        num_warmup : int
            Number of warmup (burn-in) steps.
        num_samples : int
            Number of MCMC samples to draw.
        gpu : bool or None
            If True, request GPU acceleration via
            ``numpyro.set_platform("gpu")`` (raises if none is available).
            If False, force CPU. If None (default), leave the platform
            untouched — i.e. whatever was configured before calling ``run()``.
        seed : int
            RNG seed for the NUTS sampler. Default 5567 (kept for backwards
            compatibility). When running many bins in a batch, pass distinct
            seeds per bin to avoid any correlation in the sampling chains; a
            simple convention is ``seed + bin_index`` (see ``fit_all_bins``).
        prior : {"rw1", "gaussian_core"}
            Which smoothness prior to use. ``"gaussian_core"`` (default) uses
            :func:`generate_gaussian_core_curve`, whose infinite-smoothing
            limit is a Gaussian (Merritt 1997, AJ, 114, 228) and which does
            not show the kurtosis/velocity-dispersion biases of the RW1
            prior. ``"rw1"`` is the original first-difference random walk,
            retained for comparison; its infinite-smoothing limit is a
            *uniform* LOSVD over the velocity grid, known to bias kurtosis
            high by ~+1.1 and dispersion high by ~4-8%.

        Returns
        -------
        samples : dict
            Posterior samples (e.g., "intrinsic_pdf", "smoothness_sigma").
        """
        if self.matrix is None:
            msg = "No data added."
            raise ValueError(msg)

        if prior not in ("rw1", "gaussian_core"):
            msg = f"Unknown prior {prior!r}; expected 'rw1' or 'gaussian_core'."
            raise ValueError(msg)

        if gpu is True:
            numpyro.set_platform("gpu")
        elif gpu is False:
            numpyro.set_platform("cpu")

        print("Starting NUTS MCMC...")
        model_kwargs = {
            "matrix": self.matrix,
            "n_bins": self.grid["n_bins"],
            "bin_width": self.grid["width"],
        }
        if prior == "gaussian_core":
            model_fn = model_gaussian_core
            model_kwargs["centers"] = jnp.asarray(self.grid["centers"])
            model_kwargs["rw_order"] = rw_order
        else:
            model_fn = model

        nuts_kernel = NUTS(model_fn)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)

        rng_key = jax.random.PRNGKey(int(seed))
        mcmc.run(rng_key, **model_kwargs)

        self.samples = mcmc.get_samples()
        # Invalidate any previously computed clipped summary when re-running.
        self.clipped_samples = None
        print("Inference Complete.")
        return self.samples

    def plot_result(self, ax=None, true_intrinsic=None):
        """
        Visualize the inferred LOSVD.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        true_intrinsic : array-like, optional
            True intrinsic velocities for comparison (if available).
        Returns
        -------
        ax : matplotlib.axes.Axes
            The plot axes.
        """
        # intrinsic_pdf is always probability MASS internally (each row sums
        # to 1) — this is what the likelihood, the simplex constraint, and
        # the Dynamite ECSV writer all expect. Density is purely a plotting
        # convention, so the Mass -> Density conversion happens here only.
        # Density = Mass / Bin_Width
        pdf_samples_mass = self.samples["intrinsic_pdf"]
        dx = self.grid["width"]

        # Convert to Density
        pdf_samples_density = pdf_samples_mass / dx

        mean_pdf = jnp.mean(pdf_samples_density, axis=0)
        # 68 percent credible interval
        lower = jnp.percentile(pdf_samples_density, 16, axis=0)
        upper = jnp.percentile(pdf_samples_density, 84, axis=0)

        if ax is None:
            fig, ax = plt.subplots()
        x = self.grid["centers"]

        # Plot Model, as a mid-step plot
        ax.step(x, mean_pdf, where="mid", color="tab:green", label="Inferred Intrinsic")
        ax.fill_between(
            x,
            lower,
            upper,
            color="tab:green",
            alpha=0.3,
            step="mid",
        )

        # Plot Truth
        if true_intrinsic is not None:
            if callable(true_intrinsic):
                # If function, evaluate on a fine grid for smooth curve
                x_fine = np.linspace(
                    self.grid["edges"][0],
                    self.grid["edges"][-1],
                    500,  # High resolution
                )
                y_true = true_intrinsic(x_fine)
                ax.plot(
                    x_fine,
                    y_true,
                    color="k",
                    ls="--",
                    linewidth=1.5,
                    label="True Intrinsic",
                )
            else:
                # If samples, histogram with density=True
                true_hist, _ = np.histogram(true_intrinsic, bins=self.grid["edges"], density=True)
                ax.plot(x, true_hist, color="k", ls="--", label="True Intrinsic")

        ax.set_xlabel("Velocity")
        ax.set_ylabel("Probability Density")
        ax.legend()
        return ax

    def clip_uncertainties(self, floor_fraction=0.01, abs_floor=1e-10):
        """
        Apply uncertainty floors and store LOSVD summary statistics.

        This is a **post-processing step** that does *not* modify the raw
        posterior samples in ``self.samples``.  It summarises the posterior
        as per-bin marginal medians and half-CI-widths in probability-mass
        space, then raises the uncertainties to a floor so that no bin carries
        a zero into the Dynamite output writer.

        Output format matches the Dynamite ``BayesLOSVD`` ECSV convention
        (see ``context/dynamite_format_spec.md``):

        - ``losvd_median`` stores the per-bin **marginal median** of the
          posterior probability mass.  Because the joint posterior is a
          simplex but marginals are taken independently, the median values
          typically *sum to 0.85–0.95*, not 1.  This is expected and correct.
        - ``losvd_uncertainty`` stores the **half-width** of the 68% credible
          interval: ``(p84 − p16) / 2``.  Used as symmetric ±error bars.

        Both quantities are **dimensionless probability mass per bin** — they
        are *not* divided by the bin width.

        Motivation
        ----------
        Zero uncertainties in LOSVD bins propagate into Dynamite's internal
        NNLS projection matrices and produce ``econ`` zeros that cause weight-
        solving failures in large orbit-library runs.  The relative floor
        (``floor_fraction * max_uncertainty``) is the primary safeguard; the
        absolute floor is a numerical backstop for channels where the posterior
        is pathologically tight across the board.

        Parameters
        ----------
        floor_fraction : float
            Relative floor as a fraction of the maximum per-bin half-CI-width
            across all bins.  Default 0.01 (1%).
        abs_floor : float
            Absolute floor applied after the relative floor.  Default 1e-10.

        Returns
        -------
        None
            Sets ``self.clipped_samples`` as a dict with keys:

            - ``'losvd_median'``      — per-bin marginal median, probability
              mass (dimensionless); shape (n_bins,).
            - ``'losvd_uncertainty'`` — clipped half-width of 68% CI,
              probability mass; shape (n_bins,).
        """
        if self.samples is None:
            msg = "No posterior samples found. Call run() before clip_uncertainties()."
            raise ValueError(msg)

        # Work in probability-mass space throughout.
        # self.samples["intrinsic_pdf"] has shape (n_samples, n_bins);
        # each row is a valid probability mass function (sums to 1).
        pdf_mass = np.asarray(self.samples["intrinsic_pdf"])

        # Sanity check: the MEAN of valid mass samples must also sum to ~1.
        mean_mass = np.mean(pdf_mass, axis=0)
        mean_sum = np.sum(mean_mass)
        if not np.isclose(mean_sum, 1.0, rtol=1e-3):
            msg = (
                f"Posterior mean LOSVD sums to {mean_sum:.6f}, expected ~1.0. "
                "Check that self.samples['intrinsic_pdf'] contains valid probability "
                "mass functions (each row should sum to 1)."
            )
            raise ValueError(msg)

        # Per-bin marginal statistics.
        median_mass = np.percentile(pdf_mass, 50, axis=0)
        p16 = np.percentile(pdf_mass, 16, axis=0)
        p84 = np.percentile(pdf_mass, 84, axis=0)

        # Half-width of 68% CI (used as symmetric ±uncertainty in Dynamite).
        raw_half_width = (p84 - p16) / 2.0

        # Relative floor: a fraction of the widest half-CI in this LOSVD.
        rel_floor = floor_fraction * np.max(raw_half_width)

        clipped = np.maximum(raw_half_width, rel_floor)
        clipped = np.maximum(clipped, abs_floor)

        self.clipped_samples = {
            "losvd_median": median_mass,
            "losvd_uncertainty": clipped,
        }

    def truncate_losvd(self, n_sigma=3.0, abs_floor=1e-10):
        """
        Suppress LOSVD bins beyond ``n_sigma`` dispersions from the bulk mean.

        .. note::
            **This method should only be called if tail contamination is
            observed to be causing problems** (e.g. non-negligible posterior
            mass in channels well outside the range of the input data, or
            ``econ`` zeros in Dynamite that persist after
            ``clip_uncertainties``).  It is *not* part of the standard
            pipeline and is not called by :func:`fit_all_bins`.  When in
            doubt, omit it — modifying the posterior summary without a clear
            diagnostic reason introduces unnecessary bias.

        Velocity channels far from the data carry unphysical posterior weight
        due to the random-walk prior leaking into unconstrained tails.  This
        method zeros the mean density in those channels and sets their
        uncertainty to ``abs_floor``, preventing Dynamite from trying to fit
        orbits in unphysical regimes.

        Approach follows the velocity-range truncation strategy discussed in
        Falcón-Barroso & Martig (2021, A&A).

        This is a **post-processing step** and does *not* modify
        ``self.samples``.  It operates on ``self.clipped_samples``, creating
        it first (with default floors) if it is not already present.

        Parameters
        ----------
        n_sigma : float
            Number of velocity dispersions beyond which to truncate.
            Default 3.0.
        abs_floor : float
            Uncertainty value assigned to truncated bins (must be > 0 to
            avoid ``econ`` zeros in Dynamite).  Default 1e-10.

        Returns
        -------
        None
            Updates ``self.clipped_samples`` in-place.
        """
        if self.samples is None:
            msg = "No posterior samples found. Call run() before truncate_losvd()."
            raise ValueError(msg)

        if self.clipped_samples is None:
            self.clip_uncertainties()

        centers = np.asarray(self.grid["centers"])

        # Use the posterior-mean LOSVD (probability mass) to compute the bulk
        # mean velocity and velocity dispersion of the distribution.
        mean_pdf_mass = np.mean(np.asarray(self.samples["intrinsic_pdf"]), axis=0)

        v_mean = np.dot(centers, mean_pdf_mass)
        v_var = np.dot((centers - v_mean) ** 2, mean_pdf_mass)
        v_std = np.sqrt(v_var)

        truncation_mask = np.abs(centers - v_mean) > n_sigma * v_std

        losvd_median = self.clipped_samples["losvd_median"].copy()
        losvd_unc = self.clipped_samples["losvd_uncertainty"].copy()

        losvd_median[truncation_mask] = 0.0
        losvd_unc[truncation_mask] = abs_floor

        self.clipped_samples["losvd_median"] = losvd_median
        self.clipped_samples["losvd_uncertainty"] = losvd_unc


# ==============================================================================
# Batch API
# ==============================================================================


def fit_all_bins(bin_data_list, grid_kwargs, run_kwargs=None, min_stars=10):
    """
    Run the full inference pipeline for a list of Voronoi bins.

    For each bin, this executes the ``setup_grid`` → ``add_data`` → ``run``
    → ``clip_uncertainties`` pipeline and returns a list of
    :class:`KinematicSolver` instances ready for the Dynamite output writer.
    Bins with too few stars are skipped (returning ``None`` at that position)
    so the writer can mask them.

    :meth:`~KinematicSolver.truncate_losvd` is deliberately *not* called
    here.  It is an optional diagnostic repair step; call it manually on
    individual solvers only if tail contamination is observed to be causing
    problems.

    The same velocity grid is used for every bin (``grid_kwargs`` is shared).
    Each bin receives a unique RNG seed derived as ``base_seed + bin_index``
    to avoid correlations between sampling chains.

    Parameters
    ----------
    bin_data_list : list of dict
        One dict per Voronoi bin.  Required keys:

        - ``'vel'`` — array of observed stellar velocities.
        - ``'err'`` — array of per-star measurement errors.

        Any additional keys (e.g. spatial metadata) are ignored here and
        can be passed separately to the output writer.
    grid_kwargs : dict
        Keyword arguments forwarded to :meth:`KinematicSolver.setup_grid`
        (``center``, ``width``, ``n_bins``).  Shared across all bins.
    run_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`KinematicSolver.run`
        (e.g. ``num_warmup``, ``num_samples``, ``gpu``).  The ``seed``
        key, if present, is used as the *base* seed; each bin then receives
        ``seed + bin_index``.  Defaults to ``{}`` (all ``run`` defaults
        apply).
    min_stars : int
        Minimum number of stars required to attempt inference.  Bins with
        fewer stars are skipped with a warning.  Default 10.

    Returns
    -------
    solvers : list
        One entry per input bin.  Entries are either a fully solved
        :class:`KinematicSolver` (with ``samples`` and ``clipped_samples``
        populated) or ``None`` for skipped bins.

    Examples
    --------
    >>> solvers = fit_all_bins(
    ...     bin_data_list,
    ...     grid_kwargs={"center": 0.0, "width": 600.0, "n_bins": 60},
    ...     run_kwargs={"num_warmup": 500, "num_samples": 1000, "gpu": False},
    ... )
    >>> # Pass to the output writer (Task 2):
    >>> solver.write_dynamite_kinematics(output_dir, voronoi_bin_metadata)
    """
    if run_kwargs is None:
        run_kwargs = {}

    # Extract the base seed so we can derive per-bin seeds.
    run_kwargs = dict(run_kwargs)
    base_seed = run_kwargs.pop("seed", 5567)

    n_total = len(bin_data_list)
    solvers = []

    for i, bin_data in enumerate(bin_data_list):
        print(f"Fitting bin {i + 1}/{n_total}...")

        vel = np.asarray(bin_data["vel"])
        err = np.asarray(bin_data["err"])

        if len(vel) < min_stars:
            warnings.warn(
                f"Bin {i} has only {len(vel)} star(s) (minimum is {min_stars}). "
                "Skipping — this bin will appear as None in the output list and "
                "should be masked in the Dynamite input files.",
                stacklevel=2,
            )
            solvers.append(None)
            continue

        solver = KinematicSolver()
        solver.setup_grid(**grid_kwargs)
        solver.add_data(vel=vel, err=err)
        solver.run(seed=base_seed + i, **run_kwargs)
        solver.clip_uncertainties()

        solvers.append(solver)

    n_solved = sum(s is not None for s in solvers)
    n_skipped = n_total - n_solved
    print(f"Done. {n_solved}/{n_total} bins solved" + (f", {n_skipped} skipped." if n_skipped else "."))

    return solvers


# ==============================================================================
# Dynamite Output Writer
# ==============================================================================


def write_dynamite_kinematics(
    solvers,
    output_dir,
    voronoi_bin_metadata,
    kin_filename="bayes_losvd_kins.ecsv",
    aperture_filename="aperture.dat",
    bins_filename="bins.dat",
    bin_flux_mode="nstars",
):
    """
    Write Dynamite-compatible BayesLOSVD input files from a list of solved bins.

    Produces three files that Dynamite expects for its ``BayesLOSVD`` kinematics
    representation (see ``context/dynamite_format_spec.md`` for full format
    details):

    - ``{kin_filename}`` — Astropy ECSV, one row per solved Voronoi bin,
      containing the per-bin marginal median LOSVD and ±half-CI uncertainties.
    - ``{aperture_filename}`` — pixel grid geometry.
    - ``{bins_filename}`` — pixel-to-bin mapping.

    Any ``None`` entries in ``solvers`` (bins skipped by :func:`fit_all_bins`)
    are automatically masked: their pixels are written as 0 in the bins file
    and they are omitted from the kinematics table.  The remaining bins are
    re-numbered sequentially (1-indexed) as required by Dynamite.

    :func:`clip_uncertainties` is called automatically on any solver that has
    not already had its ``clipped_samples`` populated.

    Parameters
    ----------
    solvers : list
        Solved :class:`KinematicSolver` instances (or ``None`` for skipped
        bins), as returned by :func:`fit_all_bins`.  All non-``None`` entries
        must share the same velocity grid.
    output_dir : str or path-like
        Directory in which to write the three output files.  Created if it
        does not exist.
    voronoi_bin_metadata : dict
        Spatial and observational metadata.  Required structure::

            {
                'bins': [
                    {
                        'xbin': float,       # arcsec, bin centre x
                        'ybin': float,       # arcsec, bin centre y
                        # 'bin_flux' only read when bin_flux_mode='custom'
                        'bin_flux': float,
                    },
                    ...  # one entry per entry in solvers
                ],
                'aperture': {
                    'x_start':   float,   # arcsec, lower-left corner x
                    'y_start':   float,   # arcsec, lower-left corner y
                    'x_size':    float,   # arcsec, total x extent
                    'y_size':    float,   # arcsec, total y extent
                    'angle_deg': float,   # degrees (= 90 - position_angle)
                    'nx':        int,     # pixels along x
                    'ny':        int,     # pixels along y
                },
                # 1-indexed bin IDs (bin 1 = solvers[0], bin 2 = solvers[1], …)
                # 0 = masked.  Skipped bins are re-mapped to 0 automatically.
                'pixel_bin_ids': array-like,   # shape (nx*ny,) or (ny, nx)
                'psf': {
                    'sigma':  [float, ...],   # Gaussian sigma(s) in arcsec
                    'weight': [float, ...],   # must sum to 1
                },
            }

    kin_filename : str
        File name for the kinematics ECSV.  Default ``'bayes_losvd_kins.ecsv'``.
    aperture_filename : str
        File name for the aperture file.  Default ``'aperture.dat'``.
    bins_filename : str
        File name for the bins file.  Default ``'bins.dat'``.
    bin_flux_mode : {'nstars', 'uniform', 'custom'}
        Controls what is written to the ``bin_flux`` column.

        ``'nstars'`` *(default)*
            Use the number of stars in each bin
            (``solver.n_stars``, set by :meth:`~KinematicSolver.add_data`).
            This is the physically meaningful analog of IFU surface brightness
            for discrete stellar kinematic data.  ``bin_flux`` is used by
            Dynamite **only** for flux-weighted systemic velocity centering
            (``center_v_systemic``); it does **not** enter the NNLS chi²
            (confirmed from ``NNLS.construct_nnls_matrix_and_rhs`` and
            ``BayesLOSVD.get_observed_values_and_uncertainties`` in
            Dynamite's source).  N_stars is the physically appropriate
            quantity to pass for discrete stellar data.

        ``'uniform'``
            Write 1.0 for every bin.  Use this if you want equal weighting
            of all bins in any flux-weighted calculation, or if you are
            uncertain about the right quantity and want a neutral default.

        ``'custom'``
            Read ``bin_flux`` from ``voronoi_bin_metadata['bins'][i]['bin_flux']``
            for each bin.  Useful if you have an external flux estimate
            (e.g. sum of photometric counts in each Voronoi cell).

    Raises
    ------
    ValueError
        If no solved bins are found, or if solvers share inconsistent grids.
    AssertionError
        If any LOSVD uncertainty is ≤ 0 after clipping (would cause ``econ``
        zeros in Dynamite).

    Returns
    -------
    None
    """
    try:
        from astropy.table import Table
    except ImportError as exc:
        msg = "astropy is required for write_dynamite_kinematics(). " "Install it with: pip install astropy"
        raise ImportError(msg) from exc

    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Identify solved bins and validate grid consistency
    # ------------------------------------------------------------------
    solved_indices = [i for i, s in enumerate(solvers) if s is not None]
    n_solved = len(solved_indices)

    if n_solved == 0:
        msg = "No solved bins found (all solvers are None)."
        raise ValueError(msg)

    ref_solver = solvers[solved_indices[0]]
    vcent = np.asarray(ref_solver.grid["centers"])
    dv = float(ref_solver.grid["width"])
    nvbins = int(ref_solver.grid["n_bins"])

    for idx in solved_indices[1:]:
        s = solvers[idx]
        if s.grid["n_bins"] != nvbins or not np.allclose(s.grid["centers"], vcent):
            msg = (
                f"Solver at index {idx} has a different velocity grid than "
                f"solver at index {solved_indices[0]}. All bins must share "
                "the same grid (set via setup_grid / fit_all_bins)."
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Gather LOSVD summaries (auto-clip if needed)
    # ------------------------------------------------------------------
    bin_metas = voronoi_bin_metadata["bins"]
    psf = voronoi_bin_metadata.get("psf", {"sigma": [1.0], "weight": [1.0]})

    losvd_all = np.zeros((n_solved, nvbins))
    dlosvd_all = np.zeros((n_solved, nvbins))

    for out_i, orig_i in enumerate(solved_indices):
        solver = solvers[orig_i]
        if solver.clipped_samples is None:
            solver.clip_uncertainties()
        losvd_all[out_i] = solver.clipped_samples["losvd_median"]
        dlosvd_all[out_i] = solver.clipped_samples["losvd_uncertainty"]

    # Guard: no zero uncertainties (would corrupt Dynamite's NNLS matrices).
    if not np.all(dlosvd_all > 0):
        msg = (
            "Zero or negative uncertainty found after clipping. "
            "This would cause econ zeros in Dynamite. "
            "Check clip_uncertainties() floor settings."
        )
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Compute v and sigma from normalised median LOSVD (per format spec)
    # ------------------------------------------------------------------
    losvd_sums = losvd_all.sum(axis=1, keepdims=True)  # (n_solved, 1)
    losvd_norm = losvd_all / losvd_sums  # l_hat, sums to 1
    v_vals = (vcent[np.newaxis, :] * losvd_norm).sum(axis=1)
    sigma_vals = np.sqrt((((vcent[np.newaxis, :] - v_vals[:, np.newaxis]) ** 2) * losvd_norm).sum(axis=1))

    # ------------------------------------------------------------------
    # Write kinematics ECSV
    # ------------------------------------------------------------------
    data = {}
    data["binID_BayesLOSVD"] = np.arange(n_solved)

    # Compute bin_flux according to requested mode.
    if bin_flux_mode == "nstars":
        missing = [solved_indices[k] for k, idx in enumerate(solved_indices) if solvers[idx].n_stars is None]
        if missing:
            msg = (
                f"bin_flux_mode='nstars' requires that add_data() was called "
                f"on every solver, but solvers at indices {missing} have "
                "n_stars=None.  Call add_data() before fit_all_bins(), or "
                "use bin_flux_mode='uniform' / 'custom' instead."
            )
            raise ValueError(msg)
        bin_flux_vals = np.array([float(solvers[i].n_stars) for i in solved_indices])
    elif bin_flux_mode == "uniform":
        bin_flux_vals = np.ones(n_solved, dtype=float)
    elif bin_flux_mode == "custom":
        bin_flux_vals = np.array([float(bin_metas[i].get("bin_flux", 1.0)) for i in solved_indices])
    else:
        msg = f"bin_flux_mode must be 'nstars', 'uniform', or 'custom'; " f"got {bin_flux_mode!r}."
        raise ValueError(msg)
    data["bin_flux"] = bin_flux_vals

    data["binID_dynamite"] = np.arange(1, n_solved + 1)
    data["v"] = v_vals
    data["sigma"] = sigma_vals
    data["xbin"] = np.array([bin_metas[i]["xbin"] for i in solved_indices])
    data["ybin"] = np.array([bin_metas[i]["ybin"] for i in solved_indices])

    # Interleave losvd_j and dlosvd_j columns (required column order).
    for j in range(nvbins):
        data[f"losvd_{j}"] = losvd_all[:, j]
        data[f"dlosvd_{j}"] = dlosvd_all[:, j]

    table = Table(data)
    # Metadata written in !!omap order to match BayesLOSVD convention.
    table.meta["dv"] = dv
    table.meta["vcent"] = vcent.tolist()
    table.meta["nbins"] = n_solved
    table.meta["nvbins"] = nvbins
    table.meta["PSF"] = psf

    kin_path = output_dir / kin_filename
    table.write(str(kin_path), format="ascii.ecsv", overwrite=True)
    print(f"Written kinematics ({n_solved} bins): {kin_path}")

    # ------------------------------------------------------------------
    # Write aperture.dat
    # ------------------------------------------------------------------
    ap = voronoi_bin_metadata["aperture"]
    ap_path = output_dir / aperture_filename
    with open(ap_path, "w") as f:
        f.write("#counter_rotation_boxed_aperturefile_version_2 \n")
        f.write(f"\t{ap['x_start']:f}\t{ap['y_start']:f} \n")
        f.write(f"\t{ap['x_size']:f}\t{ap['y_size']:f} \n")
        f.write(f"\t{ap['angle_deg']:f} \n")
        f.write(f"\t{ap['nx']}\t{ap['ny']} \n")
    print(f"Written aperture: {ap_path}")

    # ------------------------------------------------------------------
    # Write bins.dat
    # Re-map original 1-indexed bin IDs to sequential solved-only IDs.
    # Pixels for skipped bins become 0 (masked).
    # ------------------------------------------------------------------
    # Build mapping: original 1-indexed ID -> new 1-indexed ID (0 if skipped).
    n_total = len(solvers)
    orig_to_new = np.zeros(n_total + 1, dtype=int)  # index 0 unused
    for new_id, orig_i in enumerate(solved_indices, start=1):
        orig_to_new[orig_i + 1] = new_id  # orig_i is 0-based; +1 for 1-based

    pixel_ids = np.asarray(voronoi_bin_metadata["pixel_bin_ids"]).flatten().astype(int)
    remapped = np.where(
        (pixel_ids > 0) & (pixel_ids <= n_total),
        orig_to_new[pixel_ids],
        0,
    )
    total_pixels = len(remapped)

    bins_path = output_dir / bins_filename
    with open(bins_path, "w") as f:
        f.write("#Counterrotation_binning_version_1\n")
        f.write(f"{total_pixels}\n")
        for start in range(0, total_pixels, 10):
            chunk = remapped[start : start + 10]
            f.write("\t" + "\t".join(str(v) for v in chunk) + "\n")
    print(f"Written bins: {bins_path}")
