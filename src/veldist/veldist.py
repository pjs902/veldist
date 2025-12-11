# file src/veldist/veldist.py
"""
Bayesian Matrix-Based Kinematic Deconvolution
=============================================

This module infers the intrinsic Line-of-Sight Velocity Distribution (LOSVD)
from discrete, heteroscedastic stellar observations using a pre-computed
linear design matrix and a hierarchical smoothness prior.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

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
    edges = jnp.concatenate(
        [bin_centers - bin_width / 2, jnp.array([bin_centers[-1] + bin_width / 2])]
    )

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


def generate_smooth_curve(N_bins, smoothness_sigma):
    """
    Generates a 1D smooth curve using a Gaussian Random Walk.


    Model:
        step[i] ~ Normal(0, sigma)
        curve[i] = curve[i-1] + step[i]

    Parameters
    ----------
    N_bins : int
        Number of velocity bins.
    smoothness_sigma : float
        Controls the flexibility of the curve.
        - Low sigma: Stiff, very smooth.
        - High sigma: Flexible, jagged.

    Returns
    -------
    curve : jnp.ndarray (N_bins,)
        Latent log-density curve.
    """
    # Sample the *steps* between bins, not the bins themselves.
    # This enforces the neighbor-to-neighbor correlation.
    steps = numpyro.sample(
        "steps", dist.Normal(0.0, smoothness_sigma), sample_shape=(N_bins - 1,)
    )

    # Reconstruct the curve via cumulative sum.
    # We fix the first bin at 0.0 (relative scale).
    curve = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(steps)])

    return curve


# ==============================================================================
# Model Inference
# ==============================================================================


def model(matrix, n_bins):
    """
    The Model definition for NumPyro.

    Parameters
    ----------
    matrix : jnp.ndarray (N_stars, N_bins)
        The pre-computed Design Matrix M.
    n_bins : int
        Number of velocity bins.
    """
    # --- Hyperparameters ---

    # We infer the smoothness sigma from the data.
    # HalfNormal(0.1) is a relatively conservative prior, favoring smooth curves.
    smoothness_sigma = numpyro.sample("smoothness_sigma", dist.HalfNormal(0.1))

    # We infer the Total Flux (Number of Stars) separately from the shape.
    # TODO: test this vs simple Dirichlet prior on the weights.
    N_stars = matrix.shape[0]
    total_flux = numpyro.sample(
        "total_flux",
        dist.TruncatedNormal(
            loc=float(N_stars), scale=jnp.sqrt(float(N_stars)), low=0.0
        ),
    )

    # --- Latent Density ---
    # Generate the log-density curve using the Random Walk prior
    latent_curve = generate_smooth_curve(n_bins, smoothness_sigma)

    # --- Normalization ---

    # Fix Scale Ambiguity:
    centered_curve = latent_curve - jnp.mean(latent_curve)

    # Convert latent values to valid Probabilities (Sum = 1)
    intrinsic_pdf = jax.nn.softmax(centered_curve)

    # Save the PDF for plotting and later use
    numpyro.deterministic("intrinsic_pdf", intrinsic_pdf)

    # --- Likelihood ---

    # The probability of observing Star i is the weighted sum of the
    # probabilities of it coming from each bin.
    # P(Star_i) = DotProduct( Row_i, Intrinsic_Weights )
    per_star_prob = jnp.dot(matrix, intrinsic_pdf)

    # Poisson Log-Likelihood
    # LogLik = Sum( log( P_i * Flux ) )
    log_prob = jnp.sum(jnp.log(per_star_prob)) + N_stars * jnp.log(total_flux)

    # Register the likelihood with NumPyro
    numpyro.factor("obs_log_lik", log_prob)


# ==============================================================================
# Solver Class
# ==============================================================================


class KinematicSolver:
    def __init__(self):
        """
        Initializes the KinematicSolver instance.
        """
        self.matrix = None
        self.grid = {}
        self.samples = None

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
        """
        if not self.grid:
            msg = "Run setup_grid() first."
            raise ValueError(msg)

        print(f"Computing Design Matrix for {len(vel)} stars...")

        # This one-time computation replaces the convolution loop
        self.matrix = precompute_design_matrix(
            vel, err, self.grid["centers"], bin_width=self.grid["width"]
        )
        print(f"Matrix ready. Shape: {self.matrix.shape}")

    def run(self, num_warmup=500, num_samples=1000, gpu=True):
        """
        Run the NUTS sampler.

        Parameters
        ----------
        num_warmup : int
            Number of warmup (burn-in) steps.
        num_samples : int
            Number of MCMC samples to draw.
        gpu : bool
            Whether to use GPU acceleration (if available).
        """
        if self.matrix is None:
            msg = "No data added."
            raise ValueError(msg)

        if gpu:
            numpyro.set_platform("gpu")

        print("Starting NUTS MCMC...")
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)

        # TODO: properly seed RNG
        rng_key = jax.random.PRNGKey(5567)
        mcmc.run(rng_key, matrix=self.matrix, n_bins=self.grid["n_bins"])

        self.samples = mcmc.get_samples()
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
        # The model returns "Probability Mass" (sum = 1).
        # We need "Probability Density" (integral = 1) for plotting.
        # Density = Mass / Bin_Width
        # TODO: decide if we want to work in density space internally
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
                # If function, evaluate and normalize
                y_true = true_intrinsic(x)
                y_true /= np.sum(y_true) * dx
                ax.plot(x, y_true, color="k", ls="--", label="True Intrinsic")
            else:
                # If samples, histogram with density=True
                true_hist, _ = np.histogram(
                    true_intrinsic, bins=self.grid["edges"], density=True
                )
                ax.plot(x, true_hist, color="k", ls="--", label="True Intrinsic")

        ax.set_xlabel("Velocity")
        ax.set_ylabel("Probability Density")
        ax.legend()
        return ax
