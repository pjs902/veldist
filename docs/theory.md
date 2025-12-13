# Methodology

`vdist` infers the intrinsic velocity distribution of a stellar system using a **grid-based Bayesian deconvolution**.

Instead of assuming the velocity distribution follows a specific formula (like a Gaussian), we solve for the height of every bin in a discretized histogram. To make this mathematically tractable and robust against noise, we use two specific techniques: a **Linear Response Matrix** to handle measurement errors, and a **Gaussian Process Prior** to handle regularization.

## 1. The Generative Model (The Prior)

We model the intrinsic Velocity Distribution Function (VDF) as a vector of weights $\mathbf{w}$ on a fixed grid.

If we attempted to fit these weights directly (e.g., using a Dirichlet prior), the model would treat every velocity bin as statistically independent. This results in "white noise”—a jagged, unphysical distribution that fits the noise in the data rather than the signal.

To enforce physical continuity, we use a **Discretized Log-Gaussian Cox Process**.

### A. The Latent Random Walk
We generate a latent vector $\mathbf{u}$ using an autoregressive process (a Gaussian Random Walk). The value of bin $i$ is conditioned on the value of bin $i-1$:

$$ u_i \sim \mathcal{N}(u_{i-1}, \sigma_{\text{smooth}}) $$

The parameter $\sigma_{\text{smooth}}$ controls the "stiffness" of the distribution:
*   **Low $\sigma$:** The random walk takes tiny steps. The resulting distribution is stiff and smooth.
*   **High $\sigma$:** The random walk takes large steps. The resulting distribution is flexible and can fit sharp peaks.

Crucially, **we do not set $\sigma_{\text{smooth}}$ manually.** It is a free hyperparameter in the Bayesian model. The MCMC sampler infers the optimal smoothness from the signal-to-noise ratio of the data.

### B. The Link Function (Softmax)
The latent vector $\mathbf{u}$ exists in log-space and is unconstrained. To convert it into a valid probability density function (where all bins are positive and sum to 1), we apply the Softmax function:

$$ w_i = \frac{e^{u_i}}{\sum_{k} e^{u_k}} $$

This vector $\mathbf{w}$ represents the **Intrinsic Probability Mass** in each bin.

## 2. The Likelihood (The Response Matrix)

The core computational challenge in deconvolution is that astronomical data is **heteroscedastic**: every star has a unique measurement error $\sigma_i$. Standard convolution (FFT) cannot handle this, as the kernel changes for every data point.

We solve this by pre-computing the physics of the measurement errors into a static **Response Matrix** (often called a Design Matrix), denoted as $\mathbf{M}$.

### A. Matrix Construction
The matrix has dimensions $(N_{\text{stars}} \times K_{\text{bins}})$. Each entry $M_{ij}$ calculates the probability that Star $i$ would be observed at its current location, *conditional* on it originating from Intrinsic Bin $j$.

We calculate this using **Box Integration** (CDF Difference). Instead of evaluating the Gaussian error PDF at the center of the bin, we integrate the error distribution over the width of the bin:

$$ M_{ij} = \int_{\text{edge}_j}^{\text{edge}_{j+1}} \mathcal{N}(v \mid y_i^{\text{obs}}, \sigma_i^{\text{err}}) \, dv $$

$$ M_{ij} = \text{CDF}_i(\text{edge}_{j+1}) - \text{CDF}_i(\text{edge}_j) $$

**Why Integration Matters:**
For high-precision data (e.g., HST), the measurement error $\sigma_i$ may be smaller than the bin width. If we evaluated the PDF at the bin center, a star falling near the edge of a bin would result in near-zero probability (aliasing). Integration ensures probability mass is conserved regardless of the error size.

### B. The Linear Solve
Once $\mathbf{M}$ is constructed, the "blurring" step of deconvolution becomes a simple Matrix-Vector multiplication. The likelihood of the observed dataset $\mathbf{y}$ given the intrinsic weights $\mathbf{w}$ is:

$$ \mathcal{L} = \mathbf{M} \cdot \mathbf{w} $$

$$ \log \mathcal{L}_{\text{total}} = \sum_{i=1}^{N} \log \left( [\mathbf{M} \cdot \mathbf{w}]_i \right) $$

This operation moves the expensive transcendental functions (exponentials/integrals) **outside** the inference loop. The MCMC sampler only needs to perform linear algebra, allowing `vdist` to scale to datasets with $N > 10^5$ tracers.

## 3. Handling 2D and Missing Data

### Data Fusion (The Stacked Matrix)
Because the matrix calculation is performed row-by-row, `vdist` can fuse data from different instruments without re-gridding or degrading resolution.
*   **Rows $0 \to N$:** Can use high-precision integration math (HST data).
*   **Rows $N \to M$:** Can use broad covariance math (Gaia data).
The solver simply sees a list of constraints on the grid.

### 2D Covariance
For 2D distributions (Proper Motions $\mu_\alpha, \mu_\delta$), the Response Matrix becomes 3D conceptually (a stack of 2D images), but is flattened for computation:
$$ \mathbf{L}_{N} = \mathbf{M}_{(N \times K^2)} \cdot \mathrm{vec}(\mathbf{W}_{K \times K}) $$
This allows the method to account for the full error covariance matrix ($\rho$) of Gaia data, which creates "tilted" probability constraints that standard binning cannot capture.

## 4. Relationship to Previous Work

This method combines specific mechanical components from three established techniques in the literature:

1.  **Maximum Penalized Likelihood (MPL):**
    *   *Reference: Merritt (1997); Saha & Williams (1994).*
    *   MPL introduced the concept of solving $\mathbf{y} = \mathbf{M} \cdot \mathbf{w}$ subject to a roughness penalty. `vdist` uses this formulation but replaces the manual penalty term ($\lambda$) with a marginalized Bayesian prior, eliminating the need to hand-tune the smoothing.

2.  **Extreme Deconvolution (XD):**
    *   *Reference: Bovy, Hogg, & Roweis (2011).*
    *   XD introduced the rigorous treatment of heteroscedastic errors via individual likelihood evaluation. However, XD assumes the intrinsic distribution is a Mixture of Gaussians. `vdist` replaces the parametric Gaussian mixture with a non-parametric grid, allowing it to recover shapes (like flat-topped cores or asymmetric tails) that Gaussians cannot fit efficiently.

3.  **Log-Gaussian Cox Processes (LGCP):**
    *   *Reference: Møller et al. (1998).*
    *   Statistically, the `vdist` model is a discretized LGCP. We treat the position of stars in velocity space as a Poisson process whose intensity function is driven by a latent Gaussian field.
