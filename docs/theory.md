# Methodology

`veldist` infers the intrinsic velocity distribution of a stellar system using a **grid-based Bayesian deconvolution**.

Instead of assuming the velocity distribution follows a specific distribution (like a Gaussian, or Gauss-Hermite series), we solve for the height of every bin in a discretized histogram. To make this mathematically tractable and robust against noise, we use two specific techniques: a **Linear Response Matrix** to handle measurement errors, and a **Smoothing Prior** to handle regularization.

## The Model

We model the intrinsic Velocity Distribution Function (VDF) as a vector of weights $\mathbf{w}$ on a fixed grid.

### 1. The Latent Random Walk

We generate a latent vector $\mathbf{u}$ using an autoregressive process (a Gaussian Random Walk). The value of bin $i$ is conditioned on the value of bin $i-1$:

$$ u_i \sim \mathcal{N}(u_{i-1}, \sigma_{\text{smooth}}) $$

The parameter $\sigma_{\text{smooth}}$ controls the "stiffness" of the distribution:

* **Low $\sigma$:** The random walk takes tiny steps. The resulting distribution is stiff and smooth.
* **High $\sigma$:** The random walk takes large steps. The resulting distribution is flexible and can fit sharp peaks.

Crucially, **we do not set $\sigma_{\text{smooth}}$ manually.** It is a free hyperparameter in the Bayesian model. The MCMC sampler infers the optimal smoothness from the signal-to-noise ratio of the data.

<!-- ### B. The Link Function (Softmax)

The latent vector $\mathbf{u}$ exists in log-space and is unconstrained. To convert it into a valid probability density function (where all bins are positive and sum to 1), we apply the Softmax function:

$$ w_i = \frac{e^{u_i}}{\sum_{k} e^{u_k}} $$

This vector $\mathbf{w}$ represents the **Intrinsic Probability Mass** in each bin. 

TODO: Work on this once we decide on this whole softmax vs dirichlet thing.

-->

### 2. The Likelihood (The Response Matrix)

The core computational challenge in deconvolution is that astronomical data is **heteroscedastic**: every star has a unique measurement error $\sigma_i$. This means that for every evaluation of the likelihood, we would need to compute $N_{\text{stars}}$ integrals over $K_{\text{bins}}$ (involving exponentials), resulting in an $O(N \times K)$ operation that scales poorly for large datasets.

We solve this by pre-computing the physics of the measurement errors into a static **Response Matrix** (we also call this the Design Matrix), denoted as $\mathbf{M}$.

#### A. Matrix Construction

The matrix has dimensions $(N_{\text{stars}} \times K_{\text{bins}})$. Each entry $M_{ij}$ calculates the probability that Star $i$ would be observed at its current location, *conditional* on it originating from Intrinsic Bin $j$.

We calculate this using **Box Integration** (CDF Difference). Instead of evaluating the Gaussian error PDF at the center of the bin, we integrate the error distribution over the width of the bin:

$$ M_{ij} = \int_{\text{edge}_j}^{\text{edge}_{j+1}} \mathcal{N}(v \mid y_i^{\text{obs}}, \sigma_i^{\text{err}}) \, dv $$

$$ M_{ij} = \text{CDF}_i(\text{edge}_{j+1}) - \text{CDF}_i(\text{edge}_j) $$

#### B. The Likelihood Evaluation

Once $\mathbf{M}$ is constructed, the "blurring" step of deconvolution becomes a simple Matrix-Vector multiplication. The likelihood of the observed dataset $\mathbf{y}$ given the intrinsic weights $\mathbf{w}$ is:

$$ \mathcal{L} = \mathbf{M} \cdot \mathbf{w} $$

$$ \log \mathcal{L}_{\text{total}} = \sum_{i=1}^{N} \log \left( [\mathbf{M} \cdot \mathbf{w}]_i \right) $$

This operation moves the expensive operations (exponentials/integrals/convolutions) **outside** the inference loop. The MCMC sampler only needs to perform linear algebra, allowing `veldist` to scale to large datasets efficiently and to handle 2D or even 3D velocity distributions.

We note that the addition of a smoothing prior means that the likelihood is no longer strictly linear, so it cannot be solved with matrix inversion techniques. Instead, we use MCMC sampling to explore the posterior distribution. The core deconvolution step remains a fast matrix multiplication.

### 3. Handling 2D and Missing Data

#### 2D Covariance

For 2D distributions (Proper Motions $\mu_\alpha, \mu_\delta$), the Response Matrix becomes 3D conceptually (a stack of 2D images).

$$ \mathbf{L}_{N} = \mathbf{M}_{(N \times K^2)} \cdot \mathbf{w}_{(K^2 \times 1)} $$
This allows the method to account for the full error covariance matrix ($\rho$) of Gaia data, which creates "tilted" probability constraints that standard binning cannot capture.

### 4. Relationship to Previous Work

This method combines specific concepts from at least three prior works in the literature:

1. **Maximum Penalized Likelihood (MPL):**
    * *Reference: Merritt (1997); Saha & Williams (1994).*
    * MPL introduced the concept of solving $\mathbf{y} = \mathbf{M} \cdot \mathbf{w}$ subject to a roughness penalty. `veldist` uses this formulation but replaces the manual penalty term ($\lambda$) with a marginalized prior, eliminating the need to hand-tune the smoothing.

2. **Extreme Deconvolution (XD):**
    * *Reference: Bovy, Hogg, & Roweis (2011).*
    * XD introduced the treatment of heteroscedastic errors via individual likelihood evaluation. XD assumes the intrinsic distribution is a Mixture of Gaussians. `veldist` replaces the Gaussian mixture with a non-parametric grid, allowing it to recover shapes (like flat-topped cores or asymmetric tails) that Gaussians cannot fit efficiently.

3. **BayesLOSVD:**
    * *Reference: Falcón-Barroso & Martig (2021).*
    * BayesLOSVD introduced a Bayesian framework for non-parametric LOSVD extraction from IFU data using MCMC sampling. `veldist` uses a similar sampling approach and regularization strategy, but focuses on discrete stellar kinematics with heteroscedastic errors rather than integrated light spectra.
