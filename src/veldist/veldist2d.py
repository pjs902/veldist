# file src/veldist/veldist2d.py
"""
Bayesian Matrix-Based 2D (Proper-Motion) Kinematic Deconvolution
==================================================================

This module is the 2D analogue of ``veldist.py``: it infers the intrinsic
bivariate velocity distribution (e.g. over proper motion components
``pmra``/``pmdec``) from discrete stellar observations with per-star 2x2
measurement covariance, using a pre-computed design matrix and a hierarchical
2D GMRF (Gauss-Markov random field) smoothness prior.

Kept deliberately separate from ``veldist.py`` (see ``PLAN.md`` Part 3) --
the 1D and 2D solvers share the same broad approach (design matrix +
softmax-of-GMRF likelihood) but differ enough in the details (grid
flattening, box vs. quadrature integration, precision-matrix construction)
that premature unification would cost more than the duplication.

All GMRF / Cholesky / latent maths is done in float64
(``jax.config.update("jax_enable_x64", True)``) even though the design
matrix ``M`` is stored as float32 for memory; see the ⚠ Gotchas in
``PLAN.md`` §3.1/§3.2 for why mixing precisions here is deliberate, not an
oversight.
"""

import contextlib
import io
import warnings
from functools import cache

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 (jax's own API shape)

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .veldist import precompute_design_matrix

__all__ = [
    "KinematicSolver2D",
    "setup_grid_2d",
    "precompute_design_matrix_2d",
    "build_gmrf_precision",
    "model_2d",
    "model_gaussian_core_2d",
    "generate_gaussian_core_field_2d",
    "fit_all_bins_2d",
]

#: Rate of the Exponential prior on the non-Gaussian deviation scale. Starts
#: at the value 1D adopted after its regularisation campaign
#: (docs/superpowers/specs/2026-08-03-regularisation-decision.md). NOT yet
#: measured for 2D -- do not assume it transfers.
SIGMA3_RATE_2D = 0.35


# ==============================================================================
# Grid
# ==============================================================================


def setup_grid_2d(center, width, n_bins):
    """
    Define a square K x K velocity grid over a bivariate velocity space.

    The grid is flattened row-major (C order) using ``np.ravel_multi_index`` /
    ``np.unravel_index`` for every index conversion, never hand-written
    arithmetic, to avoid the row-major/column-major transposition bug
    flagged in ``PLAN.md`` §3.1: cell ``(ix, iy)`` maps to flat index
    ``m = ix * K + iy``.

    Parameters
    ----------
    center : (float, float)
        Center of the grid, ``(cx, cy)``.
    width : (float, float)
        Total width of the grid, ``(wx, wy)``.
    n_bins : int
        Number of bins per axis, ``K``. Total number of 2D cells is ``K**2``.

    Returns
    -------
    grid : dict
        Keys: ``centers_x`` (K,), ``centers_y`` (K,), ``edges_x`` (K+1,),
        ``edges_y`` (K+1,), ``centers_2d`` (K**2, 2) [row-major flattened],
        ``width_x``, ``width_y``, ``area`` (= width_x * width_y),
        ``n_bins`` (= K, i.e. per-axis), ``n_cells`` (= K**2), ``shape`` (K, K).
    """
    cx, cy = center
    wx, wy = width
    k = int(n_bins)

    edges_x = np.linspace(cx - wx / 2, cx + wx / 2, k + 1)
    edges_y = np.linspace(cy - wy / 2, cy + wy / 2, k + 1)
    centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
    centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])

    width_x = edges_x[1] - edges_x[0]
    width_y = edges_y[1] - edges_y[0]

    # Row-major (C order) meshgrid: flat index m = ix*K + iy.
    ix_grid, iy_grid = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    flat = np.ravel_multi_index((ix_grid.ravel(), iy_grid.ravel()), (k, k), order="C")
    # flat should just be 0..K**2-1 in this order; assemble centers_2d to match.
    centers_2d = np.empty((k * k, 2))
    centers_2d[flat, 0] = centers_x[ix_grid.ravel()]
    centers_2d[flat, 1] = centers_y[iy_grid.ravel()]

    return {
        "centers_x": centers_x,
        "centers_y": centers_y,
        "edges_x": edges_x,
        "edges_y": edges_y,
        "centers_2d": centers_2d,
        "width_x": float(width_x),
        "width_y": float(width_y),
        "area": float(width_x * width_y),
        "n_bins": k,
        "n_cells": k * k,
        "shape": (k, k),
    }


# ==============================================================================
# Design Matrix
# ==============================================================================


def _gauss_legendre_2x2_nodes():
    """2-point Gauss-Legendre nodes/weights on [-1, 1], for sub-cell quadrature."""
    node = 1.0 / np.sqrt(3.0)
    nodes = np.array([-node, node])
    weights = np.array([1.0, 1.0])
    return nodes, weights


def precompute_design_matrix_2d(pm1, pm2, cov, grid, chunk_size=5000):
    """
    Compute the 2D probability design matrix M (N, K**2).

    ``M[i, m] = integral over cell m of N(mu=(pm1_i, pm2_i), Sigma=cov_i)``.

    Three code paths, chosen per star:

    1. Diagonal ``cov_i`` (``cov_i[0,1] == 0``): exact box integration, which
       factorises into the outer product of two independent 1D
       ``precompute_design_matrix``-style erf/CDF calls (reuses
       ``veldist.precompute_design_matrix`` per axis). Exact and cheap.
    2. Correlated ``cov_i``: 2x2 Gauss-Legendre sub-cell quadrature (4
       evaluation points per cell per star), which handles arbitrary Sigma.

    The matrix is built in chunks over stars (default 5000) to bound peak
    memory, because JAX allocates intermediates during construction, so building
    the whole ``(N, K**2)`` array in one call can peak at 2-3x its final
    size. The final ``M`` is cast to float32 only at the very end of each
    chunk, after all quadrature/erf math is done in float64.

    Parameters
    ----------
    pm1, pm2 : array-like (N,)
        Observed two velocity/proper-motion components per star.
    cov : array-like (N, 2, 2)
        Per-star measurement covariance matrices.
    grid : dict
        Output of :func:`setup_grid_2d`.
    chunk_size : int
        Number of stars processed per chunk. Default 5000.

    Returns
    -------
    M : np.ndarray (N, K**2), float32
        Design matrix.
    """
    pm1 = np.asarray(pm1, dtype=np.float64)
    pm2 = np.asarray(pm2, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    n_stars = len(pm1)
    k = grid["n_bins"]
    n_cells = grid["n_cells"]

    ex = grid["edges_x"]
    ey = grid["edges_y"]
    wx = grid["width_x"]
    wy = grid["width_y"]

    out_chunks = []

    for start in range(0, n_stars, chunk_size):
        end = min(start + chunk_size, n_stars)
        p1 = pm1[start:end]
        p2 = pm2[start:end]
        c = cov[start:end]  # (n, 2, 2)
        n = len(p1)

        sx = np.sqrt(c[:, 0, 0])
        sy = np.sqrt(c[:, 1, 1])
        rho_cov = c[:, 0, 1]
        is_diag = np.isclose(rho_cov, 0.0, atol=1e-12)

        chunk_M = np.zeros((n, n_cells), dtype=np.float64)

        # --- Path 1: diagonal covariance -> exact box integration, ---
        # --- factorised as an outer product of two 1D erf/CDF calls. ---
        if np.any(is_diag):
            idx = np.where(is_diag)[0]
            Mx = np.asarray(
                precompute_design_matrix(p1[idx], sx[idx], grid["centers_x"], bin_width=wx)
            )  # (n_diag, K)
            My = np.asarray(
                precompute_design_matrix(p2[idx], sy[idx], grid["centers_y"], bin_width=wy)
            )  # (n_diag, K)
            # Outer product per star, row-major flatten to match centers_2d.
            outer = Mx[:, :, None] * My[:, None, :]  # (n_diag, K, K)
            chunk_M[idx, :] = outer.reshape(len(idx), n_cells)

        # --- Path 2: correlated covariance -> 2x2 Gauss-Legendre sub-cell ---
        if np.any(~is_diag):
            idx = np.where(~is_diag)[0]
            chunk_M[idx, :] = _design_matrix_gl_quadrature(
                p1[idx], p2[idx], c[idx], ex, ey, k
            )

        out_chunks.append(chunk_M.astype(np.float32))

    return np.concatenate(out_chunks, axis=0)


def _bivariate_gaussian_pdf(x, y, mu1, mu2, cov):
    """
    Evaluate N((x,y); mu, cov) for arrays of evaluation points and per-star
    mu/cov. Broadcasts (n_stars, n_pts).

    x, y : (n_stars, n_pts)
    mu1, mu2 : (n_stars,)
    cov : (n_stars, 2, 2)
    """
    dx = x - mu1[:, None]
    dy = y - mu2[:, None]

    a = cov[:, 0, 0][:, None]
    b = cov[:, 0, 1][:, None]
    d = cov[:, 1, 1][:, None]
    det = a * d - b * b
    det = np.maximum(det, 1e-300)

    inv_a = d / det
    inv_b = -b / det
    inv_d = a / det

    quad = inv_a * dx * dx + 2 * inv_b * dx * dy + inv_d * dy * dy
    norm = 1.0 / (2 * np.pi * np.sqrt(det))
    return norm * np.exp(-0.5 * quad)


def _design_matrix_gl_quadrature(p1, p2, cov, edges_x, edges_y, k):
    """
    2x2 Gauss-Legendre sub-cell quadrature for a chunk of (correlated) stars.

    Returns an (n, K**2) array (float64) of cell probability masses.
    """
    n = len(p1)
    n_cells = k * k

    cx0 = edges_x[:-1]
    cx1 = edges_x[1:]
    cy0 = edges_y[:-1]
    cy1 = edges_y[1:]
    hx = 0.5 * (cx1 - cx0)  # half-width per x cell, (K,)
    hy = 0.5 * (cy1 - cy0)
    mx = 0.5 * (cx1 + cx0)  # mid per x cell, (K,)
    my = 0.5 * (cy1 + cy0)

    nodes, gweights = _gauss_legendre_2x2_nodes()  # 2 nodes each axis

    # Evaluation points: for each cell, 2x2=4 points. Build full (K, K, 4) grid
    # of (x, y) coordinates and weights, then evaluate per star.
    # x_pts[ix, jnode] = mx[ix] + hx[ix]*node[jnode]
    x_pts = mx[:, None] + hx[:, None] * nodes[None, :]  # (K, 2)
    y_pts = my[:, None] + hy[:, None] * nodes[None, :]  # (K, 2)
    wx_pts = hx[:, None] * gweights[None, :]  # (K, 2); half-width already in Jacobian
    wy_pts = hy[:, None] * gweights[None, :]

    # Combine into (K, K, 4) grid of points & weights (2 nodes per axis -> 4 combos)
    # point index p in [0,4): (a,b) = divmod(p, 2)
    xs = np.empty((k, k, 4))
    ys = np.empty((k, k, 4))
    ws = np.empty((k, k, 4))
    for p in range(4):
        a, b = divmod(p, 2)
        xs[:, :, p] = x_pts[:, a][:, None]
        ys[:, :, p] = y_pts[None, :, b]
        ws[:, :, p] = wx_pts[:, a][:, None] * wy_pts[None, :, b]

    xs_flat = xs.reshape(n_cells, 4)
    ys_flat = ys.reshape(n_cells, 4)
    ws_flat = ws.reshape(n_cells, 4)

    result = np.zeros((n, n_cells), dtype=np.float64)
    # Loop over the 4 quadrature points (cheap: only 4 iterations).
    for p in range(4):
        pdf_val = _bivariate_gaussian_pdf(
            xs_flat[None, :, p].repeat(n, axis=0),
            ys_flat[None, :, p].repeat(n, axis=0),
            p1,
            p2,
            cov,
        )  # (n, n_cells)
        result += pdf_val * ws_flat[None, :, p]

    return result


# ==============================================================================
# GMRF Prior
# ==============================================================================


def build_gmrf_precision(k, diag_weight=None, edge_weight=1.0, ridge_scale=1e-6):
    """
    Build an 8-connectivity intrinsic GMRF precision matrix Q for a K x K grid.

    ``Q = D - W`` where ``W`` is the (symmetric) adjacency-weight matrix:
    weight ``edge_weight`` (default 1) for the 4 edge-neighbours and
    ``diag_weight`` (default ``1/sqrt(2)``; see the note below) for
    the 4 diagonal neighbours, and ``D = diag(row sums of W)``.

    Built via explicit ``np.ravel_multi_index``-based adjacency lists (never
    array-shift tricks), because shift-based construction silently wraps around at
    the grid boundary (periodic boundary leakage), which is the single most
    dangerous failure mode here (see ``PLAN.md`` §3.2 gotchas).

    ``Q`` is singular by construction (constant-vector null space, the
    softmax removes that direction anyway). A ridge is added for the
    Cholesky factorisation, scaled relative to ``mean(diag(Q))`` (not an
    absolute value) so its meaning does not change if the connectivity
    weights change:  ``eps = ridge_scale * mean(diag(Q))``.

    Parameters
    ----------
    k : int
        Grid size per axis (total cells = k**2).
    diag_weight : float
        Weight for diagonal (corner-touching) neighbours. Default
        ``1/sqrt(2)`` is the natural distance weighting; pass 1.0 for equal
        weighting of all 8 neighbours.
    edge_weight : float
        Weight for edge (side-touching) neighbours. Default 1.0.
    ridge_scale : float
        Relative ridge added before Cholesky:
        ``eps = ridge_scale * mean(diag(Q))``. Default 1e-6.

    Returns
    -------
    Q : np.ndarray (K**2, K**2)
    Q_reg : np.ndarray (K**2, K**2)
        Q with the ridge added.
    """
    if diag_weight is None:
        diag_weight = 1.0 / np.sqrt(2.0)

    n_cells = k * k
    W = np.zeros((n_cells, n_cells))

    ix_grid, iy_grid = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    ix_flat = ix_grid.ravel()
    iy_flat = iy_grid.ravel()

    # Edge neighbours: (dx, dy) in {(1,0), (-1,0), (0,1), (0,-1)}
    edge_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    diag_offsets = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    def add_offsets(offsets, weight):
        for dx, dy in offsets:
            nx = ix_flat + dx
            ny = iy_flat + dy
            valid = (nx >= 0) & (nx < k) & (ny >= 0) & (ny < k)
            src = np.ravel_multi_index((ix_flat[valid], iy_flat[valid]), (k, k), order="C")
            dst = np.ravel_multi_index((nx[valid], ny[valid]), (k, k), order="C")
            W[src, dst] += weight

    add_offsets(edge_offsets, edge_weight)
    add_offsets(diag_offsets, diag_weight)

    D = np.diag(W.sum(axis=1))
    Q = D - W

    eps = ridge_scale * np.mean(np.diag(Q))
    Q_reg = Q + eps * np.eye(n_cells)

    return Q, Q_reg


@cache
def _null_space_basis_2d(k):
    """Orthonormal basis of the bivariate-quadratic null space, (k*k, 6).

    Spans ``{1, x, y, x^2, xy, y^2}`` -- exactly the log-densities of
    bivariate Gaussians, which is what the prior must leave free so that the
    velocity ellipsoid is not shrunk.

    Built from tensor-product Legendre polynomials of total degree <= 2 rather
    than the raw monomials, for the conditioning reason documented in
    ``veldist.py::_legendre_basis``: the Vandermonde basis is badly conditioned
    and its QR loses precision. Legendre spans the identical space, so the
    projector is unchanged.

    Uses an index grid rather than physical cell centres. Both give the same
    projector: the orthogonal projector onto a subspace does not depend on
    which basis spans it, and for a uniformly spaced grid the index and
    physical coordinates differ only by an affine map. ``setup_grid_2d`` only
    produces uniform grids.

    Row-major flattened (``m = ix*k + iy``) to match ``setup_grid_2d``'s
    ``centers_2d``. Cached: costs an O(k^2) QR, depends only on ``k``, and is
    evaluated at JAX trace time where the result is constant-folded.
    """
    idx = np.arange(k, dtype=float)
    u = 2.0 * (idx - idx.mean()) / (k - 1)  # -> [-1, 1]
    p0 = np.ones_like(u)
    p1 = u
    p2 = 0.5 * (3.0 * u**2 - 1.0)

    ix, iy = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()

    # Tensor products with total degree <= 2.
    cols = [
        p0[ix] * p0[iy],   # 1
        p1[ix] * p0[iy],   # x
        p0[ix] * p1[iy],   # y
        p2[ix] * p0[iy],   # x^2
        p1[ix] * p1[iy],   # xy
        p0[ix] * p2[iy],   # y^2
    ]
    q, _ = np.linalg.qr(np.stack(cols, axis=1))
    return q


@cache
def _gmrf_deviation_scale_2d(k):
    """Sorbye-Rue scaling constant for the null-space-projected 2D GMRF.

    Returns the factor making the generalised variance -- the geometric mean
    of the per-cell marginal variances of the projected field -- equal to 1,
    so that ``sigma3`` means "typical log-density departure from the Gaussian
    null space" independently of grid resolution (Sorbye & Rue 2014, Spatial
    Statistics 8, 39; this is what ``scale.model=TRUE`` does in R-INLA).

    Measured size: the scale drifts about -12% from k=9 to k=21 (2.311 ->
    2.028). This is a correctness tidy so that a tuned ``SIGMA3_RATE_2D``
    transfers across grids -- it is NOT the fix for the dispersion bias, and
    an earlier hypothesis that it was has been retracted.

    Cached: O(k^6) pinv, depends only on ``k``.
    """
    q_ns = _null_space_basis_2d(k)
    proj = np.eye(k * k) - q_ns @ q_ns.T
    q_mat, _ = build_gmrf_precision(k)
    sigma = proj @ np.linalg.pinv(q_mat) @ proj.T
    var = np.clip(np.diag(sigma), 1e-300, None)
    return float(1.0 / np.sqrt(np.exp(np.mean(np.log(var)))))


# ==============================================================================
# Model Inference
# ==============================================================================


def model_2d(matrix, n_cells, L):
    """
    The 2D NumPyro model.

    Parameters
    ----------
    matrix : jnp.ndarray (N_stars, K**2)
        Pre-computed 2D design matrix.
    n_cells : int
        Number of grid cells, K**2.
    L : jnp.ndarray (K**2, K**2)
        Cholesky factor of the (ridge-regularised) GMRF precision matrix Q,
        computed once outside the model and closed over (never recomputed
        per NUTS step; Cholesky of a fixed matrix is a one-time cost).

    Notes
    -----
    Latent parameterisation is non-centred and fully generative: a real
    ``numpyro.sample`` site (``z``) followed by a deterministic transform --
    exactly as recommended in ``PLAN.md`` §1.2/§3.2, and *not* a
    ``numpyro.factor``-based penalty on an unconditioned base measure. A
    factor-based version would be invisible to ``numpyro.infer.Predictive``
    (pure ancestral / prior-predictive sampling only forward-samples through
    ``sample`` sites), silently breaking simulation-based calibration even
    though NUTS inference itself would still be numerically correct.

    We want ``x = sigma * L^-T z``, i.e.
    ``jax.scipy.linalg.solve_triangular(L.T, z, lower=False)``, solving the
    *upper* triangular system ``L.T @ x = z`` for x. Using ``L`` directly
    with ``lower=True`` would instead give ``L^-1 z``, a different (wrong)
    covariance; see ``tests/test_veldist2d.py::test_solve_triangular_direction``
    for the numerical check this is validated against.
    """
    smoothness_sigma = numpyro.sample("smoothness_sigma", dist.HalfNormal(3.0))

    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n_cells]).to_event(1))
    x = smoothness_sigma * jax.scipy.linalg.solve_triangular(L.T, z, lower=False)

    intrinsic_pdf = jax.nn.softmax(x)
    numpyro.deterministic("intrinsic_pdf", intrinsic_pdf)

    per_star_prob = jnp.dot(matrix, intrinsic_pdf)
    log_prob = jnp.sum(jnp.log(per_star_prob))
    numpyro.factor("obs_log_lik", log_prob)


def generate_gaussian_core_field_2d(k, centers_2d, L):
    """Latent log-density field: free bivariate-Gaussian core + penalised deviation.

    The infinite-smoothing limit of this prior is a bivariate Gaussian, not a
    uniform over the velocity grid. That is the whole point: the pure-GMRF
    prior in :func:`model_2d` has a uniform limit whose dispersion is
    ``grid_width/sqrt(12)`` -- 34 km/s on a 119 km/s grid against a true 17 --
    so weakly-constrained fits are pulled toward a far broader distribution and
    every recovered dispersion is biased high. Measured on the pure GMRF
    (isotropic sigma=17, err/sigma=0.014, scored against the discretised
    truth): sigma_x bias +2.34 at N=100 and +0.51 at N=500, growing with k.

    A general quadratic form in (vx, vy) softmaxes to exactly a bivariate
    Gaussian, so ``v0x``, ``v0y``, ``s0x``, ``s0y``, ``rho0`` map one-to-one
    onto the PDF's mean and covariance -- the velocity ellipsoid. Structure
    beyond second order remains penalised, exactly as h3/h4 are in 1D.

    Parameters
    ----------
    k : int
        Grid size per axis. Total cells ``k**2``.
    centers_2d : array-like, shape (k**2, 2)
        Physical cell centres from :func:`setup_grid_2d`. Required because the
        core is quadratic in *velocity*, not in cell index.
    L : jnp.ndarray, shape (k**2, k**2)
        Cholesky factor of the ridge-regularised GMRF precision.

    Returns
    -------
    field : jnp.ndarray, shape (k**2,)
        Latent log-density, to be passed through ``softmax``.
    """
    centers_2d = jnp.asarray(centers_2d)
    cx = centers_2d[:, 0]
    cy = centers_2d[:, 1]
    span_x = jnp.max(cx) - jnp.min(cx)
    span_y = jnp.max(cy) - jnp.min(cy)
    mid_x = jnp.mean(cx)
    mid_y = jnp.mean(cy)

    # --- Gaussian null space: free, unpenalised ---
    # LogNormal rather than HalfNormal on the widths: half-distributions put
    # substantial mass near zero, and a near-zero width collapses the
    # distribution onto one cell. 1D measured a prior-predictive median sigma
    # of exactly 0.00 for >99% of draws that way (veldist.py:454).
    #
    # The divisor is 6, not 1D's 8. The grid is sized at +/-3.5 sigma, so
    # span ~ 7 sigma and a prior median matching the expected dispersion wants
    # a divisor near 6.2; span/6 gives 17.7 km/s against a 17 km/s truth.
    # 1D's span/8 is 0.875 sigma, which is fine on a 37-bin grid but lands the
    # median on exactly 1.00 cell at 2D's K=9, putting half of all prior draws
    # below the grid resolution. Measured sub-cell fraction: 0.50 at
    # (span/8, 1.0) vs 0.35 at (span/6, 0.75).
    v0x = numpyro.sample("v0x", dist.Normal(mid_x, span_x / 4.0))
    v0y = numpyro.sample("v0y", dist.Normal(mid_y, span_y / 4.0))
    s0x = numpyro.sample("s0x", dist.LogNormal(jnp.log(span_x / 6.0), 0.75))
    s0y = numpyro.sample("s0y", dist.LogNormal(jnp.log(span_y / 6.0), 0.75))
    # Uniform(-0.95, 0.95) is the LKJ(2, 1) marginal with the degenerate
    # endpoints clipped, written explicitly so rho0 is a rankable site.
    rho0 = numpyro.sample("rho0", dist.Uniform(-0.95, 0.95))

    dx = (cx - v0x) / jnp.clip(s0x, 1e-3)
    dy = (cy - v0y) / jnp.clip(s0y, 1e-3)
    quad = (dx**2 - 2.0 * rho0 * dx * dy + dy**2) / (1.0 - rho0**2)
    core = -0.5 * quad

    # --- penalised non-Gaussian deviation ---
    sigma3 = numpyro.sample(
        "sigma3", dist.Exponential(SIGMA3_RATE_2D)
    ) * _gmrf_deviation_scale_2d(k)
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([k * k]).to_event(1))
    # x = sigma * L^-T z, i.e. solve the UPPER triangular system L.T @ x = z.
    # Using L with lower=True would give L^-1 z, a different covariance; see
    # test_solve_triangular_direction.
    w = sigma3 * jax.scipy.linalg.solve_triangular(L.T, z, lower=False)

    # Project out the quadratic null space. Cached constant, so a matmul
    # rather than a QR per leapfrog step.
    q_ns = jnp.asarray(_null_space_basis_2d(k))
    deviation = w - q_ns @ (q_ns.T @ w)

    return core + deviation


def model_gaussian_core_2d(matrix, n_cells, L, centers_2d):
    """The 2D NumPyro model with the Gaussian-core prior.

    Parameters
    ----------
    matrix : jnp.ndarray, shape (N_stars, k**2)
        Pre-computed 2D design matrix.
    n_cells : int
        Number of grid cells, ``k**2``.
    L : jnp.ndarray, shape (k**2, k**2)
        Cholesky factor of the ridge-regularised GMRF precision.
    centers_2d : jnp.ndarray, shape (k**2, 2)
        Physical cell centres.
    """
    k = int(round(float(n_cells) ** 0.5))
    field = generate_gaussian_core_field_2d(k, centers_2d, L)

    intrinsic_pdf = jax.nn.softmax(field)
    numpyro.deterministic("intrinsic_pdf", intrinsic_pdf)

    per_star_prob = jnp.dot(matrix, intrinsic_pdf)
    numpyro.factor("obs_log_lik", jnp.sum(jnp.log(per_star_prob)))


# ==============================================================================
# Solver Class
# ==============================================================================


class KinematicSolver2D:
    """
    High-level interface for 2D (bivariate proper-motion) Bayesian kinematic
    deconvolution. Mirrors :class:`veldist.KinematicSolver`'s API.

    Attributes
    ----------
    matrix : jnp.ndarray or None
        Pre-computed design matrix, shape (N_stars, K**2).
    grid : dict
        Metadata from :func:`setup_grid_2d`.
    Q, Q_reg, L : np.ndarray or None
        GMRF precision matrix, its ridge-regularised version, and its
        Cholesky factor. Built once by ``setup_grid`` and closed over by the
        model.
    n_stars : int or None
    samples : dict or None
    clipped_samples : dict or None
        Per-cell summary statistics (median PM-distribution mass and clipped
        uncertainties) populated by ``clip_uncertainties``.
    """

    def __init__(self):
        self.matrix = None
        self.grid = {}
        self.Q = None
        self.Q_reg = None
        self.L = None
        self.n_stars = None
        self.samples = None
        self.clipped_samples = None

    def setup_grid(
        self, center, width, n_bins, diag_weight=None, edge_weight=1.0, ridge_scale=1e-6
    ):
        """
        Define the 2D velocity grid and build/factorise the GMRF precision
        matrix.

        Parameters
        ----------
        center : (float, float)
        width : (float, float)
        n_bins : int
            Per-axis bin count K (total cells K**2).
        diag_weight, edge_weight, ridge_scale : float
            Forwarded to :func:`build_gmrf_precision`.

        Returns
        -------
        None
            Sets ``self.grid``, ``self.Q``, ``self.Q_reg``, ``self.L``.
        """
        self.grid = setup_grid_2d(center, width, n_bins)
        k = self.grid["n_bins"]

        Q, Q_reg = build_gmrf_precision(
            k, diag_weight=diag_weight, edge_weight=edge_weight, ridge_scale=ridge_scale
        )
        self.Q = Q
        self.Q_reg = Q_reg

        L = np.linalg.cholesky(Q_reg)
        # Cholesky of a near-singular matrix returns NaNs silently (unlike
        # scipy, which raises); check immediately, per PLAN.md §3.2.
        if not np.all(np.isfinite(L)):
            msg = (
                "Cholesky factorisation of the ridge-regularised GMRF "
                "precision matrix produced non-finite values. This usually "
                "means the ridge (ridge_scale) is too small relative to the "
                "connectivity weights. Try increasing ridge_scale."
            )
            raise ValueError(msg)
        self.L = L

    def add_data(self, pm1, pm2, cov, chunk_size=5000):
        """
        Load observations and pre-compute the 2D design matrix.

        Parameters
        ----------
        pm1, pm2 : array-like (N,)
            Observed two velocity/proper-motion components per star.
        cov : array-like (N, 2, 2)
            Per-star measurement covariance matrices. Build as
            ``[[sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y,
            sigma_y**2]]``. Note that catalogues such as Gaia report a
            *correlation* ``rho`` (e.g. ``pmra_pmdec_corr``), not a
            covariance; feeding rho directly in place of the covariance
            entry produces a non-positive-definite matrix for most stars.

        Returns
        -------
        None
            Sets ``self.matrix``.
        """
        if not self.grid:
            msg = "Run setup_grid() first."
            raise ValueError(msg)

        pm1 = np.asarray(pm1)
        self.n_stars = len(pm1)
        print(f"Computing 2D Design Matrix for {self.n_stars} stars...")

        self.matrix = precompute_design_matrix_2d(
            pm1, pm2, cov, self.grid, chunk_size=chunk_size
        )
        print(f"Matrix ready. Shape: {self.matrix.shape}")

    def run(self, num_warmup=500, num_samples=1000, gpu=None, seed=5567,
            prior="gaussian_core"):
        """
        Run the NUTS sampler.

        Parameters
        ----------
        num_warmup, num_samples : int
        gpu : bool or None
            See :meth:`veldist.KinematicSolver.run`.
        seed : int
        prior : {"gaussian_core", "gmrf"}
            Which prior to use. ``"gaussian_core"`` (default) gives the
            latent field a free bivariate-Gaussian core, so the velocity
            ellipsoid is unpenalised and the infinite-smoothing limit is a
            Gaussian. ``"gmrf"`` is the original pure Gauss-Markov random
            field, retained for comparison; its smoothing limit is a *uniform*
            distribution over the velocity grid, measured to bias sigma_x high
            by +0.5 (N=500) to +2.3 (N=100) km/s on a sigma=17 truth.

        Returns
        -------
        samples : dict
        """
        if self.matrix is None:
            msg = "No data added."
            raise ValueError(msg)
        if self.L is None:
            msg = "Run setup_grid() first."
            raise ValueError(msg)

        if prior not in ("gmrf", "gaussian_core"):
            msg = f"Unknown prior {prior!r}; expected 'gmrf' or 'gaussian_core'."
            raise ValueError(msg)

        if gpu is True:
            numpyro.set_platform("gpu")
        elif gpu is False:
            numpyro.set_platform("cpu")

        print("Starting NUTS MCMC (2D)...")
        L_jax = jnp.asarray(self.L)
        model_kwargs = {
            "matrix": jnp.asarray(self.matrix),
            "n_cells": self.grid["n_cells"],
            "L": L_jax,
        }
        if prior == "gaussian_core":
            model_fn = model_gaussian_core_2d
            model_kwargs["centers_2d"] = jnp.asarray(self.grid["centers_2d"])
        else:
            model_fn = model_2d

        nuts_kernel = NUTS(model_fn)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(jax.random.PRNGKey(int(seed)), **model_kwargs)

        self.samples = mcmc.get_samples()
        print("Inference Complete.")
        return self.samples

    def clip_uncertainties(self, floor_fraction=0.01, abs_floor=1e-10):
        """
        Apply uncertainty floors and store per-cell PM-distribution summary
        statistics.

        Direct port of :meth:`veldist.KinematicSolver.clip_uncertainties`;
        see that method for the full rationale (uncertainty floors, why
        marginal medians need not sum to 1). The only substantive difference
        here is naming: the quantity summarised is a bivariate proper-motion
        distribution, not a line-of-sight velocity distribution, so the keys
        are ``pdf_median`` / ``pdf_uncertainty`` rather than
        ``losvd_median`` / ``losvd_uncertainty``.

        This is a **post-processing step** that does *not* modify the raw
        posterior samples in ``self.samples``. It summarises the posterior
        as per-cell marginal medians and half-CI-widths in probability-mass
        space, then raises the uncertainties to a floor so that no cell
        carries a zero into the Dynamite output writer.

        - ``pdf_median`` stores the per-cell **marginal median** of the
          posterior probability mass. Because the joint posterior is a
          simplex but marginals are taken independently, the median values
          typically *sum to 0.85-0.95*, not 1. This is expected and correct.
        - ``pdf_uncertainty`` stores the **half-width** of the 68% credible
          interval: ``(p84 - p16) / 2``. Used as symmetric +/-error bars.

        Both quantities are **dimensionless probability mass per cell**.
        They are *not* divided by cell area.

        Motivation
        ----------
        Zero uncertainties in PM-distribution cells propagate into
        Dynamite's internal NNLS projection matrices and produce ``econ``
        zeros that cause weight-solving failures in large orbit-library
        runs. The relative floor (``floor_fraction * max_uncertainty``) is
        the primary safeguard; the absolute floor is a numerical backstop
        for channels where the posterior is pathologically tight across the
        board.

        Parameters
        ----------
        floor_fraction : float
            Relative floor as a fraction of the maximum per-cell half-CI-width
            across all cells. Default 0.01 (1%).
        abs_floor : float
            Absolute floor applied after the relative floor. Default 1e-10.

        Returns
        -------
        None
            Sets ``self.clipped_samples`` as a dict with keys:

            - ``'pdf_median'``:      per-cell marginal median, probability
              mass (dimensionless); shape (K**2,), flat row-major.
            - ``'pdf_uncertainty'``: clipped half-width of 68% CI,
              probability mass; shape (K**2,), flat row-major.
        """
        if self.samples is None:
            msg = "No posterior samples found. Call run() before clip_uncertainties()."
            raise ValueError(msg)

        # Work in probability-mass space throughout.
        # self.samples["intrinsic_pdf"] has shape (n_samples, K**2);
        # each row is a valid probability mass function (sums to 1).
        pdf_mass = np.asarray(self.samples["intrinsic_pdf"])

        # Sanity check: the MEAN of valid mass samples must also sum to ~1.
        mean_mass = np.mean(pdf_mass, axis=0)
        mean_sum = np.sum(mean_mass)
        if not np.isclose(mean_sum, 1.0, rtol=1e-3):
            msg = (
                f"Posterior mean PM distribution sums to {mean_sum:.6f}, expected ~1.0. "
                "Check that self.samples['intrinsic_pdf'] contains valid probability "
                "mass functions (each row should sum to 1)."
            )
            raise ValueError(msg)

        # Per-cell marginal statistics.
        median_mass = np.percentile(pdf_mass, 50, axis=0)
        p16 = np.percentile(pdf_mass, 16, axis=0)
        p84 = np.percentile(pdf_mass, 84, axis=0)

        # Half-width of 68% CI (used as symmetric +/-uncertainty in Dynamite).
        raw_half_width = (p84 - p16) / 2.0

        # Relative floor: a fraction of the widest half-CI in this map.
        rel_floor = floor_fraction * np.max(raw_half_width)

        clipped = np.maximum(raw_half_width, rel_floor)
        clipped = np.maximum(clipped, abs_floor)

        self.clipped_samples = {
            "pdf_median": median_mass,
            "pdf_uncertainty": clipped,
        }


# ==============================================================================
# Batch API
# ==============================================================================


def fit_all_bins_2d(bin_data_list, grid_kwargs, run_kwargs=None, min_stars=10, show_progress=True):
    """
    Run the full inference pipeline for a list of spatial (Voronoi) bins.

    Direct port of :func:`veldist.fit_all_bins` for the 2D (proper-motion)
    solver. For each bin, this executes the ``setup_grid`` -> ``add_data``
    -> ``run`` -> ``clip_uncertainties`` pipeline and returns a list of
    :class:`KinematicSolver2D` instances ready for the Dynamite output
    writer. Bins with too few stars are skipped (returning ``None`` at that
    position) so the writer can mask them.

    Unlike 1D's :func:`~veldist.fit_all_bins`, there is no ``match_grid``
    equivalent here and none will be added: every bin is fitted on the same
    shared ``grid_kwargs``. This is not a simplification made for
    convenience -- Dynamite's 2D kinematics ``.npz`` format carries a single
    scalar ``vxrange``/``vyrange`` for the whole map, so there is no
    per-aperture grid slot even at output time, and a per-bin matched grid
    would have nowhere to go.

    Each bin receives a unique RNG seed derived as ``base_seed + bin_index``
    to avoid correlations between sampling chains.

    Parameters
    ----------
    bin_data_list : list of dict
        One dict per Voronoi bin. Required keys:

        - ``'pm1'``, ``'pm2'``: arrays of observed proper-motion components.
        - ``'cov'``: array of per-star 2x2 measurement covariance matrices.

        Any additional keys (e.g. spatial metadata) are ignored here and
        can be passed separately to the output writer.
    grid_kwargs : dict
        Keyword arguments forwarded to :meth:`KinematicSolver2D.setup_grid`
        (``center``, ``width``, ``n_bins``, ...). Shared across all bins.
    run_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`KinematicSolver2D.run`
        (e.g. ``num_warmup``, ``num_samples``, ``gpu``, ``prior``). The
        ``seed`` key, if present, is used as the *base* seed; each bin then
        receives ``seed + bin_index``. Defaults to ``{}`` (all ``run``
        defaults apply).
    min_stars : int
        Minimum number of stars required to attempt inference. Bins with
        fewer stars are skipped with a warning. Default 10.
    show_progress : bool
        Show a single ``tqdm`` progress bar over bins instead of the
        default per-bin, per-chain NUTS progress bars. Default ``True``.

        Note: unlike 1D's ``KinematicSolver.run``,
        ``KinematicSolver2D.run`` currently has no ``progress_bar``
        parameter to suppress NumPyro's own per-chain bars, so this only
        controls the single outer bar over bins; it is not forwarded to
        ``run()``.

    Returns
    -------
    solvers : list
        One entry per input bin. Entries are either a fully solved
        :class:`KinematicSolver2D` (with ``samples`` and
        ``clipped_samples`` populated) or ``None`` for skipped bins.
    """
    if run_kwargs is None:
        run_kwargs = {}

    # Extract the base seed so we can derive per-bin seeds.
    run_kwargs = dict(run_kwargs)
    base_seed = run_kwargs.pop("seed", 5567)

    n_total = len(bin_data_list)
    solvers = []

    bin_iter = enumerate(bin_data_list)
    if show_progress:
        from tqdm.auto import tqdm

        bin_iter = tqdm(bin_iter, total=n_total, desc="Fitting bins", unit="bin")

    for i, bin_data in bin_iter:
        if not show_progress:
            print(f"Fitting bin {i + 1}/{n_total}...")

        pm1 = np.asarray(bin_data["pm1"])
        pm2 = np.asarray(bin_data["pm2"])
        cov = np.asarray(bin_data["cov"])

        if len(pm1) < min_stars:
            warnings.warn(
                f"Bin {i} has only {len(pm1)} star(s) (minimum is {min_stars}). "
                "Skipping. This bin will appear as None in the output list and "
                "should be masked in the Dynamite input files.",
                stacklevel=2,
            )
            solvers.append(None)
            continue

        solver = KinematicSolver2D()
        solver.setup_grid(**grid_kwargs)

        # add_data/run print progress lines of their own; redirect those to
        # keep the single outer tqdm bar clean instead of interleaving with
        # printed lines per bin.
        with contextlib.redirect_stdout(io.StringIO()) if show_progress else contextlib.nullcontext():
            solver.add_data(pm1=pm1, pm2=pm2, cov=cov)
            solver.run(seed=base_seed + i, **run_kwargs)

        solver.clip_uncertainties()

        solvers.append(solver)

    n_solved = sum(s is not None for s in solvers)
    n_skipped = n_total - n_solved
    print(f"Done. {n_solved}/{n_total} bins solved" + (f", {n_skipped} skipped." if n_skipped else "."))

    return solvers
