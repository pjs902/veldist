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
]


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
    smoothness_sigma = numpyro.sample("smoothness_sigma", dist.HalfNormal(0.1))

    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n_cells]).to_event(1))
    x = smoothness_sigma * jax.scipy.linalg.solve_triangular(L.T, z, lower=False)

    intrinsic_pdf = jax.nn.softmax(x)
    numpyro.deterministic("intrinsic_pdf", intrinsic_pdf)

    per_star_prob = jnp.dot(matrix, intrinsic_pdf)
    log_prob = jnp.sum(jnp.log(per_star_prob))
    numpyro.factor("obs_log_lik", log_prob)


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
    """

    def __init__(self):
        self.matrix = None
        self.grid = {}
        self.Q = None
        self.Q_reg = None
        self.L = None
        self.n_stars = None
        self.samples = None

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

    def run(self, num_warmup=500, num_samples=1000, gpu=None, seed=5567):
        """
        Run the NUTS sampler.

        Parameters
        ----------
        num_warmup, num_samples : int
        gpu : bool or None
            See :meth:`veldist.KinematicSolver.run`.
        seed : int

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

        if gpu is True:
            numpyro.set_platform("gpu")
        elif gpu is False:
            numpyro.set_platform("cpu")

        print("Starting NUTS MCMC (2D)...")
        L_jax = jnp.asarray(self.L)
        nuts_kernel = NUTS(model_2d)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)

        rng_key = jax.random.PRNGKey(int(seed))
        mcmc.run(
            rng_key,
            matrix=jnp.asarray(self.matrix),
            n_cells=self.grid["n_cells"],
            L=L_jax,
        )

        self.samples = mcmc.get_samples()
        print("Inference Complete.")
        return self.samples
