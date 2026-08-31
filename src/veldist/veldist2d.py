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
import json
import traceback
import warnings
from functools import cache
from pathlib import Path

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


def _as_kx_ky(n_bins):
    """Normalise a per-axis bin count to ``(kx, ky)``.

    Accepts either a scalar (square grid, ``kx == ky``) or a 2-tuple
    ``(kx, ky)`` (rectangular grid). This is the single place that
    interprets ``n_bins``; every function below that needs the per-axis
    counts goes through here (or receives ``(kx, ky)`` directly).
    """
    if np.isscalar(n_bins):
        kx = ky = int(n_bins)
    else:
        kx, ky = n_bins
        kx, ky = int(kx), int(ky)
    return kx, ky


def setup_grid_2d(center, width, n_bins):
    """
    Define a velocity grid over a bivariate velocity space.

    ``n_bins`` may be a scalar ``K`` (square ``K x K`` grid, the original
    behaviour) or a 2-tuple ``(kx, ky)`` (rectangular grid) -- see
    ``TASKS.md`` / the 2026-08-31 rectangular-grid change for why: a square
    grid forces both axes to share one resolution and one extent-in-sigma,
    which under-resolves or truncates whichever axis has the smaller
    intrinsic dispersion.

    The grid is flattened row-major (C order) using ``np.ravel_multi_index`` /
    ``np.unravel_index`` for every index conversion, never hand-written
    arithmetic, to avoid the row-major/column-major transposition bug
    flagged in ``PLAN.md`` §3.1: cell ``(ix, iy)`` maps to flat index
    ``m = ix * ky + iy`` (``ky`` is the per-axis count of the *second* axis;
    for a square grid this is the same ``m = ix * K + iy`` as before).

    Parameters
    ----------
    center : (float, float)
        Center of the grid, ``(cx, cy)``.
    width : (float, float)
        Total width of the grid, ``(wx, wy)``.
    n_bins : int or (int, int)
        Number of bins per axis. Either a scalar ``K`` (square grid, total
        cells ``K**2``) or ``(kx, ky)`` (rectangular grid, total cells
        ``kx * ky``).

    Returns
    -------
    grid : dict
        Keys: ``centers_x`` (kx,), ``centers_y`` (ky,), ``edges_x`` (kx+1,),
        ``edges_y`` (ky+1,), ``centers_2d`` (kx*ky, 2) [row-major flattened],
        ``width_x``, ``width_y``, ``area`` (= width_x * width_y),
        ``n_bins_x``, ``n_bins_y`` (per-axis counts), ``n_cells``
        (= kx * ky), ``shape`` (kx, ky).

        ``n_bins`` is set to the common per-axis count ``K`` **only when the
        grid is square** (``kx == ky``); it is intentionally *absent* from
        the dict for a rectangular grid, so any code that still assumes a
        single scalar bin count fails with a clear ``KeyError`` rather than
        silently using the wrong axis's count. New code should read
        ``n_bins_x`` / ``n_bins_y`` (or ``shape``) instead.
    """
    cx, cy = center
    wx, wy = width
    kx, ky = _as_kx_ky(n_bins)

    edges_x = np.linspace(cx - wx / 2, cx + wx / 2, kx + 1)
    edges_y = np.linspace(cy - wy / 2, cy + wy / 2, ky + 1)
    centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
    centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])

    width_x = edges_x[1] - edges_x[0]
    width_y = edges_y[1] - edges_y[0]

    n_cells = kx * ky
    # Row-major (C order) meshgrid: flat index m = ix*ky + iy.
    ix_grid, iy_grid = np.meshgrid(np.arange(kx), np.arange(ky), indexing="ij")
    flat = np.ravel_multi_index((ix_grid.ravel(), iy_grid.ravel()), (kx, ky), order="C")
    # flat should just be 0..n_cells-1 in this order; assemble centers_2d to match.
    centers_2d = np.empty((n_cells, 2))
    centers_2d[flat, 0] = centers_x[ix_grid.ravel()]
    centers_2d[flat, 1] = centers_y[iy_grid.ravel()]

    grid = {
        "centers_x": centers_x,
        "centers_y": centers_y,
        "edges_x": edges_x,
        "edges_y": edges_y,
        "centers_2d": centers_2d,
        "width_x": float(width_x),
        "width_y": float(width_y),
        "area": float(width_x * width_y),
        "n_bins_x": kx,
        "n_bins_y": ky,
        "n_cells": n_cells,
        "shape": (kx, ky),
    }
    if kx == ky:
        grid["n_bins"] = kx
    return grid


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
    kx, ky = grid["n_bins_x"], grid["n_bins_y"]
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
                p1[idx], p2[idx], c[idx], ex, ey, kx, ky
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


def _design_matrix_gl_quadrature(p1, p2, cov, edges_x, edges_y, kx, ky):
    """
    2x2 Gauss-Legendre sub-cell quadrature for a chunk of (correlated) stars.

    Returns an (n, kx*ky) array (float64) of cell probability masses.
    """
    n = len(p1)
    n_cells = kx * ky

    cx0 = edges_x[:-1]
    cx1 = edges_x[1:]
    cy0 = edges_y[:-1]
    cy1 = edges_y[1:]
    hx = 0.5 * (cx1 - cx0)  # half-width per x cell, (kx,)
    hy = 0.5 * (cy1 - cy0)
    mx = 0.5 * (cx1 + cx0)  # mid per x cell, (kx,)
    my = 0.5 * (cy1 + cy0)

    nodes, gweights = _gauss_legendre_2x2_nodes()  # 2 nodes each axis

    # Evaluation points: for each cell, 2x2=4 points. Build full (kx, ky, 4)
    # grid of (x, y) coordinates and weights, then evaluate per star.
    # x_pts[ix, jnode] = mx[ix] + hx[ix]*node[jnode]
    x_pts = mx[:, None] + hx[:, None] * nodes[None, :]  # (kx, 2)
    y_pts = my[:, None] + hy[:, None] * nodes[None, :]  # (ky, 2)
    wx_pts = hx[:, None] * gweights[None, :]  # (kx, 2); half-width already in Jacobian
    wy_pts = hy[:, None] * gweights[None, :]  # (ky, 2)

    # Combine into (kx, ky, 4) grid of points & weights (2 nodes per axis -> 4 combos)
    # point index p in [0,4): (a,b) = divmod(p, 2)
    xs = np.empty((kx, ky, 4))
    ys = np.empty((kx, ky, 4))
    ws = np.empty((kx, ky, 4))
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
    Build an 8-connectivity intrinsic GMRF precision matrix Q for a kx x ky grid.

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

    NOTE (open question, deliberately not addressed here): on a rectangular
    grid where the *cells themselves* are non-square (``width_x/kx !=
    width_y/ky``), the geometric justification for ``diag_weight =
    1/sqrt(2)`` -- the Euclidean distance to a corner-touching neighbour on a
    square lattice -- no longer strictly applies (a diagonal step covers a
    different physical distance than sqrt(2) cells). This function keeps the
    weights exactly as they were for the square case and does not attempt a
    cell-aspect-ratio correction; changing that is a separate, deliberate
    decision this task does not make.

    Parameters
    ----------
    k : int or (int, int)
        Grid size per axis. Either a scalar ``K`` (square grid, total cells
        ``K**2``) or ``(kx, ky)`` (rectangular grid, total cells
        ``kx * ky``).
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
    Q : np.ndarray (n_cells, n_cells)
    Q_reg : np.ndarray (n_cells, n_cells)
        Q with the ridge added.
    """
    if diag_weight is None:
        diag_weight = 1.0 / np.sqrt(2.0)

    kx, ky = _as_kx_ky(k)
    n_cells = kx * ky
    W = np.zeros((n_cells, n_cells))

    ix_grid, iy_grid = np.meshgrid(np.arange(kx), np.arange(ky), indexing="ij")
    ix_flat = ix_grid.ravel()
    iy_flat = iy_grid.ravel()

    # Edge neighbours: (dx, dy) in {(1,0), (-1,0), (0,1), (0,-1)}
    edge_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    diag_offsets = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    def add_offsets(offsets, weight):
        for dx, dy in offsets:
            nx = ix_flat + dx
            ny = iy_flat + dy
            valid = (nx >= 0) & (nx < kx) & (ny >= 0) & (ny < ky)
            src = np.ravel_multi_index((ix_flat[valid], iy_flat[valid]), (kx, ky), order="C")
            dst = np.ravel_multi_index((nx[valid], ny[valid]), (kx, ky), order="C")
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
    """Orthonormal basis of the bivariate-quadratic null space, (n_cells, 6).

    ``k`` is a scalar (square grid) or ``(kx, ky)`` (rectangular grid).

    Spans ``{1, x, y, x^2, xy, y^2}`` -- exactly the log-densities of
    bivariate Gaussians, which is what the prior must leave free so that the
    velocity ellipsoid is not shrunk.

    Built from tensor-product Legendre polynomials of total degree <= 2 rather
    than the raw monomials, for the conditioning reason documented in
    ``veldist.py::_null_space_basis``: the Vandermonde basis is badly
    conditioned and its QR loses precision. Legendre spans the identical
    space, so the projector is unchanged.

    Uses an index grid rather than physical cell centres. Both give the same
    projector: the orthogonal projector onto a subspace does not depend on
    which basis spans it, and for a uniformly spaced grid the index and
    physical coordinates differ only by an affine map. ``setup_grid_2d`` only
    produces uniform grids.

    Row-major flattened (``m = ix*ky + iy``) to match ``setup_grid_2d``'s
    ``centers_2d``. Cached: costs an O(n_cells^2) QR, depends only on ``k``,
    and is evaluated at JAX trace time where the result is constant-folded.
    """
    kx, ky = _as_kx_ky(k)

    idx_x = np.arange(kx, dtype=float)
    ux = 2.0 * (idx_x - idx_x.mean()) / (kx - 1)  # -> [-1, 1]
    idx_y = np.arange(ky, dtype=float)
    uy = 2.0 * (idx_y - idx_y.mean()) / (ky - 1)  # -> [-1, 1]

    p0x, p1x, p2x = np.ones_like(ux), ux, 0.5 * (3.0 * ux**2 - 1.0)
    p0y, p1y, p2y = np.ones_like(uy), uy, 0.5 * (3.0 * uy**2 - 1.0)

    ix, iy = np.meshgrid(np.arange(kx), np.arange(ky), indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()

    # Tensor products with total degree <= 2.
    cols = [
        p0x[ix] * p0y[iy],   # 1
        p1x[ix] * p0y[iy],   # x
        p0x[ix] * p1y[iy],   # y
        p2x[ix] * p0y[iy],   # x^2
        p1x[ix] * p1y[iy],   # xy
        p0x[ix] * p2y[iy],   # y^2
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
    kx, ky = _as_kx_ky(k)
    proj = np.eye(kx * ky) - q_ns @ q_ns.T
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


def generate_gaussian_core_field_2d(shape, centers_2d, L):
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
    shape : int or (int, int)
        Grid size per axis: a scalar ``K`` (square grid) or ``(kx, ky)``
        (rectangular grid). Total cells ``kx * ky``. Passed explicitly
        (never inferred from ``n_cells`` via ``sqrt``) because
        ``round(sqrt(n_cells))`` silently recovers the wrong per-axis
        counts on a rectangular grid with no error raised.
    centers_2d : array-like, shape (kx*ky, 2)
        Physical cell centres from :func:`setup_grid_2d`. Required because the
        core is quadratic in *velocity*, not in cell index.
    L : jnp.ndarray, shape (kx*ky, kx*ky)
        Cholesky factor of the ridge-regularised GMRF precision.

    Returns
    -------
    field : jnp.ndarray, shape (kx*ky,)
        Latent log-density, to be passed through ``softmax``.
    """
    kx, ky = _as_kx_ky(shape)
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
    ) * _gmrf_deviation_scale_2d(shape)
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([kx * ky]).to_event(1))
    # x = sigma * L^-T z, i.e. solve the UPPER triangular system L.T @ x = z.
    # Using L with lower=True would give L^-1 z, a different covariance; see
    # test_solve_triangular_direction.
    w = sigma3 * jax.scipy.linalg.solve_triangular(L.T, z, lower=False)

    # Project out the quadratic null space. Cached constant, so a matmul
    # rather than a QR per leapfrog step.
    q_ns = jnp.asarray(_null_space_basis_2d(shape))
    deviation = w - q_ns @ (q_ns.T @ w)

    return core + deviation


def model_gaussian_core_2d(matrix, n_cells, L, centers_2d, shape):
    """The 2D NumPyro model with the Gaussian-core prior.

    Parameters
    ----------
    matrix : jnp.ndarray, shape (N_stars, kx*ky)
        Pre-computed 2D design matrix.
    n_cells : int
        Number of grid cells, ``kx * ky``. Kept as an explicit argument
        (rather than derived from ``shape``) so the model signature matches
        :func:`model_2d`'s, but note it is *not* used to recover the
        per-axis counts -- see ``shape`` below.
    L : jnp.ndarray, shape (n_cells, n_cells)
        Cholesky factor of the ridge-regularised GMRF precision.
    centers_2d : jnp.ndarray, shape (n_cells, 2)
        Physical cell centres.
    shape : int or (int, int)
        Per-axis grid size: a scalar ``K`` (square grid) or ``(kx, ky)``
        (rectangular grid). Must be passed explicitly -- ``n_cells`` alone
        cannot be un-ambiguously factored back into ``(kx, ky)`` for a
        rectangular grid (``round(sqrt(n_cells))`` silently gives the wrong
        answer with no error), which is exactly the bug this parameter
        exists to avoid.
    """
    field = generate_gaussian_core_field_2d(shape, centers_2d, L)

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
        n_bins : int or (int, int)
            Per-axis bin count. Either a scalar ``K`` (square grid, total
            cells ``K**2``) or ``(kx, ky)`` (rectangular grid, total cells
            ``kx * ky``).
        diag_weight, edge_weight, ridge_scale : float
            Forwarded to :func:`build_gmrf_precision`.

        Returns
        -------
        None
            Sets ``self.grid``, ``self.Q``, ``self.Q_reg``, ``self.L``.
        """
        self.grid = setup_grid_2d(center, width, n_bins)
        shape = self.grid["shape"]

        Q, Q_reg = build_gmrf_precision(
            shape, diag_weight=diag_weight, edge_weight=edge_weight, ridge_scale=ridge_scale
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

    def run(self, num_warmup=500, num_samples=3000, gpu=None, seed=5567,
            prior="gaussian_core", target_accept_prob=0.95, dense_mass=False,
            max_tree_depth=10):
        """
        Run the NUTS sampler.

        Parameters
        ----------
        num_warmup : int
        num_samples : int
            **Defaults to 3000, not NumPyro's typical ~1000**, on measured
            grounds (2026-08-06, real HST data, ``dense_mass=False``,
            ``target_accept_prob=0.95``): min ESS across the six scalar
            sites (``v0x``/``v0y``/``s0x``/``s0y``/``rho0``/``sigma3``) rose
            from ~260-470 at 1000 samples to ~830-1290 at 3000, for
            essentially the *same* per-bin wall time (~1-3s, dominated by
            JIT compile, not sampling -- see ``dense_mass`` below). Drawing
            more samples here is nearly free; there is no reason to leave
            ESS on the table.
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
        target_accept_prob : float
            NUTS target acceptance rate. Defaults to 0.95 (NumPyro's own
            default is 0.8), matching 1D's ``KinematicSolver.run`` -- see
            that method's docstring for the funnel-geometry rationale
            (``docs/validation.md``). **Not yet re-validated for the 2D
            model with a full SBC campaign** the way 1D was, but *is*
            measured directly on real HST data (2026-08-06, ``dense_mass=
            False``): at ``num_samples=3000``, 0/5 test bins had any
            divergences and min ESS was ~830-1290, vs. 3 total divergences
            (out of 5 bins) at ``target_accept_prob=0.8`` with the same
            sample count -- ``0.95`` is the better-supported choice, not
            just a 1D holdover.
        dense_mass : bool
            Use a dense (full-covariance) mass matrix instead of NumPyro's
            default diagonal one. **Defaults to False** -- measured to be
            actively counterproductive on real HST data (2026-08-06).
            With ``dense_mass=True``, NUTS hits ``max_tree_depth``
            (1023 steps/sample) on essentially every sample regardless of
            ``target_accept_prob``, and a controlled comparison (same
            bins, JIT-cache warm so compile cost was hidden) still gave
            *lower* min ESS (~200-290) than ``dense_mass=False`` at the same
            1000-sample budget (~260-470) -- all those extra leapfrog steps
            buy nothing. With a cold cache (the realistic case: ~1400 bins,
            ~1400 distinct star counts, so nearly every bin needs a fresh
            XLA compile), the dense-mass kernel's compile cost alone was
            ~100s/bin (~20x ``dense_mass=False``), which is where the
            "~43 hours for a full run" estimate came from. Unlike 1D --
            where ``dense_mass=True`` *reduced* cost and improved r_hat/ESS
            -- this is the opposite result for the 2D model; do not port
            the 1D dense-mass finding here without re-measuring. Left
            overridable for anyone who wants to re-investigate, but do not
            flip the default without new evidence.
        max_tree_depth : int
            NUTS's cap on trajectory doubling; NumPyro's own default is 10
            (max 1023 leapfrog steps/sample). Exposed here (1D's
            ``KinematicSolver.run`` does not expose it) since it was needed
            to diagnose the ``dense_mass`` tree-depth blowup above.

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
            model_kwargs["shape"] = self.grid["shape"]
        else:
            model_fn = model_2d

        nuts_kernel = NUTS(
            model_fn,
            target_accept_prob=target_accept_prob,
            dense_mass=dense_mass,
            max_tree_depth=max_tree_depth,
        )
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


def _array_stats(x):
    """Small numeric summary of an array, for failure diagnostics -- plain
    floats/ints only, so this is always JSON-serialisable regardless of
    the input dtype (numpy scalars are not JSON-serialisable directly)."""
    x = np.asarray(x)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "n_nan": int(np.sum(~np.isfinite(x))),
    }


def _log_bin_failure(failure_log_path, failure):
    """Append one failure record as a JSON line. Safe under concurrent
    writers (``n_jobs`` > 1): each call opens, writes once, and closes: a
    single ``write()`` to a file opened with ``'a'`` is atomic on POSIX for
    writes below the platform pipe-buffer size (a few KB), which a single
    failure record is.
    """
    if failure_log_path is None:
        return
    with Path(failure_log_path).open("a") as f:
        f.write(json.dumps(failure) + "\n")


def _fit_one_bin_2d(i, pm1, pm2, cov, grid_kwargs, run_kwargs, seed, min_stars, failure_log_path=None):
    """Fit a single bin. Module-level (not a closure) so it's picklable for
    ``ProcessPoolExecutor`` -- see :func:`fit_all_bins_2d`'s ``n_jobs``.

    A bin whose MCMC fit raises (e.g. NumPyro's "Cannot find valid initial
    parameters", seen in practice on real HST data -- 2026-08-06) is caught
    here, logged with enough context to investigate later, and skipped
    (returned as ``None``) rather than propagating and killing every other
    bin in a multi-hour ``fit_all_bins_2d`` run. ``min_stars`` skips are a
    normal, expected outcome and are not treated as failures.

    Returns
    -------
    (int, KinematicSolver2D or None)
        Bin index and the solved solver, or ``None`` if skipped (either
        ``len(pm1) < min_stars``, or the fit raised an exception).
    """
    if len(pm1) < min_stars:
        warnings.warn(
            f"Bin {i} has only {len(pm1)} star(s) (minimum is {min_stars}). "
            "Skipping. This bin will appear as None in the output list and "
            "should be masked in the Dynamite input files.",
            stacklevel=2,
        )
        return i, None

    solver = KinematicSolver2D()
    try:
        solver.setup_grid(**grid_kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            solver.add_data(pm1=pm1, pm2=pm2, cov=cov)
            solver.run(seed=seed, **run_kwargs)
        solver.clip_uncertainties()
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: any failure here must not kill the whole run
        failure = {
            "bin": i,
            "seed": seed,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "pm1_stats": _array_stats(pm1),
            "pm2_stats": _array_stats(pm2),
            "cov00_stats": _array_stats(cov[:, 0, 0]),
            "cov11_stats": _array_stats(cov[:, 1, 1]),
            "cov01_stats": _array_stats(cov[:, 0, 1]),
            "grid_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in grid_kwargs.items()},
        }
        _log_bin_failure(failure_log_path, failure)
        warnings.warn(
            f"Bin {i} ({len(pm1)} stars) failed during the MCMC fit: "
            f"{type(exc).__name__}: {exc}. Skipping -- this bin will appear "
            "as None in the output list and should be masked in the "
            "Dynamite input files. Full diagnostics "
            + (f"logged to {failure_log_path}." if failure_log_path else "were NOT logged to disk (failure_log_path=None)."),
            stacklevel=2,
        )
        return i, None

    return i, solver


def fit_all_bins_2d(
    bin_data_list,
    grid_kwargs,
    run_kwargs=None,
    min_stars=10,
    show_progress=True,
    n_jobs=1,
    failure_log_path="fit_all_bins_2d_failures.jsonl",
):
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
        ``run()`` in the ``n_jobs=1`` path. NumPyro's per-chain bars are
        always redirected/suppressed inside ``_fit_one_bin_2d`` regardless
        of ``n_jobs``.
    n_jobs : int
        Number of bins to fit concurrently via ``ProcessPoolExecutor``.
        Default 1 (sequential, same as before this parameter existed).
        Bins are independent (own data, own posterior), so this
        parallelises over bins, **not** over chains within a bin --
        ``KinematicSolver2D.run`` has no ``num_chains`` and this does not
        add one. Each worker is a fresh process (spawned, not forked --
        JAX/XLA is not fork-safe once its backend has initialised), so
        each pays its own JIT compile cost per star-count shape it
        encounters; with ``n_jobs`` workers all potentially compiling the
        same shape independently, total compile work can exceed the
        sequential case, but wall time still drops because it happens in
        parallel. Uses ``multiprocessing.get_context("spawn")`` explicitly
        for the same fork-safety reason. Reserving JAX host devices via
        ``numpyro.set_host_device_count`` in the parent process (for
        chain-level parallelism) is unrelated to this and unaffected by
        it, since spawned workers get a fresh JAX backend, not the
        parent's.
    failure_log_path : str or path-like or None
        Where to append per-bin failure diagnostics (JSON lines: bin index,
        seed, exception type/message/traceback, summary stats on
        ``pm1``/``pm2``/the covariance diagonal/off-diagonal, and
        ``grid_kwargs``) when a bin's MCMC fit raises. A bin failing here
        is caught (see :func:`_fit_one_bin_2d`) and skipped, **not** left
        to propagate and kill the rest of a run that may be hours long --
        seen in practice on real HST data (2026-08-06): NumPyro's "Cannot
        find valid initial parameters" on one pathological bin took down
        an otherwise-healthy ~1400-bin run. Default
        ``'fit_all_bins_2d_failures.jsonl'`` (relative to the current
        working directory); pass ``None`` to disable the log file (a
        ``warnings.warn`` is still emitted either way). Safe to point
        multiple ``n_jobs`` workers at the same path: each failure is one
        atomic ``open`` + single ``write`` + ``close``, not a held-open
        file handle.

    Returns
    -------
    solvers : list
        One entry per input bin. Entries are either a fully solved
        :class:`KinematicSolver2D` (with ``samples`` and
        ``clipped_samples`` populated) or ``None`` for a skipped bin
        (either below ``min_stars``, or a failed fit -- see
        ``failure_log_path`` to tell the two apart after the fact).
    """
    if run_kwargs is None:
        run_kwargs = {}

    # Extract the base seed so we can derive per-bin seeds.
    run_kwargs = dict(run_kwargs)
    base_seed = run_kwargs.pop("seed", 5567)

    # Fresh log per call -- stale failures from a previous, now-superseded
    # run of this function (e.g. before a crash) would otherwise mix in
    # and misattribute which run a given bin's failure came from.
    if failure_log_path is not None:
        Path(failure_log_path).write_text("")

    n_total = len(bin_data_list)
    solvers = [None] * n_total

    if n_jobs == 1:
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

            _, solver = _fit_one_bin_2d(
                i, pm1, pm2, cov, grid_kwargs, run_kwargs, base_seed + i, min_stars, failure_log_path
            )
            solvers[i] = solver
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    _fit_one_bin_2d,
                    i,
                    np.asarray(bin_data["pm1"]),
                    np.asarray(bin_data["pm2"]),
                    np.asarray(bin_data["cov"]),
                    grid_kwargs,
                    run_kwargs,
                    base_seed + i,
                    min_stars,
                    failure_log_path,
                ): i
                for i, bin_data in enumerate(bin_data_list)
            }
            completed = as_completed(futures)
            if show_progress:
                from tqdm.auto import tqdm

                completed = tqdm(completed, total=n_total, desc="Fitting bins", unit="bin")
            for future in completed:
                i, solver = future.result()
                solvers[i] = solver

    n_solved = sum(s is not None for s in solvers)
    n_below_min_stars = sum(1 for bin_data in bin_data_list if len(bin_data["pm1"]) < min_stars)
    n_failed = n_total - n_solved - n_below_min_stars
    summary = f"Done. {n_solved}/{n_total} bins solved"
    if n_below_min_stars:
        summary += f", {n_below_min_stars} below min_stars"
    if n_failed:
        summary += f", {n_failed} failed during fitting"
        if failure_log_path is not None:
            summary += f" (see {failure_log_path})"
    print(summary + ".")

    return solvers
