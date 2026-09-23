# file src/veldist/veldist2d.py
"""Bayesian Matrix-Based 2D (Proper-Motion) Kinematic Deconvolution
==================================================================

The 2D counterpart of ``veldist.py``. It infers the intrinsic bivariate
velocity distribution (for example over the proper-motion components
``pmra``/``pmdec``) from stars with individual 2x2 measurement covariances,
using a precomputed design matrix and a 2D Gauss-Markov random field (GMRF)
smoothness prior.

It is deliberately kept separate from ``veldist.py`` (``PLAN.md`` Part 3).
The two solvers share the overall approach (design matrix and a softmax of
a smooth latent field) but differ in enough details (grid flattening, box
versus quadrature integration, building the precision matrix) that merging
them would cost more than the duplication.

The GMRF, Cholesky and latent-field maths run in float64
(``jax.config.update("jax_enable_x64", True)``), while the design matrix
``M`` is stored as float32 to save memory. The mixed precision is
intentional; see the Gotchas in ``PLAN.md`` §3.1/§3.2.
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
from jax.scipy.special import logsumexp
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
    """Turn a per-axis bin count into ``(kx, ky)``.

    Accepts a scalar (square grid, ``kx == ky``) or a 2-tuple ``(kx, ky)``
    (rectangular grid). This is the only place ``n_bins`` is interpreted;
    everything below that needs per-axis counts goes through it or receives
    ``(kx, ky)`` directly.
    """
    if np.isscalar(n_bins):
        kx = ky = int(n_bins)
    else:
        kx, ky = n_bins
        kx, ky = int(kx), int(ky)
    return kx, ky


def setup_grid_2d(center, width, n_bins):
    """Define a velocity grid over a 2D velocity space.

    ``n_bins`` is either a scalar ``K`` (a square ``K x K`` grid, the original
    behaviour) or a 2-tuple ``(kx, ky)`` (a rectangular grid). Rectangular grids
    were added on 2026-08-31 (see ``TASKS.md``) because a square grid forces
    both axes to share one resolution and one extent in units of sigma, which
    under-resolves or truncates whichever axis has the smaller dispersion.

    The grid is flattened in row-major (C) order, and every index conversion
    uses ``np.ravel_multi_index`` / ``np.unravel_index`` rather than
    hand-written arithmetic, to avoid the row/column transposition bug noted in
    ``PLAN.md`` §3.1. Cell ``(ix, iy)`` has flat index ``m = ix * ky + iy``,
    where ``ky`` is the count along the second axis (``m = ix * K + iy`` on a
    square grid).

    Parameters
    ----------
    center : (float, float)
        Grid centre, ``(cx, cy)``.
    width : (float, float)
        Total grid width, ``(wx, wy)``.
    n_bins : int or (int, int)
        Bins per axis: a scalar ``K`` (square, ``K**2`` cells) or
        ``(kx, ky)`` (rectangular, ``kx * ky`` cells).

    Returns
    -------
    grid : dict
        Keys: ``centers_x`` (kx,), ``centers_y`` (ky,), ``edges_x`` (kx+1,),
        ``edges_y`` (ky+1,), ``centers_2d`` (kx*ky, 2) [row-major flattened],
        ``width_x``, ``width_y``, ``area`` (= width_x * width_y),
        ``n_bins_x``, ``n_bins_y`` (per-axis counts), ``n_cells``
        (= kx * ky), ``shape`` (kx, ky).

        ``n_bins`` (the common per-axis count ``K``) is present **only for a
        square grid**. It is left out on purpose for rectangular grids, so
        code that still assumes one scalar count fails with a clear
        ``KeyError`` instead of silently using the wrong axis. New code should
        use ``n_bins_x`` / ``n_bins_y`` or ``shape``.
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
    """Compute the 2D design matrix M, shape (N, n_cells).

    ``M[i, m]`` is the integral over cell ``m`` of
    ``N(mu=(pm1_i, pm2_i), Sigma=cov_i)``.

    Each star takes one of two paths:

    1. Diagonal ``cov_i`` (``cov_i[0,1] == 0``): exact box integration. The
       integral factorises into the outer product of two 1D erf/CDF
       integrals, reusing ``veldist.precompute_design_matrix`` on each axis.
    2. Correlated ``cov_i``: 2x2 Gauss-Legendre quadrature within each cell
       (4 points per cell per star), which works for any Sigma.

    Stars are processed in chunks (default 5000) to limit peak memory: JAX
    allocates intermediates, so building the full ``(N, n_cells)`` array at
    once can take 2-3 times its final size. All quadrature and erf maths is in
    float64; each chunk is cast to float32 only at the end.

    Parameters
    ----------
    pm1, pm2 : array-like (N,)
        The two observed velocity or proper-motion components of each star.
    cov : array-like (N, 2, 2)
        Per-star measurement covariance matrices.
    grid : dict
        Output of :func:`setup_grid_2d`.
    chunk_size : int
        Stars per chunk. Default 5000.

    Returns
    -------
    M : np.ndarray (N, n_cells), float32
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
    """Evaluate N((x, y); mu, cov) for per-star means and covariances.

    Shapes: ``x``, ``y`` are (n_stars, n_pts); ``mu1``, ``mu2`` are
    (n_stars,); ``cov`` is (n_stars, 2, 2). The result is (n_stars, n_pts).
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
    """2x2 Gauss-Legendre quadrature within each cell for a chunk of correlated
    stars.

    Returns an (n, kx*ky) float64 array of cell probability masses.
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
    """Build an 8-neighbour intrinsic GMRF precision matrix Q for a kx x ky grid.

    ``Q = D - W``, where ``W`` is the symmetric adjacency-weight matrix, with
    weight ``edge_weight`` (default 1) for the 4 side neighbours and
    ``diag_weight`` (default ``1/sqrt(2)``) for the 4 diagonal neighbours, and
    ``D = diag(row sums of W)``.

    The adjacency is built from explicit ``np.ravel_multi_index`` neighbour
    lists, never from array shifts. Shifts silently wrap around at the grid
    boundary, giving a periodic grid by accident, which is the most dangerous
    failure mode here (``PLAN.md`` §3.2 gotchas).

    ``Q`` is singular by construction; its null space is the constant vector,
    which the softmax removes anyway. A ridge is added for the Cholesky
    factorisation, ``eps = ridge_scale * mean(diag(Q))``, scaled to the
    diagonal rather than absolute so that it keeps its meaning if the weights
    change.

    Open question, not addressed here: when the cells themselves are not
    square (``width_x/kx != width_y/ky``), the reasoning behind
    ``diag_weight = 1/sqrt(2)`` (the distance to a corner neighbour on a square
    lattice) no longer strictly holds. The weights are kept as for square
    cells; correcting for cell aspect ratio would be a separate decision.

    Parameters
    ----------
    k : int or (int, int)
        Grid size per axis: a scalar ``K`` (square, ``K**2`` cells) or
        ``(kx, ky)`` (rectangular, ``kx * ky`` cells).
    diag_weight : float
        Weight for diagonal neighbours. The default ``1/sqrt(2)`` weights by
        distance; 1.0 weights all 8 neighbours equally.
    edge_weight : float
        Weight for side neighbours. Default 1.0.
    ridge_scale : float
        Relative ridge added before the Cholesky:
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

    The basis spans ``{1, x, y, x^2, xy, y^2}``, exactly the log-densities of
    bivariate Gaussians. The prior must leave these free so that the velocity
    ellipsoid is not shrunk.

    It is built from tensor-product Legendre polynomials of total degree <= 2
    rather than raw monomials, for conditioning (see
    ``veldist.py::_null_space_basis``); both span the same space, so the
    projector is the same. It also uses cell indices rather than physical
    centres, which again gives the same projector because the two differ by an
    affine map on a uniform grid, and ``setup_grid_2d`` only makes uniform
    grids.

    Flattened row-major (``m = ix*ky + iy``) to match ``centers_2d``. Cached:
    it needs an O(n_cells^2) QR, depends only on ``k``, and runs at JAX trace
    time, where the result is folded in as a constant.
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
    """Sørbye-Rue scaling constant for the null-space-projected 2D GMRF.

    Returns the factor that makes the generalised variance (the geometric mean
    of the per-cell marginal variances of the projected field) equal to 1, so
    that ``sigma3`` means "typical log-density departure from the Gaussian null
    space" at any grid resolution (Sørbye & Rue 2014, Spatial Statistics 8,
    39; the same as ``scale.model=TRUE`` in R-INLA).

    The constant drifts by about -12% from k=9 to k=21 (2.311 to 2.028).
    Applying it lets a tuned ``SIGMA3_RATE_2D`` carry over between grids. It is
    a correctness tidy-up, **not** the fix for the dispersion bias; an earlier
    claim that it was has been withdrawn.

    Cached: an O(k^6) pseudo-inverse that depends only on ``k``.
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
    """2D NumPyro model with the pure GMRF prior.

    Parameters
    ----------
    matrix : jnp.ndarray (N_stars, n_cells)
        Precomputed 2D design matrix.
    n_cells : int
        Number of grid cells.
    L : jnp.ndarray (n_cells, n_cells)
        Cholesky factor of the ridge-regularised GMRF precision Q. It is
        computed once outside the model and passed in, not recomputed at
        each NUTS step.

    Notes
    -----
    The latent field is non-centred and fully generative: a real
    ``numpyro.sample`` site ``z`` followed by a deterministic transform, as
    recommended in ``PLAN.md`` §1.2/§3.2, and **not** a ``numpyro.factor``
    penalty on an unconstrained base measure. ``numpyro.infer.Predictive``
    only simulates through ``sample`` sites, so a factor-based version would
    silently break simulation-based calibration even though NUTS would still
    be correct.

    The transform is ``x = sigma * L^-T z``, i.e.
    ``jax.scipy.linalg.solve_triangular(L.T, z, lower=False)``, which solves the
    *upper*-triangular system ``L.T @ x = z``. Using ``L`` with ``lower=True``
    would give ``L^-1 z`` instead, which has the wrong covariance.
    ``tests/test_veldist2d.py::test_solve_triangular_direction`` checks this.
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
    """Latent log-density field: a free bivariate-Gaussian core plus a penalised
    deviation.

    With infinite smoothing this prior gives a bivariate Gaussian, not a
    uniform distribution over the grid. That is its purpose. The pure GMRF
    prior in :func:`model_2d` tends to a uniform distribution with dispersion
    ``grid_width/sqrt(12)``, which is 34 km/s on a 119 km/s grid against a true
    17 km/s, so weakly constrained fits are pulled toward something much
    broader and every dispersion comes out too high. Measured on the pure GMRF
    (isotropic sigma = 17, err/sigma = 0.014, scored against the discretised
    truth): a sigma_x bias of +2.34 at N=100 and +0.51 at N=500, growing
    with k.

    A general quadratic in (vx, vy) passed through a softmax is exactly a
    bivariate Gaussian, so ``v0x``, ``v0y``, ``s0x``, ``s0y`` and ``rho0`` map
    one-to-one onto the mean and covariance of the PDF, i.e. the velocity
    ellipsoid. Structure beyond second order is still penalised, as h3/h4 are
    in 1D.

    Parameters
    ----------
    shape : int or (int, int)
        Grid size per axis: a scalar ``K`` (square) or ``(kx, ky)``
        (rectangular). Passed explicitly rather than inferred from
        ``n_cells``, because ``round(sqrt(n_cells))`` silently gives the wrong
        counts for a rectangular grid.
    centers_2d : array-like, shape (kx*ky, 2)
        Cell centres in velocity, from :func:`setup_grid_2d`. Needed because
        the core is quadratic in velocity, not in cell index.
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

    sx_c = jnp.clip(s0x, 1e-3)
    sy_c = jnp.clip(s0y, 1e-3)
    dx = (cx - v0x) / sx_c
    dy = (cy - v0y) / sy_c
    omr2 = 1.0 - rho0**2

    # `intrinsic_pdf` is cell probability MASS, so the core must be a
    # Gaussian's cell mass -- NOT its density sampled at cell centres, which
    # is what softmax(-quad/2) would give. See the mass-vs-density invariant
    # in CLAUDE.md for why the two differ and why it is easy to get wrong.
    #
    # Integrate with the same 2x2 Gauss-Legendre rule the design matrix uses
    # for the error kernel (_design_matrix_gl_quadrature), so core and
    # likelihood agree on what a cell value means by construction rather than
    # by a correction a later edit could drop. A tilted bivariate Gaussian
    # over an axis-aligned cell has no closed form, hence quadrature here
    # where the 1D core gets an exact erf difference.
    #
    # Measured against the exact bivariate cell mass at K=15 (sx=13.11,
    # sy=8.44, rho=0.4): centre sampling errs by -0.074 / -0.131 on
    # sigma_x / sigma_y; GL 2x2 errs by +7e-5 / +5e-6. 3x3 buys nothing.
    #
    # NOTE ON WHAT THIS DOES *NOT* FIX: a separate, larger sigma_y
    # under-dispersion at coarse resolution (-0.250 at K=15, shrinking to
    # -0.050 at K=29) was initially attributed to this term. Measured
    # end-to-end, correcting the core moved it by 0.002 -- i.e. essentially
    # not at all. The measure bug was real and is fixed here, but it is not
    # the cause of that bias, which remains open. Do not cite this fix as the
    # explanation for the resolution-dependent dispersion bias.
    #
    # Equal GL weights and the constant cell Jacobian are dropped: both are
    # constant across cells on a uniform grid and softmax is invariant to an
    # additive constant in the log field.
    # span_x/span_y were already reduced from cx/cy above for the v0 priors.
    hx = span_x / jnp.maximum(kx - 1, 1)
    hy = span_y / jnp.maximum(ky - 1, 1)
    nodes, _ = _gauss_legendre_2x2_nodes()
    offs_x = 0.5 * hx * jnp.asarray(nodes)
    offs_y = 0.5 * hy * jnp.asarray(nodes)

    # Hoisted: omr2 is loop-invariant, but each iteration's numerator differs,
    # so XLA cannot CSE four separate divides into one. Reciprocal once.
    neg_half_over_omr2 = -0.5 / omr2
    sub = []
    for ox in offs_x:
        for oy in offs_y:
            sdx = dx + ox / sx_c
            sdy = dy + oy / sy_c
            sub.append(neg_half_over_omr2 * (sdx**2 - 2.0 * rho0 * sdx * sdy + sdy**2))
    core = logsumexp(jnp.stack(sub, axis=0), axis=0)

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
    """2D NumPyro model with the Gaussian-core prior.

    Parameters
    ----------
    matrix : jnp.ndarray, shape (N_stars, kx*ky)
        Precomputed 2D design matrix.
    n_cells : int
        Number of grid cells, ``kx * ky``. Kept so the signature matches
        :func:`model_2d`, but not used to recover the per-axis counts; see
        ``shape``.
    L : jnp.ndarray, shape (n_cells, n_cells)
        Cholesky factor of the ridge-regularised GMRF precision.
    centers_2d : jnp.ndarray, shape (n_cells, 2)
        Cell centres in velocity.
    shape : int or (int, int)
        Per-axis grid size: a scalar ``K`` (square) or ``(kx, ky)``
        (rectangular). Must be given explicitly, because ``n_cells`` alone
        cannot be factored back into ``(kx, ky)`` for a rectangular grid;
        ``round(sqrt(n_cells))`` silently gets it wrong.
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
    """High-level interface for 2D (proper-motion) Bayesian deconvolution, with
    the same API as :class:`veldist.KinematicSolver`.

    Attributes
    ----------
    matrix : jnp.ndarray or None
        Design matrix, shape (N_stars, n_cells).
    grid : dict
        Grid metadata from :func:`setup_grid_2d`.
    Q, Q_reg, L : np.ndarray or None
        GMRF precision matrix, its ridge-regularised version, and the Cholesky
        factor of the latter. Built once by ``setup_grid`` and passed to the
        model.
    n_stars : int or None
    samples : dict or None
    clipped_samples : dict or None
        Per-cell median mass and floored uncertainties, set by
        ``clip_uncertainties``.
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
        """Define the 2D velocity grid and build and factorise the GMRF precision.

        Parameters
        ----------
        center : (float, float)
        width : (float, float)
        n_bins : int or (int, int)
            Bins per axis: a scalar ``K`` (square, ``K**2`` cells) or
            ``(kx, ky)`` (rectangular, ``kx * ky`` cells).
        diag_weight, edge_weight, ridge_scale : float
            Passed to :func:`build_gmrf_precision`.

        Returns
        -------
        None
            Sets ``self.grid``, ``self.Q``, ``self.Q_reg`` and ``self.L``.
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
        """Load observations and compute the 2D design matrix.

        Parameters
        ----------
        pm1, pm2 : array-like (N,)
            The two observed velocity or proper-motion components of each star.
        cov : array-like (N, 2, 2)
            Per-star measurement covariance,
            ``[[sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y,
            sigma_y**2]]``. Catalogues such as Gaia give the *correlation*
            ``rho`` (e.g. ``pmra_pmdec_corr``), not the covariance; putting
            ``rho`` in the off-diagonal directly makes most matrices non-positive-
            definite.

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
        """Sample the posterior with NUTS.

        Parameters
        ----------
        num_warmup : int
        num_samples : int
            **Default 3000.** Measured on real HST data (2026-08-06,
            ``dense_mass=False``, ``target_accept_prob=0.95``): the minimum ESS
            over the six scalar sites (``v0x``, ``v0y``, ``s0x``, ``s0y``,
            ``rho0``, ``sigma3``) rose from about 260-470 at 1000 samples to about
            830-1290 at 3000, for essentially the same per-bin wall time (about
            1-3 s, dominated by JIT compilation, not sampling). Extra samples are
            close to free.
        gpu : bool or None
            See :meth:`veldist.KinematicSolver.run`.
        seed : int
        prior : {"gaussian_core", "gmrf"}
            ``"gaussian_core"`` (default) gives the latent field a free
            bivariate-Gaussian core, so the velocity ellipsoid is unpenalised and
            infinite smoothing gives a Gaussian. ``"gmrf"`` is the original pure
            GMRF, kept for comparison. It tends to a *uniform* distribution over
            the grid and biases sigma_x high by +0.5 km/s (N=500) to +2.3 km/s
            (N=100) on a sigma = 17 truth.
        target_accept_prob : float
            NUTS target acceptance rate. Default 0.95 rather than NumPyro's 0.8,
            as in 1D (see :meth:`veldist.KinematicSolver.run` for the funnel
            argument). It has **not** been re-validated for 2D with a full SBC
            campaign, but on real HST data (2026-08-06, ``dense_mass=False``,
            ``num_samples=3000``) it gave no divergences in 5 test bins and a
            minimum ESS of about 830-1290, against 3 divergences in the same 5
            bins at 0.8. So 0.95 is supported in 2D on its own evidence.
        dense_mass : bool
            Use a full mass matrix instead of a diagonal one. **Default False**:
            on real HST data (2026-08-06) the dense matrix made things worse.
            With ``dense_mass=True``, NUTS hit ``max_tree_depth`` (1023 steps per
            sample) on almost every sample whatever ``target_accept_prob`` was,
            and with a warm JIT cache it still gave a *lower* minimum ESS (about
            200-290) than the diagonal matrix (about 260-470) at 1000 samples.
            With a cold cache, the realistic case (about 1400 bins with about
            1400 distinct star counts, so nearly every bin compiles afresh), the
            dense kernel took about 100 s per bin to compile, 20 times the
            diagonal one, which is where the "43 hours for a full run" estimate
            came from. This is the opposite of the 1D result, where the dense
            matrix was both faster and better mixed; do not carry the 1D finding
            over without re-measuring. It remains an option, but do not change
            the default without new evidence.
        max_tree_depth : int
            NUTS's limit on trajectory doubling; NumPyro's default is 10 (at most
            1023 leapfrog steps per sample). Exposed here, unlike in 1D, because
            it was needed to diagnose the dense-mass problem above.

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
        """Summarise the posterior per cell and apply uncertainty floors.

        The 2D version of :meth:`veldist.KinematicSolver.clip_uncertainties`;
        see that method for the reasoning behind the floors and why the marginal
        medians do not sum to 1. The only real difference is naming: the result is
        a proper-motion distribution rather than an LOSVD, so the keys are
        ``pdf_median`` / ``pdf_uncertainty``.

        A post-processing step: ``self.samples`` is not changed.

        - ``pdf_median`` is the **marginal median** of each cell's mass. These
          usually **sum to 0.85-0.95**, not 1, which is expected.
        - ``pdf_uncertainty`` is the **half-width** of the 68% credible
          interval, ``(p84 - p16) / 2``, used as a symmetric error bar.

        Both are **dimensionless probability mass per cell**, not divided by the
        cell area.

        A zero uncertainty in any cell reaches Dynamite's NNLS matrices as an
        ``econ`` zero and breaks weight solving in large orbit-library runs. The
        relative floor (``floor_fraction * max_uncertainty``) is the main
        protection; the absolute floor is a numerical backstop.

        Parameters
        ----------
        floor_fraction : float
            Relative floor, as a fraction of the largest per-cell half-width.
            Default 0.01.
        abs_floor : float
            Absolute floor, applied after the relative one. Default 1e-10.

        Returns
        -------
        None
            Sets ``self.clipped_samples`` to a dict with:

            - ``'pdf_median'``: per-cell marginal median, probability mass;
              shape (n_cells,), flat row-major.
            - ``'pdf_uncertainty'``: floored 68% half-width, probability mass;
              shape (n_cells,), flat row-major.
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
    """Small numeric summary of an array for failure logs.

    Returns plain Python floats and ints so the result is always JSON-
    serialisable, whatever the input dtype.
    """
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
    """Append one failure record to the log as a JSON line.

    Safe with several writers (``n_jobs`` > 1): each call opens the file,
    writes once and closes it. On POSIX a single ``write()`` in append mode is
    atomic below the pipe-buffer size (a few KB), and one record is well under
    that.
    """
    if failure_log_path is None:
        return
    with Path(failure_log_path).open("a") as f:
        f.write(json.dumps(failure) + "\n")


def _fit_one_bin_2d(i, pm1, pm2, cov, grid_kwargs, run_kwargs, seed, min_stars, failure_log_path=None, thin=10):
    """Fit one bin. Defined at module level, not as a closure, so that it can be
    pickled for ``ProcessPoolExecutor`` (see ``n_jobs`` in
    :func:`fit_all_bins_2d`).

    If the MCMC fit raises (for example NumPyro's "Cannot find valid initial
    parameters", seen on real HST data on 2026-08-06), the error is caught,
    logged with enough context to investigate later, and the bin is returned
    as ``None``, instead of the exception killing every other bin of a
    multi-hour run. Skipping a bin below ``min_stars`` is normal and is not
    logged as a failure.

    Returns
    -------
    (int, KinematicSolver2D or None)
        The bin index and the fitted solver, or ``None`` if the bin was
        skipped (fewer than ``min_stars`` stars, or the fit raised).
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

    # ponytail: shrink the solver before it crosses the process boundary and
    # lands in fit_all_bins_2d's list. clip_uncertainties() above already ran
    # on the FULL draws, so nothing written to DYNAMITE changes.
    #
    # Why this is here and not in the caller: on the largest production set
    # (HST, 1415 bins, K=23 -> 529 cells, num_samples=3000) the untouched
    # solvers are ~26 MB each -- 36 GB in the returned list, which is the
    # whole machine. Post-hoc cleanup in a notebook is too late; peak RAM is
    # hit inside this function's callers.
    #
    #   samples["x"]  dropped outright: intrinsic_pdf is a deterministic
    #                 softmax of it (model_gaussian_core_2d), so x carries no
    #                 information the pdf does not.
    #   intrinsic_pdf thinned 10x and cast to float32: 12.7 MB -> 0.63 MB.
    #                 NUTS here runs ~31 leapfrog steps/sample, so draws are
    #                 already near-independent; 300 of 3000 against a measured
    #                 ESS of 830-1290 loses ESS, not correctness. float32 gives
    #                 ~1e-7 relative precision on values summing to 1.
    #
    # Set thin=1 to keep every draw (still float32 and still x-free).
    if thin:
        solver.samples = {
            "intrinsic_pdf": np.asarray(solver.samples["intrinsic_pdf"][::thin], dtype=np.float32)
        }
    solver.matrix = None
    solver.Q = None
    solver.Q_reg = None
    solver.L = None

    return i, solver


def fit_all_bins_2d(
    bin_data_list,
    grid_kwargs,
    run_kwargs=None,
    min_stars=10,
    show_progress=True,
    n_jobs=1,
    failure_log_path="fit_all_bins_2d_failures.jsonl",
    thin=10,
):
    """Run the full inference pipeline on a list of spatial (Voronoi) bins.

    The 2D counterpart of :func:`veldist.fit_all_bins`. For each bin it runs
    ``setup_grid``, ``add_data``, ``run`` and ``clip_uncertainties``, and
    returns the fitted :class:`KinematicSolver2D` objects ready for the
    Dynamite writer. Bins with too few stars are returned as ``None`` so the
    writer can mask them.

    There is no ``match_grid`` option as in 1D, and there will not be one.
    Dynamite's 2D ``.npz`` format has a single ``vxrange``/``vyrange`` for the
    whole map, so a per-bin grid would have nowhere to go; every bin uses the
    shared ``grid_kwargs``.

    Bin ``i`` is seeded with ``base_seed + i`` so the chains of different bins
    are independent.

    Parameters
    ----------
    bin_data_list : list of dict
        One dict per Voronoi bin, with keys:

        - ``'pm1'``, ``'pm2'``: observed proper-motion components.
        - ``'cov'``: per-star 2x2 measurement covariance matrices.

        Other keys (spatial metadata, say) are ignored here; pass them to the
        output writer separately.
    grid_kwargs : dict
        Arguments for :meth:`KinematicSolver2D.setup_grid` (``center``,
        ``width``, ``n_bins``, ...), shared by all bins.
    run_kwargs : dict, optional
        Arguments for :meth:`KinematicSolver2D.run` (``num_warmup``,
        ``num_samples``, ``gpu``, ``prior``, ...). A ``seed`` here is the base
        seed; bin ``i`` gets ``seed + i``. Default ``{}``, i.e. ``run``'s
        defaults.
    min_stars : int
        Minimum number of stars needed to fit a bin. Smaller bins are skipped
        with a warning. Default 10.
    show_progress : bool
        Show one ``tqdm`` bar over bins. Default ``True``. This controls only
        the outer bar: ``KinematicSolver2D.run`` has no ``progress_bar``
        argument, and NumPyro's per-chain bars are always suppressed inside
        ``_fit_one_bin_2d``.
    n_jobs : int
        Number of bins to fit at once in a ``ProcessPoolExecutor``. Default 1
        (one after another). This parallelises over bins, **not** over chains
        within a bin. Workers are started with ``spawn``, not ``fork``,
        because JAX is not fork-safe once its backend is running. Each worker
        therefore compiles its own copy of every star-count shape it meets, so
        total compile work can exceed the sequential case, but it happens in
        parallel and wall time still drops. Reserving host devices in the
        parent process (for chain-level parallelism) has no effect on the
        workers, which start with a fresh JAX backend.
    failure_log_path : str or path-like or None
        File to which per-bin failure diagnostics are appended as JSON lines
        when a fit raises: bin index, seed, exception type, message and
        traceback, summary statistics of ``pm1``, ``pm2`` and the covariance,
        and ``grid_kwargs``. The failed bin is skipped rather than stopping a
        run that may take hours; on real HST data (2026-08-06), one bad bin
        raising "Cannot find valid initial parameters" killed an otherwise
        healthy run of about 1400 bins. Default
        ``'fit_all_bins_2d_failures.jsonl'`` in the working directory; ``None``
        disables the file (a warning is issued either way). Several workers can
        share the path, since each record is written with a single
        open-write-close.
    thin : int
        On each returned solver, keep every ``thin``-th draw of
        ``intrinsic_pdf`` as float32 and drop the latent ``x`` draws.
        Default 10. ``clip_uncertainties`` runs on the full draws first, so the
        Dynamite output is unaffected; this only limits the memory used by the
        returned list (see :func:`_fit_one_bin_2d`). ``thin=1`` keeps every
        draw, and ``thin=0`` leaves ``samples`` untouched.

    Returns
    -------
    solvers : list
        One entry per input bin: a fitted :class:`KinematicSolver2D` (with
        ``samples`` and ``clipped_samples`` set), or ``None`` for a skipped
        bin (too few stars, or a failed fit; check ``failure_log_path`` to
        tell which).
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
                i, pm1, pm2, cov, grid_kwargs, run_kwargs, base_seed + i, min_stars, failure_log_path, thin
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
                    thin,
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
