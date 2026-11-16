"""veldist.fitstat: reference statistics for scoring model PM histograms."""

import numpy as np
import pytest

from veldist.fitstat import (
    laplace_rows,
    laplace_terms,
    profile_shift,
    shift_hist,
    sharpen_matrix,
    shift_matrix,
    stat_exact,
    stat_laplace,
)
from veldist.veldist2d import precompute_design_matrix_2d, setup_grid_2d

from scipy.special import ndtr

HST = dict(K=23, half=82.80638794003434)
GAIA = dict(K=17, half=56.45657373000141)


def _gauss_mass(edges, s, m=(0.0, 0.0)):
    """Open-border cell masses of an isotropic Gaussian (sums to 1)."""
    e = np.array(edges, float)
    e[0], e[-1] = -np.inf, np.inf
    cx = np.diff(ndtr((e - m[0]) / s))
    cy = np.diff(ndtr((e - m[1]) / s))
    return np.outer(cx, cy).ravel()


def _aperture(n, err, regime, seed=0, shift=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    g = setup_grid_2d(center=(0.0, 0.0), width=(2 * regime["half"],) * 2, n_bins=regime["K"])
    v = rng.normal(0, 17.0, (n, 2)) + rng.normal(0, err, (n, 2)) + np.asarray(shift)
    cov = np.tile(np.eye(2) * err**2, (n, 1, 1))
    M = precompute_design_matrix_2d(v[:, 0], v[:, 1], cov, g, open_edges=True).astype(float)
    return g, M


@pytest.mark.parametrize("n", [176, 430, 700])
@pytest.mark.parametrize("err,regime", [(1.5, HST), (20.0, GAIA)])
def test_laplace_matches_exact_near_h0(n, err, regime):
    g, M = _aperture(n, err, regime)
    e = g["edges_x"]
    k2 = regime["K"] ** 2
    h0 = 0.99 * _gauss_mass(e, 17.0) + 0.01 / k2
    F, gr = laplace_terms(M, h0, n_stars=n, mass=1.0)
    Lt, target = laplace_rows(F, gr, h0, n_stars=n)
    base_lap = stat_laplace(h0, Lt, target)
    base_ex = stat_exact(h0, M, n, 1.0)
    # A quadratic expansion: its error grows as the cube of the distance from h0
    # (about 10-25% of Delta at +-10% in sigma). This pins correctness at +-2%.
    for s in (17.0 * 0.98, 17.0 * 1.02):
        for scale in (0.98, 1.0, 1.02):
            h = scale * (0.99 * _gauss_mass(e, s) + 0.01 / k2)
            d_ex = stat_exact(h, M, n, 1.0) - base_ex
            d_lap = stat_laplace(h, Lt, target) - base_lap
            assert abs(d_lap - d_ex) < 0.05 * max(1.0, abs(d_ex)), (s, scale, d_ex, d_lap)


def test_laplace_rows_equal_quadratic_plus_linear():
    g, M = _aperture(430, 1.5, HST)
    h0 = 0.99 * _gauss_mass(g["edges_x"], 17.0) + 0.01 / 529
    F, gr = laplace_terms(M, h0, n_stars=430, mass=1.0)
    delta = 1e-3 * 430
    Lt, target = laplace_rows(F, gr, h0, delta=delta)
    rng = np.random.default_rng(1)
    c0 = stat_laplace(h0, Lt, target)
    for _ in range(3):
        dh = 1e-3 * rng.normal(size=529)
        lhs = stat_laplace(h0 + dh, Lt, target) - c0
        rhs = dh @ F @ dh + 2 * gr @ dh + delta * dh @ dh
        assert lhs == pytest.approx(rhs, rel=1e-7, abs=1e-8)


def test_laplace_terms_formula():
    g, M = _aperture(200, 1.5, HST)
    h0 = 0.99 * _gauss_mass(g["edges_x"], 17.0) + 0.01 / 529
    F, gr = laplace_terms(M, h0, n_stars=200, mass=2.0)
    p0 = M @ h0
    np.testing.assert_allclose(F, M.T @ ((1 / p0**2)[:, None] * M), rtol=1e-12)
    np.testing.assert_allclose(gr, 100.0 - M.T @ (1 / p0), rtol=1e-12)


def test_shift_matrix_matches_dynamite():
    # DYNAMITE kinematics.pm_shift_matrix at fork commit c1b0acb
    S = shift_matrix(np.linspace(-82.8, 82.8, 24), 3.1)
    ref = np.array([
        [-0.13496453185137933, 0.6883554389988454, 0.5087317932288252,
         -0.11492791252065589, 0.03079483160395251],
        [0.03616363731652525, -0.1349667504344979, 0.6883524071083844,
         0.5087323936259143, -0.11492787615602265],
        [-0.00969001741472312, 0.03616423178542595, -0.13496615597335287,
         0.6883532195210932, 0.5087322326725146]])
    np.testing.assert_allclose(S[10:13, 9:14], ref, rtol=1e-12)
    np.testing.assert_allclose(S[0, :3], [1.1325948432781725, 0.5991364276846846,
                                          -0.2874764405822928], rtol=1e-12)
    np.testing.assert_allclose(S[-1, -3:], [0.11916249556318158, -0.29068931309306506,
                                            0.7661532603117801], rtol=1e-12)


def test_shift_hist_conserves_mass():
    e = np.linspace(-82.8, 82.8, 24)
    h = _gauss_mass(e, 17.0)
    np.testing.assert_allclose(shift_hist(h, e, e, (2.0, -1.0)).sum(), h.sum(), rtol=1e-10)


def test_profile_shift_recovers_offset():
    g, M = _aperture(2000, 1.5, HST, shift=(2.0, -1.0))
    e = g["edges_x"]
    h = 0.99 * _gauss_mass(e, 17.0) + 0.01 / 529
    s, _ = profile_shift(lambda hh: stat_exact(hh, M, 2000, 1.0), h, e, e, cap=3.6)
    np.testing.assert_allclose(s, [2.0, -1.0], atol=0.6)  # sampling noise of 2000 stars at sigma 17: ~0.4


def test_finite_with_empty_cells():
    g, M = _aperture(176, 1.5, HST)
    h0 = 0.99 * _gauss_mass(g["edges_x"], 17.0) + 0.01 / 529
    F, gr = laplace_terms(M, h0, n_stars=176, mass=1.0)
    Lt, target = laplace_rows(F, gr, h0, n_stars=176)
    assert np.all(np.isfinite(Lt)) and np.all(np.isfinite(target))


def test_stat_laplace_offset_free_equals_difference():
    g, M = _aperture(430, 1.5, HST, seed=3)
    h0 = 0.99 * _gauss_mass(g["edges_x"], 17.0) + 0.01 / HST["K"] ** 2
    F, gr = laplace_terms(M, h0, len(M), 1.0)
    Lt, tg = laplace_rows(F, gr, h0, n_stars=len(M))
    h = _gauss_mass(g["edges_x"], 18.0, (0.5, 0.0))
    full = stat_laplace(h, Lt, tg) - stat_laplace(h0, Lt, tg)
    free = stat_laplace(h, Lt, tg, h0=h0)
    assert stat_laplace(h0, Lt, tg) > 1e5 * abs(free)  # the constant dwarfs the signal
    np.testing.assert_allclose(free, full, rtol=1e-6)
    assert stat_laplace(h0, Lt, tg, h0=h0) == 0.0


def test_sharpen_matrix_removes_flat_cell_variance():
    K, w = 21, 2.0
    S = sharpen_matrix(K)
    assert S.shape == (K * K, K * K)
    c = (np.arange(K) - K // 2) * w
    h = np.zeros((K, K))
    h[K // 2 - 3:K // 2 + 4, K // 2] = [1, 3, 6, 8, 6, 3, 1]  # interior, along axis 0
    hs = (S @ h.ravel()).reshape(K, K)
    np.testing.assert_allclose(hs.sum(), h.sum(), rtol=1e-12)
    var = lambda a: (a.sum(1) @ c**2) / a.sum() - ((a.sum(1) @ c) / a.sum()) ** 2
    np.testing.assert_allclose(var(hs), var(h) - w**2 / 6, rtol=1e-12)
    assert (hs.sum(0) @ c) == pytest.approx(0.0, abs=1e-12)  # no cross-axis spread of the mean


def test_sharpened_exact_recovers_sigma():
    # flat-in-cell M makes model cell masses look h^2/6 wider: exact fits sigma low.
    from scipy.optimize import minimize_scalar
    g, M = _aperture(20000, 1.5, HST, seed=5)
    e = np.asarray(g["edges_x"], float)
    MS = M @ sharpen_matrix(HST["K"])
    fit = lambda MM: minimize_scalar(lambda s: stat_exact(_gauss_mass(e, s), MM, len(MM), 1.0),
                                     bounds=(14, 20), method="bounded").x
    assert fit(M) < 17.0 - 0.15
    assert fit(MS) == pytest.approx(17.0, abs=0.08)
