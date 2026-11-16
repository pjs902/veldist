"""Optional per-star CSR + Laplace expansion point in the PM npz (star_data=)."""

import numpy as np

from veldist.dynamite2d import write_dynamite_kinematics_2d
from veldist.fitstat import sharpen_matrix
from veldist.veldist2d import precompute_design_matrix_2d

from tests.test_dynamite2d import _voronoi_metadata_2d, fake_solver_2d

K = 9


def _solver_with_samples(seed, n_stars):
    s = fake_solver_2d(k=K, n_stars=n_stars)
    rng = np.random.default_rng(seed)
    s.samples = {"intrinsic_pdf": rng.dirichlet(np.full(K * K, 5.0), size=50).astype(np.float32)}
    return s


def _stars(seed, n):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 15.0, (n, 2))
    v[0] = (80.0, -70.0)  # far off the +-45 grid: dropped, as the fit drops it
    cov = np.tile(np.diag([2.0**2, 3.0**2]), (n, 1, 1))
    cov[1::2, 0, 1] = cov[1::2, 1, 0] = 2.0  # half correlated
    return {"pm1": v[:, 0], "pm2": v[:, 1], "cov": cov}


def test_export_roundtrip(tmp_path):
    solvers = [_solver_with_samples(0, 30), None, _solver_with_samples(2, 25)]
    stars = [_stars(0, 30), _stars(1, 12), _stars(2, 25)]
    write_dynamite_kinematics_2d(solvers, tmp_path, _voronoi_metadata_2d(3), star_data=stars)
    z = np.load(tmp_path / "pm_2dhist.npz")
    ap = z["PM_star_aperture"]
    # the skipped bin's stars are absent; npz aperture order is 0, 1
    np.testing.assert_array_equal(np.bincount(ap), [29, 24])
    ptr, cell, mass = z["PM_star_indptr"], z["PM_star_cell"], z["PM_star_mass"]
    for a, orig in ((0, 0), (1, 2)):
        rows = np.where(ap == a)[0]
        dense = np.zeros((len(rows), K * K))
        for j, r in enumerate(rows):
            dense[j, cell[ptr[r]:ptr[r + 1]]] = mass[ptr[r]:ptr[r + 1]]
        st = stars[orig]
        ref = precompute_design_matrix_2d(st["pm1"], st["pm2"], st["cov"], solvers[orig].grid)
        ref = ref[ref.sum(axis=1) >= 1e-3]
        np.testing.assert_allclose(dense, ref, atol=1e-6 * ref.max(axis=1, keepdims=True).max())
    h0 = z["PM_lap_h0"]
    assert h0.shape == (2, K, K) and h0.dtype == np.float64
    np.testing.assert_allclose(h0.sum(axis=(1, 2)), 1.0, rtol=1e-12)
    assert np.all(h0 > 0)


def test_export_absent_by_default(tmp_path):
    write_dynamite_kinematics_2d([_solver_with_samples(0, 30)], tmp_path, _voronoi_metadata_2d(1))
    z = np.load(tmp_path / "pm_2dhist.npz")
    assert not any(k.startswith(("PM_star_", "PM_lap_")) for k in z.files)


def test_export_sharpen(tmp_path):
    solvers, stars = [_solver_with_samples(0, 30)], [_stars(0, 30)]
    write_dynamite_kinematics_2d(solvers, tmp_path, _voronoi_metadata_2d(1), star_data=stars, sharpen=True)
    z = np.load(tmp_path / "pm_2dhist.npz")
    ptr, cell, mass = z["PM_star_indptr"], z["PM_star_cell"], z["PM_star_mass"]
    dense = np.zeros((len(ptr) - 1, K * K))
    for r in range(len(ptr) - 1):
        dense[r, cell[ptr[r]:ptr[r + 1]]] = mass[ptr[r]:ptr[r + 1]]
    st = stars[0]
    ref = precompute_design_matrix_2d(st["pm1"], st["pm2"], st["cov"], solvers[0].grid).astype(float)
    ref = ref[ref.sum(axis=1) >= 1e-3] @ sharpen_matrix(K)
    assert (ref < 0).any()  # the negative side lobes must survive the sparse cut
    np.testing.assert_allclose(dense, ref, atol=1e-6 * np.abs(ref).max())
    assert bool(z["PM_star_sharpened"])
