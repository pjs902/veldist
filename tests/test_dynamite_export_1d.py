"""Optional per-star CSR + Laplace expansion point in a sidecar npz (1D, star_data=)."""

import numpy as np

from veldist.fitstat import sharpen_matrix
from veldist.veldist import precompute_design_matrix, write_dynamite_kinematics

from tests.test_pipeline import _voronoi_metadata, fake_solver

K = 9


def _solver(seed, n_stars):
    return fake_solver(n_bins=K, width=90.0, n_stars=n_stars, seed=seed)


def _stars(seed, n):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 15.0, n)
    v[0] = 200.0  # far off the +-45 grid: dropped, as the fit drops it
    return {"vel": v, "err": rng.uniform(1.0, 5.0, n)}


def _ref(st, grid, sharpen=False):
    M = np.asarray(precompute_design_matrix(st["vel"], st["err"], grid["centers"], bin_width=grid["width"]),
                   dtype=float)
    M = M[M.sum(axis=1) >= 1e-3]
    return M @ sharpen_matrix(K, ndim=1) if sharpen else M


def _dense(z):
    ptr, cell, mass = z["LOS_star_indptr"], z["LOS_star_cell"], z["LOS_star_mass"]
    d = np.zeros((len(ptr) - 1, K))
    for r in range(len(ptr) - 1):
        d[r, cell[ptr[r]:ptr[r + 1]]] = mass[ptr[r]:ptr[r + 1]]
    return d


def test_export_roundtrip(tmp_path):
    solvers = [_solver(0, 30), None, _solver(2, 25)]
    stars = [_stars(0, 30), _stars(1, 12), _stars(2, 25)]
    write_dynamite_kinematics(solvers, tmp_path, _voronoi_metadata(3), star_data=stars)
    z = np.load(tmp_path / "bayes_losvd_kins_stars.npz")
    assert z["LOS_star_indptr"].dtype == np.int64 and z["LOS_star_cell"].dtype == np.int32
    assert z["LOS_star_mass"].dtype == np.float32 and z["LOS_star_aperture"].dtype == np.int32
    ap = z["LOS_star_aperture"]
    np.testing.assert_array_equal(np.bincount(ap), [29, 24])  # skipped bin absent
    dense = _dense(z)
    for a, orig in ((0, 0), (1, 2)):
        ref = _ref(stars[orig], solvers[orig].grid)
        np.testing.assert_allclose(dense[ap == a], ref, atol=1e-6 * ref.max())
    h0 = z["LOS_lap_h0"]
    assert h0.shape == (2, K) and h0.dtype == np.float64
    np.testing.assert_allclose(h0.sum(axis=1), 1.0, rtol=1e-12)
    assert np.all(h0 > 0)
    assert not bool(z["LOS_star_sharpened"])


def test_export_absent_by_default_and_ecsv_unchanged(tmp_path):
    solvers = [_solver(0, 30), None, _solver(2, 25)]
    a, b = tmp_path / "a", tmp_path / "b"
    write_dynamite_kinematics(solvers, a, _voronoi_metadata(3))
    write_dynamite_kinematics(solvers, b, _voronoi_metadata(3), star_data=[_stars(0, 30), None, _stars(2, 25)])
    assert not list(a.glob("*.npz"))
    for f in a.iterdir():
        assert f.read_bytes() == (b / f.name).read_bytes()


def test_export_sharpen(tmp_path):
    solvers, stars = [_solver(0, 30)], [_stars(0, 30)]
    write_dynamite_kinematics(solvers, tmp_path, _voronoi_metadata(1), star_data=stars, sharpen=True)
    z = np.load(tmp_path / "bayes_losvd_kins_stars.npz")
    ref = _ref(stars[0], solvers[0].grid, sharpen=True)
    assert (ref < 0).any()  # the negative side lobes must survive the sparse cut
    np.testing.assert_allclose(_dense(z), ref, atol=1e-6 * np.abs(ref).max())
    assert bool(z["LOS_star_sharpened"])
