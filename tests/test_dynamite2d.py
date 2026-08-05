"""Tests for the DYNAMITE 2D (proper-motion) .npz output writer.

Covers `write_dynamite_kinematics_2d`, per
docs/superpowers/specs/2026-08-05-dynamite-2d-writer-design.md.

Uses hand-built `KinematicSolver2D` instances with `clipped_samples` set
directly (no MCMC), so these tests are fast.
"""

import numpy as np
import pytest

from veldist.veldist2d import KinematicSolver2D
from veldist.dynamite2d import write_dynamite_kinematics_2d


def fake_solver_2d(k=9, center=(0.0, 0.0), width=(90.0, 90.0),
                    pdf_median=None, pdf_uncertainty=None, n_stars=100):
    """A KinematicSolver2D with clipped_samples set directly (no MCMC)."""
    s = KinematicSolver2D()
    s.setup_grid(center=center, width=width, n_bins=k)
    s.n_stars = n_stars
    n_cells = k * k
    if pdf_median is None:
        pdf_median = np.full(n_cells, 0.9 / n_cells)
    if pdf_uncertainty is None:
        pdf_uncertainty = np.full(n_cells, 0.01)
    s.clipped_samples = {
        "pdf_median": np.asarray(pdf_median, dtype=float),
        "pdf_uncertainty": np.asarray(pdf_uncertainty, dtype=float),
    }
    return s


def _voronoi_metadata_2d(n_bins_total, nx=2, ny=2):
    bins_meta = [{"xbin": float(i) * 1.5, "ybin": float(i) * 2.5} for i in range(n_bins_total)]
    n_pixels = nx * ny
    pixel_bin_ids = np.array([(i % n_bins_total) + 1 for i in range(n_pixels)], dtype=int)
    return {
        "bins": bins_meta,
        "aperture": {
            "x_start": -5.0, "y_start": -5.0, "x_size": 10.0, "y_size": 10.0,
            "angle_deg": 0.0, "nx": nx, "ny": ny,
        },
        "pixel_bin_ids": pixel_bin_ids,
        "psf": {"sigma": [1.0], "weight": [1.0]},
    }


class TestReshapeOrdering:
    def test_arange_pattern_row_major(self):
        """clip_uncertainties_2d's (K,K) reshape must be row-major: m = ix*K+iy."""
        k = 5
        flat = np.arange(k * k, dtype=float)
        reshaped = flat.reshape(k, k, order="C")
        np.testing.assert_array_equal(reshaped[0, :], np.arange(k))
        np.testing.assert_array_equal(reshaped[:, 0], np.arange(0, k * k, k))


class TestTransposition:
    def test_off_diagonal_mass_lands_at_expected_cell(self, tmp_path):
        """Strong transposition guard: place all mass in one off-diagonal
        cell (ix=0, iy=k-1) and verify it lands there after write+reload,
        not at (k-1, 0)."""
        k = 9
        ix_true, iy_true = 0, k - 1
        m_true = ix_true * k + iy_true

        pdf_median = np.full(k * k, 1e-6)
        pdf_median[m_true] = 1.0
        pdf_uncertainty = np.full(k * k, 0.01)

        s0 = fake_solver_2d(k=k, pdf_median=pdf_median, pdf_uncertainty=pdf_uncertainty)
        meta = _voronoi_metadata_2d(n_bins_total=1, nx=1, ny=1)

        write_dynamite_kinematics_2d([s0], tmp_path, meta)

        npz = np.load(tmp_path / "pm_2dhist.npz")
        hist = npz["PM_2dhist"][0]  # (K, K)

        max_idx = np.unravel_index(np.argmax(hist), hist.shape)
        assert max_idx == (ix_true, iy_true)
        assert max_idx != (iy_true, ix_true)


class TestIDCompaction:
    def test_dense_1indexed_with_skipped_bins_interleaved(self, tmp_path):
        s0 = fake_solver_2d()
        s2 = fake_solver_2d()
        s3 = fake_solver_2d()
        solvers = [s0, None, s2, s3]
        meta = _voronoi_metadata_2d(n_bins_total=4, nx=2, ny=2)

        write_dynamite_kinematics_2d(solvers, tmp_path, meta)

        npz = np.load(tmp_path / "pm_2dhist.npz")
        binID = npz["binID_dynamite"]
        np.testing.assert_array_equal(binID, [1, 2, 3])

        assert npz["PM_2dhist"].shape[0] == 3

        # xbin/ybin correspond to original indices [0, 2, 3]
        expected_xbin = [meta["bins"][i]["xbin"] for i in (0, 2, 3)]
        np.testing.assert_allclose(npz["xbin"], expected_xbin)


class TestGridMismatch:
    def test_mismatched_widths_raise(self, tmp_path):
        s0 = fake_solver_2d(width=(90.0, 90.0))
        s1 = fake_solver_2d(width=(80.0, 80.0))
        meta = _voronoi_metadata_2d(n_bins_total=2, nx=1, ny=2)

        with pytest.raises(ValueError):
            write_dynamite_kinematics_2d([s0, s1], tmp_path, meta)


class TestGridCentring:
    def test_off_centre_grid_raises(self, tmp_path):
        s0 = fake_solver_2d(center=(5.0, 0.0))
        meta = _voronoi_metadata_2d(n_bins_total=1, nx=1, ny=1)

        with pytest.raises(ValueError, match="centred"):
            write_dynamite_kinematics_2d([s0], tmp_path, meta)

    def test_off_centre_y_axis_raises(self, tmp_path):
        s0 = fake_solver_2d(center=(0.0, -3.0))
        meta = _voronoi_metadata_2d(n_bins_total=1, nx=1, ny=1)

        with pytest.raises(ValueError, match="centred"):
            write_dynamite_kinematics_2d([s0], tmp_path, meta)


class TestOddBinsConstraint:
    def test_even_k_raises(self, tmp_path):
        s0 = fake_solver_2d(k=8)
        meta = _voronoi_metadata_2d(n_bins_total=1, nx=1, ny=1)

        with pytest.raises(ValueError):
            write_dynamite_kinematics_2d([s0], tmp_path, meta)

    def test_even_k_raises_before_touching_filesystem(self, tmp_path):
        s0 = fake_solver_2d(k=8)
        meta = _voronoi_metadata_2d(n_bins_total=1, nx=1, ny=1)

        with pytest.raises(ValueError):
            write_dynamite_kinematics_2d([s0], tmp_path, meta)

        assert not (tmp_path / "pm_2dhist.npz").exists()


class TestAllNone:
    def test_all_none_raises(self, tmp_path):
        meta = _voronoi_metadata_2d(n_bins_total=2, nx=1, ny=2)
        with pytest.raises(ValueError):
            write_dynamite_kinematics_2d([None, None], tmp_path, meta)


class TestUncertaintyFloor:
    def test_no_zero_or_negative_sigma(self, tmp_path):
        k = 9
        pdf_uncertainty = np.zeros(k * k)  # pathologically tight posterior
        s0 = fake_solver_2d(k=k, pdf_uncertainty=pdf_uncertainty)
        meta = _voronoi_metadata_2d(n_bins_total=1, nx=1, ny=1)

        with pytest.raises(ValueError):
            # Zero uncertainty everywhere means max(raw)==0, so no floor
            # mechanism inside clip_uncertainties_2d can rescue it if fed
            # directly like this -- the writer's own floor guard should
            # still reject silently-broken input.
            write_dynamite_kinematics_2d([s0], tmp_path, meta)


class TestRoundTrip:
    def test_npz_keys_shapes_dtypes(self, tmp_path):
        k = 9
        s0 = fake_solver_2d(k=k, n_stars=42)
        s1 = fake_solver_2d(k=k, n_stars=17)
        meta = _voronoi_metadata_2d(n_bins_total=2, nx=1, ny=2)

        write_dynamite_kinematics_2d([s0, s1], tmp_path, meta)

        npz = np.load(tmp_path / "pm_2dhist.npz")
        for key in ("PM_2dhist", "PM_2dhist_sigma", "binID_dynamite",
                    "nstarbin", "vxrange", "vyrange", "xbin", "ybin"):
            assert key in npz

        assert npz["PM_2dhist"].shape == (2, k, k)
        assert npz["PM_2dhist_sigma"].shape == (2, k, k)
        assert npz["binID_dynamite"].shape == (2,)
        assert npz["nstarbin"].shape == (2,)
        assert npz["xbin"].shape == (2,)
        assert npz["ybin"].shape == (2,)

        # scalars, not arrays with a shape
        assert np.asarray(npz["vxrange"]).ndim == 0
        assert np.asarray(npz["vyrange"]).ndim == 0

        np.testing.assert_array_equal(npz["nstarbin"], [42, 17])

        assert (tmp_path / "aperture.dat").exists()
        assert (tmp_path / "bins.dat").exists()

    def test_bins_dat_masks_skipped_bins(self, tmp_path):
        s0 = fake_solver_2d()
        s2 = fake_solver_2d()
        s3 = fake_solver_2d()
        solvers = [s0, None, s2, s3]
        meta = _voronoi_metadata_2d(n_bins_total=4, nx=2, ny=2)

        write_dynamite_kinematics_2d(solvers, tmp_path, meta)

        bins_text = (tmp_path / "bins.dat").read_text()
        lines = bins_text.strip().split("\n")
        header, count_line, *data_lines = lines
        remapped = [int(v) for line in data_lines for v in line.split()]
        # pixel_bin_ids cycles 1,2,3,4 over 4 pixels; orig bin id 2 (0-indexed
        # bin 1, skipped) -> 0; others get their new sequential id.
        assert remapped == [1, 0, 2, 3]
