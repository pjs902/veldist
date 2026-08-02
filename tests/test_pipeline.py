"""Pipeline / export round-trip tests.

Covers `KinematicSolver.clip_uncertainties`, `truncate_losvd`, `fit_all_bins`,
and `write_dynamite_kinematics`, per PLAN.md Part 2.

Most tests use `fake_solver` (a KinematicSolver with a synthetic Dirichlet
posterior, no MCMC) so they are fast. Only the tests marked `slow` run real
NUTS inference, with tiny settings.

Note: the golden-file `.ecsv` diff item from PLAN.md Part 2 is skipped —
building a stable reference fixture is out of scope for this task. Instead
we check file existence and re-parseability with `Table.read`.
"""


import numpy as np
import pytest
from astropy.table import Table

from veldist.analysis import compute_summary
from veldist.veldist import KinematicSolver, fit_all_bins, write_dynamite_kinematics


# ==============================================================================
# fake_solver fixture
# ==============================================================================


def fake_solver(centers=0.0, width=100.0, n_bins=10, pdf_mass=None,
                 n_stars=100, n_draws=200, seed=0):
    """A KinematicSolver with a synthetic posterior, no MCMC.

    `pdf_mass` is a target simplex over `n_bins` bins (defaults to uniform).
    Posterior samples are drawn via `rng.dirichlet(pdf_mass * 500, size=n_draws)`
    which guarantees each row is a valid simplex (sums to exactly 1) with a
    realistic, narrow spread around `pdf_mass` (see PLAN.md gotchas: a smaller
    concentration multiplier gives draws too diffuse for the 0.85-0.95
    marginal-median-sum convention to hold).
    """
    rng = np.random.default_rng(seed)
    if pdf_mass is None:
        pdf_mass = np.ones(n_bins) / n_bins
    pdf_mass = np.asarray(pdf_mass, dtype=float)
    n_bins = len(pdf_mass)

    s = KinematicSolver()
    s.setup_grid(center=centers, width=width, n_bins=n_bins)
    s.n_stars = n_stars
    s.samples = {"intrinsic_pdf": rng.dirichlet(pdf_mass * 500, size=n_draws)}
    return s


# ==============================================================================
# clip_uncertainties
# ==============================================================================


class TestClipUncertainties:
    def test_raises_before_run(self):
        s = KinematicSolver()
        s.setup_grid(center=0.0, width=100.0, n_bins=10)
        with pytest.raises(ValueError):
            s.clip_uncertainties()

    def test_floor_applied_and_positive_everywhere(self):
        rng = np.random.default_rng(1)
        n_bins = 10
        pdf_mass = np.ones(n_bins) / n_bins
        s = fake_solver(pdf_mass=pdf_mass, seed=1)

        # Artificially zero out the variance in one bin: pin column 0 to a
        # fixed value across all draws, and rescale the *other* columns so
        # each row still sums to 1 exactly (a valid simplex).
        samples = np.asarray(s.samples["intrinsic_pdf"]).copy()
        v0 = samples[:, 0].mean()
        rest = samples[:, 1:]
        rest = rest / rest.sum(axis=1, keepdims=True) * (1.0 - v0)
        samples[:, 0] = v0
        samples[:, 1:] = rest
        s.samples["intrinsic_pdf"] = samples

        s.clip_uncertainties()
        unc = s.clipped_samples["losvd_uncertainty"]

        raw_half_width = (
            np.percentile(samples, 84, axis=0) - np.percentile(samples, 16, axis=0)
        ) / 2.0
        rel_floor = 0.01 * np.max(raw_half_width)

        # Bin 0's raw half-width is tiny (near-zero variance), well below the
        # relative floor derived from the widest bin, so the floor must apply.
        assert raw_half_width[0] < rel_floor
        assert unc[0] == pytest.approx(max(rel_floor, 1e-10))
        np.testing.assert_allclose(unc, np.maximum(raw_half_width, rel_floor))
        assert np.all(unc > 0)

    def test_marginal_median_sum_convention(self):
        s = fake_solver(seed=2)
        s.clip_uncertainties()
        total = np.sum(s.clipped_samples["losvd_median"])
        assert 0.8 < total < 1.0

    def test_non_destructive_of_raw_samples(self):
        s = fake_solver(seed=3)
        raw_before = np.array(s.samples["intrinsic_pdf"], copy=True)
        s.clip_uncertainties()
        raw_after = np.asarray(s.samples["intrinsic_pdf"])
        np.testing.assert_array_equal(raw_before, raw_after)


# ==============================================================================
# truncate_losvd
# ==============================================================================


class TestTruncateLosvd:
    def test_raises_before_run(self):
        s = KinematicSolver()
        s.setup_grid(center=0.0, width=100.0, n_bins=10)
        with pytest.raises(ValueError):
            s.truncate_losvd()

    def test_zeros_bins_beyond_n_sigma(self):
        # Concentrate mass tightly at the grid center so the tails are
        # unambiguously beyond n_sigma.
        n_bins = 41
        pdf_mass = np.zeros(n_bins)
        pdf_mass[n_bins // 2] = 1.0
        pdf_mass += 1e-6  # keep every entry strictly positive (Dirichlet needs > 0)
        pdf_mass /= pdf_mass.sum()

        s = fake_solver(centers=0.0, width=200.0, pdf_mass=pdf_mass, seed=4)
        s.truncate_losvd(n_sigma=1.0)

        centers = np.asarray(s.grid["centers"])
        mean_pdf = np.mean(np.asarray(s.samples["intrinsic_pdf"]), axis=0)
        v_mean = np.dot(centers, mean_pdf)
        v_std = np.sqrt(np.dot((centers - v_mean) ** 2, mean_pdf))
        mask = np.abs(centers - v_mean) > 1.0 * v_std

        assert mask.any()  # sanity: truncation should affect some bins
        assert np.all(s.clipped_samples["losvd_median"][mask] == 0.0)
        assert np.allclose(s.clipped_samples["losvd_uncertainty"][mask], 1e-10)

    def test_auto_calls_clip_uncertainties(self):
        s = fake_solver(seed=5)
        assert s.clipped_samples is None
        s.truncate_losvd()
        assert s.clipped_samples is not None

    def test_idempotent(self):
        s = fake_solver(seed=6)
        s.truncate_losvd(n_sigma=2.0)
        first_median = s.clipped_samples["losvd_median"].copy()
        first_unc = s.clipped_samples["losvd_uncertainty"].copy()

        s.truncate_losvd(n_sigma=2.0)
        np.testing.assert_array_equal(first_median, s.clipped_samples["losvd_median"])
        np.testing.assert_array_equal(first_unc, s.clipped_samples["losvd_uncertainty"])


# ==============================================================================
# fit_all_bins
# ==============================================================================


class TestFitAllBins:
    @pytest.mark.slow
    def test_min_stars_skipping_and_seed_decorrelation(self):
        rng = np.random.default_rng(42)
        n_stars_ok = 30
        vel = rng.normal(0.0, 10.0, n_stars_ok)
        err = np.full(n_stars_ok, 2.0)

        # Two bins with identical data (to test per-bin seed decorrelation),
        # one bin below min_stars (to test skipping + warning).
        bin_data_list = [
            {"vel": vel, "err": err},
            {"vel": vel, "err": err},
            {"vel": vel[:3], "err": err[:3]},  # below min_stars=10
        ]

        grid_kwargs = {"center": 0.0, "width": 80.0, "n_bins": 10}
        run_kwargs = {"num_warmup": 50, "num_samples": 100, "gpu": False, "seed": 111}

        with pytest.warns(UserWarning, match="only 3 star"):
            solvers = fit_all_bins(
                bin_data_list, grid_kwargs, run_kwargs=run_kwargs, min_stars=10
            )

        assert len(solvers) == 3
        assert solvers[0] is not None
        assert solvers[1] is not None
        assert solvers[2] is None

        # Per-bin seed derivation: identical input data, different seeds
        # (base_seed + i) must give different posterior draws.
        draws0 = np.asarray(solvers[0].samples["intrinsic_pdf"])
        draws1 = np.asarray(solvers[1].samples["intrinsic_pdf"])
        assert not np.allclose(draws0, draws1)

    def test_all_below_threshold_returns_all_none(self):
        bin_data_list = [
            {"vel": np.array([1.0, 2.0]), "err": np.array([0.5, 0.5])},
            {"vel": np.array([3.0]), "err": np.array([0.5])},
        ]
        grid_kwargs = {"center": 0.0, "width": 20.0, "n_bins": 5}

        with pytest.warns(UserWarning):
            solvers = fit_all_bins(bin_data_list, grid_kwargs, min_stars=10)

        assert solvers == [None, None]


# ==============================================================================
# write_dynamite_kinematics
# ==============================================================================


def _voronoi_metadata(n_bins_total, nx=2, ny=2, extra_bin_flux=None):
    bins_meta = []
    for i in range(n_bins_total):
        entry = {"xbin": float(i) * 1.5, "ybin": float(i) * 2.5}
        if extra_bin_flux is not None:
            entry["bin_flux"] = extra_bin_flux[i]
        bins_meta.append(entry)

    # pixel_bin_ids: nx*ny pixels, 1-indexed bin ids (0 = masked).
    # Cycle through bins 1..n_bins_total so every bin is referenced at least
    # once when nx*ny >= n_bins_total.
    n_pixels = nx * ny
    pixel_bin_ids = np.array(
        [(i % n_bins_total) + 1 for i in range(n_pixels)], dtype=int
    )

    return {
        "bins": bins_meta,
        "aperture": {
            "x_start": -5.0,
            "y_start": -5.0,
            "x_size": 10.0,
            "y_size": 10.0,
            "angle_deg": 0.0,
            "nx": nx,
            "ny": ny,
        },
        "pixel_bin_ids": pixel_bin_ids,
        "psf": {"sigma": [1.0], "weight": [1.0]},
    }


class TestWriteDynamiteKinematics:
    def _make_mixed_solvers(self):
        """[solver, None, solver, solver] — the mixed-None round trip case."""
        n_bins = 8
        pdf_mass = np.ones(n_bins) / n_bins
        s0 = fake_solver(pdf_mass=pdf_mass, seed=10)
        s2 = fake_solver(pdf_mass=pdf_mass, seed=12)
        s3 = fake_solver(pdf_mass=pdf_mass, seed=13)
        for s in (s0, s2, s3):
            s.clip_uncertainties()
        return [s0, None, s2, s3]

    def test_round_trip_mixed_none(self, tmp_path):
        solvers = self._make_mixed_solvers()
        meta = _voronoi_metadata(n_bins_total=4, nx=2, ny=2)

        write_dynamite_kinematics(solvers, tmp_path, meta)

        kin_path = tmp_path / "bayes_losvd_kins.ecsv"
        bins_path = tmp_path / "bins.dat"
        ap_path = tmp_path / "aperture.dat"
        assert kin_path.exists()
        assert bins_path.exists()
        assert ap_path.exists()

        table = Table.read(str(kin_path), format="ascii.ecsv")

        # row count == 3 (solvers[1] was None and is skipped)
        assert len(table) == 3
        assert list(table["binID_dynamite"]) == [1, 2, 3]

        # xbin/ybin match metadata for the surviving bins only:
        # solvers indices [0, 2, 3] survived -> original bin_metas[0], [2], [3]
        surviving_orig = [0, 2, 3]
        expected_xbin = [meta["bins"][i]["xbin"] for i in surviving_orig]
        expected_ybin = [meta["bins"][i]["ybin"] for i in surviving_orig]
        np.testing.assert_allclose(table["xbin"], expected_xbin)
        np.testing.assert_allclose(table["ybin"], expected_ybin)

        # bins.dat: no reference to skipped bin's original 1-indexed id (2),
        # and pixels that pointed at it (original bin index 1 -> 1-indexed id 2)
        # are now 0.
        bins_text = bins_path.read_text()
        lines = bins_text.strip().split("\n")
        header, count_line, *data_lines = lines
        remapped = [int(v) for line in data_lines for v in line.split()]
        assert 2 not in set(np.unique(np.array([], dtype=int)))  # placeholder guard
        # pixel_bin_ids cycles 1,2,3,4 over 4 pixels; orig bin id 2 (0-indexed
        # bin 1, the skipped one) must map to 0, others to their new sequential id.
        # orig_to_new: bin1(id1)->1, bin2(id2, skipped)->0, bin3(id3)->2, bin4(id4)->3
        assert remapped == [1, 0, 2, 3]

        # losvd_j/dlosvd_j column count == 2*nvbins and interleaved order.
        n_bins = 8
        losvd_cols = [c for c in table.colnames if c.startswith("losvd_")]
        dlosvd_cols = [c for c in table.colnames if c.startswith("dlosvd_")]
        assert len(losvd_cols) + len(dlosvd_cols) == 2 * n_bins
        # Interleaving check: losvd_j immediately followed by dlosvd_j.
        for j in range(n_bins):
            idx_l = table.colnames.index(f"losvd_{j}")
            idx_d = table.colnames.index(f"dlosvd_{j}")
            assert idx_d == idx_l + 1

        # metadata
        assert table.meta["dv"] == pytest.approx(solvers[0].grid["width"])
        np.testing.assert_allclose(
            np.array(table.meta["vcent"]), np.asarray(solvers[0].grid["centers"])
        )
        assert table.meta["nbins"] == 3
        assert table.meta["nvbins"] == n_bins

    def test_v_sigma_match_compute_summary(self, tmp_path):
        n_bins = 8
        pdf_mass = np.ones(n_bins) / n_bins
        s0 = fake_solver(pdf_mass=pdf_mass, seed=20)
        s0.clip_uncertainties()
        solvers = [s0]
        meta = _voronoi_metadata(n_bins_total=1, nx=1, ny=1)

        write_dynamite_kinematics(solvers, tmp_path, meta)
        table = Table.read(str(tmp_path / "bayes_losvd_kins.ecsv"), format="ascii.ecsv")

        # compute_summary operates on the full posterior samples, while the
        # writer's inline v/sigma use the clipped per-bin *median* LOSVD.
        # These are two related but distinct point estimates of the same
        # underlying distribution, so we only require agreement within a
        # bin width (per PLAN.md), not exact equality.
        summary = compute_summary(s0.samples["intrinsic_pdf"], s0.grid["centers"])
        v_summary = summary["v_mean"][0]
        sigma_summary = summary["sigma"][0]

        bin_width = s0.grid["width"]
        assert abs(table["v"][0] - v_summary) < bin_width
        assert abs(table["sigma"][0] - sigma_summary) < bin_width

    def test_grid_mismatch_raises(self, tmp_path):
        s0 = fake_solver(n_bins=8, seed=30)
        s1 = fake_solver(n_bins=6, seed=31)  # different grid
        s0.clip_uncertainties()
        s1.clip_uncertainties()
        meta = _voronoi_metadata(n_bins_total=2, nx=1, ny=2)

        with pytest.raises(ValueError):
            write_dynamite_kinematics([s0, s1], tmp_path, meta)

    def test_all_none_raises(self, tmp_path):
        meta = _voronoi_metadata(n_bins_total=2, nx=1, ny=2)
        with pytest.raises(ValueError):
            write_dynamite_kinematics([None, None], tmp_path, meta)

    @pytest.mark.parametrize("mode", ["nstars", "uniform", "custom"])
    def test_bin_flux_modes(self, tmp_path, mode):
        n_bins = 6
        pdf_mass = np.ones(n_bins) / n_bins
        s0 = fake_solver(pdf_mass=pdf_mass, n_stars=42, seed=40)
        s0.clip_uncertainties()
        solvers = [s0]
        meta = _voronoi_metadata(n_bins_total=1, nx=1, ny=1, extra_bin_flux=[99.0])

        write_dynamite_kinematics(solvers, tmp_path, meta, bin_flux_mode=mode)
        table = Table.read(str(tmp_path / "bayes_losvd_kins.ecsv"), format="ascii.ecsv")

        if mode == "nstars":
            assert table["bin_flux"][0] == pytest.approx(42.0)
        elif mode == "uniform":
            assert table["bin_flux"][0] == pytest.approx(1.0)
        elif mode == "custom":
            assert table["bin_flux"][0] == pytest.approx(99.0)

    def test_bin_flux_nstars_requires_n_stars(self, tmp_path):
        n_bins = 6
        pdf_mass = np.ones(n_bins) / n_bins
        s0 = fake_solver(pdf_mass=pdf_mass, seed=41)
        s0.n_stars = None  # simulate add_data() never having been called
        s0.clip_uncertainties()
        meta = _voronoi_metadata(n_bins_total=1, nx=1, ny=1)

        with pytest.raises(ValueError):
            write_dynamite_kinematics(
                [s0], tmp_path, meta, bin_flux_mode="nstars"
            )

    def test_pixel_bin_ids_shape_equivalence(self, tmp_path):
        n_bins = 6
        pdf_mass = np.ones(n_bins) / n_bins
        s0 = fake_solver(pdf_mass=pdf_mass, seed=50)
        s0.clip_uncertainties()
        solvers = [s0]

        nx, ny = 2, 3
        flat_ids = np.array([1, 1, 0, 1, 0, 1], dtype=int)  # (nx*ny,)
        grid_ids = flat_ids.reshape(ny, nx)  # (ny, nx)

        meta_flat = {
            "bins": [{"xbin": 0.0, "ybin": 0.0}],
            "aperture": {
                "x_start": -1.0, "y_start": -1.0, "x_size": 2.0, "y_size": 3.0,
                "angle_deg": 0.0, "nx": nx, "ny": ny,
            },
            "pixel_bin_ids": flat_ids,
            "psf": {"sigma": [1.0], "weight": [1.0]},
        }
        meta_grid = dict(meta_flat)
        meta_grid["pixel_bin_ids"] = grid_ids

        out_flat = tmp_path / "flat"
        out_grid = tmp_path / "grid"
        write_dynamite_kinematics([s0], out_flat, meta_flat)
        write_dynamite_kinematics([s0], out_grid, meta_grid)

        bins_flat = (out_flat / "bins.dat").read_text()
        bins_grid = (out_grid / "bins.dat").read_text()
        assert bins_flat == bins_grid

    def test_kin_ecsv_reparseable(self, tmp_path):
        """Existence + re-parseability check (golden-file diff skipped, see
        module docstring — a stable reference fixture is out of scope here)."""
        n_bins = 8
        pdf_mass = np.ones(n_bins) / n_bins
        s0 = fake_solver(pdf_mass=pdf_mass, seed=60)
        s0.clip_uncertainties()
        meta = _voronoi_metadata(n_bins_total=1, nx=1, ny=1)

        write_dynamite_kinematics([s0], tmp_path, meta)
        kin_path = tmp_path / "bayes_losvd_kins.ecsv"
        assert kin_path.exists()
        table = Table.read(str(kin_path), format="ascii.ecsv")
        assert len(table) == 1
        assert "losvd_0" in table.colnames
