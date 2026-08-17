"""
DYNAMITE 2D (Proper-Motion) Output Writer
==========================================

The 2D analogue of ``veldist.write_dynamite_kinematics``: turns a list of
solved :class:`~veldist.veldist2d.KinematicSolver2D` instances into the
``.npz`` archive Dynamite's ``ProperMotions``/``Histogram2D`` kinematics
representation consumes, plus the usual ``aperture.dat``/``bins.dat`` pair
(format unchanged from 1D).

Kept in its own module rather than appended to ``veldist2d.py`` -- see
``docs/superpowers/specs/2026-08-05-dynamite-2d-writer-design.md`` §1. This
is pure post-processing I/O with no shared state with the sampler, has a
different dependency footprint (numpy-only, no astropy), and targets an
upstream format that is still moving (see the version-pinning note below),
so it is kept trivially swappable/deletable independent of
``veldist2d.py``'s tested modelling code.

**Target format**: Dynamite PR #442, commit ``9ccc416``, merged to
``main`` 2026-06-03. **Not yet in any tagged release** (``v5.0.0`` predates
it). This writer's ``.npz`` key/shape contract is data, not code, and has
no runtime dependency on Dynamite itself -- but the contract may need
revision if the upstream interface changes before a tagged release.
"""

from pathlib import Path

import numpy as np

__all__ = ["write_dynamite_kinematics_2d"]


def _write_aperture_and_bins_files(
    solvers, output_dir, voronoi_bin_metadata, aperture_filename, bins_filename
):
    """Write aperture.dat / bins.dat. Format unchanged from 1D's writer
    (veldist.write_dynamite_kinematics); duplicated here rather than shared
    because that function is not importable without pulling in astropy.

    ``ap['angle_deg']`` is written verbatim: this writer does no frame handling
    and no validation. DYNAMITE's rule is ``angle_deg = -theta_maj`` with
    ``theta_maj`` the receding major axis measured CCW from +x **in the caller's
    own frame** -- not a sky position angle (the two agree only mod 180, and
    substituting one for the other silently inverts every fitted rotation).

    The frame of the ``pm1``/``pm2`` histograms matters too and is likewise the
    caller's responsibility: DYNAMITE's projection is right-handed with the LOS
    along ``x' x y'``, so ``(x, y, v_los)`` must be right-handed. In particular
    ``pm2`` (the minor-axis component) flips sign under an East/West mirror while
    ``pm1`` does not. See ``omegaCen/dynamite_dataprep/dynamite_frame.py``.
    """
    ap = voronoi_bin_metadata["aperture"]
    ap_path = output_dir / aperture_filename
    with open(ap_path, "w") as f:
        f.write("#counter_rotation_boxed_aperturefile_version_2 \n")
        f.write(f"\t{ap['x_start']:f}\t{ap['y_start']:f} \n")
        f.write(f"\t{ap['x_size']:f}\t{ap['y_size']:f} \n")
        f.write(f"\t{ap['angle_deg']:f} \n")
        f.write(f"\t{ap['nx']}\t{ap['ny']} \n")
    print(f"Written aperture: {ap_path}")

    solved_indices = [i for i, s in enumerate(solvers) if s is not None]
    n_total = len(solvers)
    orig_to_new = np.zeros(n_total + 1, dtype=int)  # index 0 unused
    for new_id, orig_i in enumerate(solved_indices, start=1):
        orig_to_new[orig_i + 1] = new_id  # orig_i is 0-based; +1 for 1-based

    pixel_ids = np.asarray(voronoi_bin_metadata["pixel_bin_ids"]).flatten().astype(int)
    remapped = np.where(
        (pixel_ids > 0) & (pixel_ids <= n_total),
        orig_to_new[pixel_ids],
        0,
    )
    total_pixels = len(remapped)

    bins_path = output_dir / bins_filename
    with open(bins_path, "w") as f:
        f.write("#Counterrotation_binning_version_1\n")
        f.write(f"{total_pixels}\n")
        for start in range(0, total_pixels, 10):
            chunk = remapped[start : start + 10]
            f.write("\t" + "\t".join(str(v) for v in chunk) + "\n")
    print(f"Written bins: {bins_path}")


def write_dynamite_kinematics_2d(
    solvers,
    output_dir,
    voronoi_bin_metadata,
    npz_filename="pm_2dhist.npz",
    aperture_filename="aperture.dat",
    bins_filename="bins.dat",
    uncertainty_floor_fraction=0.01,
    uncertainty_abs_floor=1e-10,
):
    """
    Write Dynamite-compatible ProperMotions/Histogram2D input files from a
    list of solved spatial (Voronoi) bins.

    Produces three files:

    - ``{npz_filename}``: NumPy ``.npz`` archive with keys ``PM_2dhist``,
      ``PM_2dhist_sigma`` (both ``(n_apertures, K, K)``), ``binID_dynamite``,
      ``nstarbin`` (all ``(n_apertures,)``), ``vxrange``, ``vyrange``
      (scalars -- one shared velocity grid for every aperture), ``xbin``,
      ``ybin`` (``(n_apertures,)``).
    - ``{aperture_filename}``: pixel grid geometry. Format unchanged from 1D.
    - ``{bins_filename}``: pixel-to-bin mapping. Format unchanged from 1D.

    Any ``None`` entries in ``solvers`` (bins skipped by
    :func:`~veldist.dynamite2d.fit_all_bins_2d`) are automatically masked:
    their pixels are written as 0 in the bins file and they are omitted
    from ``PM_2dhist``/``PM_2dhist_sigma``/``nstarbin``/``xbin``/``ybin``.
    The remaining bins are re-numbered sequentially (1-indexed) in
    ``binID_dynamite`` -- **required**, not merely conventional: Dynamite's
    legacy orbit-library reader (``orblib_f.f90``, ``LegacyOrbitLibrary.
    read_orbit_base``) assumes 1-indexed, gap-free bin IDs.

    :meth:`~veldist.veldist2d.KinematicSolver2D.clip_uncertainties` is
    called automatically (with ``uncertainty_floor_fraction``/
    ``uncertainty_abs_floor``) on any solver that has not already had its
    ``clipped_samples`` populated.

    **Normalisation**: ``PM_2dhist``/``PM_2dhist_sigma`` are written exactly
    as ``clip_uncertainties`` produces them -- per-bin marginal medians that
    typically sum to ~0.85-0.95, not 1. Do **not** pre-normalise. Dynamite's
    ``ProperMotions.normalise()`` divides both arrays by the identical
    per-aperture ``hist_scale`` factor on load, so the value/uncertainty
    correspondence is preserved regardless of the pre-normalisation sum
    (see the design doc §9 for the full trace through Dynamite's source).

    **Axis order**: ``PM_2dhist[a, ix, iy]`` -- axis 1 is vx, axis 2 is vy,
    matching ``setup_grid_2d``'s ``(ix, iy)`` convention with no transpose.
    Verified against ``ProperMotions.as_histogram2d()`` (Dynamite PR #442,
    commit ``9ccc416``); see the design doc §5.

    Parameters
    ----------
    solvers : list
        Solved :class:`~veldist.veldist2d.KinematicSolver2D` instances (or
        ``None`` for skipped bins), as returned by
        :func:`~veldist.dynamite2d.fit_all_bins_2d`. All non-``None``
        entries must share the same (square) velocity grid -- same per-axis
        bin count K and the same ``(vx, vy)`` bin edges.
    output_dir : str or path-like
        Directory in which to write the three output files. Created if it
        does not exist.
    voronoi_bin_metadata : dict
        Spatial metadata. Same structure as
        :func:`veldist.veldist.write_dynamite_kinematics`'s
        ``voronoi_bin_metadata`` argument: ``'bins'`` (list of dicts with
        ``'xbin'``/``'ybin'``), ``'aperture'``, ``'pixel_bin_ids'``.
    npz_filename : str
        File name for the kinematics ``.npz``. Default ``'pm_2dhist.npz'``.
    aperture_filename : str
        File name for the aperture file. Default ``'aperture.dat'``.
    bins_filename : str
        File name for the bins file. Default ``'bins.dat'``.
    uncertainty_floor_fraction, uncertainty_abs_floor : float
        Forwarded to :meth:`~veldist.veldist2d.KinematicSolver2D.clip_uncertainties`
        when auto-invoked. Defaults match 1D's validated values (not yet
        re-measured for 2D).

    Raises
    ------
    ValueError
        If no solved bins are found; if solvers share inconsistent grids;
        if any solved solver's per-axis bin count K is even (Dynamite's
        ``set_default_hist_bins`` rejects even bin counts); or if any
        ``PM_2dhist_sigma`` entry is <= 0 after clipping.

    Returns
    -------
    None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Identify solved bins and validate grid consistency
    # ------------------------------------------------------------------
    solved_indices = [i for i, s in enumerate(solvers) if s is not None]
    n_solved = len(solved_indices)

    if n_solved == 0:
        msg = "No solved bins found (all solvers are None)."
        raise ValueError(msg)

    ref_solver = solvers[solved_indices[0]]
    k = int(ref_solver.grid["n_bins"])

    if k % 2 == 0:
        msg = (
            f"Solver at index {solved_indices[0]} has an even per-axis bin "
            f"count K={k}. Dynamite requires odd bin counts per axis "
            "(set_default_hist_bins raises otherwise). Rebuild the grid "
            "with an odd n_bins before writing."
        )
        raise ValueError(msg)

    edges_x = np.asarray(ref_solver.grid["edges_x"])
    edges_y = np.asarray(ref_solver.grid["edges_y"])

    for idx in solved_indices[1:]:
        s = solvers[idx]
        s_k = int(s.grid["n_bins"])
        if s_k % 2 == 0:
            msg = (
                f"Solver at index {idx} has an even per-axis bin count "
                f"K={s_k}. Dynamite requires odd bin counts per axis."
            )
            raise ValueError(msg)
        if (
            s_k != k
            or not np.allclose(s.grid["edges_x"], edges_x)
            or not np.allclose(s.grid["edges_y"], edges_y)
        ):
            msg = (
                f"Solver at index {idx} has a different velocity grid than "
                f"solver at index {solved_indices[0]}. All bins must share "
                "the same (vx, vy) grid -- Dynamite's .npz format carries a "
                "single scalar vxrange/vyrange for the whole map."
            )
            raise ValueError(msg)

    # DYNAMITE reconstructs the velocity axis as linspace(-vxrange, +vxrange,
    # K+1), so a grid that is not centred on zero would be silently shifted on
    # read-back -- every mean proper motion displaced, with nothing raising.
    # Check both axes against a tolerance scaled to the grid, not an absolute
    # epsilon, so this behaves the same at any velocity scale.
    for axis_name, edges in (("x", edges_x), ("y", edges_y)):
        centre = 0.5 * (edges[0] + edges[-1])
        span = edges[-1] - edges[0]
        if abs(centre) > 1e-9 * span:
            msg = (
                f"velocity grid is not centred on zero: {axis_name}-axis centre "
                f"is {centre:.6g} over a span of {span:.6g}. DYNAMITE assumes a "
                f"symmetric [-v{axis_name}range, +v{axis_name}range] axis, so an "
                "off-centre grid would be silently shifted on read-back. "
                "Re-fit with center=(0.0, 0.0) in setup_grid."
            )
            raise ValueError(msg)

    # vxrange/vyrange are half-widths (Dynamite builds vxedg = linspace(
    # -vxrange, vxrange, K+1) in ProperMotions.as_histogram2d()).
    vxrange = float((edges_x[-1] - edges_x[0]) / 2.0)
    vyrange = float((edges_y[-1] - edges_y[0]) / 2.0)

    # ------------------------------------------------------------------
    # Gather per-cell PM-distribution summaries (auto-clip if needed)
    # ------------------------------------------------------------------
    bin_metas = voronoi_bin_metadata["bins"]

    PM_2dhist = np.zeros((n_solved, k, k), dtype=np.float64)
    PM_2dhist_sigma = np.zeros((n_solved, k, k), dtype=np.float64)
    nstarbin = np.zeros(n_solved, dtype=np.int64)
    xbin = np.zeros(n_solved, dtype=np.float64)
    ybin = np.zeros(n_solved, dtype=np.float64)

    missing_nstars = []
    for out_i, orig_i in enumerate(solved_indices):
        solver = solvers[orig_i]
        if solver.clipped_samples is None:
            solver.clip_uncertainties(
                floor_fraction=uncertainty_floor_fraction, abs_floor=uncertainty_abs_floor
            )

        median_flat = np.asarray(solver.clipped_samples["pdf_median"])
        unc_flat = np.asarray(solver.clipped_samples["pdf_uncertainty"])

        # Row-major reshape: flat index m = ix*K+iy -> [ix, iy]. This is
        # exactly setup_grid_2d's convention; do NOT use order="F" and do
        # NOT transpose after (that would silently swap vx/vy).
        PM_2dhist[out_i] = median_flat.reshape(k, k, order="C")
        PM_2dhist_sigma[out_i] = unc_flat.reshape(k, k, order="C")

        if solver.n_stars is None:
            missing_nstars.append(orig_i)
        else:
            nstarbin[out_i] = int(solver.n_stars)

        xbin[out_i] = bin_metas[orig_i]["xbin"]
        ybin[out_i] = bin_metas[orig_i]["ybin"]

    if missing_nstars:
        msg = (
            f"nstarbin requires that add_data() was called on every solver, "
            f"but solvers at indices {missing_nstars} have n_stars=None."
        )
        raise ValueError(msg)

    # Guard: no zero/negative uncertainties (would corrupt Dynamite's NNLS
    # matrices exactly as in 1D -- K**2 cells per aperture means more
    # exposure to this failure mode than 1D's n_bins).
    if not np.all(PM_2dhist_sigma > 0):
        msg = (
            "Zero or negative uncertainty found in PM_2dhist_sigma after "
            "clipping. This would cause econ zeros in Dynamite's NNLS "
            "projection. Check clip_uncertainties() floor settings."
        )
        raise ValueError(msg)

    binID_dynamite = np.arange(1, n_solved + 1)

    # ------------------------------------------------------------------
    # Write the .npz archive
    # ------------------------------------------------------------------
    npz_path = output_dir / npz_filename
    np.savez(
        npz_path,
        PM_2dhist=PM_2dhist,
        PM_2dhist_sigma=PM_2dhist_sigma,
        binID_dynamite=binID_dynamite,
        nstarbin=nstarbin,
        vxrange=vxrange,
        vyrange=vyrange,
        xbin=xbin,
        ybin=ybin,
    )
    print(f"Written kinematics ({n_solved} bins): {npz_path}")

    # ------------------------------------------------------------------
    # Write aperture.dat / bins.dat (format unchanged from 1D)
    # ------------------------------------------------------------------
    _write_aperture_and_bins_files(
        solvers, output_dir, voronoi_bin_metadata, aperture_filename, bins_filename
    )
