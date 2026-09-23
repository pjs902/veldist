"""
DYNAMITE 2D (Proper-Motion) Output Writer
==========================================

The 2D counterpart of ``veldist.write_dynamite_kinematics``. It takes a list
of fitted :class:`~veldist.veldist2d.KinematicSolver2D` objects and writes
the ``.npz`` archive read by Dynamite's ``ProperMotions``/``Histogram2D``
kinematics, plus the usual ``aperture.dat`` and ``bins.dat`` (same format as
in 1D).

It lives in its own module rather than in ``veldist2d.py`` (see
``docs/superpowers/specs/2026-08-05-dynamite-2d-writer-design.md`` §1). It
is pure I/O that shares no state with the sampler, needs only numpy (no
astropy), and targets an upstream format that may still change, so it
should be easy to replace or remove without touching the modelling code.

**Target format**: Dynamite PR #442, commit ``9ccc416``, merged to ``main``
on 2026-06-03 and **not yet in a tagged release** (``v5.0.0`` is older).
The writer does not import Dynamite; it only follows the ``.npz`` keys and
shapes, which may need updating if the upstream interface changes before a
release.
"""

from pathlib import Path

import numpy as np

__all__ = ["write_dynamite_kinematics_2d"]


def _write_aperture_and_bins_files(
    solvers, output_dir, voronoi_bin_metadata, aperture_filename, bins_filename
):
    """Write aperture.dat and bins.dat in the same format as the 1D writer.

    The code is duplicated from ``veldist.write_dynamite_kinematics`` because
    importing that function would pull in astropy.

    ``ap['angle_deg']`` is written as given, with no frame handling or
    checks. Dynamite expects ``angle_deg = -theta_maj``, where ``theta_maj``
    is the receding major axis measured counter-clockwise from +x **in the
    caller's own frame**. This is not a sky position angle: the two agree
    only modulo 180, and using one in place of the other silently flips every
    fitted rotation.

    The frame of the ``pm1``/``pm2`` histograms is also the caller's
    responsibility. Dynamite's projection is right-handed with the line of
    sight along ``x' x y'``, so ``(x, y, v_los)`` must be right-handed.
    ``pm2`` (the minor-axis component) changes sign under an East/West
    mirror; ``pm1`` does not. See
    ``omegaCen/dynamite_dataprep/dynamite_frame.py``.
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
    Write Dynamite ProperMotions/Histogram2D input files for a set of
    fitted spatial (Voronoi) bins.

    Three files are written:

    - ``{npz_filename}``: a NumPy ``.npz`` archive with ``PM_2dhist`` and
      ``PM_2dhist_sigma`` (both ``(n_apertures, K, K)``), ``binID_dynamite``,
      ``nstarbin``, ``xbin`` and ``ybin`` (all ``(n_apertures,)``), and the
      scalars ``vxrange`` and ``vyrange`` (every aperture shares one velocity
      grid).
    - ``{aperture_filename}``: pixel grid geometry, same format as 1D.
    - ``{bins_filename}``: pixel-to-bin map, same format as 1D.

    ``None`` entries in ``solvers`` (bins skipped by
    :func:`~veldist.veldist2d.fit_all_bins_2d`) are masked: their pixels are
    written as 0 in the bins file and they are left out of every array in
    the ``.npz``. The remaining bins are renumbered 1, 2, 3, ... in
    ``binID_dynamite``. This is required: Dynamite's legacy orbit-library
    reader (``orblib_f.f90``, ``LegacyOrbitLibrary.read_orbit_base``) assumes
    bin IDs start at 1 with no gaps.

    Any solver whose ``clipped_samples`` is not yet set has
    :meth:`~veldist.veldist2d.KinematicSolver2D.clip_uncertainties` called on
    it with ``uncertainty_floor_fraction`` and ``uncertainty_abs_floor``.

    **Normalisation.** ``PM_2dhist`` and ``PM_2dhist_sigma`` are written
    exactly as ``clip_uncertainties`` returns them. They are per-cell
    marginal medians and usually sum to about 0.85-0.95, not 1. Do **not**
    normalise them first: Dynamite's ``ProperMotions.normalise()`` divides
    both arrays by the same per-aperture ``hist_scale`` when loading, so
    values and uncertainties stay consistent whatever the sum (see §9 of the
    design doc).

    **Axis order.** ``PM_2dhist[a, ix, iy]``: axis 1 is vx and axis 2 is vy,
    the same ``(ix, iy)`` order as ``setup_grid_2d``, with no transpose.
    Checked against ``ProperMotions.as_histogram2d()`` in Dynamite PR #442,
    commit ``9ccc416`` (design doc §5).

    Parameters
    ----------
    solvers : list
        Solved :class:`~veldist.veldist2d.KinematicSolver2D` instances (or
        ``None`` for skipped bins), as returned by
        :func:`~veldist.veldist2d.fit_all_bins_2d`. Every non-``None``
        entry must use the same square velocity grid: the same per-axis bin
        count K and the same ``(vx, vy)`` bin edges.
    output_dir : str or path-like
        Directory for the three output files; created if needed.
    voronoi_bin_metadata : dict
        Spatial metadata, structured as for
        :func:`veldist.veldist.write_dynamite_kinematics`: ``'bins'`` (list of dicts with
        ``'xbin'``/``'ybin'``), ``'aperture'``, ``'pixel_bin_ids'``.
    npz_filename : str
        File name for the kinematics ``.npz``. Default ``'pm_2dhist.npz'``.
    aperture_filename : str
        File name for the aperture file. Default ``'aperture.dat'``.
    bins_filename : str
        File name for the bins file. Default ``'bins.dat'``.
    uncertainty_floor_fraction, uncertainty_abs_floor : float
        Forwarded to :meth:`~veldist.veldist2d.KinematicSolver2D.clip_uncertainties`
        when it is called automatically. The defaults are the values
        validated in 1D; they have not been re-measured for 2D.

    Raises
    ------
    ValueError
        If there are no fitted bins, if the solvers' grids differ, if any
        per-axis bin count K is even (Dynamite's ``set_default_hist_bins``
        rejects even counts), or if any ``PM_2dhist_sigma`` entry is <= 0
        after clipping.

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
