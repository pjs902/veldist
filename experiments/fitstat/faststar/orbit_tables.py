"""Model tables for the fast-star term from a finished model's own orbits (docs/fast-star-term.md section 3).

Re-integrates a model's orbits with the patched Fortran (dynamite-fitstat feature/fast-star-hist: input
histogram dimension 3 = 2D histogram over (|v_2D|, |v_los|)) and one extra aperture set: annuli about the
model centre on a fine pixel grid. The integration is order-independent per orbit (per-orbit seeds, per-orbit
DOP853 start step), so the existing weights apply unchanged. Nothing is written into the model directory: the
run directory symlinks its inputs.

    python orbit_tables.py prepare MODEL_DIR RUN_DIR [--with-kin] [--start S --number N | --chunks K]
        writes RUN_DIR/infil/{orblib,orblibbox}_fs.in, the annulus aperture/bins files and symlinks; prints the
        two Fortran commands (run them from RUN_DIR through launch.sh). --with-kin keeps the model's own four
        kinematic sets (for the gate); without it only the fast-star set is binned.
    python orbit_tables.py gate MODEL_DIR RUN_DIR
        bit-for-bit comparison of the re-integrated 2D (PM) records of the first N orbits with the stored library.
    python orbit_tables.py table CONFIG MODEL_DIRECTORY RUN_DIR OUT.npz
        weighted annular tables P[a, i, j] (each annulus sums to 1) with velocity edges scaled to the model's ml.
"""
import bz2
import os
import sys
from pathlib import Path

import numpy as np

R_EDGES = np.array([0, .25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6.5, 8, 10, 12.5])
NB, WIDTH = 81, 405.0                     # odd: Fortran 2D bin index = iv + (NB - 1) // 2; cells of 5 km/s from 0
PIX, HALF = 0.05, 12.5                    # annulus pixel grid [arcsec]


def _write_annuli(infil):
    n = int(round(2 * HALF / PIX))
    c = -HALF + PIX * (np.arange(n) + 0.5)
    X, Y = np.meshgrid(c, c, indexing="ij")
    a = np.searchsorted(R_EDGES, np.hypot(X, Y), side="right")          # 1-based annulus, 0 = unused
    a[a > len(R_EDGES) - 1] = 0
    (infil / "faststar_aperture.dat").write_text(
        "#counter_rotation_boxed_aperturefile_version_2\n"
        f"\t{-HALF:.6f}\t{-HALF:.6f}\n\t{2 * HALF:.6f}\t{2 * HALF:.6f}\n\t0.000000\n\t{n}\t{n}\n")
    flat = a.T.ravel()                         # x fastest, as in the existing bins files
    lines = ["\t" + "\t".join(str(v) for v in flat[i:i + 10]) for i in range(0, flat.size, 10)]
    (infil / "faststar_bins.dat").write_text("#Counterrotation_binning_version_1\n" f"{flat.size}\n" + "\n".join(lines) + "\n")


def _rewrite(src, box, with_kin, start, number, tag=""):
    """The model's orblib(.box).in with the fast-star set appended (or alone), outputs into RUN_DIR/datfil."""
    L = src.read_text().splitlines()
    head, i = L[:8], 8
    if start is not None:
        head[5] = f"{start}\t\t\t\t\t\t\t\t[starting orbit]"
        head[6] = f"{number}\t\t\t\t\t\t\t\t[orbits to integrate; -1 --> all orbits]"
    npsf = int(L[i].split()[0]); i += 1
    ngauss = L[i:i + npsf]; i += npsf
    nglines = sum(int(x.split()[0]) for x in ngauss)
    gauss = L[i:i + nglines]; i += nglines
    nap = int(L[i].split()[0]); i += 1
    aps = L[i:i + 3 * nap]; i += 3 * nap
    dims = [int(aps[3 * k + 2].split()[0]) for k in range(nap)]
    nh = sum(max(d, 1) for d in dims)
    hist = L[i:i + nh]; i += nh
    usebin = L[i:i + nap]; i += nap
    binf = L[i:i + nap]; i += nap
    outs = L[i:i + 4]
    assert len(L) == i + 4, (src, len(L), i)
    fam = ("orblibbox" if box else "orblib") + tag
    fs_ap = ['"infil/faststar_aperture.dat"', f"{npsf + 1 if with_kin else 1}\t\t\t\t\t\t\t\t[psf for faststar]",
             "3\t\t\t\t\t\t\t\t[histogram dimension for faststar: speed histogram]"]
    fs_hist = [f"{WIDTH} {WIDTH / 2} {NB}\t\t\t\t\t\tdim {k} of 2 [faststar |v_2D|, |v_los|]" for k in (1, 2)]
    if with_kin:
        body = ([f"{npsf + 1}\t\t\t\t\t\t\t\t[number of psfs]"] + ngauss + ["1\t\t\t\t\t\t\t\t[# gaussians psf faststar]"]
                + gauss + ["1.0 0.0\t\t\t\t\t\t\t[weight, sigma faststar psf]"]
                + [f"{nap + 1}\t\t\t\t\t\t\t\t[# of apertures]"] + aps + fs_ap + hist + fs_hist
                + usebin + ["1\t\t\t\t\t\t\t\t[use binning faststar]"] + binf + ['"infil/faststar_bins.dat"'])
    else:
        body = (["1\t\t\t\t\t\t\t\t[number of psfs]", "1\t\t\t\t\t\t\t\t[# gaussians psf faststar]",
                 "1.0 0.0\t\t\t\t\t\t\t[weight, sigma faststar psf]", "1\t\t\t\t\t\t\t\t[# of apertures]"]
                + fs_ap + fs_hist + ["1\t\t\t\t\t\t\t\t[use binning faststar]", '"infil/faststar_bins.dat"'])
    outs = [f'"datfil/{fam}_qgrid.dat"', f'"datfil/{fam}_losvd_hist.dat"', f'"datfil/{fam}_pm_hist.dat"',
            f'"datfil/{fam}.dat_orbclass.out"']
    return "\n".join(head + body + outs) + "\n"


def prepare(mdir, run, with_kin=False, start=None, number=None, chunks=None, norb=15000):
    mdir, run = Path(mdir).resolve(), Path(run).resolve()
    assert "/models/" in str(mdir) and run != mdir and not str(run).startswith(str(mdir))
    (run / "infil").mkdir(parents=True, exist_ok=True)
    (run / "datfil").mkdir(exist_ok=True)
    for f in (mdir / "infil").iterdir():                                 # potential, aperture and bins files
        if f.suffix in (".in", ".dat") and not f.name.startswith("orblib"):
            d = run / "infil" / f.name
            if not d.exists():
                d.symlink_to(f)
    for f in ("begin.dat", "beginbox.dat"):
        d = run / "datfil" / f
        if not d.exists():
            d.symlink_to(mdir / "datfil" / f)
    for f in ("interpolgrid", "fort.30"):                                # potential interpolation inputs
        if (mdir / f).exists() and not (run / f).exists():
            (run / f).symlink_to(mdir / f)
    _write_annuli(run / "infil")
    exe = "$FS_ORBLIB"            # the patched orblib_new_mirror; set by the job script
    if chunks:                    # contiguous orbit ranges per family; outputs datfil/{fam}_c{k}_*
        edges = np.linspace(0, norb, chunks + 1).round().astype(int)
        for box in (False, True):
            fam = "orblibbox" if box else "orblib"
            for k in range(chunks):
                (run / "infil" / f"{fam}_fs_c{k}.in").write_text(_rewrite(
                    mdir / "infil" / f"{fam}.in", box, with_kin, edges[k] + 1, edges[k + 1] - edges[k], f"_c{k}"))
                print(f"cd {run} && {exe} < infil/{fam}_fs_c{k}.in > {fam}_fs_c{k}.log")
        (run / "chunks.txt").write_text(" ".join(map(str, edges)))
        return
    for box in (False, True):
        fam = "orblibbox" if box else "orblib"
        (run / "infil" / f"{fam}_fs.in").write_text(_rewrite(mdir / "infil" / f"{fam}.in", box, with_kin, start, number))
    for fam in ("orblib", "orblibbox"):
        print(f"cd {run} && {exe} < infil/{fam}_fs.in > {fam}_fs.log")


def walk(raw, n_pairs):
    """(start, ivmin0, ivmin1, n0, nv) per (orbit, aperture) pair of a raw 2D-histogram stream (fork format)."""
    mv = memoryview(raw)[: len(raw) // 4 * 4].cast("i")     # a prefix read of the stored file can end mid-record
    p, out = 0, np.zeros((5, n_pairs), np.int64)
    for k in range(n_pairs):
        i0, i1, a0, a1 = mv[p + 1], mv[p + 2], mv[p + 3], mv[p + 4]
        p += 6
        if i0 <= a0 and i1 <= a1:
            nb = mv[p]
            out[:, k] = (p + 1, i0, i1, a0 - i0 + 1, nb >> 3)
            p += 2 + (nb >> 2)
    return out, p * 4


def _raw(path, nbytes=None):
    if str(path).endswith(".bz2"):
        with bz2.open(path, "rb") as f:
            return f.read(nbytes) if nbytes else f.read()
    with open(path, "rb") as f:
        return f.read(nbytes) if nbytes else f.read()


def gate(mdir, run, n_ap2d=1555):
    """First-N-orbit records of the model's own 2D sets must match the stored library bit for bit."""
    mdir, run = Path(mdir), Path(run)
    for fam in ("orblib", "orblibbox"):
        new = _raw(run / "datfil" / f"{fam}_pm_hist.dat")
        n_orb = int((run / "infil" / f"{fam}_fs.in").read_text().splitlines()[6].split()[0])
        n_fs = len(R_EDGES) - 1
        rec_new, used = walk(new, n_orb * (n_ap2d + n_fs))
        old = _raw(mdir / "datfil" / f"{fam}_pm_hist.dat.bz2", nbytes=used * 2 + (64 << 20))
        rec_old, _ = walk(old, n_orb * n_ap2d)
        sel = (np.arange(n_orb * (n_ap2d + n_fs)) % (n_ap2d + n_fs)) < n_ap2d       # the model's own sets
        rn = rec_new[:, sel]
        assert np.array_equal(rn[1:], rec_old[1:]), f"{fam}: record shapes differ"
        f64n, f64o = np.frombuffer(new, np.uint8), np.frombuffer(old, np.uint8)
        bad = 0
        for k in np.nonzero(rn[4] > 0)[0]:
            a = f64n[rn[0, k] * 4: rn[0, k] * 4 + rn[4, k] * 8]
            b = f64o[rec_old[0, k] * 4: rec_old[0, k] * 4 + rec_old[4, k] * 8]
            bad += not np.array_equal(a, b)
        print(f"{fam}: {n_orb} orbits, {int((rn[4] > 0).sum())} non-empty 2D records, {bad} differ")
        assert bad == 0


def speed_hist(run, fam, wt):
    """sum_o wt[o] h_o: (n_annuli, NB, NB) weighted speed histograms of the fast-star set (the only 2D set in a
    fast-only run); wt has one entry per orbit of this family. Chunked runs (chunks.txt) are summed chunk by chunk."""
    ch = Path(run) / "chunks.txt"
    if ch.exists():
        e = [int(x) for x in ch.read_text().split()]
        assert e[-1] == len(wt), (e[-1], len(wt))
        return sum(speed_hist_file(Path(run) / "datfil" / f"{fam}_c{k}_pm_hist.dat", wt[e[k]:e[k + 1]])
                   for k in range(len(e) - 1))
    return speed_hist_file(Path(run) / "datfil" / f"{fam}_pm_hist.dat", wt)


def speed_hist_file(path, wt):
    raw = _raw(path)
    n_ap = len(R_EDGES) - 1
    rec, _ = walk(raw, len(wt) * n_ap)
    f64 = np.frombuffer(raw, np.uint8)
    M = np.zeros((n_ap, NB, NB))
    c = (NB - 1) // 2
    for k in np.nonzero((rec[4] > 0) & (np.repeat(wt, n_ap) > 0))[0]:
        st, i0, i1, n0, nv = rec[:, k]
        v = np.frombuffer(f64[st * 4: st * 4 + nv * 8].tobytes(), np.float64).reshape((nv // n0, n0)).T  # Fortran order
        M[k % n_ap, i0 + c: i0 + c + n0, i1 + c: i1 + c + v.shape[1]] += wt[k // n_ap] * v
    return M


def table(cfg, mdirectory, run, out):
    sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
    import grid_env  # noqa: F401
    import dynamite as dyn
    from astropy.table import Table
    os.chdir(Path(cfg).parent)
    c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
    rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdirectory.rstrip("/")]
    assert len(rows) == 1, mdirectory
    model = c.all_models.get_model_from_row(rows[0])
    sf = c.all_models.get_model_velocity_scaling_factor(model_id=rows[0])
    w = np.asarray(Table.read(Path(model.directory) / "orbit_weights.ecsv")["weights"], float)
    ch = Path(run) / "chunks.txt"
    if ch.exists():
        n_tube = n_box = int(ch.read_text().split()[-1])
    else:
        n_tube = int((Path(run) / "infil" / "orblib_fs.in").read_text().splitlines()[6].split()[0])
        n_box = int((Path(run) / "infil" / "orblibbox_fs.in").read_text().splitlines()[6].split()[0])
    if n_tube < 0:
        n_tube = n_box = c.settings.orblib_settings["nE"] * c.settings.orblib_settings["nI2"] * c.settings.orblib_settings["nI3"]
    assert w.size == 2 * n_tube + n_box, (w.size, n_tube, n_box)
    # tube columns are [orbit, mirrored orbit] pairs; mirroring flips signs, leaving the speeds unchanged
    M = speed_hist(run, "orblib", w[0:2 * n_tube:2] + w[1:2 * n_tube:2]) + speed_hist(run, "orblibbox", w[2 * n_tube:])
    mass = M.sum((1, 2))
    edges = np.linspace(0, WIDTH, NB + 1) * sf                          # library velocities scaled to the model's ml
    from scipy.ndimage import gaussian_filter
    Ms = np.stack([gaussian_filter(m, 2.0, mode="nearest") for m in M])   # 2 cells: table noise (smooth_test.py)
    np.savez(out, P=M / mass[:, None, None], P_s2=Ms / Ms.sum((1, 2))[:, None, None], mass=mass, R_edges=R_EDGES, v_edges=edges, w_edges=edges,
             sf=sf, directory=mdirectory)
    print(f"{mdirectory}: sf {sf:.5f}; annulus masses {np.round(mass / mass.sum(), 4)}")


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "prepare":
        o = dict(with_kin="--with-kin" in a)
        if "--start" in a:
            o.update(start=int(a[a.index("--start") + 1]), number=int(a[a.index("--number") + 1]))
        if "--chunks" in a:
            o.update(chunks=int(a[a.index("--chunks") + 1]))
        prepare(a[0], a[1], **o)
    elif cmd == "gate":
        gate(a[0], a[1])
    elif cmd == "table":
        table(*a[:4])
