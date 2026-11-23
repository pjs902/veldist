"""Tests for reusing an orbit library across M/L by stretching the data (docs/ml-reuse-investigation.md).

The claim under test: a library built at ml0 and reused at ml = k ml0 has the same cell masses m on cells s*e
(s = sqrt(k)), so the data likelihood in km/s is
    -2 ln L(s) = -2 sum_i ln( rows_i(v_i / s, C_i / s^2; e) . m / h^d ) + 2 N d ln s
with rows built by the production veldist design-matrix functions on the native cells e. Run:
    pytest experiments/fitstat/test_ml_stretch.py -q
"""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import integrate  # noqa: E402
from scipy.stats import multivariate_normal, norm  # noqa: E402

from veldist.veldist import precompute_design_matrix  # noqa: E402
from veldist.veldist2d import precompute_design_matrix_2d, setup_grid_2d  # noqa: E402

S_VALUES = (0.9, 0.97, 1.0, 1.04, 1.12)
rng = np.random.default_rng(7)


def _stars_1d(n=400):
    return rng.normal(3.0, 18.0, n), rng.uniform(1.0, 9.0, n)


def _stars_2d(n=300):
    pm = rng.multivariate_normal([2.0, -1.0], [[300.0, 40.0], [40.0, 200.0]], n)
    a = rng.uniform(1.0, 6.0, n)
    b = rng.uniform(1.0, 6.0, n)
    r = rng.uniform(-0.6, 0.6, n)
    cov = np.stack([np.stack([a * a, r * a * b], -1), np.stack([r * a * b, b * b], -1)], -2)
    return pm[:, 0], pm[:, 1], cov


@pytest.mark.parametrize("s", S_VALUES)
def test_rows_1d_stars_over_s_equal_kernels_over_stretched_cells(s):
    """T1a: dividing the stars by s on cells e == the unscaled stars on cells s*e (production 1D function)."""
    v, err = _stars_1d()
    c, h = np.arange(-80.0, 80.1, 8.0), 8.0
    a = np.asarray(precompute_design_matrix(v / s, err / s, c, bin_width=h))
    b = np.asarray(precompute_design_matrix(v, err, s * c, bin_width=s * h))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.parametrize("s", S_VALUES)
def test_rows_2d_correlated_stars_over_s_equal_kernels_over_stretched_cells(s):
    """T1b: the same for 2D rows with correlated covariances (the Owen's-T path) and diagonal ones."""
    p1, p2, cov = _stars_2d()
    cov[:50, 0, 1] = cov[:50, 1, 0] = 0.0  # exercise the diagonal path too
    g = setup_grid_2d((0.0, 0.0), (165.6, 165.6), 23)
    gs = setup_grid_2d((0.0, 0.0), (s * 165.6, s * 165.6), 23)
    a = np.asarray(precompute_design_matrix_2d(p1 / s, p2 / s, cov / s ** 2, g), dtype=float)
    b = np.asarray(precompute_design_matrix_2d(p1, p2, cov, gs), dtype=float)
    np.testing.assert_allclose(a, b, rtol=0, atol=2e-7)  # the function casts each chunk to float32


@pytest.mark.parametrize("s", (0.93, 1.0, 1.07))
def test_absolute_likelihood_1d_with_jacobian_equals_direct_integral(s):
    """T2a: the stretch formula (rows of v/s, /h, + 2 N ln s) equals -2 ln L in km/s computed independently by
    quadrature of the model density (flat m_c / (s h) on the stretched cells) against each star's kernel."""
    v, err = _stars_1d(25)
    h, c = 8.0, np.arange(-80.0, 80.1, 8.0)
    m = rng.dirichlet(np.ones(c.size))
    rows = np.asarray(precompute_design_matrix(v / s, err / s, c, bin_width=h))
    stretch = -2 * np.sum(np.log(rows @ m / h)) + 2 * len(v) * np.log(s)
    edges = s * np.r_[c - h / 2, c[-1] + h / 2]
    direct = 0.0
    for vi, ei in zip(v, err):
        p = sum(mc / (s * h) * integrate.quad(lambda u: norm.pdf(vi, u, ei), lo, hi, epsabs=1e-14)[0]
                for mc, lo, hi in zip(m, edges[:-1], edges[1:]))
        direct += -2 * np.log(p)
    assert abs(stretch - direct) < 1e-6 * abs(direct), (stretch, direct)


@pytest.mark.parametrize("s", (0.93, 1.07))
def test_absolute_likelihood_2d_with_jacobian_equals_direct_integral(s):
    """T2b: the same in 2D (Jacobian 2 N * 2 ln s), against a dblquad of the stretched flat-cell density."""
    p1, p2, cov = _stars_2d(4)
    K, width = 5, 40.0
    g = setup_grid_2d((0.0, 0.0), (width, width), K)
    h = width / K
    m = rng.dirichlet(np.ones(K * K))
    rows = np.asarray(precompute_design_matrix_2d(p1 / s, p2 / s, cov / s ** 2, g), dtype=float)
    stretch = -2 * np.sum(np.log(rows @ m / h ** 2)) + 2 * len(p1) * 2 * np.log(s)
    e = s * (np.arange(K + 1) * h - width / 2)
    direct = 0.0
    for i in range(len(p1)):
        k = multivariate_normal([p1[i], p2[i]], cov[i])
        p = 0.0
        for ix in range(K):
            for iy in range(K):
                mass = m[np.ravel_multi_index((ix, iy), (K, K))]
                if mass < 1e-4:
                    continue
                f = lambda y, x: k.pdf([x, y])  # noqa: E731
                p += mass / (s * h) ** 2 * integrate.dblquad(f, e[ix], e[ix + 1], e[iy], e[iy + 1], epsabs=1e-12)[0]
        direct += -2 * np.log(p)
    m_small = m < 1e-4
    assert not m_small.all()
    # cells skipped in the quadrature (mass < 1e-4) bound the difference
    assert abs(stretch - direct) < 2e-3 * abs(direct) + 1e-3, (stretch, direct)


def test_jacobian_sign_with_a_known_answer():
    """T2c: a Gaussian model N(0, (s sigma)^2) binned on s*e, wide window, tiny errors: -2 ln L(s) from the stretch
    formula must match the analytic Gaussian -2 ln L of the same stars at every s to discretisation accuracy, so the
    minimum over s is at the data's own scale. Catches a wrong sign or a missing factor of the Jacobian."""
    sig, h = 15.0, 1.0
    v = rng.normal(0, sig * 0.95, 5000)  # data at s_true = 0.95
    err = np.full(v.size, 1.0)  # >= h: no cell aliasing (that is test_aliasing_*), so only the Jacobian is tested
    c = np.arange(-120.0, 120.1, h)
    S = np.linspace(0.88, 1.03, 31)
    stretch, analytic = [], []
    for s in S:
        m = np.diff(norm.cdf(np.r_[c - h / 2, c[-1] + h / 2], 0, sig))  # library at s = 1, native cells
        rows = np.asarray(precompute_design_matrix(v / s, err / s, c, bin_width=h))
        stretch.append(-2 * np.sum(np.log(rows @ m / h)) + 2 * v.size * np.log(s))
        analytic.append(-2 * np.sum(norm.logpdf(v, 0, np.hypot(s * sig, err))))
    stretch, analytic = np.array(stretch), np.array(analytic)
    assert abs(S[np.argmin(stretch)] - S[np.argmin(analytic)]) <= 0.005
    assert np.max(np.abs(stretch - analytic)) < 1.0, np.max(np.abs(stretch - analytic))


def test_jacobian_error_would_be_caught():
    """T2d: the same setup without the Jacobian (or with the wrong sign) misses the minimum by far: the test above
    is sensitive to it."""
    sig, h = 15.0, 1.0
    v = rng.normal(0, sig * 0.95, 5000)
    err = np.full(v.size, 1.0)
    c = np.arange(-120.0, 120.1, h)
    m = np.diff(norm.cdf(np.r_[c - h / 2, c[-1] + h / 2], 0, sig))
    S = np.linspace(0.88, 1.03, 31)
    none, wrong = [], []
    for s in S:
        base = -2 * np.sum(np.log(np.asarray(precompute_design_matrix(v / s, err / s, c, bin_width=h)) @ m / h))
        none.append(base)
        wrong.append(base - 2 * v.size * np.log(s))
    assert abs(S[np.argmin(none)] - 0.95) > 0.02 and abs(S[np.argmin(wrong)] - 0.95) > 0.02


def test_aliasing_sharpened_stretch_close_to_fresh_library():
    """T3: production-like axis (HST errors ~1.5 km/s, 7.2 km/s cells, sharpened rows), fixed smooth model: the
    stretch's detrended noise and trend over a Υ scan are within 2x and 60 of a freshly integrated library's,
    and its minimum within 0.005 of the exact one. (ml_stretch_alias.py at 610k stars: rms 10.0 vs 5.6.)"""
    import subprocess
    import sys
    from pathlib import Path
    here = Path(__file__).parent
    out = subprocess.run([sys.executable, str(here / "ml_stretch_alias.py"), "150000"], cwd=here,
                         capture_output=True, text=True, check=True).stdout
    rows = {ln.split()[0]: [float(x) for x in ln.split()[1:]] for ln in out.splitlines()
            if ln.split() and ln.split()[0] in ("fresh_sh", "stretch_sh")}
    f, st = rows["fresh_sh"], rows["stretch_sh"]
    # the stretch's noise does not shrink with N (9.6 at 150k, 10.0 at 610k: a few window-edge stars and the
    # fast, precise tail stars, whose cells move under the stretch), so the bound is absolute: ~0.6% in Υ at HST
    assert st[1] < 15, out
    assert abs(st[3]) < abs(f[3]) + 60, out
    am = {ln.split(":")[0]: float(ln.split("=")[1]) for ln in out.splitlines() if ln.endswith(tuple("0123456789")) and "_sh: argmin s =" in ln}
    exact = float(out.split("exact: argmin s = ")[1].split()[0])
    assert abs(am["stretch_sh"] - exact) <= 0.005, out


@pytest.fixture(scope="module")
def hst_subset(tmp_path_factory):
    """The first 6 apertures of the real HST dump (raw pm1, pm2, cov, grid; h0)."""
    import pickle
    from pathlib import Path
    src = Path(__file__).parent / "work" / "hst_add_dump.pkl"
    if not src.exists():
        pytest.skip("work/hst_add_dump.pkl not present")
    d = pickle.load(open(src, "rb"))
    p = tmp_path_factory.mktemp("hst") / "sub.pkl"
    pickle.dump({"bins": d["bins"][:6], "h0": d["h0"][:6], "relative": d["relative"]}, open(p, "wb"))
    return p


def _build(pkl, out, *extra):
    import subprocess
    import sys
    from pathlib import Path
    here = Path(__file__).parent
    subprocess.run([sys.executable, str(here / "c3_hst_terms.py"), "build", str(pkl), str(out), "sharpen", *extra],
                   cwd=here, check=True, capture_output=True)
    return dict(np.load(out))


def test_c3_build_stretch_1_is_the_production_build(hst_subset, tmp_path):
    """T4a: stretch=1 reproduces the unstretched build exactly (rows, h0, star counts)."""
    a = _build(hst_subset, tmp_path / "a.npz")
    b = _build(hst_subset, tmp_path / "b.npz", "stretch=1.0")
    for k in ("PM_star_indptr", "PM_star_cell", "PM_star_mass", "PM_star_aperture", "PM_lap_h0", "n_star_used"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.mark.parametrize("s", (0.95, 1.03))
def test_c3_build_stretch_freezes_stars_and_matches_direct_rows(hst_subset, tmp_path, s):
    """T4b: real HST stars. The stretched build keeps exactly the s = 1 stars, its rows equal the production 2D
    design matrix of pm / s, cov / s^2 (x-East flip, sharpened, sparsified), records the Jacobian 2 N d ln s, and
    and leaves h0 as production's (ml_stretch_pack.py makes the stretched one)."""
    import pickle
    from veldist.fitstat import sharpen_matrix
    a = _build(hst_subset, tmp_path / "a.npz")
    b = _build(hst_subset, tmp_path / "b.npz", f"stretch={s}")
    np.testing.assert_array_equal(a["n_star_used"], b["n_star_used"])
    np.testing.assert_array_equal(a["PM_star_aperture"], b["PM_star_aperture"])
    N = int(a["n_star_used"].sum())
    assert np.isclose(float(b["jacobian"]), 2 * N * 2 * np.log(s))
    d = pickle.load(open(hst_subset, "rb"))
    st = d["bins"][0]
    cov = st["cov"].copy()
    cov[:, 0, 1] *= -1
    cov[:, 1, 0] *= -1
    M1 = np.asarray(precompute_design_matrix_2d(st["pm1"], -st["pm2"], cov, st["grid"]), dtype=float)
    keep = M1.sum(1) >= 1e-3
    Ms = np.asarray(precompute_design_matrix_2d(st["pm1"] / s, -st["pm2"] / s, cov / s ** 2, st["grid"]),
                    dtype=float)[keep] @ sharpen_matrix(23)
    ptr, cell, mass = b["PM_star_indptr"], b["PM_star_cell"], b["PM_star_mass"]
    for j in range(int(b["n_star_used"][0])):
        row = np.zeros(23 * 23)
        row[cell[ptr[j]:ptr[j + 1]]] = mass[ptr[j]:ptr[j + 1]]
        big = np.abs(Ms[j]) >= 1e-6 * np.abs(Ms[j]).max()
        np.testing.assert_allclose(row[big], Ms[j][big], rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(a["PM_lap_h0"], b["PM_lap_h0"])


# ---- pack level: ml_stretch_export.py outputs in work/stretch (skip if not built) ----------------------------------
def _packs():
    from pathlib import Path
    root = Path(__file__).parent / "work" / "stretch"
    return root, sorted(p for p in root.glob("ml*") if (p / "xeast").exists())


def _centroids(z, pre, centres):
    ptr, cell, mass = z[f"{pre}_star_indptr"], z[f"{pre}_star_cell"], z[f"{pre}_star_mass"].astype(float)
    seg = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    tot = np.bincount(seg, mass)
    cs = [np.bincount(seg, mass * c[cell]) / tot for c in centres]
    sec = [np.bincount(seg, mass * c[cell] ** 2) / tot - m ** 2 for c, m in zip(centres, cs)]
    return cs, sec, tot


@pytest.mark.parametrize("f,pre", [("gaia_veldist_pm_2dhist.npz", "PM"), ("veldist_kinematics_stars.npz", "LOS")])
def test_export_s1_is_the_production_input_bit_for_bit(f, pre):
    """T5: the stretched exporter at s = 1 reproduces the production sidecar (stars_pack2 = the regrid input on the
    node, md5-checked 2026-11-23) in every array."""
    root, _ = _packs()
    a_p, b_p = root / "s1.000" / "xeast" / f, root.parent / "stars_pack2" / "xeast" / f
    if not (a_p.exists() and b_p.exists()):
        pytest.skip("packs not built")
    a, b = dict(np.load(a_p)), dict(np.load(b_p))
    for k in b:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.mark.parametrize("f,pre,dim", [("gaia_veldist_pm_2dhist.npz", "PM", 2), ("veldist_kinematics_stars.npz", "LOS", 1)])
def test_stretched_packs_freeze_stars_and_scale_each_star_by_1_over_s(f, pre, dim):
    """T6: every stretched pack keeps exactly the s = 1 stars per aperture, records 2 N d ln s, and each interior
    star's row centroid is its s = 1 centroid / s (and its row variance / s^2, up to the cell term) -- a check of
    the stretch on the real Gaia/MUSE stars that does not use the code that made it."""
    root, packs = _packs()
    if not packs:
        pytest.skip("no stretched packs")
    z1 = dict(np.load(root / "s1.000" / "xeast" / f))
    if pre == "PM":
        K, half = z1["PM_2dhist"].shape[1], float(z1["vxrange"])
        c = np.linspace(-half, half, K + 1)
        c = 0.5 * (c[1:] + c[:-1])
        G = [g.ravel() for g in np.meshgrid(c, c, indexing="ij")]
        h = c[1] - c[0]
    else:
        from astropy.table import Table
        meta = Table.read(root / "s1.000" / "xeast" / "veldist_kinematics.ecsv").meta
        K, dv = int(meta["nvbins"]), float(meta["dv"])
        G = [(np.arange(K) - (K - 1) / 2) * dv]
        h = dv
    c1, v1, t1 = _centroids(z1, pre, G)
    for p in packs:
        z = dict(np.load(p / "xeast" / f))
        s = float(z["stretch"])
        np.testing.assert_array_equal(z[f"{pre}_star_aperture"], z1[f"{pre}_star_aperture"])
        N = len(z1[f"{pre}_star_aperture"])
        assert np.isclose(float(z["jacobian"]), 2 * N * dim * np.log(s))
        cs, vs, ts = _centroids(z, pre, G)
        inner = np.ones(N, bool)
        for a in c1:
            inner &= np.abs(a) < 0.6 * np.abs(G[0]).max()
        inner &= (t1 > 0.99) & (ts > 0.99)  # kernel fully on the grid at both s
        # a row's centroid on cell centres is a binned mean: for kernels narrower than a cell it aliases (that is
        # ml_stretch_alias.py), so test the stars whose kernel spans >= ~2 cells per axis
        for a, b in zip(c1, cs):  # every interior star: aliasing averages out of the slope and the median
            assert abs(np.polyfit(a[inner], b[inner], 1)[0] * s - 1) < 1e-3
            assert np.median(np.abs(b[inner] - a[inner] / s)) < 0.02 * h
        broad = inner.copy()
        for v in v1:
            broad &= v >= (1.5 * h) ** 2
        if broad.sum() > 100:  # Gaia: per star, where the kernel spans >= ~2 cells (MUSE has none)
            for a, b in zip(c1, cs):
                np.testing.assert_allclose(b[broad], a[broad] / s, atol=0.03 * h)


def test_assembled_packs_reference_costs_the_same_at_every_s():
    """T7 (ml_stretch_pack.py outputs, work/stretch/*/replace): at s = 1 the assembled h0 is production's; at every s
    each set's h0 is positive and normalized, and the data reference costs the same as at
    s = 1 once the Jacobian is added: E0(s) + 2 N d ln s - E0(1) is small against the Jacobian itself (measured at
    ml 3.065: HST +106 of 40524, Gaia +4.7 of 4282, MUSE -0.3 of 827). A wrong sign or a missing factor would leave
    the full Jacobian. LVM: v, dv, sigma, dsigma divided by s, h3..h6 untouched."""
    from astropy.table import Table
    root, _ = _packs()
    base = root / "s1.000" / "final"
    reps = sorted(root.glob("ml*/replace/stretch_meta.npz"))
    if not (base / "stretch_meta.npz").exists() or not reps:
        pytest.skip("assembled packs not built")
    m1 = np.load(base / "stretch_meta.npz")
    prod = dict(np.load(root / "hst_prod.npz"))
    np.testing.assert_allclose(dict(np.load(base / "hst_veldist_pm_2dhist.npz"))["PM_lap_h0"], prod["PM_lap_h0"],
                               atol=1e-15)
    lvm1 = Table.read(root.parent / "stars_pack2" / "xeast" / "lvm_kins.ecsv")
    bound = {"hst": 300.0, "gaia": 30.0, "veldist": 15.0}
    for r in reps:
        m = np.load(r)
        s = float(m["stretch"])
        for k, lim in bound.items():
            d = float(m[f"e0_{k}"]) + float(m[f"jac_{k}"]) - float(m1[f"e0_{k}"])
            assert abs(d) < lim and abs(d) < 0.01 * abs(float(m[f"jac_{k}"])), (r.parent.parent.name, k, d)
        for f, pre in (("hst_veldist_pm_2dhist.npz", "PM"), ("gaia_veldist_pm_2dhist.npz", "PM"),
                       ("veldist_kinematics_stars.npz", "LOS")):
            h = np.load(r.parent / f)[f"{pre}_lap_h0"]
            assert np.all(h > 0) and np.allclose(h.sum(axis=tuple(range(1, h.ndim))), 1)
        lvm = Table.read(r.parent / "lvm_kins.ecsv")
        for c in ("v", "dv", "sigma", "dsigma"):
            np.testing.assert_allclose(lvm[c], lvm1[c] / s, rtol=1e-12)
        for c in ("h3", "dh3", "h4", "dh4"):
            np.testing.assert_array_equal(lvm[c], lvm1[c])
