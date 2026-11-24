"""Analytic test bed for the fast-star term (docs/fast-star-term.md): an isotropic Eddington DF for a tracer
around a point-mass BH inside a Plummer cluster, used both to draw mock stars and to build the model's
annular (v2D, |v_los|) mass tables, i.e. what the orbit library will provide for real models.

Units: pc, km/s, Msun. The tracer is massless (its density need not match the potential).
"""
import numpy as np
from scipy.interpolate import interp1d

G = 4.30091e-3
ARCSEC_PC = 5494.0 * np.pi / 180 / 3600          # pc per arcsec at the model distance


class Cluster:
    """Potential psi(r) = G M_bh / r + G M_cl / sqrt(r^2 + b^2); tracer rho ~ (1 + r^2/b^2)^-5/2 (1 + (r_c/r)^gamma).

    v_esc,cl(0) = sqrt(2 G M_cl / b) = 62 km/s by default. The cusp (gamma >= 1/2, r_c >= r_h) keeps the
    Eddington f(E) positive around the BH (a cored tracer around a BH has no isotropic DF).
    """

    def __init__(self, m_bh, b=2.4, vesc0=62.0, gamma=1.0, r_c=0.3, n=4000):
        self.m_bh, self.b, self.gamma, self.r_c = m_bh, b, gamma, r_c
        self.m_cl = vesc0 ** 2 / 2 * b / G
        r = np.logspace(-5, 3, n)
        psi, rho = self.psi(r), self.rho(r)
        # Eddington: f(eps) = 1/(sqrt(8) pi^2) dF/deps, F(eps) = 2 int_0^sqrt(eps) drho/dpsi(eps - t^2) dt
        lpsi = np.log(psi[::-1])
        drho_dpsi = np.gradient(rho[::-1], psi[::-1])
        g = interp1d(lpsi, drho_dpsi, bounds_error=False, fill_value=(0.0, drho_dpsi[-1]))
        eps = np.logspace(np.log10(psi.min()) + 0.01, np.log10(psi.max()) - 0.01, 600)
        t = np.linspace(0, 1, 2001)[None, :]
        x = eps[:, None] - (t * np.sqrt(eps[:, None])) ** 2
        F = 2 * np.sqrt(eps) * np.trapezoid(g(np.log(np.maximum(x, 1e-300))) * (x > 0), t, axis=1)
        f = np.gradient(F, eps) / (np.sqrt(8) * np.pi ** 2)
        assert np.all(f[5:-5] > 0), "Eddington f(E) < 0: tracer cusp too shallow for this BH"
        self._lf = interp1d(np.log(eps), np.log(np.maximum(f, 1e-300)), bounds_error=False,
                            fill_value=(np.log(max(f[0], 1e-300)), -np.inf))

    def psi(self, r):
        return G * self.m_bh / r + G * self.m_cl / np.sqrt(r ** 2 + self.b ** 2)

    def rho(self, r):
        return (1 + (r / self.b) ** 2) ** -2.5 * (1 + (self.r_c / r) ** self.gamma)

    def f(self, eps):
        out = np.zeros_like(eps)
        m = eps > 0
        out[m] = np.exp(self._lf(np.log(eps[m])))
        return out

    def sample_speed(self, r, rng, n_grid=400, chunk=4000):
        r = np.asarray(r, float)
        if len(r) > chunk:
            return np.concatenate([self.sample_speed(r[s:s + chunk], rng, n_grid) for s in range(0, len(r), chunk)])
        return self._sample_speed(r, rng, n_grid)

    def _sample_speed(self, r, rng, n_grid):
        """3D speed at each radius r (array) from p(v) ~ v^2 f(psi - v^2/2)."""
        r = np.asarray(r, float)
        ve = np.sqrt(2 * self.psi(r))
        u = np.linspace(0, 1, n_grid)[None, :]
        v = u * ve[:, None]
        p = v ** 2 * self.f(self.psi(r)[:, None] - v ** 2 / 2)
        c = np.cumsum(p, axis=1)
        c /= c[:, -1:]
        x = rng.random(len(r))[:, None]
        k = np.clip((c < x).sum(axis=1), 1, n_grid - 1)
        c0, c1 = c[np.arange(len(r)), k - 1], c[np.arange(len(r)), k]
        w = (x[:, 0] - c0) / np.where(c1 > c0, c1 - c0, 1)
        return (u[0, k - 1] + w * (u[0, k] - u[0, k - 1])) * ve

    def sample_los(self, R_pc, rng, zmax=60.0, n_grid=800, chunk=2000):
        """Line-of-sight offset z for stars at projected R, from rho(sqrt(R^2 + z^2)) (chunked: memory)."""
        zg = np.concatenate([[0], np.logspace(-5, np.log10(zmax), n_grid)])
        z = np.empty(len(R_pc))
        for s in range(0, len(R_pc), chunk):
            Rc = R_pc[s:s + chunk]
            p = self.rho(np.hypot(Rc[:, None], zg[None, :]))
            c = np.concatenate([np.zeros((len(Rc), 1)), np.cumsum(0.5 * (p[:, 1:] + p[:, :-1]) * np.diff(zg), 1)], 1)
            c /= c[:, -1:]
            x = rng.random(len(Rc))[:, None]
            k = np.clip((c < x).sum(1), 1, len(zg) - 1)
            i = np.arange(len(Rc))
            c0, c1 = c[i, k - 1], c[i, k]
            z[s:s + chunk] = zg[k - 1] + (x[:, 0] - c0) / np.where(c1 > c0, c1 - c0, 1) * (zg[k] - zg[k - 1])
        return z * rng.choice([-1, 1], len(R_pc))

    def sample_at(self, R_arcsec, rng):
        """(v2D, v_los, vx, vy) in km/s for stars at projected radii R_arcsec (isotropic velocities)."""
        Rp = np.asarray(R_arcsec, float) * ARCSEC_PC
        r = np.hypot(Rp, self.sample_los(Rp, rng))
        v = self.sample_speed(r, rng)
        mu, ph = rng.uniform(-1, 1, len(r)), rng.uniform(0, 2 * np.pi, len(r))
        st = np.sqrt(1 - mu ** 2)
        vx, vy, vz = v * st * np.cos(ph), v * st * np.sin(ph), v * mu
        return np.hypot(vx, vy), vz, vx, vy

    def sigma(self, R_arcsec):
        """Surface density (arbitrary units) at projected R."""
        Rp = np.atleast_1d(np.asarray(R_arcsec, float)) * ARCSEC_PC
        zg = np.concatenate([[0], np.logspace(-5, np.log10(60.0), 600)])
        return np.trapezoid(self.rho(np.hypot(Rp[:, None], zg[None, :])), zg, axis=1)

    def projected_R(self, n, rmax_arcsec, rng, n_grid=2000):
        """Projected radii of n tracer stars inside rmax, from the surface density Sigma(R)."""
        Rg = np.linspace(1e-4, rmax_arcsec, n_grid)
        c = np.cumsum(self.sigma(Rg) * Rg)
        c /= c[-1]
        return np.interp(rng.random(n), c, Rg)


def model_table(cl, R_edges, v_edges, w_edges, n_per=20000, seed=0):
    """Mass fractions P[a, i, j] of (v2D, |v_los|) cells per annulus (each annulus sums to 1), by Monte Carlo:
    the analogue of the orbit library's annular histograms (with its sampling noise)."""
    rng = np.random.default_rng(seed)
    P = np.zeros((len(R_edges) - 1, len(v_edges) - 1, len(w_edges) - 1))
    for a in range(len(R_edges) - 1):
        R = np.sqrt(rng.uniform(R_edges[a] ** 2, R_edges[a + 1] ** 2, n_per))   # uniform in area, then
        wt = cl.sigma(R)                                                          # weighted by Sigma(R)
        v2, vz, _, _ = cl.sample_at(R, rng)
        H, _, _ = np.histogram2d(v2, np.abs(vz), [v_edges, w_edges], weights=wt)
        P[a] = H / wt.sum()
    return P


CUT = 2.41 * 4.74047 * 5.494          # Haeberle's 2.41 mas/yr at the model distance [km/s]
R_EDGES = np.array([0, .25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6.5, 8, 10, 12.5])
V_EDGES = np.arange(0, 405, 5.0)
W_EDGES = np.arange(0, 405, 5.0)


def draw_observed(cl, real, rng, bh=(0.0, 0.0), rmax=10.0, n_rv=102, rv_box=1.5, erv_med=5.8):
    """Mock of the real data set: len(real['x']) quality stars inside rmax of the frame origin (AvdM10), the
    tracer centred on the BH at `bh`; PM errors resampled from the real sample; per-component Gaussian PM
    errors; Haeberle's selection; contaminants at the real rate with the real background v2D distribution;
    an RV subsample of n_rv members inside a (2 rv_box)^2 box (NIRSpec-like), lognormal errors."""
    n = len(real["dx"])
    R = cl.projected_R(4 * n, rmax + np.hypot(*bh) + 0.5, rng)
    ph = rng.uniform(0, 2 * np.pi, len(R))
    x, y = bh[0] + R * np.cos(ph), bh[1] + R * np.sin(ph)
    keep = np.nonzero(np.hypot(x, y) < rmax)[0][:n]
    assert len(keep) == n
    x, y, Rb = x[keep], y[keep], R[keep]
    e = rng.choice(np.asarray(real["e2d"]), n)
    _, vz, vx, vy = cl.sample_at(Rb, rng)
    v2 = np.hypot(vx + e * rng.standard_normal(n), vy + e * rng.standard_normal(n))
    cont = rng.random(n) < float(real["bg_per_star"])
    v2[cont] = rng.choice(np.asarray(real["bg_v2d"]), cont.sum())
    sel = v2 - 3 * e > CUT
    vlos, elos = np.full(n, np.nan), np.full(n, np.nan)
    box = np.nonzero((np.abs(x) < rv_box) & (np.abs(y) < rv_box) & ~cont)[0]
    rvs = rng.choice(box, min(n_rv, len(box)), replace=False)
    elos[rvs] = erv_med * np.exp(0.45 * rng.standard_normal(len(rvs)))
    vlos[rvs] = vz[rvs] + elos[rvs] * rng.standard_normal(len(rvs))
    return dict(x=x, y=y, v2d=v2, e2d=e, selected=sel, vlos=vlos, elos=elos, contaminant=cont)


def bg_density(real, edges=V_EDGES):
    """(eta, edges, density) of contaminant v2D from the real 20-90" selected stars."""
    h, _ = np.histogram(np.asarray(real["bg_v2d"]), edges, density=True)
    return float(real["bg_per_star"]), edges, h
