"""Fast-star likelihood for the IMBH mass (docs/fast-star-term.md, sections 2-4).

A model enters only as annular mass tables P[a, i, j] over (v2D, |v_los|) cells: one table per model, from the
orbit library for real models (dfmock.model_table for tests). Conditional on the positions of a velocity-blind
quality sample:

  term (a)  Bernoulli per quality star: selected (v2D_obs - 3 e > cut) or not, with the selected stars'
            observed v2D density; contaminant mixture (rate eta per star, v2D density g_bg above the cut).
  term (b)  RV stars: ln p(v_los,obs | v2D,obs, R), the chain-rule factor of the joint, both from the same table.

  chi2 = -2 ln L, with the IMBH position marginalised over a Gaussian prior on a grid: the table's annuli are
  centred on the BH, so each trial position only changes each star's annulus.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import rice


def _sub(edges, ns):
    """Sub-sample points (n_cells, ns) uniformly inside each cell, for the in-cell integrals."""
    lo, hi = edges[:-1, None], edges[1:, None]
    return lo + (hi - lo) * (np.arange(ns)[None, :] + 0.5) / ns


def v2d_kernel(v_obs, e, v_edges, ns=8):
    """K[k, i] = mean over cell i of the Rice density of v2D_obs given true v2D (2D Gaussian PM error e)."""
    vs = _sub(v_edges, ns)                                             # (nv, ns)
    v_obs, e = np.asarray(v_obs, float)[:, None, None], np.asarray(e, float)[:, None, None]
    return rice.pdf(v_obs / e, vs[None] / e) .mean(-1) / e[..., 0]


def sel_prob(cut, e, v_edges, ns=8):
    """S[k, i] = mean over cell i of P(v2D_obs > cut_k) given true v2D, cut_k = cut + 3 e_k (Haeberle's 3 sigma)."""
    vs = _sub(v_edges, ns)
    c = (cut + 3 * np.asarray(e, float))[:, None, None]
    e = np.asarray(e, float)[:, None, None]
    return rice.sf(c / e, vs[None] / e).mean(-1)


def vlos_kernel(vz_obs, ez, w_edges, ns=8):
    """Kz[k, j] = mean over |v_los| cell j of the density of v_los,obs, the model being symmetric in sign."""
    ws = _sub(w_edges, ns)
    x, s = np.asarray(vz_obs, float)[:, None, None], np.asarray(ez, float)[:, None, None]
    g = lambda m: np.exp(-0.5 * ((x - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
    return 0.5 * (g(ws[None]) + g(-ws[None])).mean(-1)


class FastStarTerm:
    """Precomputes everything that does not depend on the model table or the BH position.

    stars: dict with x, y [arcsec], v2d [km/s], e2d [km/s, the error of v2D], selected [bool]; optional
    vlos, elos [km/s, relative to systemic, NaN where missing]. bg: (eta per quality star, v2D edges, density
    of contaminant v2D above the cut on those edges).
    """

    def __init__(self, stars, R_edges, v_edges, w_edges, cut, bg=None):
        self.x, self.y = np.asarray(stars["x"], float), np.asarray(stars["y"], float)
        self.R_edges, self.cut = np.asarray(R_edges, float), cut
        sel = np.asarray(stars["selected"], bool)
        e = np.asarray(stars["e2d"], float)
        self.sel = sel
        self.S = sel_prob(cut, e, v_edges)                             # (n, nv)
        self.K = np.zeros_like(self.S)
        self.K[sel] = v2d_kernel(np.asarray(stars["v2d"])[sel], e[sel], v_edges)
        self.eta, self.gbg = 0.0, np.zeros(len(e))
        if bg is not None:
            self.eta, be, dens = bg
            self.gbg[sel] = np.interp(np.asarray(stars["v2d"])[sel], 0.5 * (be[1:] + be[:-1]), dens, left=0, right=0)
        vz = np.asarray(stars.get("vlos", np.full(len(e), np.nan)), float)
        self.rv = np.isfinite(vz)
        if self.rv.any():
            self.Kv = v2d_kernel(np.asarray(stars["v2d"])[self.rv], e[self.rv], v_edges)        # (m, nv)
            self.Kz = vlos_kernel(vz[self.rv], np.asarray(stars["elos"])[self.rv], w_edges)   # (m, nw)

    def annulus(self, dx=0.0, dy=0.0):
        R = np.hypot(self.x - dx, self.y - dy)
        a = np.searchsorted(self.R_edges, R, side="right") - 1
        return a, (a >= 0) & (a < len(self.R_edges) - 1)

    def lnL(self, P, dx=0.0, dy=0.0, terms=("a", "b")):
        """ln L at one BH position for the model table P[a, i, j] (each annulus sums to 1)."""
        a, inside = self.annulus(dx, dy)
        assert inside.all(), "a quality star lies outside the table's annuli"
        P2 = P.sum(-1)                                                  # (na, nv): v2D marginal per annulus
        out = 0.0
        if "a" in terms:
            psel = np.einsum("ki,ki->k", P2[a], self.S)                 # P(selected | member)
            pnot = (1 - self.eta) * (1 - psel)                          # contaminants are selected by construction
            out += np.log(np.maximum(pnot[~self.sel], 1e-300)).sum()
            dens = (1 - self.eta) * np.einsum("ki,ki->k", P2[a][self.sel], self.K[self.sel]) + self.eta * self.gbg[self.sel]
            out += np.log(np.maximum(dens, 1e-300)).sum()
        if "b" in terms and self.rv.any():
            ar = a[self.rv]
            num = np.einsum("kij,ki,kj->k", P[ar], self.Kv, self.Kz)
            den = np.einsum("ki,ki->k", P2[ar], self.Kv)
            out += np.log(np.maximum(num, 1e-300) / np.maximum(den, 1e-300)).sum()
        return out

    def chi2(self, P, pos_sigma=None, half=1.5, step=0.1, terms=("a", "b")):
        """-2 ln L; with pos_sigma [arcsec], marginalised over a Gaussian prior on the BH position (grid)."""
        if pos_sigma is None:
            return -2 * self.lnL(P, terms=terms)
        g = np.arange(-half, half + step / 2, step)
        X, Y = np.meshgrid(g, g)
        lp = -0.5 * (X ** 2 + Y ** 2) / pos_sigma ** 2
        lp -= logsumexp(lp)
        ll = np.array([self.lnL(P, dx, dy, terms) for dx, dy in zip(X.ravel(), Y.ravel())])
        return -2 * logsumexp(ll + lp.ravel())
