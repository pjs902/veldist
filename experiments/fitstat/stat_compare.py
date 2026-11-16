"""Compare fit statistics for one synthetic HST-like aperture.

N=430 stars, isotropic Gaussian sigma=17 km/s, per-axis err 1.5 km/s, the
production HST grid (K=23, width 165.61). Scores model histograms against the
data with: dynamite's diagonal chi2 (floored / unfloored), a full-covariance
chi2 from the posterior draws, the exact per-star -2 log L, and its Laplace
(quadratic) expansion about the posterior mean.
"""
import sys
import numpy as np
from scipy.special import erf
from veldist.veldist2d import KinematicSolver2D

SIG, ERR, N, K, W = 17.0, 1.5, 430, 23, 165.61277588006868


def gauss_mass(edges, sx, sy=None, mx=0.0, my=0.0):
    sy = sx if sy is None else sy
    cx = 0.5 * (1 + erf((edges - mx) / (np.sqrt(2) * sx)))
    cy = 0.5 * (1 + erf((edges - my) / (np.sqrt(2) * sy)))
    return np.outer(np.diff(cx), np.diff(cy)).ravel()


def ring_mass(edges, v0, dv):
    c = 0.5 * (edges[1:] + edges[:-1])
    vx, vy = np.meshgrid(c, c, indexing="ij")
    r = np.hypot(vx, vy).ravel()
    m = np.exp(-0.5 * ((r - v0) / dv) ** 2)
    return m / m.sum()


def run(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, SIG, (N, 2)) + rng.normal(0, ERR, (N, 2))
    cov = np.tile(np.eye(2) * ERR**2, (N, 1, 1))
    s = KinematicSolver2D()
    s.setup_grid(center=(0.0, 0.0), width=(W, W), n_bins=K)
    s.add_data(v[:, 0], v[:, 1], cov)
    s.run(num_samples=3000, seed=seed)
    s.clip_uncertainties()
    M = np.asarray(s.matrix)
    e = np.asarray(s.grid["edges_x"])
    draws = np.asarray(s.samples["intrinsic_pdf"], dtype=float)

    # dynamite's data vector: median and sigma, both divided by the median's sum
    med, sig = s.clipped_samples["pdf_median"], s.clipped_samples["pdf_uncertainty"]
    scale = med.sum()
    d, sd = med / scale, sig / scale
    raw = (np.percentile(draws, 84, 0) - np.percentile(draws, 16, 0)) / 2 / scale
    at_floor = np.isclose(sig, sig.min()).mean()

    # full covariance: shrink toward its diagonal, invert on the sum-to-1 subspace
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(draws)
    C = lw.covariance_
    P = np.linalg.pinv(C, rcond=1e-10, hermitian=True)
    pm = draws.mean(0)

    # likelihood-only (Fisher) at the posterior mean
    p0 = M @ pm
    g = -(M.T @ (1 / p0))
    F = M.T @ ((1 / p0**2)[:, None] * M)

    truth = gauss_mass(e, SIG)
    models = {
        "truth": truth,
        "sigma+5%": gauss_mass(e, SIG * 1.05),
        "sigma-5%": gauss_mass(e, SIG * 0.95),
        "tail 1% @60": 0.99 * truth + 0.01 * ring_mass(e, 60, 4),
        "tail 0.3% @60": 0.997 * truth + 0.003 * ring_mass(e, 60, 4),
        "aniso 1.1": gauss_mass(e, SIG * 1.05, SIG / 1.05),
    }
    out = {}
    for k, h in models.items():
        h = h / h.sum()
        r = h - d
        out[k] = dict(
            diag_floor=np.sum((r / sd) ** 2),
            diag_raw=np.sum((r / np.maximum(raw, 1e-12)) ** 2),
            fullcov=r @ P @ r,
            exact=-2 * np.sum(np.log(np.maximum(M @ h, 1e-300))),
            laplace=2 * (g @ (h - pm) + 0.5 * (h - pm) @ F @ (h - pm)),
        )
    base = out["truth"]
    rel = {k: {m: v[m] - base[m] for m in v} for k, v in out.items() if k != "truth"}

    # covariance structure diagnostics
    sdp = np.sqrt(np.diag(C))
    corr = C / np.outer(sdp, sdp)
    idx = np.arange(K * K).reshape(K, K)
    nn = np.concatenate([corr[idx[:-1, :].ravel(), idx[1:, :].ravel()],
                         corr[idx[:, :-1].ravel(), idx[:, 1:].ravel()]])
    Fd = np.sqrt(np.abs(np.diag(F)))
    Fcorr = F / np.outer(Fd, Fd)
    Fnn = np.concatenate([Fcorr[idx[:-1, :].ravel(), idx[1:, :].ravel()],
                          Fcorr[idx[:, :-1].ravel(), idx[:, 1:].ravel()]])
    # cells with real data: expected count > 1
    occ = (M.sum(0) > 1)
    # posterior precision vs likelihood precision, per cell (diag of the
    # posterior precision restricted to the cell; prior share = 1 - F/Ppost)
    share = np.diag(F)[occ] / np.diag(P)[occ]
    tail = ~occ
    return dict(rel=rel, at_floor=at_floor, nn_corr_med=np.median(nn),
                nn_corr_occ=np.median(nn), F_nn_med=np.median(np.abs(Fnn)),
                lik_share_occ=np.median(share), n_occ=occ.sum(),
                post_sd_over_poisson=np.median((sdp * 1)[occ] / np.sqrt(pm[occ] * (1 - pm[occ]) / N)),
                tail_post_sd_vs_floor=np.median(sdp[tail] / sd[tail]))


if __name__ == "__main__":
    for seed in map(int, sys.argv[1:] or ["1"]):
        R = run(seed)
        print(f"\n=== seed {seed}")
        for k in ["at_floor", "n_occ", "nn_corr_med", "F_nn_med", "lik_share_occ",
                  "post_sd_over_poisson", "tail_post_sd_vs_floor"]:
            print(f"{k:24s} {R[k]:.3f}")
        print(f"{'model (Δ vs truth)':16s}" + "".join(f"{m:>12s}" for m in
              ["diag_floor", "diag_raw", "fullcov", "exact", "laplace"]))
        for k, v in R["rel"].items():
            print(f"{k:16s}" + "".join(f"{v[m]:12.1f}" for m in
                  ["diag_floor", "diag_raw", "fullcov", "exact", "laplace"]))
