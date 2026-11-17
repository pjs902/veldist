"""Mock test of the exact-likelihood floor rules for sharpened rows: do they bias the fitted dispersion?

usage: python floor_mock.py OUT.npz [NMOCK]
HST-like mocks as in paper figure m6 (N = 430 stars, errors 1.5 km/s, sigma 18/16 km/s, K = 23 on +-82.8 km/s),
plus a fraction of outlier stars uniform in the outer ring of the grid (|v| beyond 4 sigma on either axis). The
model is a Gaussian of dispersions (s*18, s*16) in exact cell masses, TRUNCATED to exact zeros where its cell mass is
below 1e-6 of the peak (orbit models have exactly empty cells; the real-data p_M = 0 stars sit there). s is fitted
by minimising -2 ln L with p per star:
  plain        M h                                    (floored at 1e-300, as stat_exact)
  sharp_raw    M S h                                  (floored at 1e-300)
  f{f}_e{eps}  max(M S h, f M h) + eps * hbar,   hbar = sum(h) / K^2
and the extended term 2 N sum(h) (aperture mass 1). Reported: mean bias of sigma_1 and its standard error.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1 import HST, gmass  # noqa: E402

from veldist.fitstat import sharpen_matrix  # noqa: E402
from veldist.veldist2d import KinematicSolver2D, precompute_design_matrix_2d  # noqa: E402

S1, S2, N, K = 18.0, 16.0, 430, 23
NMOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 200
OUTF = (0.0, 0.003, 0.01)
RULES = [("plain", None, None), ("sharp_raw", None, None)] + [
    (f"f{f:g}_e{e:g}", f, e) for f in (0.0, 0.1) for e in (0.0, 1e-6, 1e-4, 1e-3, 1e-2)]
rng = np.random.default_rng(11)
sol = KinematicSolver2D()
sol.setup_grid(center=(0.0, 0.0), width=(2 * HST["half"],) * 2, n_bins=K)
grid = sol.grid
e = np.asarray(grid["edges_x"], float)
Sh = sharpen_matrix(K)


def model(s):
    h = gmass(e, S1 * s, S2 * s)
    return np.where(h >= 1e-6 * h.max(), h, 0.0)


def m2lnl(h, M, MS, f, eps):
    pM = M @ h
    if f is None:
        p = pM if eps == "plain" else MS @ h
        p = np.maximum(p, 1e-300)
    else:
        p = np.maximum(MS @ h, f * pM) + eps * h.sum() / K ** 2
        p = np.where(p > 0, p, 1e-300)
    return 2.0 * (-np.log(p).sum() + len(M) * h.sum())


res = {}
for of in OUTF:
    bias = {r[0]: [] for r in RULES}
    for _ in range(NMOCK):
        v = rng.normal(0, 1, (N, 2)) * [S1, S2] + rng.normal(0, HST["err"], (N, 2))
        n_out = rng.binomial(N, of)
        if n_out:
            u = rng.uniform(-HST["half"], HST["half"], (20 * n_out + 50, 2))
            u = u[(np.abs(u[:, 0]) > 4 * S1) | (np.abs(u[:, 1]) > 4 * S2)][:n_out]
            v[:len(u)] = u
        cov = np.tile(np.eye(2) * HST["err"] ** 2, (N, 1, 1))
        M = np.asarray(precompute_design_matrix_2d(v[:, 0], v[:, 1], cov, grid), float)
        M = M[M.sum(1) >= 1e-3]
        MS = M @ Sh
        for name, f, eps in RULES:
            tag = "plain" if name == "plain" else eps
            r = minimize_scalar(lambda s: m2lnl(model(s), M, MS, f, tag), bounds=(0.7, 1.3), method="bounded",
                                options=dict(xatol=1e-5))
            bias[name].append(S1 * r.x - S1)
    for name in bias:
        b = np.asarray(bias[name])
        res[f"{name}|out{of:g}"] = b
        print(f"outliers {of:5.3f}  {name:14s} bias {b.mean():+.3f} +- {b.std(ddof=1) / np.sqrt(len(b)):.3f} "
              f"(scatter {b.std(ddof=1):.3f})", flush=True)
np.savez(sys.argv[1], **res)
