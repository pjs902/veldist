"""sigma-hat bias and calibration per statistic, addback setting (mean known = 0).

For each seed: fit veldist, then scan model Gaussian sigma; sigma_hat = argmin,
claimed 1-sigma error from the Delta=1 width of a parabola fit. Truth sigma=17.
Calibrated statistic: std(sigma_hat across seeds) ~= median claimed error.
"""
import sys, numpy as np
from stat_compare import KinematicSolver2D, gauss_mass, SIG, ERR, N, K, W
from sklearn.covariance import LedoitWolf

grid = np.linspace(15.0, 19.0, 81)
STATS = ["diag_floor", "diag_raw", "fullcov", "exact", "laplace"]
res = {k: [] for k in STATS}
for seed in map(int, sys.argv[1:]):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, SIG, (N, 2)) + rng.normal(0, ERR, (N, 2))
    s = KinematicSolver2D(); s.setup_grid(center=(0.0, 0.0), width=(W, W), n_bins=K)
    s.add_data(v[:, 0], v[:, 1], np.tile(np.eye(2) * ERR**2, (N, 1, 1)))
    s.run(num_samples=3000, seed=seed); s.clip_uncertainties()
    M = np.asarray(s.matrix); e = np.asarray(s.grid["edges_x"])
    dr = np.asarray(s.samples["intrinsic_pdf"], float)
    med, sig = s.clipped_samples["pdf_median"], s.clipped_samples["pdf_uncertainty"]
    sc = med.sum(); d, sd = med / sc, sig / sc
    raw = np.maximum((np.percentile(dr, 84, 0) - np.percentile(dr, 16, 0)) / 2 / sc, 1e-12)
    P = np.linalg.pinv(LedoitWolf().fit(dr).covariance_, rcond=1e-10, hermitian=True)
    pm = dr.mean(0); p0 = M @ pm; g = -(M.T @ (1 / p0)); F = M.T @ ((1 / p0**2)[:, None] * M)
    curves = {k: [] for k in STATS}
    for sgm in grid:
        h = gauss_mass(e, sgm); h /= h.sum(); r = h - d
        curves["diag_floor"].append(np.sum((r / sd) ** 2))
        curves["diag_raw"].append(np.sum((r / raw) ** 2))
        curves["fullcov"].append(r @ P @ r)
        curves["exact"].append(-2 * np.sum(np.log(M @ h)))
        curves["laplace"].append(2 * (g @ (h - pm) + 0.5 * (h - pm) @ F @ (h - pm)))
    for k in STATS:
        c = np.array(curves[k]); i = np.argmin(c)
        lo, hi = max(i - 6, 0), min(i + 7, len(grid))
        a = np.polyfit(grid[lo:hi], c[lo:hi], 2)[0]
        res[k].append((grid[i], 1 / np.sqrt(max(a, 1e-12))))
print(f"{'stat':12s}{'mean bias':>11s}{'scatter':>9s}{'claimed':>9s}{'ratio':>7s}")
for k in STATS:
    x = np.array(res[k]); b = x[:, 0] - SIG
    print(f"{k:12s}{b.mean():11.3f}{b.std(ddof=1):9.3f}{np.median(x[:,1]):9.3f}{b.std(ddof=1)/np.median(x[:,1]):7.1f}")
