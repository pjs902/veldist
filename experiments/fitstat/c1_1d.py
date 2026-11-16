"""C1 in 1D: the c1c.py comparison for MUSE-like line-of-sight data.

Truth: a Gaussian LOSVD, sigma S0 (mean M0; 0 here), N stars with log-normal
per-star errors. MUSE numbers are veldist.calibration.OMEGACAT_MEASURED (the
replay of the production MUSE notebook, 2026-09-01): median error 4.0 km/s,
log-sigma 0.62, 152 stars per bin, sigma 10.4-20.0, rotation span 17.9, grid
width = 2*4*sigma_max + span = 178 km/s, bin width = 2 x median error = 8 km/s,
i.e. K = 22 (cell ~ 8.1 km/s). Truth sigma 15 km/s is mid-range.

Model family: Gaussian cell masses (s, mu) on the closed grid, plus a free mean
m in relative mode ("addback" fixes m = 0, as c1.py). Versions per mode, as c1c.py:
diag, exact, exact_S, lap, lap_S, lap_S_it1 (one re-expansion at lap_S's fit).

usage: python c1_1d.py run SEED [SEED ...]    (one @@-prefixed JSON line per seed)
       python c1_1d.py summarise OUT.jsonl
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
from c1 import fit  # noqa: E402

from veldist.fitstat import (laplace_rows, laplace_terms, sharpen_matrix, stat_diag,  # noqa: E402
                             stat_exact, stat_laplace)
from veldist.veldist import KinematicSolver, _cell_mass_correction, precompute_design_matrix  # noqa: E402

K, WIDTH = 22, 178.0
N, S0, ERR_MED, ERR_LOGSIG = 152, 15.0, 4.0, 0.62


def gmass(e, s, m=0.0):
    h = np.diff(ndtr((e - m) / s))
    return h / h.sum()


def family(e, relative):
    h = lambda t: t[1] * gmass(e, t[0], t[2] if relative else 0.0)  # noqa: E731
    return [S0, 1.0] + ([0.0] if relative else []), [(5, 30), (0.5, 1.5)] + ([(-8, 8)] if relative else []), h


def lap(M, h0, n):
    F, g = laplace_terms(M, h0, n, 1.0)
    Lt, tg = laplace_rows(F, g, h0, n_stars=n)
    return lambda h: stat_laplace(h, Lt, tg, h0=h0)


def run(seed):
    t0 = time.time()
    rng = np.random.default_rng([seed, 1])
    err = np.exp(rng.normal(np.log(ERR_MED), ERR_LOGSIG, N))
    v = rng.normal(0, S0, N) + rng.normal(0, 1, N) * err
    sol = KinematicSolver()
    sol.setup_grid(center=0.0, width=WIDTH, n_bins=K)
    sol.add_data(v, err)
    sol.run(num_samples=2000, seed=seed, num_chains=2, ncpu=2, progress_bar=False)
    sol.clip_uncertainties()
    e = np.asarray(sol.grid["edges"], float)
    med, sig = sol.clipped_samples["losvd_median"], sol.clipped_samples["losvd_uncertainty"]
    d, sd = med / med.sum(), sig / med.sum()
    mean = _cell_mass_correction(np.asarray(sol.samples["intrinsic_pdf"], float), (K,)).mean(0)
    h0 = 0.99 * mean / mean.sum() + 0.01 / K
    M = np.asarray(precompute_design_matrix(v, err, sol.grid["centers"], bin_width=sol.grid["width"]), float)
    M = M[M.sum(axis=1) >= 1e-3]
    MS = M @ sharpen_matrix(K, ndim=1)
    n = len(M)
    stats = {"diag": lambda h: stat_diag(h, d, sd),
             "exact": lambda h: stat_exact(h, M, n, 1.0),
             "exact_S": lambda h: stat_exact(h, MS, n, 1.0),
             "lap": lap(M, h0, n),
             "lap_S": lap(MS, h0, n)}
    fits = {}
    for mode in ("addback", "relative"):
        x0, b, hf = family(e, mode == "relative")
        for name, st in stats.items():
            fits[f"{mode}_{name}"] = fit(lambda t, st=st, hf=hf: st(hf(t)), x0, b)
        hm = hf(fits[f"{mode}_lap_S"]["x"])
        st = lap(MS, 0.99 * hm / hm.sum() + 0.01 / K, n)
        fits[f"{mode}_lap_S_it1"] = fit(lambda t, st=st, hf=hf: st(hf(t)), x0, b)
    return dict(seed=seed, truth=dict(s=S0, mu=1.0, m=0.0, N=N, n_kept=n), secs=round(time.time() - t0, 1), fits=fits)


def summarise(path):
    rs = []
    for x in Path(path).read_text().splitlines():
        x = x[2:] if x.startswith("@@") else x
        try:
            rs.append(json.loads(x))
        except json.JSONDecodeError:
            pass
    bad = [r for r in rs if "error" in r]
    rs = [r for r in rs if "fits" in r]
    print(f"C1 1D (MUSE-like): n = {len(rs)} seeds ({len(bad)} errors), K={K}, width={WIDTH}, N={N}, "
          f"sigma={S0}, err median {ERR_MED} log-sigma {ERR_LOGSIG}; median {np.median([r['secs'] for r in rs]):.0f} s/seed")
    print("bias = mean(fit - truth) +- s.e.; vs exact = paired difference; scatter = std over seeds; "
          "claimed = median Hessian error at Delta=1; ratio = scatter / claimed (1 = calibrated)")
    for mode, pars in (("addback", ["s", "mu"]), ("relative", ["s", "mu", "m"])):
        names = [k.split("_", 1)[1] for k in rs[0]["fits"] if k.startswith(mode)]
        print(f"\n--- {mode}")
        print(f"{'variant':10s}{'par':>4s}{'bias':>16s}{'vs exact':>16s}{'scatter':>9s}{'claimed':>9s}{'ratio':>7s}")
        for v in names:
            for i, p in enumerate(pars):
                x = np.array([r["fits"][f"{mode}_{v}"]["x"][i] for r in rs])
                xe = np.array([r["fits"][f"{mode}_exact"]["x"][i] for r in rs])
                tr = {"s": S0, "mu": 1.0, "m": 0.0}[p]
                c = np.array([r["fits"][f"{mode}_{v}"]["err"][i] for r in rs])
                se = lambda a: a.std(ddof=1) / np.sqrt(len(a))  # noqa: E731
                sc, cl = x.std(ddof=1), np.nanmedian(c)
                print(f"{v:10s}{p:>4s}{(x - tr).mean():+9.3f}±{se(x - tr):.3f}{(x - xe).mean():+9.3f}±{se(x - xe):.3f}"
                      f"{sc:9.3f}{cl:9.3f}{sc / cl:7.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "summarise":
        summarise(sys.argv[2])
        sys.exit()
    for seed in map(int, sys.argv[2:]):
        try:
            out = run(seed)
        except Exception as exc:
            out = dict(seed=seed, error=repr(exc), tb=traceback.format_exc()[-2000:])
        print("@@" + json.dumps(out), flush=True)
