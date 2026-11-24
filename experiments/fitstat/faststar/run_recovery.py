"""Mock recovery and calibration study of the fast-star term (docs/fast-star-term.md). Node job.

    python run_recovery.py QUALITY.npz OUT.npz [N_SEEDS]

Truth M = 2.5e4. For each seed: one mock data set per case, chi2 on an M grid with tables built once per M
(two independent table seeds measure the Monte Carlo noise the orbit library will also have). Cases:
  centred    BH at the frame origin, no position marginalisation
  offset     BH at (-0.4", +0.4"), chi2 assuming the origin (bias check)
  offset_mg  same data, BH position marginalised (Gaussian prior sigma 0.5" about the origin)
and per case the terms a, b and a+b. Output chi2[case, terms, table_seed, seed, M].
"""
import sys
import time

import numpy as np

import dfmock as d
import likelihood as lk

real = dict(np.load(sys.argv[1]))
out = sys.argv[2]
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 30
M_TRUE = 2.5e4
GRID = np.array([0.6, 0.9, 1.3, 1.7, 2.1, 2.5, 3.0, 3.6, 4.4, 5.5, 7.0, 9.0]) * 1e4
CASES = {"centred": ((0.0, 0.0), None), "offset": ((-0.4, 0.4), None), "offset_mg": ((-0.4, 0.4), 0.5)}
TERMS = [("a",), ("b",), ("a", "b")]
bg = d.bg_density(real)

t0 = time.time()
tables = [[d.model_table(d.Cluster(m), d.R_EDGES, d.V_EDGES, d.W_EDGES, n_per=20000, seed=100 * ts + i)
           for i, m in enumerate(GRID)] for ts in range(2)]
print(f"tables {time.time() - t0:.0f} s", flush=True)

chi2 = np.full((len(CASES), len(TERMS), 2, n_seeds, len(GRID)), np.nan)
nsel = np.zeros((len(CASES), n_seeds), int)
for s in range(n_seeds):
    for c, (name, (bh, ps)) in enumerate(CASES.items()):
        rng = np.random.default_rng(1000 * s + (0 if name == "centred" else 1))    # offset and offset_mg share data
        obs = d.draw_observed(d.Cluster(M_TRUE), real, rng, bh=bh)
        nsel[c, s] = obs["selected"].sum()
        t = lk.FastStarTerm(obs, d.R_EDGES, d.V_EDGES, d.W_EDGES, d.CUT, bg=bg)
        for k, terms in enumerate(TERMS):
            if ps is not None and terms != ("a", "b"):
                continue                                                  # marginalised case: combined term only
            for ts in range(2):
                chi2[c, k, ts, s] = [t.chi2(P, pos_sigma=ps, terms=terms) for P in tables[ts]]
    print(f"seed {s} done {time.time() - t0:.0f} s, selected {nsel[:, s]}", flush=True)
    np.savez(out, chi2=chi2, nsel=nsel, grid=GRID, m_true=M_TRUE, cases=list(CASES), terms=[" ".join(t) for t in TERMS])


def summary():
    it = int(np.argmin(np.abs(GRID - M_TRUE)))
    for c, name in enumerate(CASES):
        for k, terms in enumerate(TERMS):
            x = chi2[c, k, 0]
            if np.isnan(x).all():
                continue
            d0 = x[:, it] - x.min(1)                                     # Delta chi2 at the truth (grid profile)
            mhat = GRID[np.argmin(x, 1)]
            noise = np.std(chi2[c, k, 1] - chi2[c, k, 0], axis=0).max() / np.sqrt(2)
            print(f"{name:9s} {'+'.join(terms):4s}: median M_hat {np.median(mhat):.2e} "
                  f"(16-84% {np.percentile(mhat, 16):.1e}-{np.percentile(mhat, 84):.1e}); "
                  f"P(dchi2_true<1) {np.mean(d0 < 1):.2f} [0.68], <4 {np.mean(d0 < 4):.2f} [0.95]; "
                  f"table MC noise sd {noise:.2f}")


summary()
print(f"done {time.time() - t0:.0f} s")
