"""Misspecification study of the fast-star term (docs/fast-star-term.md). Node job.

    python run_misspec.py QUALITY.npz OUT.npz [N_SEEDS]

Truth M = 2.5e4. Data from the isotropic-cusp (iso) or the tangential-core (tan, beta = -1/2, rho ~ r^-0.2)
family; chi2 on the M grid with tables from each family. The diagonal is the correctly specified fit; the
off-diagonal measures how much of the term's information on M is really about the inner DF (cusp shape and
anisotropy), which the Schwarzschild model fits but the analytic route assumes.
Output chi2[data_family, model_family, terms, seed, M].
"""
import sys
import time

import numpy as np

import dfmock as d
import likelihood as lk

real = dict(np.load(sys.argv[1]))
out = sys.argv[2]
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 12
M_TRUE = 2.5e4
GRID = np.array([0.6, 0.9, 1.3, 1.7, 2.1, 2.5, 3.0, 3.6, 4.4, 5.5, 7.0, 9.0, 12.0]) * 1e4
FAM = {"iso": d.Cluster, "tan": d.TangentialCluster}
TERMS = [("a",), ("b",), ("a", "b")]
bg = d.bg_density(real)
t0 = time.time()
tables = {f: [d.model_table(C(m), d.R_EDGES, d.V_EDGES, d.W_EDGES, n_per=20000, seed=i) for i, m in enumerate(GRID)]
          for f, C in FAM.items()}
print(f"tables {time.time() - t0:.0f} s", flush=True)
chi2 = np.full((2, 2, len(TERMS), n_seeds, len(GRID)), np.nan)
for s in range(n_seeds):
    for i, (fd, Cd) in enumerate(FAM.items()):
        obs = d.draw_observed(Cd(M_TRUE), real, np.random.default_rng(5000 + s))
        t = lk.FastStarTerm(obs, d.R_EDGES, d.V_EDGES, d.W_EDGES, d.CUT, bg=bg)
        for j, fm in enumerate(FAM):
            for k, terms in enumerate(TERMS):
                chi2[i, j, k, s] = [t.chi2(P, terms=terms) for P in tables[fm]]
    print(f"seed {s} {time.time() - t0:.0f} s", flush=True)
    np.savez(out, chi2=chi2, grid=GRID, m_true=M_TRUE, fam=list(FAM), terms=[" ".join(t) for t in TERMS])
it = int(np.argmin(np.abs(GRID - M_TRUE)))
for i, fd in enumerate(FAM):
    for j, fm in enumerate(FAM):
        for k, terms in enumerate(TERMS):
            x = chi2[i, j, k]
            mhat = GRID[np.argmin(x, 1)]
            print(f"data {fd} model {fm} {'+'.join(terms):4s}: median M_hat {np.median(mhat):.2e} "
                  f"(16-84% {np.percentile(mhat, 16):.1e}-{np.percentile(mhat, 84):.1e}); "
                  f"median dchi2(truth) {np.median(x[:, it] - x.min(1)):.1f}; median min chi2 {np.median(x.min(1)):.1f}")
print(f"done {time.time() - t0:.0f} s")
