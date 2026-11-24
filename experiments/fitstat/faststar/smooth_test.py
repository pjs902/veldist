"""Term-b table noise vs smoothing (mock; seconds). Noise = sd over mocks of chi2(table seed 1) - chi2(seed 0)."""
import numpy as np
from scipy.ndimage import gaussian_filter

import dfmock as d
import likelihood as lk

real = dict(np.load("../work/quality_R10.npz"))
bg = d.bg_density(real)
M = [2.1e4, 2.5e4, 3.0e4]
tab = {ts: [d.model_table(d.Cluster(m), d.R_EDGES, d.V_EDGES, d.W_EDGES, n_per=20000, seed=10 * ts + i)
            for i, m in enumerate(M)] for ts in (0, 1)}


def smooth(P, s):
    if s == 0:
        return P
    Q = np.stack([gaussian_filter(p, s, mode="nearest") for p in P])
    return Q / Q.sum((1, 2), keepdims=True)


for s in (0, 0.7, 1.0, 1.5, 2.0):
    noise, curv = [], []
    for seed in range(8):
        t = lk.FastStarTerm(d.draw_observed(d.Cluster(2.5e4), real, np.random.default_rng(seed)),
                            d.R_EDGES, d.V_EDGES, d.W_EDGES, d.CUT, bg=bg)
        c = {ts: np.array([t.chi2(smooth(P, s), terms=("b",)) for P in tab[ts]]) for ts in (0, 1)}
        noise.append(c[1] - c[0])
        curv.append(c[0][0] + c[0][2] - 2 * c[0][1])               # signal: curvature across +-20% in M
    noise, curv = np.array(noise), np.array(curv)
    print(f"sigma {s:3.1f} cells: term-b table noise sd {noise.std():6.2f}; median curvature (+-20% M) {np.median(curv):6.2f}")
