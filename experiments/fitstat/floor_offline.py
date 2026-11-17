"""Score exact-likelihood floor rules offline from the h saved by rescore_v4.py (no orbit-library read).

    python floor_offline.py RESCORE_DIR SHARP_TERMS.npz PLAIN_TERMS.npz OUT.npz

Per model, per star: p_S = (M S) h, p_M = M h (mass units; the extended-Poisson term is unchanged by the rule).
Rules, each -2 ln L summed over all HST stars:
  floor[f, eps]  p = max(p_S, f p_M) + eps * hbar_a,   hbar_a = mean model mass per cell in the star's aperture
                 (eps: outlier floor, a uniform density over the aperture's grid; eps = 0 is the bare floor)
  plain[eps]     p = p_M + eps * hbar_a
Also per model: stars with p_M <= 0 (model has no mass under the star) and their star indices, so the same
stars can be recognised across models.
"""
import glob
import sys

import numpy as np

FLOORS = (0.0, 1e-2, 0.1, 0.3)
EPS = (1e-6, 1e-4, 1e-3, 1e-2)
files = sorted(glob.glob(f"{sys.argv[1]}/rescore_f_*.npz"))
S, P = np.load(sys.argv[2]), np.load(sys.argv[3])
apert = S["PM_star_aperture"]
np.testing.assert_array_equal(P["PM_star_aperture"], apert)
n_star = len(apert)


def rows(T):
    ptr, cell, mass = T["PM_star_indptr"], T["PM_star_cell"], T["PM_star_mass"].astype(np.float64)
    star = np.repeat(np.arange(n_star), np.diff(ptr))
    return star, cell, mass


sS, cS, mS = rows(S)
sP, cP, mP = rows(P)
out = dict(floors=np.asarray(FLOORS), eps=np.asarray(EPS), models=[], floor=[], plain=[], ext=[], n_zero=[],
           zero_stars=[])
for f in files:
    z = np.load(f, allow_pickle=True)
    h = z["h"].astype(np.float64)                     # n_ap x K^2, mass units
    pS = np.bincount(sS, weights=mS * h[apert[sS], cS], minlength=n_star)
    pM = np.bincount(sP, weights=mP * h[apert[sP], cP], minlength=n_star)
    hbar = h.mean(1)[apert]
    out["models"].append(str(z["directory"]))
    out["floor"].append([[-2 * np.log(np.maximum(pS, fl * pM) + e * hbar).sum() for e in EPS] for fl in FLOORS])
    out["plain"].append([-2 * np.log(pM + e * hbar).sum() for e in EPS])
    nst = np.bincount(apert, minlength=h.shape[0])
    out["ext"].append(2 * np.sum(nst * z["mratio"]))  # 2 sum_a N_a h_a.sum()/M_a (rule-independent)
    zero = np.flatnonzero(pM <= 0)
    out["n_zero"].append(len(zero))
    out["zero_stars"].append(zero)
    print(f"{out['models'][-1]}: p_M<=0 {len(zero)}, p_S<=0 {int(np.sum(pS <= 0))}", flush=True)
np.savez(sys.argv[4], **{k: np.asarray(v, dtype=object if k == "zero_stars" else None)
                                for k, v in out.items()})
zs = [set(s.tolist()) for s in out["zero_stars"]]
common = set.intersection(*zs) if zs else set()
print(f"stars with p_M<=0 in every model: {len(common)} -> {sorted(common)[:40]}")
print(f"their apertures: {np.unique(apert[sorted(common)]).tolist() if common else []}")
