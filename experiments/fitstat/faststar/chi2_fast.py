"""Fast-star term (a) for real orbit-library tables (docs/fast-star-term.md). Node job, light.

    python chi2_fast.py QUALITY.npz TABLE.npz [TABLE.npz ...] [--sigma 0.5] [--raw]

QUALITY: quality_sample.py output (positions relative to the adopted centre). TABLE: orbit_tables.py table
output (P_s2 by default, --raw for the unsmoothed P). Prints chi2 with the BH at the centre and marginalised
over a Gaussian prior (sigma arcsec) on its position, plus each selected star's density and P(selected) at
its annulus (centred), to look at star A and the per-star contributions.
"""
import sys

import numpy as np

import dfmock as d
import likelihood as lk

args = [a for a in sys.argv[1:] if not a.startswith("--")]
sigma = float(sys.argv[sys.argv.index("--sigma") + 1]) if "--sigma" in sys.argv else 0.5
if "--sigma" in sys.argv:
    args.remove(str(sys.argv[sys.argv.index("--sigma") + 1]))
key = "P" if "--raw" in sys.argv else "P_s2"
q = dict(np.load(args[0]))
stars = dict(x=q["dx"], y=q["dy"], v2d=q["v2d"], e2d=q["e2d"], selected=q["selected"])
print(f"quality stars {len(q['dx'])}, selected {q['selected'].sum()}, bg per star {float(q['bg_per_star']):.2e}")
for path in args[1:]:
    T = np.load(path)
    P, ve = T[key], T["v_edges"]
    bg = d.bg_density(q, ve)
    t = lk.FastStarTerm(stars, T["R_edges"], ve, T["w_edges"], d.CUT, bg=bg)
    c0, cm = t.chi2(P, terms=("a",)), t.chi2(P, pos_sigma=sigma, terms=("a",))
    print(f"{str(T['directory'])}: sf {float(T['sf']):.4f}  chi2_a centred {c0:.2f}  marginalised(sigma {sigma}\") {cm:.2f}")
    a, _ = t.annulus()
    P2 = P.sum(-1)
    s = t.sel
    psel = np.einsum("ki,ki->k", P2[a], t.S)
    dens = np.einsum("ki,ki->k", P2[a][s], t.K[s])
    for i, k in enumerate(np.flatnonzero(s)):
        print(f"   sel star {int(q['id'][k])}: R {np.hypot(q['dx'][k], q['dy'][k]):5.2f}\" v2d {q['v2d'][k]:6.1f}"
              f"+-{q['e2d'][k]:4.1f}  member dens {dens[i]:.3e}  bg dens {t.gbg[k]:.3e}  P(sel) {psel[k]:.3e}")
    print(f"   sum P(sel) over quality stars (expected members selected) {psel.sum():.2f}")
