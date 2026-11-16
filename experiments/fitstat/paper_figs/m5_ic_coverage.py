"""Paper figure m5: action-space coverage of the orbit starting points, orbitstart vs action-space grid.

usage: python m5_ic_coverage.py CANDIDATES_NPZ OUT.pdf
CANDIDATES_NPZ: probabilistic_dynamics work/ic_design/candidates.npz (scripts/ic_coverage.py on the node).
Design A = DYNAMITE's orbitstart grid; design C = rows evenly spaced in J_r/J_tot along each (E, theta) ray,
the rule of dynamite/start_grid.py (initial_conditions: actions), both nE 30, nI2 25, nI3 20, dithering 3
(45,000 orbits), on the omega Cen test potential. Per energy shell, a (J_phi/J_tot, J_r/J_tot) cell is covered
if any starting orbit of that shell falls in it; maps show the fraction of shells covering each cell.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
xb, yb = np.linspace(-1, 1, 41), np.linspace(0, 1, 21)
lJ = np.array([np.median(np.log10(np.abs(z["A_J"][e]).sum(-1))) for e in range(z["A_J"].shape[0])])
w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(1, 3, figsize=(3.2 * w, h), constrained_layout=True)
for a, key, ttl in ((ax[0], "A", "DYNAMITE grid"), (ax[1], "C", "action-space grid")):
    im = a.pcolormesh(xb, yb, z[f"{key}_ma"].T, cmap="viridis", vmin=0, vmax=1)
    a.plot([-1, 0, 1, -1], [0, 1, 0, 0], "k-", lw=0.8)
    a.set_xlabel(r"$J_\phi/J_{\rm tot}$")
    a.set_ylabel(r"$J_r/J_{\rm tot}$")
    a.set_title(ttl)
fig.colorbar(im, ax=ax[1], label="fraction of energy shells", pad=0.02)
for key, lab, c, m in (("A", "DYNAMITE grid", "k", "s"), ("C", "action-space grid", "#0072B2", "o")):
    ax[2].plot(lJ, z[f"{key}_fa"], marker=m, color=c, label=lab)
ax[2].set_ylim(0, 1.05)
ax[2].set_xlabel(r"energy shell: median $\log_{10} J_{\rm tot}$ [pc km s$^{-1}$]")
ax[2].set_ylabel("fraction of action triangle covered")
ax[2].legend(frameon=False, loc="lower left")
fig.savefig(sys.argv[2])
fig.savefig(sys.argv[2].replace(".pdf", ".png"), dpi=200)
print("saved", sys.argv[2], "median covered A %.2f C %.2f" % (np.median(z["A_fa"]), np.median(z["C_fa"])))
