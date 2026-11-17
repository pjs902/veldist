"""Talk figure (schematic): two velocity distributions with identical v_R and v_T distributions but different orbits.

usage: python t_2d_toy.py OUT.pdf
A: half the stars on near-circular, rotating orbits (narrow in v_R, broad in v_T, mean v_T > 0), half on radial
orbits (broad in v_R, narrow in v_T, not rotating). B: the product of A's two marginals, i.e. one population with
the same v_R and v_T distributions. Exact densities on a grid (illustration, not data). Right: the marginals of A and
B, identical: a fit to v_R and v_T separately cannot tell the orbit mix from the single population.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

v = np.linspace(-60, 60, 361)
dv = v[1] - v[0]
VR, VT = np.meshgrid(v, v, indexing="ij")
g = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
circ = g(VR, 0, 5) * g(VT, 8, 14)  # near-circular orbits: little radial motion, rotating, broad in v_T
rad = g(VR, 0, 18) * g(VT, 0, 5)  # radial orbits: broad in v_R, no rotation, little tangential motion
A = 0.5 * circ + 0.5 * rad
A /= A.sum() * dv * dv
pR, pT = A.sum(1) * dv, A.sum(0) * dv
B = np.outer(pR, pT)
assert np.allclose(B.sum(1) * dv, pR) and np.allclose(B.sum(0) * dv, pT)

w, h = plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(1, 3, figsize=(1.3 * 2.9 * h, 0.5 * 2.9 * h), constrained_layout=True,
                       subplot_kw=dict(box_aspect=1))
lev = A.max() * np.array([0.03, 0.1, 0.25, 0.5, 0.75])
for a, D, t in ((ax[0], A, r"\textbf{A}: circular $+$ radial orbits"), (ax[1], B, r"\textbf{B}: one population")):
    a.contourf(v, v, D.T, levels=np.r_[lev, D.max() * 1.01], cmap="Blues")
    a.contour(v, v, D.T, levels=lev, colors="#0072B2", linewidths=0.6)
    a.axhline(0, color="0.6", lw=0.5)
    a.axvline(0, color="0.6", lw=0.5)
    a.set_xlim(-50, 50)
    a.set_ylim(-50, 50)
    a.set_xlabel(r"$v_R$ [km s$^{-1}$]")
    a.set_title(t, fontsize="medium")
ax[0].set_ylabel(r"$v_T$ [km s$^{-1}$]")
ax[0].annotate("radial orbits", (32, 1), (18, -38), fontsize="small", arrowprops=dict(arrowstyle="->", lw=0.8))
ax[0].annotate("near-circular,\nrotating", (2, 30), (12, 38), fontsize="small", arrowprops=dict(arrowstyle="->", lw=0.8))
ax[2].plot(v, pR, color="#0072B2", lw=2.5, label=r"$v_R$: A")
ax[2].plot(v, B.sum(1) * dv, color="k", lw=1.2, ls="--", label=r"$v_R$: B")
ax[2].plot(v, pT, color="#D55E00", lw=2.5, label=r"$v_T$: A")
ax[2].plot(v, B.sum(0) * dv, color="k", lw=1.2, ls=":", label=r"$v_T$: B")
ax[2].set_xlim(-50, 50)
ax[2].set_ylim(0, 1.3 * max(pR.max(), pT.max()))
ax[2].set_xlabel(r"velocity [km s$^{-1}$]")
ax[2].set_ylabel("fraction of stars")
ax[2].set_title(r"$v_R$ and $v_T$ separately: \textbf{A = B}", fontsize="medium")
ax[2].legend(frameon=False, fontsize="small", ncol=2, loc="upper left")
fig.savefig(sys.argv[1])
fig.savefig(sys.argv[1].replace(".pdf", ".png"), dpi=200)
print("max |marginal difference|:", np.abs(B.sum(1) * dv - pR).max(), np.abs(B.sum(0) * dv - pT).max())
