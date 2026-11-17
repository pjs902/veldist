"""Talk figure (schematic): two velocity distributions with identical v_R and v_T histograms but different orbits.

usage: python t_2d_toy.py OUT.pdf
Mirrored columns. A: two orbit families, radial orbits (broad in v_R, narrow in v_T, not rotating) and near-circular
rotating orbits (narrow in v_R, broad in v_T, mean v_T > 0), half the stars each. B: one population, the product of
A's v_R and v_T distributions. Top: 2D distributions. Bottom: each column's own v_R and v_T distributions, identical.
Exact densities on a grid (illustration, not data).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

v = np.linspace(-60, 60, 361)
dv = v[1] - v[0]
VR, VT = np.meshgrid(v, v, indexing="ij")
g = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
rad = 0.5 * g(VR, 0, 18) * g(VT, 0, 5)  # radial orbits
circ = 0.5 * g(VR, 0, 5) * g(VT, 8, 14)  # near-circular, rotating orbits
norm = (rad + circ).sum() * dv * dv  # unit integral on the grid, so B has exactly A's marginals
rad, circ = rad / norm, circ / norm
A = rad + circ
pR, pT = A.sum(1) * dv, A.sum(0) * dv
B = np.outer(pR, pT)
assert np.allclose(B.sum(1) * dv, pR) and np.allclose(B.sum(0) * dv, pT)

CR, CC = "#D55E00", "#0072B2"
w, h = plt.rcParams["figure.figsize"]
fig = plt.figure(figsize=(1.25 * 2.9 * h, 0.95 * 2.9 * h), constrained_layout=True)
gs = fig.add_gridspec(3, 4, height_ratios=[2.2, 0.12, 1])
top = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4])]
bot = [fig.add_subplot(gs[2, k]) for k in range(4)]
mid = fig.add_subplot(gs[1, :])
mid.set_axis_off()
lev = lambda D: D.max() * np.array([0.05, 0.2, 0.5, 0.8])
for D, c in ((rad, CR), (circ, CC)):
    top[0].contourf(v, v, D.T, levels=np.r_[lev(D), D.max() * 1.01], colors=c, alpha=0.18)
    top[0].contour(v, v, D.T, levels=lev(D), colors=c, linewidths=1.2)
top[0].text(30, -12, "radial orbits", color=CR, ha="center", fontsize="medium")
top[0].text(9, 38, "rotating,\nnear-circular orbits", color=CC, ha="left", fontsize="medium")
top[1].contourf(v, v, B.T, levels=np.r_[lev(B), B.max() * 1.01], colors="0.35", alpha=0.18)
top[1].contour(v, v, B.T, levels=lev(B), colors="0.25", linewidths=1.2)
top[0].set_title(r"\textbf{A}: two orbit families", fontsize="large")
top[1].set_title(r"\textbf{B}: one population", fontsize="large")
for a in top:
    a.set_box_aspect(1)
    a.set_xlim(-50, 50)
    a.set_ylim(-50, 50)
    a.axhline(0, color="0.7", lw=0.5)
    a.axvline(0, color="0.7", lw=0.5)
    a.set_xlabel(r"$v_R$ [km s$^{-1}$]")
    a.set_ylabel(r"$v_T$ [km s$^{-1}$]")
for k, (p, lab) in enumerate(((pR, r"$v_R$"), (pT, r"$v_T$"), (pR, r"$v_R$"), (pT, r"$v_T$"))):
    a = bot[k]
    a.fill_between(v, p, color="0.75")
    a.plot(v, p, color="k", lw=1.5)
    a.set_xlim(-50, 50)
    a.set_ylim(0, 1.15 * max(pR.max(), pT.max()))
    a.set_yticks([])
    a.set_xlabel(lab + r" [km s$^{-1}$]")
bot[0].set_ylabel("stars")
bot[2].set_ylabel("stars")
mid.text(0.5, 0.0, r"$v_R$ and $v_T$ fitted separately: \textbf{A and B look the same}", ha="center", va="center",
         fontsize="large", transform=mid.transAxes)
fig.savefig(sys.argv[1])
fig.savefig(sys.argv[1].replace(".pdf", ".png"), dpi=200)
