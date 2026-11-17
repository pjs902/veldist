"""Paper figure d2: the binned kinematics that enter the fit, every dataset on one map per quantity.

usage: python d2_maps.py PACK_DIR OUT.pdf   (x-East production pack: *_pm_2dhist.npz, veldist/lvm ecsv)
Each dataset's own DYNAMITE bins (aperture.dat + bins.dat pixel maps; single unused pixels enclosed by one bin
filled for display) are drawn with radius stretched as 100" asinh(R/100") at the true position angle, so the HST/MUSE core and the Gaia/LVM outskirts share one panel. PM moments come from the cell masses (Sheppard-corrected dispersions) after rotating
(pm1, pm2) from the aperture frame (receding major axis at PA_KIN) back to (East, North).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

PA_KIN = 104.3  # deg, N through E (omegaCen dynamite_frame.PA_KIN): pm1 along it, pm2 along +90 deg E->N


def pm_moments(path):
    z = np.load(path)
    H, K, half = z["PM_2dhist"], z["PM_2dhist"].shape[1], float(z["vxrange"])
    h = 2 * half / K
    c = np.linspace(-half + h / 2, half - h / 2, K)
    w = H / H.sum((1, 2), keepdims=True)
    P1, P2 = np.meshgrid(c, c, indexing="ij")
    m1, m2 = (w * P1).sum((1, 2)), (w * P2).sum((1, 2))
    s11 = (w * (P1 - m1[:, None, None]) ** 2).sum((1, 2)) - h ** 2 / 12
    s22 = (w * (P2 - m2[:, None, None]) ** 2).sum((1, 2)) - h ** 2 / 12
    s12 = (w * (P1 - m1[:, None, None]) * (P2 - m2[:, None, None])).sum((1, 2))
    pa = np.radians(PA_KIN)
    maj, mnr = np.array([np.sin(pa), np.cos(pa)]), np.array([-np.cos(pa), np.sin(pa)])
    T = np.stack([maj, mnr], 1)  # (E,N) = T @ (pm1, pm2)
    x, y = z["xbin"], z["ybin"]
    R = np.hypot(x, y)
    er, et = np.stack([x, y], 1) / R[:, None], np.stack([-y, x], 1) / R[:, None]
    A_r, A_t = er @ T, et @ T  # projections of (pm1, pm2) onto radial / tangential
    mr, mt = (A_r * np.stack([m1, m2], 1)).sum(1), (A_t * np.stack([m1, m2], 1)).sum(1)
    S = np.stack([np.stack([s11, s12], 1), np.stack([s12, s22], 1)], 1)
    sr = np.sqrt(np.einsum("ni,nij,nj->n", A_r, S, A_r))
    st = np.sqrt(np.einsum("ni,nij,nj->n", A_t, S, A_t))
    return dict(x=x, y=y, R=R, mr=mr, mt=-mt, sr=sr, st=st, n=z["nstarbin"])  # mt>0 along the rotation


def load(P):
    hst, gaia = pm_moments(P / "hst_veldist_pm_2dhist.npz"), pm_moments(P / "gaia_veldist_pm_2dhist.npz")
    m = Table.read(P / "veldist_kinematics.ecsv")
    l = Table.read(P / "lvm_kins.ecsv")
    muse = dict(x=np.asarray(m["xbin"]), y=np.asarray(m["ybin"]), v=np.asarray(m["v"]), s=np.asarray(m["sigma"]))
    lvm = dict(x=np.asarray(l["x"]), y=np.asarray(l["y"]), v=np.asarray(l["v"]), dv=np.asarray(l["dv"]))
    return hst, gaia, muse, lvm


import os

STRETCH = os.environ.get("D2_STRETCH", "asinh100")  # pow<p> (R^p) or asinh<R0> (R0 asinh(R/R0))
EDGES = os.environ.get("D2_EDGES", "none")  # none | bins


def srad(R):
    if STRETCH.startswith("pow"):
        return np.asarray(R, float) ** float(STRETCH[3:])
    r0 = float(STRETCH[5:])
    return r0 * np.arcsinh(np.asarray(R, float) / r0)


def stretch(x, y):
    """radial stretch srad(R) at fixed position angle."""
    R = np.hypot(x, y)
    f = srad(R) / np.where(R > 0, R, 1)
    return x * f, y * f


def binmap(P, stem):
    """DYNAMITE aperture grid: pixel corner coordinates (sky x-East, y-North, arcsec) and bin ids (0 = unused)."""
    L = (P / f"{stem}_aperture.dat").read_text().split("\n")
    x0, y0 = map(float, L[1].split())
    wx, wy = map(float, L[2].split())
    nx, ny = map(int, L[4].split())
    b = np.array((P / f"{stem}_bins.dat").read_text().split()[2:], int).reshape((nx, ny), order="F")
    return np.linspace(x0, x0 + wx, nx + 1), np.linspace(y0, y0 + wy, ny + 1), b


def fill_holes(b):
    """Display only: give an unused pixel (id 0) the bin id of its four neighbours when they all agree."""
    p = np.pad(b, 1)
    nb = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
    hole = (b == 0) & (nb[0] > 0) & np.all(nb == nb[0], axis=0)
    return np.where(hole, nb[0], b)


def edges(xe, ye, b):
    """Line segments separating pixels of different bins (outer boundary included)."""
    segs = []
    bp = np.pad(b, 1)
    for i, j in zip(*np.nonzero(bp[1:, 1:-1] != bp[:-1, 1:-1])):  # vertical edges at xe[i]
        if bp[i + 1, j + 1] or bp[i, j + 1]:
            segs.append([(xe[i], ye[j]), (xe[i], ye[j + 1])])
    for i, j in zip(*np.nonzero(bp[1:-1, 1:] != bp[1:-1, :-1])):  # horizontal edges at ye[j]
        if bp[i + 1, j + 1] or bp[i + 1, j]:
            segs.append([(xe[i], ye[j]), (xe[i + 1], ye[j])])
    out = []
    for (a, c) in segs:  # subdivide so the stretch bends long edges correctly
        t = np.linspace(0, 1, 6)[:, None]
        p = np.array(a) + t * (np.array(c) - np.array(a))
        out.append(np.stack(stretch(p[:, 0], p[:, 1]), 1))
    return out


def panel(ax, layers, cmap, vmin, vmax, label):
    """layers: (xe, ye, binid, values-by-bin-id) drawn in order (outer datasets first)."""
    for xe, ye, b, val in layers:
        X, Y = np.meshgrid(xe, ye, indexing="ij")
        SX, SY = stretch(X, Y)
        b = fill_holes(b)
        v = np.full(b.shape, np.nan)
        v[b > 0] = val[b[b > 0] - 1]
        im = ax.pcolormesh(SX, SY, np.ma.masked_invalid(v), cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
                           rasterized=True)
        if EDGES == "bins":
            ax.add_collection(LineCollection(edges(xe, ye, b), colors="k", linewidths=0.15, alpha=0.5))
    smax = float(srad(1550))
    for r in (100, 300, 1000):
        ax.add_patch(Circle((0, 0), float(srad(r)), fill=False, color="0.4", lw=0.5, ls=":"))
        ax.text(float(srad(r)) * np.cos(2.4), float(srad(r)) * np.sin(2.4), rf"{r}$^{{\prime\prime}}$", color="0.2",
                fontsize="small", ha="center", va="center", bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.5))
    ax.set_xlim(smax, -smax)  # East to the left
    ax.set_ylim(-smax, smax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, label=label, shrink=0.85, pad=0.02)


def main():
    P = Path(sys.argv[1])
    hst, gaia, muse, lvm = load(P)
    w, h = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(2, 3, figsize=(3 * w, 2 * h), constrained_layout=True)
    G = {k: binmap(P, k) for k in ("hst_veldist", "gaia_veldist", "veldist", "lvm")}
    lay = lambda key, d, q: (*G[key], d[q])
    vlim, slo, shi = 12, 5, 22
    panel(ax[0, 0], [lay("lvm", lvm, "v"), lay("veldist", muse, "v")], "RdBu_r", -vlim, vlim,
          r"$v_{\rm LOS}$ [km s$^{-1}$]")
    for k, (q, lab, cm, lo, hi) in enumerate([("mt", r"$\mu_T$", "RdBu_r", -vlim, vlim),
                                              ("mr", r"$\mu_R$", "RdBu_r", -vlim, vlim)]):
        panel(ax[0, k + 1], [lay("gaia_veldist", gaia, q), lay("hst_veldist", hst, q)], cm, lo, hi,
              lab + r" [km s$^{-1}$]")
    panel(ax[1, 0], [lay("veldist", muse, "s")], "viridis", slo, shi, r"$\sigma_{\rm LOS}$ [km s$^{-1}$]")
    for k, (q, lab) in enumerate([("st", r"$\sigma_T$"), ("sr", r"$\sigma_R$")]):
        panel(ax[1, k + 1], [lay("gaia_veldist", gaia, q), lay("hst_veldist", hst, q)], "viridis", slo, shi,
              lab + r" [km s$^{-1}$]")
    for a, t in zip(ax.flat, ["MUSE + LVM", r"\textit{HST} + \textit{Gaia}", r"\textit{HST} + \textit{Gaia}",
                              "MUSE", r"\textit{HST} + \textit{Gaia}", r"\textit{HST} + \textit{Gaia}"]):
        a.set_title(t)
    ax[0, 0].annotate("", xy=(0.12, 0.95), xytext=(0.12, 0.8), xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[0, 0].annotate("", xy=(0.04, 0.8), xytext=(0.12, 0.8), xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[0, 0].text(0.12, 0.97, "N", transform=ax[0, 0].transAxes, ha="center", va="bottom", fontsize="small")
    ax[0, 0].text(0.02, 0.8, "E", transform=ax[0, 0].transAxes, ha="right", va="center", fontsize="small")
    fig.savefig(sys.argv[2])
    fig.savefig(sys.argv[2].replace(".pdf", ".png"), dpi=200)
    print("saved", sys.argv[2], "HST mean mu_R %.2f, Gaia %.2f" % (np.median(hst["mr"]), np.median(gaia["mr"])))


if __name__ == "__main__":
    main()
