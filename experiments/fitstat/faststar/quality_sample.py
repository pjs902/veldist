"""Step 0 of docs/fast-star-term.md: rebuild the Haeberle+24 high-quality PM sample and its fast stars.

    python quality_sample.py OUT.npz

Cuts (Haeberle et al. 2024, Methods): >= 20 measurements, baseline >= 20 yr, rejected fraction < 15%,
PM error < 0.194 mas/yr and reduced chi2 < 10 in both components. Uncorrected PMs (they did not apply the
local corrections). PM errors are inflated by sqrt(max(chi2, 1)) per component: that reproduces their A-G
speeds AND errors exactly (G 66.2 +- 1.9, B 66.6 +- 4.1) and the 0-3" selection = A C D E F exactly.
No CMD cut by default: their CMD boundaries are not available (Bella's reject D) and the term only needs a
velocity-blind sample; the background is then measured in this sample (selected density at 20-90").
Counts within 90" without CMD: 171,850 (theirs with CMD: 157,320 of 241,133).
"""
import sys

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

D = "/Users/pesmith/research/fast-stars/bella_fitting/"
RA0, DE0 = 201.696833, -47.479569                     # AvdM10
MU_CUT = 2.41                                           # mas/yr (62 km/s at 5.43 kpc)
KMS = 4.74047 * 5.494                                  # km/s per mas/yr at the model distance
FAST = dict(A=532867, B=475236, C=1422379, D=476492, E=509644, F=476467, G=510061)

a = fits.open(D + "astrometry.fits", memmap=True)[1].data
col = lambda h, c: np.asarray(h[c])
ids = col(a, "ID")
dx = (col(a, "RA") - RA0) * np.cos(np.radians(DE0)) * 3600
dy = (col(a, "DEC") - DE0) * 3600
R = np.hypot(dx, dy)
pmx, pmy, ex, ey = (col(a, c) for c in ("pmra", "pmdec", "pmra_err", "pmdec_err"))
nf, nu, base = col(a, "nfound"), col(a, "nused"), col(a, "baseline")
c2x, c2y = col(a, "chi2x"), col(a, "chi2y")
with np.errstate(invalid="ignore", divide="ignore"):
    q_ast = ((nu >= 20) & (base >= 20) & ((nf - nu) / nf < 0.15) & (ex < 0.194) & (ey < 0.194)
             & (c2x < 10) & (c2y < 10) & np.isfinite(pmx) & np.isfinite(pmy))

mag = {}
for f in ("F606W", "F814W"):
    p = fits.open(D + f"{f}_photometry.fits", memmap=True)[1].data
    pid, o = np.asarray(p["ID"]), None
    o = np.argsort(pid)
    k = np.searchsorted(pid, ids, sorter=o).clip(0, len(pid) - 1)
    hit = pid[o[k]] == ids
    m = np.full(len(ids), np.nan)
    m[hit] = np.asarray(p["corrected_mag"], float)[o[k[hit]]]
    print(f"{f}: matched {hit.sum()} of {len(ids)}")
    mag[f] = m
L = np.loadtxt(D + "cmd_f606w_f814w_left.txt", delimiter=",")
Rt = np.loadtxt(D + "cmd_f606w_f814w_right.txt", delimiter=",")
fl = interp1d(L[:, 1], L[:, 0], fill_value="extrapolate")
fr = interp1d(Rt[:, 1], Rt[:, 0], fill_value="extrapolate")
col_ = mag["F606W"] - mag["F814W"]
with np.errstate(invalid="ignore"):
    q_cmd = (fl(mag["F606W"]) < col_) & (col_ < fr(mag["F606W"])) & (mag["F606W"] < 24) & (mag["F606W"] > 16)

inn = R < 90
print(f"within 90\": {inn.sum()} (Haeberle 241,133)")
print(f"  astrometric cuts: {(inn & q_ast).sum()}   + CMD (Bella boundaries): {(inn & q_ast & q_cmd).sum()} (Haeberle 157,320)")
Q = q_ast & (q_cmd if "--cmd" in sys.argv else True)
ex, ey = ex * np.sqrt(np.maximum(c2x, 1)), ey * np.sqrt(np.maximum(c2y, 1))
mu = np.hypot(pmx, pmy)
emu = np.sqrt((pmx * ex) ** 2 + (pmy * ey) ** 2) / np.where(mu > 0, mu, 1)
sel = Q & (mu - 3 * emu > MU_CUT)
ring = Q & (R > 20) & (R < 90)
rho_bg = np.sum(sel & ring) / (np.pi * (90**2 - 20**2))
print(f"background (selected per arcsec^2 at 20-90\"): {rho_bg:.4f} (Haeberle 0.0026 with CMD); "
      f"per quality star: {np.sum(sel & ring) / ring.sum():.2e}")
for n, i in FAST.items():
    j = np.nonzero(ids == i)[0][0]
    print(f"  {n}: R={R[j]:.2f}\" mu={mu[j]:.3f}±{emu[j]:.3f}  ast={q_ast[j]} cmd={q_cmd[j]} selected={sel[j]}")
for r0, r1 in ((0, 3), (3, 10)):
    m = (R >= r0) & (R < r1)
    print(f"  {r0}-{r1}\": quality {np.sum(Q & m)} (Haeberle 3-10\": 2090), selected {np.sum(sel & m)}: "
          + ", ".join(f"{ids[j]}@{R[j]:.2f}\"" for j in np.nonzero(sel & m)[0]))
out = [x for x in sys.argv[1:] if not x.startswith("--")]
if out:
    m = Q & (R < 10)
    np.savez(out[0], id=ids[m], dx=dx[m], dy=dy[m], pmx=pmx[m], pmy=pmy[m], ex=ex[m], ey=ey[m], selected=sel[m], c2x=c2x[m], c2y=c2y[m], v2d=mu[m] * KMS, e2d=emu[m] * KMS, bg_v2d=mu[sel & ring] * KMS, rho_bg=rho_bg, bg_per_star=np.sum(sel & ring) / ring.sum(),
             f606w=mag["F606W"][m], f814w=mag["F814W"][m])
    print("wrote", out[0], m.sum())
