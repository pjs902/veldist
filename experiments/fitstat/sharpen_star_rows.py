"""Sharpen the per-star rows of an existing PM star-data npz in place of a re-export: rows -> rows @ S.

usage: python sharpen_star_rows.py IN.npz OUT.npz
Same operation and sparsification as veldist.dynamite2d._laplace_export(sharpen=True). S (sharpen_matrix) is a
Kronecker product of one symmetric persymmetric 1D stencil, so it commutes with axis order and with the x-East flip
of pm2: the stored (flipped) rows can be sharpened directly. h0 is unchanged (the writer does not sharpen it).
"""
import sys

import numpy as np

from veldist.fitstat import sharpen_matrix

z = dict(np.load(sys.argv[1]))
assert not bool(z.get("PM_star_sharpened", False)), "already sharpened"
K = z["PM_lap_h0"].shape[1]
S = sharpen_matrix(K)
ptr, cell, mass = z["PM_star_indptr"], z["PM_star_cell"], z["PM_star_mass"]
n = len(ptr) - 1
indptr, cells, masses = [0], [], []
for s0 in range(0, n, 8192):
    s1 = min(n, s0 + 8192)
    D = np.zeros((s1 - s0, K * K))
    for i in range(s0, s1):
        D[i - s0, cell[ptr[i]:ptr[i + 1]]] = mass[ptr[i]:ptr[i + 1]]
    for row in D @ S:
        a = np.abs(row)
        nz = np.nonzero(a >= 1e-6 * a.max())[0]
        cells.append(nz.astype(np.int32))
        masses.append(row[nz].astype(np.float32))
        indptr.append(indptr[-1] + len(nz))
# self-check: interior rows keep their mass (S preserves the sum of a row away from the edges)
before = np.add.reduceat(mass.astype(float), ptr[:-1])
z.update(PM_star_indptr=np.asarray(indptr, np.int64), PM_star_cell=np.concatenate(cells),
         PM_star_mass=np.concatenate(masses), PM_star_sharpened=np.asarray(True))
after = np.add.reduceat(z["PM_star_mass"].astype(float), z["PM_star_indptr"][:-1])
print(f"{n} stars; row-sum change median {np.median(after - before):+.2e}, max |.| {np.abs(after - before).max():.2e}")
np.savez(sys.argv[2], **z)
