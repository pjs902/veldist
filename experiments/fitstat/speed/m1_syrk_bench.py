"""M1: dsyrk/ssyrk thread scaling at production size (one 8192-row block into a 45k Gram).

    MKL_NUM_THREADS=T python m1_syrk_bench.py T OUT.json [N_ORBS ROWS]

Times scipy.linalg.blas {d,s}syrk(trans=1, lower=0) exactly as NormalEquationAccumulator.add
calls it (C Fortran-ordered, in place, beta=1), two repeats each; reports GFLOP/s (n^2 k flops).
"""
import json, sys, time
import numpy as np
from scipy.linalg.blas import dsyrk, ssyrk

T, out = int(sys.argv[1]), sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 45000
k = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
rng = np.random.default_rng(0)
res = {"threads": T, "n": n, "k": k}
for name, f, dt in (("dsyrk", dsyrk, np.float64), ("ssyrk", ssyrk, np.float32)):
    A = np.asfortranarray(rng.random((k, n), dtype=np.float64).astype(dt))
    G = np.zeros((n, n), dtype=dt, order="F")
    ts = []
    for _ in range(2):
        t = time.perf_counter()
        G = f(1.0, A, beta=1.0, c=G, trans=1, lower=0, overwrite_c=1)
        ts.append(time.perf_counter() - t)
    del A, G
    res[name] = {"s": ts, "gflops": n * n * k / min(ts) / 1e9}
    print(name, res[name], flush=True)
json.dump(res, open(out, "w"), indent=1)
