"""Quick checks of the fast-star term (seconds; the recovery study is run_recovery.py on the node)."""
import numpy as np

from dfmock import _trapz

import dfmock as d
import likelihood as lk

REAL = "../work/quality_R10.npz"


def test_kernels():
    rng = np.random.default_rng(0)
    e, vt = 2.0, 70.0
    vo = np.hypot(vt + e * rng.standard_normal(200000), e * rng.standard_normal(200000))
    edges = np.array([69.0, 71.0])
    # selection: P(v_obs > cut + 3e) for true v2D in [69, 71] vs Monte Carlo at 70
    s = lk.sel_prob(d.CUT, np.array([e]), edges)[0, 0]
    assert abs(s - np.mean(vo > d.CUT + 3 * e)) < 0.01, s
    # v2D kernel integrates to 1 over v_obs
    vgrid = np.linspace(40, 100, 3001)
    K = lk.v2d_kernel(vgrid, np.full(vgrid.size, e), edges)[:, 0]
    assert abs(_trapz(K, vgrid) - 1) < 1e-3
    # v_los kernel of a |v| cell integrates to 1 over signed v_los
    Kz = lk.vlos_kernel(np.linspace(-150, 150, 3001), np.full(3001, 5.0), np.array([20.0, 30.0]))[:, 0]
    assert abs(_trapz(Kz, np.linspace(-150, 150, 3001)) - 1) < 1e-3


def test_rv_conditional_normalised():
    """term (b) is p(v_los | v2D): integrating its exp over v_los must give 1 for any table."""
    rng = np.random.default_rng(1)
    P = rng.random((len(d.R_EDGES) - 1, len(d.V_EDGES) - 1, len(d.W_EDGES) - 1)) ** 4
    P /= P.sum((1, 2), keepdims=True)
    vz = np.linspace(-450, 450, 1801)
    st = dict(x=np.full(vz.size, 0.6), y=np.zeros(vz.size), v2d=np.full(vz.size, 80.0), e2d=np.full(vz.size, 1.5),
              selected=np.zeros(vz.size, bool), vlos=vz, elos=np.full(vz.size, 6.0))
    t = lk.FastStarTerm(st, d.R_EDGES, d.V_EDGES, d.W_EDGES, d.CUT)
    a, _ = t.annulus()
    num = np.einsum("kij,ki,kj->k", P[a], t.Kv, t.Kz)
    den = np.einsum("ki,ki->k", P.sum(-1)[a], t.Kv)
    assert abs(_trapz(num / den, vz) - 1) < 2e-3


def test_mock_recovery_quick():
    """One mock at M = 2.5e4: the chi2 minimum over a coarse grid must bracket the truth."""
    real = dict(np.load(REAL))
    rng = np.random.default_rng(2)
    obs = d.draw_observed(d.Cluster(2.5e4), real, rng)
    t = lk.FastStarTerm(obs, d.R_EDGES, d.V_EDGES, d.W_EDGES, d.CUT, bg=d.bg_density(real))
    grid = [0.8e4, 1.6e4, 2.5e4, 4e4, 8e4]
    c2 = [t.chi2(d.model_table(d.Cluster(m), d.R_EDGES, d.V_EDGES, d.W_EDGES, n_per=4000, seed=7)) for m in grid]
    print("selected", obs["selected"].sum(), "contaminants", obs["contaminant"].sum(),
          "chi2 - min:", np.round(np.array(c2) - min(c2), 1))
    k = int(np.argmin(c2))
    assert 0 < k < len(grid) - 1 or grid[k] == 2.5e4, (grid[k], c2)


if __name__ == "__main__":
    for f in (test_kernels, test_rv_conditional_normalised, test_mock_recovery_quick):
        f()
        print(f.__name__, "ok")
