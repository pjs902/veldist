"""Prior-predictive measurement: does a mode-order split scale free h3/h4?

No MCMC, no data. Pure numpy reimplementation of generate_gaussian_core_curve
with the single sigma3 replaced by two scales acting on a mode-order
decomposition of the projected RW-k deviation.
"""
import numpy as np

N_BINS = 37
GRID_CENTER, GRID_WIDTH = 200.0, 400.0
RW_ORDER = 3
NDRAW = 20000
TIGHT = 50.0  # Exp rate meaning "switched off"
BASE = 0.35  # SIGMA3_RATE

bin_width = GRID_WIDTH / N_BINS
centers = GRID_CENTER - GRID_WIDTH / 2 + (np.arange(N_BINS) + 0.5) * bin_width
span = centers.max() - centers.min()
mid = centers.mean()


def deviation_basis(n, order):
    """Modes of the projected k-fold-integrated RW, ordered smooth -> wiggly.

    Returns columns B[:, j] with std 1 per mode amplitude, Sorbye-Rue scaled
    so the geometric mean of per-bin marginal variance of the full sum is 1.
    """
    idx = np.arange(n, dtype=float)
    u = (idx - idx.mean()) / (n - 1)
    q, _ = np.linalg.qr(np.stack([u**k for k in range(order)], axis=1))
    proj = np.eye(n) - q @ q.T
    a = np.linalg.matrix_power(np.tril(np.ones((n, n))), order)
    cov = proj @ (a @ a.T) @ proj.T
    scale = 1.0 / np.sqrt(np.exp(np.mean(np.log(np.clip(np.diag(cov), 1e-300, None)))))
    w, v = np.linalg.eigh(cov)
    o = np.argsort(w)[::-1]  # descending eigenvalue == smoothest first
    return scale * v[:, o] * np.sqrt(np.clip(w[o], 0, None))


B = deviation_basis(N_BINS, RW_ORDER)


def draw(rate_lo, rate_hi, split, n=NDRAW, seed=0):
    """Prior-predictive PDFs. split=0 => single scale rate_hi (baseline)."""
    rng = np.random.default_rng(seed)
    v0 = rng.normal(mid, span / 4.0, n)
    s0 = np.exp(rng.normal(np.log(span / 8.0), 1.0, n))
    core = -0.5 * ((centers[None, :] - v0[:, None]) / np.clip(s0, 1e-3, None)[:, None]) ** 2
    z = rng.normal(size=(n, N_BINS))
    s = np.empty((n, N_BINS))
    s[:, :split] = rng.exponential(1.0 / rate_lo, n)[:, None]
    s[:, split:] = rng.exponential(1.0 / rate_hi, n)[:, None]
    curve = core + (z * s) @ B.T
    curve -= curve.max(axis=1, keepdims=True)
    e = np.exp(curve)
    return e / e.sum(axis=1, keepdims=True)


def stats(pdf):
    m = pdf @ centers
    d = centers[None, :] - m[:, None]
    var = np.einsum("ij,ij->i", pdf, d**2)
    sk = np.einsum("ij,ij->i", pdf, d**3) / var**1.5
    ku = np.einsum("ij,ij->i", pdf, d**4) / var**2 - 3.0
    ok = np.isfinite(sk) & np.isfinite(ku)
    degen = float(np.mean(np.sqrt(var[np.isfinite(var)]) < bin_width))
    return (
        float(np.percentile(np.abs(sk[ok]), 90)),
        float(np.percentile(np.abs(ku[ok]), 90)),
        degen,
    )


# Reference denominators: the current adopted prior (single scale, Exp(0.35)).
ref_sk, ref_ku, ref_dg = stats(draw(BASE, BASE, 0))
off_sk, off_ku, off_dg = stats(draw(TIGHT, TIGHT, 0))
print(f"baseline single-scale Exp({BASE}): p90|h3|={ref_sk:.3f} p90|h4|={ref_ku:.3f} degen={ref_dg:.3f}")
print(f"deviation off (Exp({TIGHT})):      p90|h3|={off_sk:.3f} p90|h4|={off_ku:.3f} degen={off_dg:.3f}")
print(f"=> published-style retention h3={off_sk/ref_sk:.3f} h4={off_ku/ref_ku:.3f}\n")

print(f"{'rate_lo':>8} {'rate_hi':>8} {'split':>6} {'p90|h3|':>8} {'p90|h4|':>8} {'h3_ret':>7} {'h4_ret':>7} {'degen':>6}")
def roughness(pdf):
    """p90 of RMS 3rd difference of the PDF -- the wiggle we want suppressed."""
    m = pdf @ centers
    d = centers[None, :] - m[:, None]
    keep = np.sqrt(np.einsum("ij,ij->i", pdf, d**2)) > bin_width  # drop degenerate spikes
    r = np.sqrt(np.mean(np.diff(pdf, n=3, axis=1) ** 2, axis=1))[keep]
    return float(np.percentile(r, 90))


print("roughness (p90 RMS 3rd difference of PDF):")
print(f"  baseline single-scale Exp({BASE})   : {roughness(draw(BASE, BASE, 0)):.5f}")
for _split in (1, 2, 3, 4):
    print(f"  split={_split} lo=Exp({BASE}) hi=Exp({TIGHT}): {roughness(draw(BASE, TIGHT, _split)):.5f}")
print(f"  deviation fully off              : {roughness(draw(TIGHT, TIGHT, 0)):.5f}\n")

for rate_hi in (TIGHT, 5.0):
    for rate_lo in (BASE, 0.15, 0.05):
        for split in (1, 2, 3, 4, 6, 10):
            sk, ku, dg = stats(draw(rate_lo, rate_hi, split))
            print(
                f"{rate_lo:8.2f} {rate_hi:8.2f} {split:6d} {sk:8.3f} {ku:8.3f} "
                f"{sk/ref_sk:7.3f} {ku/ref_ku:7.3f} {dg:6.3f}"
            )
