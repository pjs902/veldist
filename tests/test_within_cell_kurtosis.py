"""The within-cell (Sheppard) correction for the FOURTH cumulant.

`compute_summary` used to divide an UNCORRECTED 4th central moment by a
CORRECTED sigma**4. Those two choices are inconsistent, and the inconsistency
is a deterministic bias of roughly `-0.5 * (h/sigma)**2` in excess kurtosis --
on MUSE's grid at its narrowest bins (h/sigma = 0.78) it reported -0.272
excess kurtosis for an exact, noiseless Gaussian.

The reference direction is the part that is easy to get backwards, and getting
it backwards is what produced the h**2/6 gap fixed on 2026-09-01:

  * the FITTED pdf P satisfies `P * U ~= f`, so `kappa_n(f) = kappa_n(P) +
    kappa_n(U)` -- ADD `h**2/12` to k2 and ADD `-h**4/120` to k4.
  * the TRUTH's cell masses are the binned f, so the same terms are
    SUBTRACTED.

`compute_summary` is applied to fitted pdfs, so it takes the first. These
tests pin that, without needing a fit: pick discrete masses P on the grid,
define `f = P * U` analytically, and require the estimator applied to P to
return f's continuous cumulants.
"""

import numpy as np
import pytest

from veldist.analysis import compute_summary

H = 8.09  # MUSE's measured cell width, km/s
CENTERS = np.arange(-11, 12) * H


def _continuous_cumulants(p, centers, h, n=2_000_001):
    """kappa_2..4 of `f = sum_m p_m Uniform(centre_m, h)`, by dense quadrature."""
    v = np.linspace(centers[0] - h, centers[-1] + h, n)
    dens = np.zeros_like(v)
    for pm, c in zip(p, centers):
        dens += pm * ((v >= c - h / 2) & (v < c + h / 2)) / h
    dens /= np.trapezoid(dens, v)
    m1 = np.trapezoid(v * dens, v)
    d = v - m1
    m2 = np.trapezoid(d**2 * dens, v)
    m3 = np.trapezoid(d**3 * dens, v)
    m4 = np.trapezoid(d**4 * dens, v)
    return m2, m3, m4 - 3 * m2**2


CASES = {
    "gaussian_ish": np.exp(-0.5 * (CENTERS / 22.0) ** 2),
    "narrow": np.exp(-0.5 * (CENTERS / 11.0) ** 2),
    "heavy_tailed": 1.0 / (1.0 + (CENTERS / 18.0) ** 2) ** 1.5,
    "skewed": np.exp(-0.5 * (CENTERS / 20.0) ** 2) * (1 + 0.8 * np.tanh(CENTERS / 25.0)),
    "flat_top": np.exp(-0.5 * (CENTERS / 26.0) ** 4),
}


@pytest.mark.parametrize("name", list(CASES))
def test_moments_recover_the_convolved_truth(name):
    """Exact for every shape: the correction is additive in cumulants, so it
    carries no shape dependence at this order."""
    p = CASES[name] / CASES[name].sum()
    k2, k3, k4 = _continuous_cumulants(p, CENTERS, H)
    s = compute_summary(p[None, :], CENTERS, bin_width=H)

    # 1e-4 rather than machine precision: the residual is trapezoid error on a
    # piecewise-constant density, not estimator error.
    assert s["sigma"][0] == pytest.approx(np.sqrt(k2), abs=1e-4)
    assert s["skewness"][0] == pytest.approx(k3 / k2**1.5, abs=1e-4)
    assert s["kurtosis"][0] == pytest.approx(k4 / k2**2, abs=1e-4)


def test_uncorrected_normalisation_would_bias_a_gaussian():
    """The regression this guards. Dividing the raw 4th moment by the
    corrected sigma**4 gives ~-0.5*(h/sigma)**2; at MUSE's narrowest bins that
    is -0.27 excess kurtosis on a noiseless Gaussian, which the MUSE recovery
    curve was misreading as prior shrinkage (it is flat in information
    content, exactly like real shrinkage)."""
    p = CASES["narrow"] / CASES["narrow"].sum()
    s = compute_summary(p[None, :], CENTERS, bin_width=H)

    means = p @ CENTERS
    d = CENTERS - means
    var_pm = p @ d**2
    sigma_corrected = np.sqrt(var_pm + H**2 / 12)
    old = (p @ d**4) / sigma_corrected**4 - 3.0

    assert old < -0.15, "the old estimator's bias should be large here"
    assert abs(s["kurtosis"][0] - old) > 0.15
    assert abs(s["kurtosis"][0]) < 0.05
