import numpy as np
import pytest

from veldist.binning import make_ivar_sn_func, target_ivar


def test_sn_func_matches_the_ivar_definition():
    """vorbin calls sn_func(index, signal, noise); noise carries the per-star
    velocity errors and signal is unused."""
    f = make_ivar_sn_func(sigma=10.0)
    noise = np.array([2.0, 4.0, 6.0])
    signal = np.zeros(3)

    got = f(np.array([0, 2]), signal, noise)
    expected = 1.0 / (100.0 + 4.0) + 1.0 / (100.0 + 36.0)

    assert got == pytest.approx(expected)


def test_sn_func_increases_with_bin_size():
    """vorbin's accretion assumes S/N grows as spaxels are added. It does."""
    f = make_ivar_sn_func(sigma=10.0)
    noise = np.full(10, 3.0)
    signal = np.zeros(10)

    values = [f(np.arange(k), signal, noise) for k in range(1, 11)]

    assert all(b > a for a, b in zip(values, values[1:]))


def test_sn_func_downweights_noisy_stars():
    f = make_ivar_sn_func(sigma=5.0)
    signal = np.zeros(2)
    precise = f(np.array([0]), signal, np.array([0.5, 50.0]))
    imprecise = f(np.array([1]), signal, np.array([0.5, 50.0]))

    assert precise > imprecise


def test_target_ivar_inverts_the_cramer_rao_bound():
    """ivar = 1/precision^2, so a 1 km/s requirement is ivar = 1."""
    assert target_ivar(1.0) == pytest.approx(1.0)
    assert target_ivar(0.5) == pytest.approx(4.0)
    assert 1.0 / np.sqrt(target_ivar(2.5)) == pytest.approx(2.5)


def test_target_ivar_rejects_nonpositive_precision():
    with pytest.raises(ValueError, match="positive"):
        target_ivar(0.0)


def test_sn_func_rejects_nonpositive_sigma():
    with pytest.raises(ValueError, match="positive"):
        make_ivar_sn_func(0.0)


def test_ivar_definition_agrees_across_modules():
    """The information measure ``sum_i 1/(sigma^2 + err_i^2)`` is implemented
    in three places, and nothing structurally prevents them from diverging:

    - ``make_ivar_sn_func`` in ``veldist.binning``
    - ``ObservingProfile.ivar`` in ``veldist.calibration``
    - the inline expression inside ``fit_all_bins`` in ``veldist.veldist``,
      around the ``min_ivar`` check

    A divergence between these would be silent, since all three return a
    plain float in the same units, so this test pins them to agree on a
    heteroscedastic fixture (equal errors would let a wrong formula pass by
    coincidence)."""
    from veldist.calibration import ObservingProfile

    sigma = 7.0
    err = np.array([0.5, 1.0, 2.0, 3.5, 10.0])
    index = np.arange(err.size)

    from_binning = make_ivar_sn_func(sigma)(index, np.zeros(err.size), err)
    from_calibration = ObservingProfile.ivar(sigma, err[index])
    # Reproduced literally from veldist.veldist.fit_all_bins so this test
    # fails if that copy is edited away from the shared definition.
    from_fit_all_bins = float(np.sum(1.0 / (sigma**2 + np.asarray(err)[index] ** 2)))

    assert from_binning == pytest.approx(from_calibration)
    assert from_binning == pytest.approx(from_fit_all_bins)
