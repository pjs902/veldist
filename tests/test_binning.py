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
