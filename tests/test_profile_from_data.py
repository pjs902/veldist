import json

import numpy as np
import pytest

from veldist.calibration import OMEGACAT, ObservingProfile


def _synthetic_field(rng, n_bins=40, n_per_bin=150, err_median=2.5, err_log_sigma=0.4):
    """A fake catalogue with known profile parameters.

    Dispersion falls linearly from 22 to 7 km/s outwards and the mean velocity
    swings over a 10 km/s rotation span, matching OMEGACAT by construction so
    the test asserts against numbers with a physical meaning.
    """
    sigmas = np.linspace(22.0, 7.0, n_bins)
    means = np.linspace(-5.0, 5.0, n_bins)
    vel, err, bin_ids = [], [], []
    for i in range(n_bins):
        e = np.exp(rng.normal(np.log(err_median), err_log_sigma, size=n_per_bin))
        v = rng.normal(means[i], sigmas[i], size=n_per_bin) + rng.normal(0.0, e)
        vel.append(v)
        err.append(e)
        bin_ids.append(np.full(n_per_bin, i))
    return np.concatenate(vel), np.concatenate(err), np.concatenate(bin_ids)


def test_from_data_recovers_known_profile():
    rng = np.random.default_rng(20260811)
    vel, err, bin_ids = _synthetic_field(rng)

    prof = ObservingProfile.from_data(vel, err, bin_ids, name="synthetic")

    assert prof.name == "synthetic"
    assert prof.n_stars == 150
    assert prof.err_median == pytest.approx(2.5, rel=0.1)
    assert prof.err_log_sigma == pytest.approx(0.4, rel=0.15)
    assert prof.sigma_max == pytest.approx(22.0, rel=0.15)
    assert prof.sigma_min == pytest.approx(7.0, rel=0.25)
    assert prof.rotation_span == pytest.approx(10.0, abs=4.0)


def test_from_data_skips_underpopulated_bins():
    """A 3-star bin gives a meaningless dispersion; including it would drag
    sigma_min to near zero and corrupt every downstream grid calculation."""
    rng = np.random.default_rng(1)
    vel, err, bin_ids = _synthetic_field(rng, n_bins=10)
    vel = np.concatenate([vel, np.array([0.0, 1.0, -1.0])])
    err = np.concatenate([err, np.array([2.0, 2.0, 2.0])])
    bin_ids = np.concatenate([bin_ids, np.array([99, 99, 99])])

    prof = ObservingProfile.from_data(vel, err, bin_ids, min_stars=10)

    assert prof.sigma_min > 5.0


def test_from_data_rejects_too_few_bins():
    rng = np.random.default_rng(2)
    vel, err, bin_ids = _synthetic_field(rng, n_bins=1)
    with pytest.raises(ValueError, match="at least 2"):
        ObservingProfile.from_data(vel, err, bin_ids)


def test_json_round_trip(tmp_path):
    path = tmp_path / "profile.json"
    OMEGACAT.to_json(path)
    restored = ObservingProfile.from_json(path)

    assert restored == OMEGACAT


def test_json_is_human_readable(tmp_path):
    """The fixture gets code-reviewed, so it must be diffable."""
    path = tmp_path / "profile.json"
    OMEGACAT.to_json(path)
    payload = json.loads(path.read_text())

    assert payload["name"] == "oMEGACat"
    assert payload["n_stars"] == 150
    assert "\n" in path.read_text()  # indented, not one line
