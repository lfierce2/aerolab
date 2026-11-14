import numpy as np
import pytest
from part2pop import build_population

import pop2obs
from pop2obs.instruments import smps

CONFIG = {
    "type": "binned_lognormals",
    "N": [1e9],
    "GMD": [100e-9],
    "GSD": [1.6],
    "N_bins": 40,
    "aero_spec_names": [["SO4"]],
    "aero_spec_fracs": [[1.0]],
}


@pytest.fixture(scope="module")
def population():
    return build_population(CONFIG)


def test_noise_free_observation(population):
    obs = pop2obs.observe("smps", population)
    assert obs.name == "size_distribution"
    assert obs.availability == "common"
    assert obs.provenance["instrument"] == "smps"
    assert obs.noise_free is None
    assert obs.values.shape == (len(smps.EDGES) - 1,)
    assert obs.uncertainty.shape == obs.values.shape


def test_noise_is_layered_on_the_same_observation(population):
    clean = pop2obs.observe("smps", population)
    noisy = pop2obs.observe("smps", population, noise=True, seed=1)
    assert np.array_equal(noisy.noise_free, clean.values)
    assert np.array_equal(noisy.expected, clean.expected)
    assert not np.array_equal(noisy.values, clean.values)


def test_noise_is_reproducible_by_seed(population):
    a = pop2obs.observe("smps", population, noise=True, seed=5).values
    b = pop2obs.observe("smps", population, noise=True, seed=5).values
    assert np.array_equal(a, b)


def test_instruments_are_listed_and_looked_up_by_name(population):
    assert pop2obs.list_instruments() == ["acsm", "optics", "smps"]
    with pytest.raises(ValueError, match="unknown instrument"):
        pop2obs.observe("nope", population)
