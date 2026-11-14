import numpy as np
import pytest
from part2pop import build_population

import pop2obs
from pop2obs.instruments import optics as optics_instrument


@pytest.fixture(scope="module")
def population():
    return build_population(
        {
            "type": "binned_lognormals",
            "N": [1e9],
            "GMD": [100e-9],
            "GSD": [1.6],
            "N_bins": 40,
            "aero_spec_names": [["SO4", "BC"]],
            "aero_spec_fracs": [[0.9, 0.1]],
        }
    )


def test_reports_scattering_and_absorption_at_three_wavelengths(population):
    obs = pop2obs.observe("optics", population)
    assert obs.name == "optics"
    assert obs.values.shape == (2, 3)
    assert obs.availability == "common"
    assert obs.provenance["instrument"] == "optics"
    assert obs.noise_free is None
    assert obs.uncertainty.shape == obs.values.shape


def test_values_below_the_detection_limit_are_masked(population):
    obs = pop2obs.observe("optics", population)
    assert np.array_equal(obs.mask, obs.values >= optics_instrument.LIMIT)


def test_noise_is_layered_on_and_reproducible(population):
    clean = pop2obs.observe("optics", population)
    noisy = pop2obs.observe("optics", population, noise=True, seed=3)
    again = pop2obs.observe("optics", population, noise=True, seed=3)
    assert np.array_equal(noisy.noise_free, clean.values)
    assert not np.array_equal(noisy.values, clean.values)
    assert np.array_equal(noisy.values, again.values)
