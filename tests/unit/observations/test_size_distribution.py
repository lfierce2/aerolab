import numpy as np
import pytest
from part2pop import build_population

from pop2obs.observations import size_distribution

CONFIG = {
    "type": "binned_lognormals",
    "N": [1e9],
    "GMD": [100e-9],
    "GSD": [1.6],
    "N_bins": 40,
    "aero_spec_names": [["SO4"]],
    "aero_spec_fracs": [[1.0]],
}
EDGES = np.logspace(-8, -6, 31)


def total_number(obs):
    return np.sum(obs.values * np.diff(np.log(obs.coords["edges"])))


def test_integral_recovers_the_total_number():
    population = build_population(CONFIG)
    obs = size_distribution(population, EDGES)
    assert total_number(obs) == pytest.approx(population.get_Ntot(), rel=1e-9)


def test_record_describes_the_grid():
    obs = size_distribution(build_population(CONFIG), EDGES)
    assert obs.name == "size_distribution"
    assert obs.dims == ("diameter",)
    assert obs.values.shape == (30,)
    assert np.array_equal(obs.coords["edges"], EDGES)
    assert obs.noise_free is None
    assert obs.availability == "model only"


def test_particles_outside_the_edges_are_dropped():
    population = build_population(CONFIG)
    edges = np.logspace(np.log10(200e-9), -6, 11)
    assert total_number(size_distribution(population, edges)) < population.get_Ntot()
