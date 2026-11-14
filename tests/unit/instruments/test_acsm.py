import numpy as np
import pytest

import pop2obs
from pop2obs.instruments import acsm

PARTICLES = [
    (50e-9, {"SO4": 0.6, "NH4": 0.2, "OC": 0.2}, 1e9),
    (200e-9, {"SO4": 0.5, "BC": 0.5}, 5e8),
    (2000e-9, {"SO4": 1.0}, 1e7),
]


def test_reports_the_groups_it_sees(population_from):
    obs = pop2obs.observe("acsm", population_from(PARTICLES))
    assert obs.name == "bulk_composition"
    assert list(obs.coords["species"]) == ["Org", "SO4", "NO3", "NH4", "Chl"]
    assert obs.availability == "common"
    assert obs.provenance["instrument"] == "acsm"
    assert obs.noise_free is None
    assert obs.uncertainty.shape == obs.values.shape


def test_is_blind_to_bc_and_to_particles_above_the_cut(population_from):
    population = population_from(PARTICLES)
    obs = pop2obs.observe("acsm", population)
    seen = sum(
        w * m[:3].sum()  # SO4, NH4 and OC columns; BC is the fourth
        for w, m in zip(
            population.num_concs[:2], population.spec_masses[:2], strict=True
        )
    )
    assert obs.values.sum() == pytest.approx(seen)


def test_absent_species_fall_below_the_detection_limit(population_from):
    obs = pop2obs.observe("acsm", population_from(PARTICLES))
    groups = list(obs.coords["species"])
    assert not obs.mask[groups.index("NO3")]
    assert obs.mask[groups.index("SO4")]
    assert np.array_equal(obs.mask, obs.values >= acsm.LIMIT)


def test_noise_is_layered_on_and_reproducible(population_from):
    population = population_from(PARTICLES)
    clean = pop2obs.observe("acsm", population)
    noisy = pop2obs.observe("acsm", population, noise=True, seed=2)
    again = pop2obs.observe("acsm", population, noise=True, seed=2)
    assert np.array_equal(noisy.noise_free, clean.values)
    assert not np.array_equal(noisy.values, clean.values)
    assert np.array_equal(noisy.values, again.values)
