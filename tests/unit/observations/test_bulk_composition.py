import numpy as np
import pytest

from pop2obs.observations import bulk_composition

PARTICLES = [
    (50e-9, {"SO4": 0.6, "NH4": 0.2, "OC": 0.2}, 1e9),
    (200e-9, {"SO4": 0.5, "BC": 0.5}, 5e8),
    (2000e-9, {"SO4": 1.0}, 1e7),
]


def test_species_masses_match_part2pop(population_from):
    population = population_from(PARTICLES)
    obs = bulk_composition(population)
    names = list(obs.coords["species"])
    assert names == ["SO4", "NH4", "OC", "BC"]
    expected = [population.get_mass_conc(name) for name in names]
    assert np.allclose(obs.values, expected, rtol=1e-12)


def test_groups_sum_their_members_and_absent_species_add_nothing(population_from):
    population = population_from(PARTICLES)
    obs = bulk_composition(population, {"Org": ["OC", "ARO1"], "Sulfate": "SO4"})
    assert list(obs.coords["species"]) == ["Org", "Sulfate"]
    assert obs.values[0] == pytest.approx(population.get_mass_conc("OC"))
    assert obs.values[1] == pytest.approx(population.get_mass_conc("SO4"))


def test_a_size_cut_drops_larger_particles(population_from):
    population = population_from(PARTICLES)
    cut = bulk_composition(population, ["SO4"], max_diameter=1e-6)
    full = bulk_composition(population, ["SO4"])
    dropped = population.num_concs[2] * population.spec_masses[2, 0]
    assert full.values[0] - cut.values[0] == pytest.approx(dropped)
    assert cut.provenance["max_diameter"] == 1e-6


def test_water_cannot_be_requested(population_from):
    with pytest.raises(ValueError, match="water"):
        bulk_composition(population_from(PARTICLES), ["H2O"])


def test_record_describes_the_species(population_from):
    obs = bulk_composition(population_from(PARTICLES), ["SO4", "BC"])
    assert obs.name == "bulk_composition"
    assert obs.dims == ("species",)
    assert obs.units == "kg m^-3"
    assert obs.noise_free is None
    assert obs.uncertainty is None
    assert obs.availability == "model only"
