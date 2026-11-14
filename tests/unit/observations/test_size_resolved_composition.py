import numpy as np
import pytest
from part2pop import ParticlePopulation, get_species

from pop2obs.observations import size_resolved_composition
from pop2obs.observations.size_resolved_composition import (
    dry_diameters,
    mass_fractions_by_size,
)

NAMES = ("SO4", "BC", "H2O")
EDGES = np.array([10e-9, 100e-9, 1000e-9, 10000e-9])
# (dry diameter in m, SO4 mass fraction of the dry mass, number concentration in 1/m^3)
PARTICLES = [(30e-9, 1.0, 2e9), (60e-9, 0.0, 1e9), (300e-9, 0.5, 1e8)]


def make_population(particles):
    species = tuple(get_species(name, None) for name in NAMES)
    rho_so4, rho_bc = species[0].density, species[1].density
    masses = []
    for diameter, f_so4, _ in particles:
        volume = np.pi / 6 * diameter**3
        mass = volume / (f_so4 / rho_so4 + (1 - f_so4) / rho_bc)
        masses.append([f_so4 * mass, (1 - f_so4) * mass, 0.0])
    return ParticlePopulation(
        species=species,
        spec_masses=np.array(masses),
        num_concs=np.array([p[2] for p in particles]),
        ids=list(range(len(particles))),
    )


def reference_fractions(population, members):
    weights = population.num_concs[members]
    dry = (weights[:, None] * population.spec_masses[members][:, :2]).sum(axis=0)
    return dry / dry.sum()


def test_fractions_follow_the_weighted_mass_formula():
    population = make_population(PARTICLES)
    fractions, bin_dry_mass, bin_number, names = mass_fractions_by_size(
        population, EDGES
    )
    assert names == ["SO4", "BC"]
    assert np.allclose(fractions[:, 0], reference_fractions(population, [0, 1]))
    assert np.allclose(fractions[:, 1], reference_fractions(population, [2]))
    assert bin_number[:2] == pytest.approx([3e9, 1e8])
    assert bin_dry_mass[2] == 0


def test_empty_bins_are_nan_and_masked():
    obs = size_resolved_composition(make_population(PARTICLES), EDGES)
    assert np.isnan(obs.values[:, 2]).all()
    assert not obs.mask[:, 2].any()
    assert obs.mask[:, :2].all()
    assert obs.extras["bin_number"][2] == 0


def test_fractions_of_all_dry_species_sum_to_one():
    obs = size_resolved_composition(make_population(PARTICLES), EDGES)
    assert np.allclose(obs.values[:, :2].sum(axis=0), 1.0)


def test_a_subset_is_of_the_total_dry_mass_and_absent_species_are_zero():
    population = make_population(PARTICLES)
    subset, *_ = mass_fractions_by_size(population, EDGES, species=["bc", "NH4"])
    full, *_ = mass_fractions_by_size(population, EDGES)
    assert np.allclose(subset[0], full[1], equal_nan=True)
    assert (subset[1, :2] == 0).all()


def test_water_changes_neither_sizes_nor_fractions():
    dry = make_population(PARTICLES)
    wet = make_population(PARTICLES)
    wet.spec_masses[:, 2] = 1e-17
    assert np.allclose(dry_diameters(dry), dry_diameters(wet))
    a, *_ = mass_fractions_by_size(dry, EDGES)
    b, *_ = mass_fractions_by_size(wet, EDGES)
    assert np.allclose(a, b, equal_nan=True)


def test_requesting_water_is_an_error():
    with pytest.raises(ValueError, match="water"):
        mass_fractions_by_size(make_population(PARTICLES), EDGES, species=["H2O"])


def test_particles_outside_the_edges_are_ignored():
    population = make_population([*PARTICLES, (20000e-9, 0.5, 1e9)])
    _, _, bin_number, _ = mass_fractions_by_size(population, EDGES)
    assert bin_number.sum() == pytest.approx(3.1e9)


def test_edges_must_increase():
    with pytest.raises(ValueError, match="increasing"):
        mass_fractions_by_size(make_population(PARTICLES), EDGES[::-1])


def test_dry_diameters_match_part2pop():
    population = make_population(PARTICLES)
    part2pop_values = [population.get_particle(i).get_Ddry() for i in population.ids]
    assert np.allclose(dry_diameters(population), part2pop_values, rtol=1e-12)
    assert np.allclose(dry_diameters(population), [p[0] for p in PARTICLES], rtol=1e-9)


def test_record_describes_the_grid():
    obs = size_resolved_composition(make_population(PARTICLES), EDGES)
    assert obs.name == "size_resolved_composition"
    assert obs.dims == ("species", "diameter")
    assert list(obs.coords["species"]) == ["SO4", "BC"]
    assert obs.values.shape == (2, 3)
    assert obs.noise_free is None
    assert obs.uncertainty is None
    assert obs.availability == "model only"
