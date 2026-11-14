import numpy as np
import pytest
from part2pop import ParticlePopulation, get_species


@pytest.fixture
def population_from():
    """Build a population from (dry diameter, {species: mass fraction}, number conc)."""

    def build(particles):
        names = []
        for _, fractions, _ in particles:
            names += [name for name in fractions if name not in names]
        species = tuple(get_species(name, None) for name in [*names, "H2O"])
        density = {
            name: s.density for name, s in zip(names, species[:-1], strict=True)
        }
        masses = np.zeros((len(particles), len(species)))
        for i, (diameter, fractions, _) in enumerate(particles):
            volume = np.pi / 6 * diameter**3
            total = volume / sum(f / density[name] for name, f in fractions.items())
            for name, fraction in fractions.items():
                masses[i, names.index(name)] = fraction * total
        return ParticlePopulation(
            species=species,
            spec_masses=masses,
            num_concs=np.array([p[2] for p in particles], dtype=float),
            ids=list(range(len(particles))),
        )

    return build
