"""Per-particle helpers shared by the observation types."""

from __future__ import annotations

import numpy as np


def dry_species(population) -> np.ndarray:
    """Boolean mask of the species that count as dry: not water, with a density."""
    return np.array(
        [s.name.upper() != "H2O" and s.density > 0 for s in population.species]
    )


def dry_diameters(population) -> np.ndarray:
    """Dry diameter of every particle in m, from species masses and densities."""
    dry = dry_species(population)
    density = np.array([s.density for s in population.species], dtype=float)
    masses = np.asarray(population.spec_masses, dtype=float)
    volume = (masses[:, dry] / density[dry]).sum(axis=1)
    return (6.0 * volume / np.pi) ** (1.0 / 3.0)
