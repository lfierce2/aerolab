"""Species mass fractions by dry particle size, computed from a population.

Parked here until the definition settles. mass_fractions_by_size takes and returns
plain arrays so that it can move to part2pop unchanged.
"""

from __future__ import annotations

import numpy as np

from ..records import Observation, package_versions
from ._particles import dry_diameters, dry_species


def mass_fractions_by_size(population, edges, species=None):
    """Species mass fractions per dry-diameter bin.

    Returns fractions (n_species, n_bins), the dry mass concentration of each bin in
    kg/m^3, its number concentration in 1/m^3, and the species names reported.

    Fractions are of the total dry mass in the bin, so a subset of species does not
    sum to 1. Species missing from the population report 0. Empty bins give NaN, and
    particles outside the edges are ignored. Pass species to fix the list across a
    dataset; by default every dry species in the population is reported.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be increasing with at least two values")
    n_bins = edges.size - 1

    names = [s.name for s in population.species]
    upper = [name.upper() for name in names]
    dry = dry_species(population)
    if species is None:
        species = [name for name, is_dry in zip(names, dry, strict=True) if is_dry]
    if any(name.upper() == "H2O" for name in species):
        raise ValueError("fractions are on a dry basis; water cannot be requested")

    diameters = dry_diameters(population)
    index = np.searchsorted(edges, diameters, side="right") - 1
    index[diameters == edges[-1]] = n_bins - 1  # like numpy.histogram
    inside = (index >= 0) & (index < n_bins)
    index = index[inside]
    weights = np.asarray(population.num_concs, dtype=float)[inside]
    masses = np.asarray(population.spec_masses, dtype=float)[inside]

    def per_bin(values):
        return np.bincount(index, weights=values, minlength=n_bins)

    bin_dry_mass = per_bin(weights * masses[:, dry].sum(axis=1))
    bin_number = per_bin(weights)
    fractions = np.zeros((len(species), n_bins))
    for k, name in enumerate(species):
        if name.upper() in upper:
            fractions[k] = per_bin(weights * masses[:, upper.index(name.upper())])
    with np.errstate(invalid="ignore", divide="ignore"):
        fractions = fractions / bin_dry_mass
    return fractions, bin_dry_mass, bin_number, list(species)


def size_resolved_composition(population, edges, species=None) -> Observation:
    """Species mass fractions per dry-diameter bin, as an observation.

    Empty bins are NaN and masked. extras holds bin_dry_mass (kg/m^3) and
    bin_number (1/m^3), which are useful as weights.
    """
    fractions, bin_dry_mass, bin_number, names = mass_fractions_by_size(
        population, edges, species
    )
    edges = np.asarray(edges, dtype=float)
    return Observation(
        name="size_resolved_composition",
        values=fractions,
        dims=("species", "diameter"),
        coords={
            "species": np.array(names),
            "diameter": np.sqrt(edges[:-1] * edges[1:]),
            "edges": edges,
        },
        units="1",
        mask=np.broadcast_to(bin_dry_mass > 0, fractions.shape).copy(),
        provenance={
            "observation": "size_resolved_composition",
            "basis": "dry mass, dry diameter",
            "versions": package_versions(),
        },
        extras={"bin_dry_mass": bin_dry_mass, "bin_number": bin_number},
    )
