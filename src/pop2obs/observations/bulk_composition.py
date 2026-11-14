"""Bulk species mass concentrations, optionally for particles below a size cut."""

from __future__ import annotations

import numpy as np

from ..records import Observation, package_versions
from ._particles import dry_diameters, dry_species


def bulk_composition(population, groups=None, max_diameter=None) -> Observation:
    """Mass concentration in kg/m^3 of each species group, on a dry basis.

    groups maps a reported name to the population species it sums, for example
    {"Org": ["OC", "ARO1"], "SO4": ["SO4"]}. A list reports each species on its own,
    and None reports every dry species. Species missing from the population add 0.
    max_diameter (m) keeps only particles with a dry diameter at or below it.
    """
    names = [s.name for s in population.species]
    upper = [name.upper() for name in names]
    if groups is None:
        dry = dry_species(population)
        groups = [name for name, is_dry in zip(names, dry, strict=True) if is_dry]
    if not isinstance(groups, dict):
        groups = {name: [name] for name in groups}
    groups = {
        name: [members] if isinstance(members, str) else list(members)
        for name, members in groups.items()
    }
    if any(m.upper() == "H2O" for members in groups.values() for m in members):
        message = "bulk composition is on a dry basis; water cannot be requested"
        raise ValueError(message)

    weights = np.asarray(population.num_concs, dtype=float)
    masses = np.asarray(population.spec_masses, dtype=float)
    if max_diameter is not None:
        keep = dry_diameters(population) <= max_diameter
        weights, masses = weights[keep], masses[keep]
    totals = (weights[:, None] * masses).sum(axis=0)
    values = np.array(
        [
            sum(totals[upper.index(m.upper())] for m in members if m.upper() in upper)
            for members in groups.values()
        ],
        dtype=float,
    )
    return Observation(
        name="bulk_composition",
        values=values,
        dims=("species",),
        coords={"species": np.array(list(groups))},
        units="kg m^-3",
        provenance={
            "observation": "bulk_composition",
            "groups": groups,
            "max_diameter": max_diameter,
            "versions": package_versions(),
        },
    )
