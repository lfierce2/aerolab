"""ACSM-like instrument: non-refractory PM1 bulk composition with Gaussian noise.

Placeholders for the team to review: the species groups, the 1 um dry-diameter cut,
the relative error and the detection limit. Collection-efficiency bias and the lens
transmission curve are not modelled.
"""

from __future__ import annotations

from dataclasses import replace

from ..noise import detection_limit, gaussian_noise, gaussian_uncertainty
from ..observations import bulk_composition
from ..records import Observation

GROUPS = {
    "Org": [
        "OC", "MSA", "ARO1", "ARO2", "ALK1", "OLE1", "API1", "API2", "LIM1", "LIM2",
        "IEPOX_SOA", "tetrol", "tetrol_olig", "IEPOX_OS", "IEPOX_OH_SOA",
    ],
    "SO4": ["SO4"],
    "NO3": ["NO3"],
    "NH4": ["NH4"],
    "Chl": ["Cl"],
}
MAX_DIAMETER = 1e-6  # dry diameter cut, m
RELATIVE = 0.1  # relative one-sigma error
LIMIT = 1e-10  # detection limit per group, kg/m^3 (0.1 ug/m^3)
FLOOR = LIMIT / 3  # absolute one-sigma noise floor


def observe(population, noise: bool = False, seed: int | None = None) -> Observation:
    """What the instrument reports for a population; noise is layered on if asked."""
    obs = bulk_composition(population, GROUPS, max_diameter=MAX_DIAMETER)
    obs = replace(
        obs,
        uncertainty=gaussian_uncertainty(obs, RELATIVE, FLOOR),
        availability="common",
        provenance={**obs.provenance, "instrument": "acsm"},
    )
    if noise:
        obs = gaussian_noise(obs, RELATIVE, FLOOR, seed)
    return detection_limit(obs, LIMIT)
