"""Optics-like instrument: dry scattering and absorption at three wavelengths.

Placeholders for the team to review: the wavelengths, the relative error and the
detection limit. Truncation and filter-based corrections are not modelled.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..noise import detection_limit, gaussian_noise, gaussian_uncertainty
from ..observations import optics as optics_coefficients
from ..records import Observation

WAVELENGTHS = np.array([450e-9, 550e-9, 700e-9])  # m
RH = 0.0  # dry sample
MORPHOLOGY = "core_shell"
RELATIVE = 0.1  # relative one-sigma error
LIMIT = 1e-6  # detection limit, m^-1 (1 Mm^-1)
FLOOR = LIMIT / 3  # absolute one-sigma noise floor


def observe(population, noise: bool = False, seed: int | None = None) -> Observation:
    """What the instrument reports for a population; noise is layered on if asked."""
    obs = optics_coefficients(population, WAVELENGTHS, rh=RH, morphology=MORPHOLOGY)
    obs = replace(
        obs,
        uncertainty=gaussian_uncertainty(obs, RELATIVE, FLOOR),
        availability="common",
        provenance={**obs.provenance, "instrument": "optics"},
    )
    if noise:
        obs = gaussian_noise(obs, RELATIVE, FLOOR, seed)
    return detection_limit(obs, LIMIT)
