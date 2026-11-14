"""SMPS-like instrument: dry number size distribution with Poisson counting noise.

Not modelled yet: DMA transfer function, charging and diffusion losses.
The size range, bins and counted volume are placeholders.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..noise import counting_noise, counting_uncertainty
from ..observations import size_distribution
from ..records import Observation

EDGES = np.logspace(np.log10(10e-9), np.log10(500e-9), 61)  # m
VOLUME = 1e-5  # counted air volume per bin, m^3


def observe(population, noise: bool = False, seed: int | None = None) -> Observation:
    """What the instrument reports for a population; noise is layered on if asked."""
    obs = size_distribution(population, EDGES, wetsize=False)
    obs = replace(
        obs,
        uncertainty=counting_uncertainty(obs, VOLUME),
        availability="common",
        provenance={**obs.provenance, "instrument": "smps"},
    )
    return counting_noise(obs, VOLUME, seed) if noise else obs
