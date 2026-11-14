"""Noise layered on a noise-free observation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .records import Observation


def _scale(obs: Observation, volume: float) -> np.ndarray:
    """Counts per unit of dN/dlnD in each bin: ln-width times counted volume."""
    if volume <= 0:
        raise ValueError("volume must be positive")
    try:
        edges = np.asarray(obs.coords["edges"], dtype=float)
    except KeyError:
        message = "counting noise needs bin edges in obs.coords['edges']"
        raise ValueError(message) from None
    return np.log(edges[1:] / edges[:-1]) * volume


def counting_uncertainty(obs: Observation, volume: float) -> np.ndarray:
    """One-sigma Poisson counting noise on a number size distribution.

    volume is the air volume counted per bin in m^3, including transmission and
    detection efficiency.
    """
    scale = _scale(obs, volume)
    return np.sqrt(obs.expected * scale) / scale


def counting_noise(
    obs: Observation, volume: float, seed: int | None = None
) -> Observation:
    """Draw Poisson counting noise. The noise-free values move to noise_free."""
    if obs.noise_free is not None:
        raise ValueError("noise was already applied to this observation")
    scale = _scale(obs, volume)
    noisy = np.random.default_rng(seed).poisson(obs.values * scale) / scale
    noise = {"model": "counting", "volume": volume, "seed": seed}
    return replace(
        obs,
        values=noisy,
        noise_free=obs.values,
        uncertainty=counting_uncertainty(obs, volume),
        provenance={**obs.provenance, "noise": noise},
    )


def gaussian_uncertainty(obs: Observation, relative=0.0, absolute=0.0) -> np.ndarray:
    """One-sigma Gaussian noise: a relative error and an absolute floor in quadrature.

    relative and absolute may be arrays that broadcast against the values, for
    example one floor per species.
    """
    return np.hypot(relative * np.abs(obs.expected), absolute)


def gaussian_noise(
    obs: Observation, relative=0.0, absolute=0.0, seed: int | None = None
) -> Observation:
    """Draw Gaussian noise. The noise-free values move to noise_free."""
    if obs.noise_free is not None:
        raise ValueError("noise was already applied to this observation")
    sigma = gaussian_uncertainty(obs, relative, absolute)
    rng = np.random.default_rng(seed)
    noisy = obs.values + sigma * rng.normal(size=obs.values.shape)
    noise = {
        "model": "gaussian",
        "relative": np.asarray(relative, dtype=float).tolist(),
        "absolute": np.asarray(absolute, dtype=float).tolist(),
        "seed": seed,
    }
    return replace(
        obs,
        values=noisy,
        noise_free=obs.values,
        uncertainty=sigma,
        provenance={**obs.provenance, "noise": noise},
    )


def detection_limit(obs: Observation, limit) -> Observation:
    """Mask values below a detection limit; the values themselves are kept.

    limit is a number or an array that broadcasts against the values.
    """
    mask = obs.values >= limit
    if obs.mask is not None:
        mask = mask & obs.mask
    detection = np.asarray(limit, dtype=float).tolist()
    return replace(
        obs, mask=mask, provenance={**obs.provenance, "detection_limit": detection}
    )
