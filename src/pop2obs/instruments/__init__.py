"""Instruments: one module per instrument, looked up by name."""

from __future__ import annotations

from . import acsm, optics, smps

_INSTRUMENTS = {"acsm": acsm, "optics": optics, "smps": smps}


def list_instruments() -> list[str]:
    """Names of the available instruments."""
    return sorted(_INSTRUMENTS)


def observe(name: str, population, noise: bool = False, seed: int | None = None):
    """What the named instrument reports for a population."""
    try:
        instrument = _INSTRUMENTS[name]
    except KeyError:
        available = ", ".join(list_instruments())
        message = f"unknown instrument {name!r}; available: {available}"
        raise ValueError(message) from None
    return instrument.observe(population, noise=noise, seed=seed)
