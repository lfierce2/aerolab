"""An observation: values on a grid, with units, noise level and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

import numpy as np


def package_versions() -> dict[str, str]:
    """Versions of pop2obs and part2pop, recorded in provenance."""
    found = {}
    for name in ("pop2obs", "part2pop"):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "unknown"
    return found


@dataclass
class Observation:
    """One observation of one type, on a grid.

    values is what is reported, noisy if noise was applied. noise_free holds the values
    before noise and is None if none was applied. mask is True where values are valid.
    availability says where this exists in real data: "common", "special campaigns"
    or "model only".
    """

    name: str
    values: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]
    units: str
    uncertainty: np.ndarray | None = None
    noise_free: np.ndarray | None = None
    mask: np.ndarray | None = None
    availability: str = "model only"
    provenance: dict = field(default_factory=dict)
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def expected(self) -> np.ndarray:
        """The noise-free values."""
        return self.values if self.noise_free is None else self.noise_free

    def to_dict(self) -> dict:
        """Arrays and metadata as a plain dict, for saving in any format."""
        return {
            "name": self.name,
            "values": self.values,
            "expected": self.expected,
            "uncertainty": self.uncertainty,
            "mask": self.mask,
            "dims": list(self.dims),
            "coords": dict(self.coords),
            "units": self.units,
            "availability": self.availability,
            "provenance": dict(self.provenance),
            "extras": dict(self.extras),
        }
