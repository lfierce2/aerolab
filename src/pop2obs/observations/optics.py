"""Scattering and absorption coefficients, from part2pop's optics."""

from __future__ import annotations

import numpy as np
from part2pop import build_optical_population

from ..records import Observation, package_versions

COEFFICIENTS = {"scattering": "b_scat", "absorption": "b_abs"}


def optics(
    population,
    wavelengths,
    rh: float = 0.0,
    morphology: str = "core_shell",
    temperature: float = 298.15,
) -> Observation:
    """Scattering and absorption coefficients in m^-1 at each wavelength in m.

    rh is a fraction, 0 for dry. morphology is a part2pop optics type:
    "homogeneous", "core_shell" or "fractal".
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    config = {
        "type": morphology,
        "rh_grid": [rh],
        "wvl_grid": wavelengths.tolist(),
        "temp": temperature,
    }
    optical = build_optical_population(population, config)
    values = np.array(
        [
            optical.get_optical_coeff(key, rh=None, wvl=None)[0]
            for key in COEFFICIENTS.values()
        ]
    )
    return Observation(
        name="optics",
        values=values,
        dims=("coefficient", "wavelength"),
        coords={"coefficient": np.array(list(COEFFICIENTS)), "wavelength": wavelengths},
        units="m^-1",
        provenance={
            "observation": "optics",
            "rh": rh,
            "morphology": morphology,
            "temperature": temperature,
            "versions": package_versions(),
        },
    )
