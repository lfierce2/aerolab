import numpy as np
import pytest
from part2pop import build_optical_population, build_population

from pop2obs.observations import optics

WAVELENGTHS = [450e-9, 550e-9, 700e-9]


@pytest.fixture(scope="module")
def population():
    return build_population(
        {
            "type": "binned_lognormals",
            "N": [1e9],
            "GMD": [100e-9],
            "GSD": [1.6],
            "N_bins": 40,
            "aero_spec_names": [["SO4", "BC"]],
            "aero_spec_fracs": [[0.9, 0.1]],
        }
    )


def test_record_describes_the_coefficients(population):
    obs = optics(population, WAVELENGTHS)
    assert obs.name == "optics"
    assert obs.dims == ("coefficient", "wavelength")
    assert list(obs.coords["coefficient"]) == ["scattering", "absorption"]
    assert np.array_equal(obs.coords["wavelength"], WAVELENGTHS)
    assert obs.values.shape == (2, 3)
    assert obs.units == "m^-1"


def test_values_match_part2pop_optics(population):
    config = {
        "type": "core_shell",
        "rh_grid": [0.0],
        "wvl_grid": WAVELENGTHS,
        "temp": 298.15,
    }
    reference = build_optical_population(population, config)
    obs = optics(population, WAVELENGTHS)
    assert np.allclose(obs.values[0], reference.get_optical_coeff("b_scat")[0])
    assert np.allclose(obs.values[1], reference.get_optical_coeff("b_abs")[0])


def test_scattering_falls_with_wavelength_and_bc_absorbs(population):
    scattering, absorption = optics(population, WAVELENGTHS).values
    assert (np.diff(scattering) < 0).all()
    assert (absorption > 0).all()


def test_a_population_without_bc_does_not_absorb():
    population = build_population(
        {
            "type": "binned_lognormals",
            "N": [1e9],
            "GMD": [100e-9],
            "GSD": [1.6],
            "N_bins": 20,
            "aero_spec_names": [["SO4"]],
            "aero_spec_fracs": [[1.0]],
        }
    )
    absorption = optics(population, WAVELENGTHS).values[1]
    assert np.allclose(absorption, 0.0, atol=1e-12)
