from dataclasses import replace

import numpy as np
import pytest

from pop2obs import Observation
from pop2obs.noise import (
    counting_noise,
    counting_uncertainty,
    detection_limit,
    gaussian_noise,
    gaussian_uncertainty,
)

EDGES = np.logspace(-8, -6, 11)
VOLUME = 1e-5


def size_distribution(values):
    return Observation(
        name="size_distribution",
        values=np.asarray(values, dtype=float),
        dims=("diameter",),
        coords={"diameter": np.sqrt(EDGES[:-1] * EDGES[1:]), "edges": EDGES},
        units="m^-3",
    )


def test_noise_keeps_the_noise_free_values():
    obs = size_distribution(np.full(10, 1e9))
    noisy = counting_noise(obs, VOLUME, seed=1)
    assert np.array_equal(noisy.noise_free, obs.values)
    assert not np.array_equal(noisy.values, obs.values)
    assert np.array_equal(noisy.uncertainty, counting_uncertainty(obs, VOLUME))


def test_noise_is_reproducible_by_seed():
    obs = size_distribution(np.full(10, 1e9))
    a = counting_noise(obs, VOLUME, seed=3).values
    b = counting_noise(obs, VOLUME, seed=3).values
    c = counting_noise(obs, VOLUME, seed=4).values
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_noise_matches_the_stated_uncertainty():
    obs = size_distribution(np.full(10, 1e9))
    draws = np.array([counting_noise(obs, VOLUME, seed=s).values for s in range(500)])
    sigma = counting_uncertainty(obs, VOLUME)
    assert np.allclose(draws.mean(axis=0), obs.values, rtol=0.01)
    assert np.allclose(draws.std(axis=0), sigma, rtol=0.2)


def test_empty_bins_stay_empty():
    values = np.zeros(10)
    values[4] = 1e9
    noisy = counting_noise(size_distribution(values), VOLUME, seed=1)
    assert np.count_nonzero(noisy.values) == 1
    assert noisy.values[4] > 0


def test_noise_cannot_be_applied_twice():
    noisy = counting_noise(size_distribution(np.full(10, 1e9)), VOLUME, seed=1)
    with pytest.raises(ValueError, match="already"):
        counting_noise(noisy, VOLUME)


def test_counting_noise_needs_bin_edges():
    obs = Observation(name="x", values=np.ones(3), dims=("d",), coords={}, units="1")
    with pytest.raises(ValueError, match="edges"):
        counting_noise(obs, VOLUME)


def test_counted_volume_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        counting_noise(size_distribution(np.ones(10)), 0.0)


def composition(values):
    return Observation(
        name="bulk_composition",
        values=np.asarray(values, dtype=float),
        dims=("species",),
        coords={"species": np.array(["a", "b", "c"])},
        units="kg m^-3",
    )


def test_gaussian_uncertainty_adds_relative_and_floor_in_quadrature():
    obs = composition([3.0, 0.0, 3.0])
    sigma = gaussian_uncertainty(obs, relative=1.0, absolute=4.0)
    assert np.allclose(sigma, [5.0, 4.0, 5.0])


def test_gaussian_noise_matches_the_stated_uncertainty():
    obs = composition([10.0, 20.0, 30.0])
    seeds = range(2000)
    draws = np.array([gaussian_noise(obs, 0.1, 1.0, seed=s).values for s in seeds])
    sigma = gaussian_uncertainty(obs, 0.1, 1.0)
    assert (np.abs(draws.mean(axis=0) - obs.values) < 0.1 * sigma).all()
    assert np.allclose(draws.std(axis=0), sigma, rtol=0.1)


def test_gaussian_noise_is_reproducible_and_keeps_the_noise_free_values():
    obs = composition([10.0, 20.0, 30.0])
    a = gaussian_noise(obs, 0.1, seed=1)
    b = gaussian_noise(obs, 0.1, seed=1)
    assert np.array_equal(a.values, b.values)
    assert np.array_equal(a.noise_free, obs.values)
    with pytest.raises(ValueError, match="already"):
        gaussian_noise(a, 0.1)


def test_detection_limit_masks_but_keeps_the_values():
    obs = composition([1.0, 5.0, 10.0])
    flagged = detection_limit(obs, 4.0)
    assert flagged.mask.tolist() == [False, True, True]
    assert np.array_equal(flagged.values, obs.values)


def test_detection_limit_can_differ_per_entry_and_keeps_an_existing_mask():
    obs = replace(composition([1.0, 5.0, 10.0]), mask=np.array([True, True, False]))
    flagged = detection_limit(obs, np.array([0.5, 6.0, 1.0]))
    assert flagged.mask.tolist() == [True, False, False]
