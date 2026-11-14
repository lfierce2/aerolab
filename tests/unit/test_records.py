import numpy as np

from pop2obs import Observation


def make(**changes):
    fields = {
        "name": "size_distribution",
        "values": np.array([1.0, 2.0]),
        "dims": ("diameter",),
        "coords": {"diameter": np.array([1e-8, 2e-8])},
        "units": "m^-3",
    }
    return Observation(**{**fields, **changes})


def test_expected_is_the_values_when_no_noise_was_applied():
    obs = make()
    assert obs.expected is obs.values


def test_expected_is_the_noise_free_values_after_noise():
    obs = make(values=np.array([1.5, 1.5]), noise_free=np.array([1.0, 2.0]))
    assert np.array_equal(obs.expected, [1.0, 2.0])


def test_to_dict_holds_arrays_and_metadata():
    obs = make(uncertainty=np.array([0.1, 0.1]), availability="common")
    out = obs.to_dict()
    assert out["name"] == "size_distribution"
    assert out["availability"] == "common"
    assert out["dims"] == ["diameter"]
    assert np.array_equal(out["expected"], obs.values)
    assert np.array_equal(out["uncertainty"], [0.1, 0.1])


def test_availability_defaults_to_model_only():
    assert make().availability == "model only"
