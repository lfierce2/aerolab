"""The README's training-pair flow: noisy instrument inputs, noise-free targets."""

import numpy as np
from part2pop import build_population

import pop2obs
from pop2obs.observations import size_resolved_composition

EDGES = np.logspace(-8, -6, 21)  # target size bins, 10 nm to 1 um
INSTRUMENTS = ["smps", "acsm", "optics"]


def make_pairs(geometric_mean_diameters):
    inputs = {name: [] for name in INSTRUMENTS}
    targets, masks = [], []
    for i, gmd in enumerate(geometric_mean_diameters):
        population = build_population(
            {
                "type": "binned_lognormals",
                "N": [1e9],
                "GMD": [gmd],
                "GSD": [1.6],
                "N_bins": 40,
                "aero_spec_names": [["SO4", "BC"]],
                "aero_spec_fracs": [[0.7, 0.3]],
            }
        )
        for k, name in enumerate(INSTRUMENTS):
            x = pop2obs.observe(name, population, noise=True, seed=100 * i + k)
            inputs[name].append(x.values)
        y = size_resolved_composition(population, EDGES, species=["SO4", "BC"])
        targets.append(y.values)
        masks.append(y.mask)
    inputs = {name: np.array(values) for name, values in inputs.items()}
    return inputs, np.array(targets), np.array(masks)


def test_pairs_have_consistent_shapes_and_valid_targets():
    inputs, targets, masks = make_pairs([60e-9, 90e-9, 130e-9])
    assert inputs["smps"].shape == (3, 60)
    assert inputs["acsm"].shape == (3, 5)
    assert inputs["optics"].shape == (3, 2, 3)
    assert all(np.isfinite(values).all() for values in inputs.values())
    assert targets.shape == masks.shape == (3, 2, 20)
    assert np.array_equal(np.isnan(targets), ~masks)
    assert np.allclose(targets[:, 0][masks[:, 0]], 0.7)
    assert np.allclose(targets[:, 1][masks[:, 1]], 0.3)


def test_pairs_are_reproducible():
    first, _, _ = make_pairs([60e-9, 90e-9])
    second, _, _ = make_pairs([60e-9, 90e-9])
    assert all(np.array_equal(first[name], second[name]) for name in INSTRUMENTS)
