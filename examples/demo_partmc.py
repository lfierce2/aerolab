"""Demo: synthetic observations from a PartMC-MOSAIC population.

PartMC output file -> part2pop population -> pop2obs observations -> figure.

The bundled output (data/partmc) is the last snapshot (timestep 61, t = 3600 s) of
part2pop's example scenario: an hour of Brownian coagulation of dry ammonium sulfate
in Aitken, accumulation and coarse modes, started from 1000 computational particles.
It is a copy of one file from examples/example_data/model_output/partmc in part2pop.
Instrument settings are placeholders.

Usage:
    python examples/demo_partmc.py [PARTMC_DIR] [TIMESTEP] [--seed N] [--output FILE]

PARTMC_DIR is a run folder that contains out/ (default: the bundled example).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from part2pop import build_population

import pop2obs
from pop2obs.observations import size_resolved_composition

HERE = Path(__file__).resolve().parent
EDGES = np.logspace(-8, -5, 31)  # target size bins, 10 nm to 10 um
INSTRUMENTS = ["smps", "acsm", "optics"]


def load_population(partmc_dir, timestep, repeat=1):
    config = {
        "type": "partmc",
        "partmc_dir": str(partmc_dir),
        "timestep": timestep,
        "repeat": repeat,
    }
    return build_population(config)


def observe_all(population, seed):
    """Each instrument's noise-free observation, and one with noise layered on."""
    clean = {name: pop2obs.observe(name, population) for name in INSTRUMENTS}
    noisy = {
        name: pop2obs.observe(name, population, noise=True, seed=seed + k)
        for k, name in enumerate(INSTRUMENTS)
    }
    return clean, noisy


def print_summary(population, clean, noisy, target):
    names = [s.name for s in population.species]
    print(f"population: {len(population.ids)} particles, ", end="")
    print(f"{population.get_Ntot() * 1e-6:.4g} cm^-3, species {', '.join(names)}")

    smps = clean["smps"]
    edges = smps.coords["edges"]
    in_range = np.sum(smps.values * np.diff(np.log(edges))) * 1e-6
    print(f"smps: {in_range:.4g} cm^-3 between {edges[0] * 1e9:.0f} and ", end="")
    print(f"{edges[-1] * 1e9:.0f} nm")

    acsm = noisy["acsm"]
    print("acsm (ug m^-3):   noise-free   with noise   above detection limit")
    for group, expected, value, seen in zip(
        acsm.coords["species"], acsm.expected, acsm.values, acsm.mask, strict=True
    ):
        flag = "yes" if seen else "no"
        print(f"  {group:<5}{expected * 1e9:14.3f}{value * 1e9:13.3f}   {flag}")

    optics = noisy["optics"]
    print("optics (Mm^-1) at", np.round(optics.coords["wavelength"] * 1e9), "nm")
    for name, expected, values in zip(
        optics.coords["coefficient"], optics.expected, optics.values, strict=True
    ):
        print(f"  {name:<11} noise-free {np.round(expected * 1e6, 3)}", end="")
        print(f"  with noise {np.round(values * 1e6, 3)}")

    mass = target.extras["bin_dry_mass"]
    overall = np.nansum(target.values * mass, axis=1) / mass.sum()
    fractions = ", ".join(
        f"{name} {value:.2f}"
        for name, value in zip(target.coords["species"], overall, strict=True)
        if value > 0
    )
    occupied = int(target.mask[0].sum())
    print(f"target: {occupied} of {target.mask.shape[1]} size bins occupied; ", end="")
    print(f"overall dry-mass fractions {fractions}")


def plot(clean, noisy, target, path, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(title)

    ax = axes[0, 0]
    smps, smps_noisy = clean["smps"], noisy["smps"]
    diameter = smps.coords["diameter"] * 1e9
    ax.plot(diameter, smps.expected * 1e-6, color="0.3", label="noise-free")
    ax.errorbar(
        diameter,
        smps_noisy.values * 1e-6,
        yerr=smps_noisy.uncertainty * 1e-6,
        fmt=".",
        color="C0",
        label="with noise",
    )
    ax.set(
        xscale="log",
        xlabel="dry diameter (nm)",
        ylabel="dN/dlnD (cm$^{-3}$)",
        title="SMPS: size distribution",
    )
    ax.legend()

    ax = axes[0, 1]
    acsm, acsm_noisy = clean["acsm"], noisy["acsm"]
    x = np.arange(len(acsm.values))
    seen = acsm_noisy.mask
    ax.bar(x, acsm.expected * 1e9, color="0.8", label="noise-free")
    for subset, fill, label in [
        (seen, "C0", "with noise"),
        (~seen, "none", "below limit"),
    ]:
        ax.errorbar(
            x[subset],
            acsm_noisy.values[subset] * 1e9,
            yerr=acsm_noisy.uncertainty[subset] * 1e9,
            fmt="o",
            color="C0",
            mfc=fill,
            label=label,
        )
    ax.set_xticks(x, acsm.coords["species"])
    ax.set(
        ylabel="mass concentration ($\\mu$g m$^{-3}$)",
        title="ACSM: bulk composition",
    )
    ax.legend()

    ax = axes[1, 0]
    optics, optics_noisy = clean["optics"], noisy["optics"]
    wavelength = optics.coords["wavelength"] * 1e9
    for row, name in enumerate(optics.coords["coefficient"]):
        ax.plot(wavelength, optics.expected[row] * 1e6, color=f"C{row}", label=name)
        ax.errorbar(
            wavelength,
            optics_noisy.values[row] * 1e6,
            yerr=optics_noisy.uncertainty[row] * 1e6,
            fmt="o",
            color=f"C{row}",
        )
    ax.set(
        xlabel="wavelength (nm)",
        ylabel="coefficient (Mm$^{-1}$)",
        title="Optics: scattering and absorption",
    )
    ax.legend()

    ax = axes[1, 1]
    present = np.nansum(target.values, axis=1) > 0
    ax.stackplot(
        target.coords["diameter"] * 1e9,
        target.values[present],
        labels=target.coords["species"][present],
    )
    ax.set(
        xscale="log",
        ylim=(0, 1),
        xlabel="dry diameter (nm)",
        ylabel="dry-mass fraction",
        title="Target: size-resolved composition",
    )
    ax.legend(loc="lower left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_dir = HERE / "data" / "partmc"
    parser.add_argument("partmc_dir", nargs="?", type=Path, default=default_dir)
    parser.add_argument("timestep", nargs="?", type=int, default=61)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "out" / "demo_partmc.png")
    args = parser.parse_args(argv)

    plt.switch_backend("Agg")
    population = load_population(args.partmc_dir, args.timestep)
    clean, noisy = observe_all(population, args.seed)
    target = size_resolved_composition(population, EDGES)
    print_summary(population, clean, noisy, target)
    title = f"PartMC-MOSAIC timestep {args.timestep}: synthetic observations"
    plot(clean, noisy, target, args.output, title + " (placeholder settings)")
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
