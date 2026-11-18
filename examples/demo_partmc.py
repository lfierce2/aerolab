"""Demo: synthetic observations from a PartMC-MOSAIC population.

PartMC output file -> part2pop population -> pop2obs observations -> figure.

The bundled output (data/partmc) is one snapshot (timestep 12, t = 39600 s, 6205
computational particles, 20 species) from a PartMC 2.1.5 run named urban_plume.
Instrument settings are placeholders.

Usage:
    python examples/demo_partmc.py [PARTMC_DIR] [TIMESTEP] [--seed N]
        [--output FILE] [--combined] [--smps-volume-factor N]
        [--noise-free-output FILE] [--with-noise-output FILE]

PARTMC_DIR is a run folder that contains out/ (default: the bundled example).
If TIMESTEP is omitted, the latest matching NetCDF timestep is used.
"""

from __future__ import annotations

import argparse
import csv
import re
import tempfile
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from part2pop import build_population

import pop2obs
from pop2obs.instruments.smps import VOLUME as SMPS_VOLUME
from pop2obs.noise import counting_noise, gaussian_noise
from pop2obs.observations import size_resolved_composition

HERE = Path(__file__).resolve().parent
EDGES = np.logspace(-8, -5, 31)  # target size bins, 10 nm to 10 um
INSTRUMENTS = ["smps", "acsm", "optics"]


NC_PATTERN = re.compile(
    r"^(?P<prefix>.+)_(?P<repeat>\d+)_(?P<timestep>\d+)\.nc$"
)


def find_partmc_file(partmc_dir, timestep=None, repeat=1):
    """Find a PartMC NetCDF file without assuming its filename prefix."""
    output_dir = Path(partmc_dir) / "out"
    if not output_dir.is_dir():
        raise FileNotFoundError(f"PartMC output directory does not exist: {output_dir}")

    candidates = []
    for path in output_dir.rglob("*.nc"):
        match = NC_PATTERN.match(path.name)
        if match and int(match["repeat"]) == repeat:
            candidates.append((int(match["timestep"]), path))
    if not candidates:
        raise FileNotFoundError(
            f"No PartMC NetCDF files for repeat {repeat} found in {output_dir}"
        )

    if timestep is None:
        return max(candidates, key=lambda item: item[0])

    matches = [item for item in candidates if item[0] == timestep]
    if not matches:
        available = sorted(item[0] for item in candidates)
        raise FileNotFoundError(
            f"No PartMC NetCDF file for repeat {repeat}, timestep {timestep}; "
            f"available timesteps: {available}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple PartMC NetCDF files match repeat {repeat}, timestep "
            f"{timestep}: {[str(item[1]) for item in matches]}"
        )
    return matches[0]


def load_population(partmc_dir, timestep=None, repeat=1):
    """Load PartMC output with an explicit file-to-part2pop handoff.

    part2pop 1.1 intentionally constructs the filename from a small set of
    known prefixes. The demo resolves the actual PartMC file first, then
    presents that file using a supported PartMC filename pattern. The user's
    PartMC directory is never changed.
    """
    selected_timestep, source_path = find_partmc_file(partmc_dir, timestep, repeat)
    with tempfile.TemporaryDirectory(prefix="pop2obs-partmc-") as staging:
        staging_root = Path(staging)
        staged_out = staging_root / "out"
        staged_out.mkdir()
        staged_path = staged_out / (
            f"urban_plume_{repeat:04d}_{selected_timestep:08d}.nc"
        )
        staged_path.symlink_to(source_path.resolve())
        population = build_population(
            {
                "type": "partmc",
                "partmc_dir": str(staging_root),
                "timestep": selected_timestep,
                "repeat": repeat,
            }
        )
    return population, selected_timestep


def _as_ssa(observation):
    """Convert scattering/absorption coefficients to single-scattering albedo."""
    scattering, absorption = observation.values
    denominator = scattering + absorption
    values = np.divide(
        scattering,
        denominator,
        out=np.full_like(scattering, np.nan),
        where=denominator != 0,
    )
    values = np.clip(values, 0.0, 1.0)
    if observation.noise_free is None:
        expected = values
        uncertainty = np.zeros_like(values)
    else:
        expected_scattering, expected_absorption = observation.noise_free
        expected_denominator = expected_scattering + expected_absorption
        expected = np.divide(
            expected_scattering,
            expected_denominator,
            out=np.full_like(expected_scattering, np.nan),
            where=expected_denominator != 0,
        )
        expected = np.clip(expected, 0.0, 1.0)
        sigma_s, sigma_a = observation.uncertainty
        uncertainty = np.sqrt(
            (absorption / denominator**2 * sigma_s) ** 2
            + (scattering / denominator**2 * sigma_a) ** 2
        )
    mask = np.all(observation.mask, axis=0) if observation.mask is not None else None
    return replace(
        observation,
        name="single_scattering_albedo",
        values=values,
        dims=("wavelength",),
        coords={"wavelength": observation.coords["wavelength"]},
        units="1",
        uncertainty=uncertainty,
        noise_free=expected if observation.noise_free is not None else None,
        mask=mask,
        provenance={**observation.provenance, "quantity": "ssa"},
    )


def observe_all(population, seed, smps_volume_factor=10, optics_output="coefficients"):
    """Each instrument's noise-free observation, and one with noise layered on."""
    clean = {name: pop2obs.observe(name, population) for name in INSTRUMENTS}
    noisy = {
        name: pop2obs.observe(name, population, noise=True, seed=seed + k)
        for k, name in enumerate(INSTRUMENTS)
    }
    # A smaller counted volume produces visibly larger Poisson error bars.
    if smps_volume_factor <= 0:
        raise ValueError("smps_volume_factor must be positive")
    noisy["smps"] = counting_noise(
        clean["smps"], SMPS_VOLUME / smps_volume_factor, seed=seed
    )
    if optics_output == "ssa":
        clean["optics"] = _as_ssa(clean["optics"])
        noisy["optics"] = _as_ssa(noisy["optics"])
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
    if optics.provenance.get("quantity") == "ssa":
        print("optics SSA at", np.round(optics.coords["wavelength"] * 1e9), "nm")
        print("  noise-free", np.round(optics.expected, 4))
        print("  with noise ", np.round(optics.values, 4))
    else:
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


COMBINED_FIELDS = [
    "product",
    "quantity",
    "index_0",
    "index_1",
    "index_2",
    "coord_0",
    "coord_1",
    "coord_2",
    "edge_left",
    "edge_right",
    "value",
    "expected",
    "uncertainty",
    "mask",
    "units",
]


def _csv_rows(
    product,
    quantity,
    values,
    dims,
    coords,
    expected=None,
    uncertainty=None,
    mask=None,
    units="",
):
    """Yield tidy CSV rows for an array-valued observation."""
    values = np.asarray(values)
    expected = None if expected is None else np.asarray(expected)
    uncertainty = None if uncertainty is None else np.asarray(uncertainty)
    mask = None if mask is None else np.asarray(mask)
    coordinate_arrays = {name: np.asarray(value) for name, value in coords.items()}
    edges = coordinate_arrays.get("edges")

    for index in np.ndindex(values.shape):
        row = {field: "" for field in COMBINED_FIELDS}
        row.update(
            {
                "product": product,
                "quantity": quantity,
                "value": values[index],
                "units": units,
            }
        )
        for axis, position in enumerate(index):
            row[f"index_{axis}"] = position
            if axis < len(dims) and dims[axis] in coordinate_arrays:
                row[f"coord_{axis}"] = coordinate_arrays[dims[axis]][position]
            if axis == 0 and edges is not None and dims[axis] == "diameter":
                row["edge_left"] = edges[position]
                row["edge_right"] = edges[position + 1]
        if expected is not None:
            row["expected"] = expected[index]
        if uncertainty is not None and uncertainty.size:
            row["uncertainty"] = uncertainty[index]
        if mask is not None:
            row["mask"] = mask[index]
        yield row


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _combined_observation_rows(name, observation, noise_free=False):
    rows = list(
        _csv_rows(
            "instrument",
            name,
            observation.values,
            observation.dims,
            observation.coords,
            expected=observation.expected,
            uncertainty=(
                np.zeros_like(observation.values)
                if noise_free
                else observation.uncertainty
            ),
            mask=observation.mask,
            units=observation.units,
        )
    )
    for extra_name, extra_values in observation.extras.items():
        rows.extend(
            _csv_rows(
                "instrument_extra",
                f"{name}:{extra_name}",
                extra_values,
                ("diameter",),
                observation.coords,
                units=observation.units,
            )
        )
    return rows


def _combined_target_rows(target, noise_free=False):
    rows = list(
        _csv_rows(
            "target",
            "size_resolved_composition",
            target.values,
            target.dims,
            target.coords,
            expected=target.expected,
            uncertainty=(
                np.zeros_like(target.values) if noise_free else target.uncertainty
            ),
            mask=target.mask,
            units=target.units,
        )
    )
    for extra_name, extra_values in target.extras.items():
        rows.extend(
            _csv_rows(
                "target_extra",
                f"target:{extra_name}",
                extra_values,
                ("diameter",),
                target.coords,
            )
        )
    return rows


def _combined_case_rows(observations, target, noise_free=False):
    """Return all observation and target rows in the combined-file order."""
    rows = []
    for name, observation in observations.items():
        rows.extend(_combined_observation_rows(name, observation, noise_free))
    rows.extend(_combined_target_rows(target, noise_free))
    return rows


def _valid_values(observation):
    if observation.mask is None:
        return np.ones_like(observation.values, dtype=bool)
    return np.asarray(observation.mask, dtype=bool)


def _human_observation_rows(name, observation, noise_free):
    """Return Excel-friendly rows with physical coordinates and units."""
    values = observation.values
    expected = observation.expected
    uncertainty = np.zeros_like(values) if noise_free else observation.uncertainty
    valid = _valid_values(observation)

    if name == "smps":
        diameter = np.asarray(observation.coords["diameter"]) * 1e9
        edges = np.asarray(observation.coords["edges"]) * 1e9
        scale = 1e-6  # m^-3 to cm^-3
        return [
            {
                "diameter_nm": diameter[i],
                "diameter_bin_left_nm": edges[i],
                "diameter_bin_right_nm": edges[i + 1],
                "number_concentration_cm3_per_dlnD": values[i] * scale,
                "expected_number_concentration_cm3_per_dlnD": expected[i] * scale,
                "uncertainty_cm3_per_dlnD": uncertainty[i] * scale,
                "valid": valid[i],
            }
            for i in range(values.size)
        ]

    if name == "acsm":
        species = np.asarray(observation.coords["species"])
        scale = 1e9  # kg m^-3 to ug m^-3
        return [
            {
                "species": species[i],
                "mass_concentration_ug_m3": values[i] * scale,
                "expected_mass_concentration_ug_m3": expected[i] * scale,
                "uncertainty_ug_m3": uncertainty[i] * scale,
                "above_detection_limit": valid[i],
            }
            for i in range(values.size)
        ]

    if name == "optics":
        if observation.provenance.get("quantity") == "ssa":
            wavelengths = np.asarray(observation.coords["wavelength"]) * 1e9
            return [
                {
                    "wavelength_nm": wavelengths[i],
                    "single_scattering_albedo": values[i],
                    "expected_single_scattering_albedo": expected[i],
                    "uncertainty_ssa": uncertainty[i],
                    "valid": valid[i],
                }
                for i in range(values.size)
            ]
        coefficients = np.asarray(observation.coords["coefficient"])
        wavelengths = np.asarray(observation.coords["wavelength"]) * 1e9
        scale = 1e6  # m^-1 to Mm^-1
        return [
            {
                "coefficient": coefficients[i],
                "wavelength_nm": wavelengths[j],
                "coefficient_Mm^-1": values[i, j] * scale,
                "expected_coefficient_Mm^-1": expected[i, j] * scale,
                "uncertainty_Mm^-1": uncertainty[i, j] * scale,
                "valid": valid[i, j],
            }
            for i in range(values.shape[0])
            for j in range(values.shape[1])
        ]

    raise ValueError(f"No human-readable CSV layout is defined for {name!r}")


def _human_target_rows(target, noise_free):
    """Return one row per diameter bin with one column per species."""
    diameter = np.asarray(target.coords["diameter"]) * 1e9
    edges = np.asarray(target.coords["edges"]) * 1e9
    valid = np.asarray(target.mask).any(axis=0)
    uncertainty = (
        np.zeros_like(target.values) if noise_free else target.uncertainty
    )
    rows = []
    for i in range(target.values.shape[1]):
        row = {
            "diameter_nm": diameter[i],
            "diameter_bin_left_nm": edges[i],
            "diameter_bin_right_nm": edges[i + 1],
            "valid": valid[i],
            "bin_dry_mass_kg_m3": target.extras["bin_dry_mass"][i],
            "bin_number_m3": target.extras["bin_number"][i],
        }
        for species_index, species in enumerate(target.coords["species"]):
            row[f"{species}_dry_mass_fraction"] = target.values[species_index, i]
            row[f"{species}_uncertainty"] = (
                uncertainty[species_index, i]
                if uncertainty is not None
                else 0.0
            )
        rows.append(row)
    return rows


def save_case(path_stem, observations, target, combined=False, noise_free=False):
    """Save a case as readable separate CSVs, or one combined tidy CSV."""
    path_stem = Path(path_stem)
    if path_stem.suffix == ".csv":
        path_stem = path_stem.with_suffix("")

    if combined:
        path = path_stem.with_suffix(".csv")
        _write_csv(
            path,
            _combined_case_rows(observations, target, noise_free),
            COMBINED_FIELDS,
        )
        return {"combined": path}

    paths = {}
    for name, observation in observations.items():
        rows = _human_observation_rows(name, observation, noise_free)
        path = path_stem.with_name(f"{path_stem.name}_{name}.csv")
        _write_csv(path, rows, list(rows[0]))
        paths[name] = path

    rows = _human_target_rows(target, noise_free)
    path = path_stem.with_name(f"{path_stem.name}_size_resolved_composition.csv")
    _write_csv(path, rows, list(rows[0]))
    paths["size_resolved_composition"] = path
    return paths


def plot(clean, noisy, target, path, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(title)

    ax = axes[0, 0]
    smps, smps_noisy = clean["smps"], noisy["smps"]
    diameter = smps.coords["diameter"] * 1e9
    smps_values = smps_noisy.values * 1e-6
    smps_uncertainty = smps_noisy.uncertainty * 1e-6
    # Poisson counts are nonnegative, but a symmetric one-sigma interval can
    # extend below zero. Use an asymmetric lower bar for this visualization.
    smps_yerr = np.vstack(
        [np.minimum(smps_uncertainty, smps_values), smps_uncertainty]
    )
    ax.plot(diameter, smps.expected * 1e-6, color="0.3", label="noise-free")
    ax.errorbar(
        diameter,
        smps_values,
        yerr=smps_yerr,
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
    if optics.provenance.get("quantity") == "ssa":
        ax.plot(wavelength, optics.expected, color="0.3", label="noise-free")
        ax.errorbar(
            wavelength,
            optics_noisy.values,
            yerr=optics_noisy.uncertainty,
            fmt="o",
            color="C0",
            label="with noise",
        )
        ax.set(
            xlabel="wavelength (nm)",
            ylabel="single-scattering albedo",
            ylim=(0, 1),
            title="Optics: single-scattering albedo",
        )
        ax.legend()
    else:
        scattering = np.asarray(optics.coords["coefficient"]) == "scattering"
        absorption = np.asarray(optics.coords["coefficient"]) == "absorption"
        ax_abs = ax.twinx()
        for mask, axis, color, label in [
            (scattering, ax, "C0", "scattering"),
            (absorption, ax_abs, "C1", "absorption"),
        ]:
            row = np.flatnonzero(mask)[0]
            axis.plot(
                wavelength,
                optics.expected[row] * 1e6,
                color=color,
                label=f"{label} noise-free",
            )
            axis.errorbar(
                wavelength,
                optics_noisy.values[row] * 1e6,
                yerr=optics_noisy.uncertainty[row] * 1e6,
                fmt="o",
                color=color,
                label=f"{label} with noise",
            )
        ax.set(xlabel="wavelength (nm)", ylabel="scattering (Mm$^{-1}$)", color='C0')
        ax_abs.set_ylabel("absorption (Mm$^{-1}$)", color="C1")
        ax.set_title("Optics: scattering and absorption")
        handles, labels = ax.get_legend_handles_labels()
        handles_abs, labels_abs = ax_abs.get_legend_handles_labels()
        ax.legend(handles + handles_abs, labels + labels_abs, loc="best")

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
    parser.add_argument(
        "timestep",
        nargs="?",
        type=int,
        default=None,
        help="PartMC timestep; defaults to the latest file in out/",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smps-volume-factor",
        type=float,
        default=10,
        help="divide the nominal SMPS counted volume by this factor (default: 10)",
    )
    parser.add_argument(
        "--optics-output",
        choices=("coefficients", "ssa"),
        default="coefficients",
        help="write scattering/absorption coefficients or single-scattering albedo",
    )
    parser.add_argument("--output", type=Path, default=HERE / "out" / "demo_partmc.png")
    parser.add_argument(
        "--combined",
        action="store_true",
        help="write one CSV for each case instead of one CSV per variable",
    )
    parser.add_argument("--noise-free-output", type=Path)
    parser.add_argument("--with-noise-output", type=Path)
    args = parser.parse_args(argv)

    noise_free_output = args.noise_free_output or args.output.with_name(
        "demo_partmc_noise_free"
    )
    with_noise_output = args.with_noise_output or args.output.with_name(
        "demo_partmc_with_noise"
    )

    plt.switch_backend("Agg")
    population, selected_timestep = load_population(args.partmc_dir, args.timestep)
    clean, noisy = observe_all(
        population,
        args.seed,
        smps_volume_factor=args.smps_volume_factor,
        optics_output=args.optics_output,
    )
    target = size_resolved_composition(population, EDGES)
    noisy_target = gaussian_noise(
        target, relative=0.2, absolute=0.02, seed=args.seed + 3
    )
    print_summary(population, clean, noisy, target)
    noise_free_paths = save_case(
        noise_free_output, clean, target, combined=args.combined, noise_free=True
    )
    with_noise_paths = save_case(
        with_noise_output, noisy, noisy_target, combined=args.combined
    )
    title = f"PartMC-MOSAIC timestep {selected_timestep}: synthetic observations"
    plot(clean, noisy, target, args.output, title + " (placeholder settings)")
    print(f"figure: {args.output}")
    for name, path in noise_free_paths.items():
        print(f"noise-free {name}: {path}")
    for name, path in with_noise_paths.items():
        print(f"with-noise {name}: {path}")


if __name__ == "__main__":
    main()
