# pop2obs

Synthetic observations from aerosol populations.

pop2obs takes a population built with [part2pop](https://github.com/pnnl/part2pop) and returns what an instrument would observe from it, with noise layered on top. It also computes quantities that common instruments do not measure, such as size-resolved composition, from the same population. The paired results are inputs and targets for developing and testing methods that infer population properties from observations, where the true answer is known.

## How it works

```
population  ->  observation (noise-free)  ->  noise  ->  noisy observation
```

- **Population:** a part2pop `ParticlePopulation`, from any source part2pop can build (PartMC, MAM4, lognormal modes, single-particle data).
- **Observation:** a noise-free quantity of one type, computed from the population. Each type is a function in `pop2obs.observations`: `size_distribution`, `bulk_composition`, `optics` (scattering and absorption) and `size_resolved_composition`.
- **Instrument:** one module per instrument in `pop2obs.instruments`: `smps` (size distribution), `acsm` (bulk composition) and `optics` (scattering and absorption). Each names the observation type it reports and applies its own settings (size range, species seen, wavelengths) and its noise. `pop2obs.observe("smps", population)` looks one up by name.
- **Noise:** `pop2obs.noise` adds random error to a noise-free observation (Poisson counting noise, or Gaussian noise with a relative error and an absolute floor) and masks values below a detection limit. A `seed` makes noise reproducible.
- **Record:** every result is an `Observation`: values, grid, units, noise level (`uncertainty`), `mask`, `availability` (where it exists in real data) and provenance. After noise is applied, the noise-free values stay in `noise_free`. `to_dict()` gives plain arrays and metadata to save in any format.

A truth target needs no instrument. It is the same kind of observation, computed directly with no noise.

## How it fits with part2pop

part2pop describes what the population is; pop2obs describes what can be observed of it. pop2obs takes whatever part2pop builds, and calls part2pop where it already computes a quantity (the size distribution uses `dNdlnD`). Quantities part2pop lacks, such as size-resolved composition, are parked here and are expected to move to part2pop once their definition settles. Bulk composition is also computed here, because it needs a size cut and part2pop's `spec_mass_conc` fails in 1.1.0.

## Using it to train a model

A training script loops over populations. It takes the observations the model will see as inputs and the quantity to infer as the target; which is which is the script's choice, not the package's.

```python
import numpy as np
from part2pop import build_population

import pop2obs
from pop2obs.observations import size_resolved_composition

edges = np.logspace(-8, -6, 21)  # target size bins, 10 nm to 1 um
instruments = ["smps", "acsm", "optics"]
inputs = {name: [] for name in instruments}
targets, masks = [], []
for i, gmd in enumerate([60e-9, 90e-9, 130e-9]):
    population = build_population({
        "type": "binned_lognormals", "N": [1e9], "GMD": [gmd], "GSD": [1.6],
        "N_bins": 40, "aero_spec_names": [["SO4", "BC"]],
        "aero_spec_fracs": [[0.7, 0.3]],
    })
    for k, name in enumerate(instruments):  # a different seed for each instrument
        x = pop2obs.observe(name, population, noise=True, seed=100 * i + k)  # input
        inputs[name].append(x.values)
    y = size_resolved_composition(population, edges, species=["SO4", "BC"])  # target
    targets.append(y.values)
    masks.append(y.mask)
```

- Split by scenario rather than by sample, so held-out scenarios are independent simulations.
- Use a different seed per instrument and sample so noise is independent, and pass `species=` so every sample has the same species list.
- Targets carry a `mask` for empty bins; inputs carry one for values below the detection limit. Each instrument keeps its own array and units, so scaling and combining them is the model's job.
- Writing arrays to files (NetCDF, NumPy) is left to the script; colleagues training their own model need only those files. A training package can instead import pop2obs and use the noise-free values and noise level kept in each record.

## Demo

`python examples/demo_partmc.py` runs the three instruments on a bundled PartMC-MOSAIC snapshot, prints a summary and writes a four-panel figure to `examples/out/`. It also accepts any PartMC run folder and timestep: `python examples/demo_partmc.py PARTMC_DIR TIMESTEP`. If the timestep is omitted, the demo selects the latest matching NetCDF file in `PARTMC_DIR/out/` and discovers its filename prefix rather than assuming a particular PartMC scenario name.

By default, the demo writes separate, Excel-friendly CSV files for each
instrument and the target beside the figure. The separated files use physical
quantities rather than array indices:

- SMPS: diameter and bin edges in nm; number concentration in
  `cm^-3 per dlnD`.
- ACSM: species and mass concentration in `ug m^-3`.
- Optics: coefficient, wavelength in nm, and coefficient in `Mm^-1`.
- Size-resolved composition: diameter and bin edges in nm, one dry-mass
  fraction column per species, and the bin mass/number weights.

Each noisy file contains the reported value, the noise-free expected value, the
one-sigma uncertainty, and a validity/detection-limit flag. The explicitly
noise-free files contain the noise-free value in both value and expected-value
columns and have uncertainty set to zero. The instrument's theoretical
uncertainty is still available in the combined machine-oriented output, but it
is not presented as realized noise in the noise-free human-readable files.

The default SMPS demo uses a counted volume 10 times smaller than the nominal
placeholder volume, making the Poisson error bars visible without reducing
most bins to zero. Adjust it with
`--smps-volume-factor`; larger values mean more noise. For example:

```bash
python examples/demo_partmc.py --smps-volume-factor 1000
```

The optics figure and CSV contain scattering and absorption in separate
physical units/scales. Use `--optics-output ssa` to compute single-scattering
albedo instead:

```bash
python examples/demo_partmc.py --optics-output ssa
```

SSA is computed as `scattering / (scattering + absorption)`, with uncertainty
propagated from the noisy scattering and absorption coefficients.

Use `--combined` to write one tidy CSV per case instead:

```
python examples/demo_partmc.py --combined
```

This writes `demo_partmc_noise_free.csv` and `demo_partmc_with_noise.csv`.
Custom filename stems can be supplied with `--noise-free-output` and
`--with-noise-output` in either mode.

## Workflows

pop2obs is a library: it needs a population and nothing else. Running ensembles of
simulations and turning them into datasets, on a laptop or a cluster, is the job of a
separate workflow package, `popflow`. It chains
AMBRS, PartMC-MOSAIC, part2pop and pop2obs:

```text
AMBRS → PartMC-MOSAIC → part2pop → pop2obs
```

The workflow package depends on pop2obs; pop2obs does not know about it, about Slurm,
or about any machine.

## Status

Working: the SMPS-, ACSM- and optics-like instruments from the workflow figure, plus the size-resolved composition target. All instrument settings are placeholders for the team to review, and instrument physics (SMPS transfer function and charging, ACSM collection efficiency and lens transmission, optics truncation) is not modelled. Not yet: AMS and SEM-EDX instruments, and missing data. For a 10,000-particle population, optics takes about 8 s and the SMPS about 3 s, mostly part2pop's per-particle lookups. The architecture for the population-constraint problem is still being worked out with the team and is not part of pop2obs.

## Development

```bash
conda env create -f environment.yml
conda activate pop2obs
python -m pip install -e ".[dev]"
python -m pytest
```

Sources for the workflow figures are in `docs/figures/workflow/`.
