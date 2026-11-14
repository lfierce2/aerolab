"""Number size distribution, from part2pop's dNdlnD."""

from __future__ import annotations

import numpy as np
from part2pop.analysis import build_variable

from ..records import Observation, package_versions


def size_distribution(population, edges, wetsize: bool = False) -> Observation:
    """dN/dlnD in m^-3 on diameter bins; dry sizes unless wetsize is True."""
    edges = np.asarray(edges, dtype=float)
    cfg = {"method": "hist", "wetsize": wetsize, "edges": edges}
    out = build_variable("dNdlnD", var_cfg=cfg).compute(population, as_dict=True)
    return Observation(
        name="size_distribution",
        values=out["dNdlnD"],
        dims=("diameter",),
        coords={"diameter": out["D"], "edges": out["edges"]},
        units="m^-3",
        provenance={
            "observation": "size_distribution",
            "wetsize": wetsize,
            "versions": package_versions(),
        },
    )
