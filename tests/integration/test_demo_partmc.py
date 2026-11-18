"""The PartMC demo runs end to end on its bundled example output."""

import csv
import runpy
from pathlib import Path

DEMO = Path(__file__).resolve().parents[2] / "examples" / "demo_partmc.py"


def test_demo_prints_a_summary_and_writes_a_figure(tmp_path, capsys):
    main = runpy.run_path(str(DEMO))["main"]
    figure = tmp_path / "demo.png"
    main(["--output", str(figure)])
    assert figure.stat().st_size > 0
    noise_free = tmp_path / "demo_partmc_noise_free_smps.csv"
    with_noise = tmp_path / "demo_partmc_with_noise_smps.csv"
    assert noise_free.stat().st_size > 0
    assert with_noise.stat().st_size > 0
    with noise_free.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {
        "diameter_nm",
        "diameter_bin_left_nm",
        "diameter_bin_right_nm",
        "number_concentration_cm3_per_dlnD",
        "uncertainty_cm3_per_dlnD",
    } <= set(rows[0])
    assert all(float(row["uncertainty_cm3_per_dlnD"]) == 0 for row in rows)
    with with_noise.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert any(float(row["uncertainty_cm3_per_dlnD"]) > 0 for row in rows)
    assert any(
        float(row["number_concentration_cm3_per_dlnD"])
        != float(row["expected_number_concentration_cm3_per_dlnD"])
        for row in rows
    )
    printed = capsys.readouterr().out
    assert " particles" in printed
    assert "acsm" in printed

    combined_figure = tmp_path / "combined.png"
    main(["--output", str(combined_figure), "--combined"])
    assert combined_figure.stat().st_size > 0
    combined_noise_free = tmp_path / "demo_partmc_noise_free.csv"
    combined_with_noise = tmp_path / "demo_partmc_with_noise.csv"
    assert combined_noise_free.stat().st_size > 0
    assert combined_with_noise.stat().st_size > 0
    with combined_noise_free.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["quantity"] for row in rows} >= {
        "smps",
        "acsm",
        "optics",
        "size_resolved_composition",
    }

    ssa_figure = tmp_path / "ssa.png"
    main(["--output", str(ssa_figure), "--optics-output", "ssa"])
    ssa = tmp_path / "demo_partmc_with_noise_optics.csv"
    with ssa.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {
        "wavelength_nm",
        "single_scattering_albedo",
        "expected_single_scattering_albedo",
        "uncertainty_ssa",
    } <= set(rows[0])
    assert all(
        0 <= float(row["expected_single_scattering_albedo"]) <= 1 for row in rows
    )
