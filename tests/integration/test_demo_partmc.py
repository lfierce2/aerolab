"""The PartMC demo runs end to end on its bundled example output."""

import runpy
from pathlib import Path

DEMO = Path(__file__).resolve().parents[2] / "examples" / "demo_partmc.py"


def test_demo_prints_a_summary_and_writes_a_figure(tmp_path, capsys):
    main = runpy.run_path(str(DEMO))["main"]
    figure = tmp_path / "demo.png"
    main(["--output", str(figure)])
    assert figure.stat().st_size > 0
    printed = capsys.readouterr().out
    assert "733 particles" in printed
    assert "acsm" in printed
