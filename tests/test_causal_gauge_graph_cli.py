from __future__ import annotations

from prob4d.cli import main as cli_main


def test_grouped_causal_gauge_graph_help(capsys) -> None:
    assert cli_main(["diagnostic", "gauge-graph", "--help"]) == 0
    output = capsys.readouterr().out
    assert "--predictions" in output
    assert "--calibration-predictions" in output
    assert "--minimum-edge-weight" in output
