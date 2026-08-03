from __future__ import annotations

import pytest

from prob4d.cli import main as cli_main


def test_grouped_causal_gauge_graph_help(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        cli_main(["diagnostic", "gauge-graph", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--predictions" in output
    assert "--calibration-predictions" in output
    assert "--minimum-edge-weight" in output
