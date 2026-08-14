from __future__ import annotations

import pytest

from prob4d import prediction_cli
from prob4d.provider_adapter import load_provider_adapter_request


def test_prediction_help_lists_adapter_and_matrix_routes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as help_exit:
        prediction_cli.main(["--help"])
    assert help_exit.value.code == 0
    output = capsys.readouterr().out
    assert "adapter-conformance" in output
    assert "readiness-matrix" in output


def test_extension_subcommand_help_is_available() -> None:
    with pytest.raises(SystemExit) as conformance:
        prediction_cli.main(["adapter-conformance", "--help"])
    assert conformance.value.code == 0

    with pytest.raises(SystemExit) as matrix:
        prediction_cli.main(["readiness-matrix", "--help"])
    assert matrix.value.code == 0


def test_adapter_request_cli_builds_valid_artifact(tmp_path) -> None:
    output = tmp_path / "request.json"
    assert (
        prediction_cli.main(
            [
                "adapter-conformance",
                "build-request",
                "--sequence-id",
                "case-a",
                "--causal-frame-stop",
                "6",
                "--input-family-id",
                "a" * 64,
                "--input-snapshot-id",
                "b" * 64,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    request = load_provider_adapter_request(output)
    assert request.sequence_id == "case-a"
    assert request.causal_frame_stop == 6
