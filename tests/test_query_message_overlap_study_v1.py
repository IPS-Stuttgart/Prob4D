from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_query_message_overlap_study_v1.py"
PROTOCOL = ROOT / "protocols/query-message-overlap-study-v1.json"


def _module():
    spec = importlib.util.spec_from_file_location(
        "query_message_overlap_study_v1",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load overlap study")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_controlled_study_passes() -> None:
    module = _module()
    protocol = module._load_protocol(PROTOCOL)
    result = module.run(protocol)

    assert result["decision"] == "controlled-overlap-passed"
    assert all(result["checks"].values())
    assert result["protocol_id"] == protocol["protocol_id"]
    assert len(result["result_id"]) == 64
    assert len(result["rows"]) == 6
    assert result["summary"]["ci_improves_over_either_single_window"] is True
    assert result["summary"]["naive_high_correlation_minimum_normalized_nees"] >= 1.5
    assert result["summary"]["naive_high_correlation_maximum_coverage"] <= 0.75
    assert result["summary"]["ci_maximum_normalized_nees"] <= 1.0
    assert result["summary"]["ci_minimum_coverage"] >= 0.9
    assert result["summary"]["maximum_duplicate_mean_error"] == 0.0
    assert result["summary"]["maximum_duplicate_covariance_error"] == 0.0
    for row in result["rows"]:
        assert row["ci_weights"] == pytest.approx(
            {"window_a": 0.5, "window_b": 0.5}
        )


def test_protocol_identity_fails_closed(tmp_path: Path) -> None:
    module = _module()
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["sample_count"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        module._load_protocol(path)


def test_output_directory_must_not_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--protocol",
            str(PROTOCOL),
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError):
        module.main()
