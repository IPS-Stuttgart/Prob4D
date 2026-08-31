from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_PATH = ROOT / "scripts/science/materialize_dot_rope_query_selective_request.py"
RUNNER_PATH = ROOT / "scripts/science/run_dot_rope_query_selective_heldout.py"
PROTOCOL_PATH = ROOT / "protocols/dot-rope-query-selective-heldout-v1.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MATERIALIZER = _load("dot_request_materializer_test", MATERIALIZER_PATH)
RUNNER = _load("dot_query_runner_validation_test", RUNNER_PATH)


def _artifact() -> dict:
    return {
        "id": 987654321,
        "name": "dot-rope-cut3r-heldout-evaluation-gpuserver6000-1",
        "digest": "sha256:" + "a" * 64,
        "expired": False,
        "workflow_run": {"id": 33434695566},
    }


def test_built_request_passes_existing_frozen_validator(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    blob = "1" * 40
    request = MATERIALIZER.build_request(
        protocol=protocol,
        protocol_git_blob_sha=blob,
        artifact=_artifact(),
        verification={"decision": "heldout-strong-positive", "result_id": "b" * 64},
        marker_support_id="c" * 64,
    )
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    validated = RUNNER.validate_request(
        path,
        Path("protocols/dot-rope-query-selective-heldout-v1.json"),
        blob,
    )
    assert validated["request_id"] == request["request_id"]
    assert validated["prerequisite"]["run_id"] == 33434695566
    assert request["target_sequences"] == [f"R{index:02d}" for index in range(11, 31)]
    assert request["reserved_sequences"] == "R31-R70"


def test_non_strong_prerequisite_fails_closed() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="not strong-positive"):
        MATERIALIZER.build_request(
            protocol=protocol,
            protocol_git_blob_sha="1" * 40,
            artifact=_artifact(),
            verification={
                "decision": "heldout-directional-positive",
                "result_id": "b" * 64,
            },
            marker_support_id="c" * 64,
        )


def test_materialize_verifies_then_refuses_overwrite(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    support = tmp_path / "marker-support.json"
    artifact = tmp_path / "artifact.json"
    output = tmp_path / "request.json"
    result.write_text('{"placeholder": true}', encoding="utf-8")
    support.write_text(json.dumps({"support_id": "c" * 64}), encoding="utf-8")
    artifact.write_text(json.dumps(_artifact()), encoding="utf-8")
    calls = []

    def verifier(value, marker_support):
        calls.append((value, marker_support))
        return {"decision": "heldout-strong-positive", "result_id": "b" * 64}

    request = MATERIALIZER.materialize(
        protocol_path=PROTOCOL_PATH,
        protocol_git_blob_sha="1" * 40,
        result_path=result,
        marker_support_path=support,
        artifact_metadata_path=artifact,
        output_path=output,
        verifier=verifier,
    )
    assert len(calls) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == request
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MATERIALIZER.materialize(
            protocol_path=PROTOCOL_PATH,
            protocol_git_blob_sha="1" * 40,
            result_path=result,
            marker_support_path=support,
            artifact_metadata_path=artifact,
            output_path=output,
            verifier=verifier,
        )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"expired": True}, "unexpired"),
        ({"digest": "sha256:bad"}, "64 lowercase"),
        ({"workflow_run": {"id": 0}}, "workflow_run.id"),
    ],
)
def test_invalid_artifact_metadata_is_rejected(change: dict, match: str) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    artifact = _artifact()
    artifact.update(change)
    with pytest.raises(ValueError, match=match):
        MATERIALIZER.build_request(
            protocol=protocol,
            protocol_git_blob_sha="1" * 40,
            artifact=artifact,
            verification={"decision": "heldout-strong-positive", "result_id": "b" * 64},
            marker_support_id="c" * 64,
        )
