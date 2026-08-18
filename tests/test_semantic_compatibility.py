from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from prob4d.semantic_compatibility import (
    PROVIDED_CAPABILITIES,
    SEMANTIC_COMPATIBILITY_SCHEMA,
    SEMANTIC_COMPATIBILITY_VERSION,
    build_semantic_compatibility_manifest,
    load_semantic_compatibility_manifest,
    main,
    semantic_compatibility_report,
    validate_semantic_compatibility_manifest,
    write_semantic_compatibility_manifest,
)


def _rehash(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(manifest)
    payload.pop("manifest_id", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload["manifest_id"] = hashlib.sha256(encoded).hexdigest()
    return payload


def test_semantic_manifest_is_deterministic_and_named_vector_based() -> None:
    first = build_semantic_compatibility_manifest()
    second = build_semantic_compatibility_manifest()

    assert first == second
    assert first["schema"] == SEMANTIC_COMPATIBILITY_SCHEMA
    assert first["schema_version"] == SEMANTIC_COMPATIBILITY_VERSION == 1
    assert first["claim_bearing_evidence_pin_required"] is True
    assert first["capabilities"] == list(PROVIDED_CAPABILITIES)
    assert set(first["contracts"]["observation_belief"]["required_vectors"]) == {
        "minimal",
        "zero_rank",
    }
    assert set(first["contracts"]["provider_v2_factors"]["required_vectors"]) == {
        "minimal"
    }
    assert "bundle_sha256" not in first["contracts"]["observation_belief"]
    assert validate_semantic_compatibility_manifest(first) == first


def test_additive_capability_does_not_break_existing_requirement() -> None:
    required = build_semantic_compatibility_manifest()
    provided = deepcopy(required)
    provided["capabilities"].append("z-additive-capability-v1")
    provided["capabilities"].sort()
    provided = _rehash(provided)

    report = semantic_compatibility_report(required, provided)

    assert report["compatible"] is True
    assert report["missing_capabilities"] == []
    assert report["missing_or_changed_vectors"] == []


def test_missing_capability_and_changed_named_vector_fail_closed() -> None:
    required = build_semantic_compatibility_manifest()
    provided = deepcopy(required)
    provided["capabilities"] = provided["capabilities"][1:]
    provided["contracts"]["observation_belief"]["required_vectors"]["minimal"] = (
        "0" * 64
    )
    provided = _rehash(provided)

    report = semantic_compatibility_report(required, provided)

    assert report["compatible"] is False
    assert report["missing_capabilities"] == [required["capabilities"][0]]
    assert report["missing_or_changed_vectors"] == ["observation_belief:minimal"]
    assert "mandatory-vector-mismatch" in report["reasons"]
    assert "missing-capability" in report["reasons"]


def test_manifest_round_trip_is_atomic_and_no_clobber(tmp_path: Path) -> None:
    destination = tmp_path / "semantic-compatibility.json"
    expected = build_semantic_compatibility_manifest()

    assert write_semantic_compatibility_manifest(destination) == expected
    assert load_semantic_compatibility_manifest(destination) == expected
    assert write_semantic_compatibility_manifest(destination) == expected

    different = deepcopy(expected)
    different["capabilities"].append("z-different-v1")
    different["capabilities"].sort()
    different = _rehash(different)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_semantic_compatibility_manifest(destination, different)


def test_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_semantic_compatibility_manifest(path)


def test_cli_print_build_verify_and_check(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["print"]) == 0
    assert json.loads(capsys.readouterr().out) == build_semantic_compatibility_manifest()

    required = tmp_path / "required.json"
    provided = tmp_path / "provided.json"
    assert main(["build", "--output", str(required)]) == 0
    built_id = capsys.readouterr().out.strip()
    assert built_id == build_semantic_compatibility_manifest()["manifest_id"]
    assert main(["verify", str(required), "--require-current"]) == 0
    assert capsys.readouterr().out.strip() == built_id

    write_semantic_compatibility_manifest(provided)
    assert main(["check", str(required), str(provided)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["compatible"] is True
