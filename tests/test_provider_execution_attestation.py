from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from prob4d._immutable_json import plain_json
from prob4d.provider_execution_attestation import (
    PROVIDER_EXECUTION_ATTESTATION_SCHEMA,
    build_provider_execution_attestation,
    load_provider_execution_attestation,
    main,
    validate_provider_execution_attestation,
    write_provider_execution_attestation,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def _spec() -> dict[str, object]:
    return {
        "provider_repository": "example/provider",
        "provider_revision": "1" * 40,
        "provider_run_id": DIGEST_A,
        "model_set_id": DIGEST_B,
        "loader_id": DIGEST_C,
        "execution_mode": "recurrent-online",
        "command_argv": ["python", "demo.py", "--revisit", "1"],
        "causal_declarations": {
            "source_order_preserved": True,
            "online_prefix_only": True,
            "revisit_count": 1,
            "global_alignment": False,
            "future_frame_postprocessing": False,
        },
        "runtime": {
            "python_version": "3.12.11",
            "implementation": "CPython",
            "platform": "Linux-x86_64",
            "container_image_digest": f"sha256:{DIGEST_D}",
            "environment_lock_sha256": DIGEST_A,
        },
        "environment_variables": [
            {"name": "CUDA_VISIBLE_DEVICES", "value_sha256": DIGEST_B},
            {"name": "PYTHONHASHSEED", "value_sha256": DIGEST_A},
        ],
        "input_artifacts": [
            {"name": "input-video", "sha256": DIGEST_C, "byte_count": 123},
            {"name": "checkpoint", "sha256": DIGEST_B, "byte_count": 456},
        ],
        "output_artifacts": [
            {"name": "provider-manifest", "sha256": DIGEST_D, "byte_count": 789},
        ],
        "execution_evidence_mode": "wrapper-observed-v1",
        "execution_evidence_sha256": DIGEST_C,
        "prediction_provider_manifest_id": DIGEST_D,
        "started_at_utc": "2026-08-18T01:02:03Z",
        "completed_at_utc": "2026-08-18T01:04:05.123456Z",
        "terminal_status": "succeeded",
        "metadata": {"runner": "workstation2", "attempt": 1},
    }


def test_build_is_content_addressed_canonical_and_immutable() -> None:
    specification = _spec()
    specification["environment_variables"] = list(
        reversed(specification["environment_variables"])  # type: ignore[arg-type]
    )
    attestation = build_provider_execution_attestation(specification)

    assert attestation["schema"] == PROVIDER_EXECUTION_ATTESTATION_SCHEMA
    assert attestation["execution_evidence_complete"] is True
    names = [row["name"] for row in attestation["environment_variables"]]
    assert names == ["CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED"]
    assert len(attestation["provider_execution_attestation_id"]) == 64
    with pytest.raises(TypeError):
        attestation["metadata"]["runner"] = "other"  # type: ignore[index]


def test_round_trip_is_idempotent_and_refuses_different_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "attestation.json"
    attestation = build_provider_execution_attestation(_spec())

    written = write_provider_execution_attestation(destination, attestation)
    repeated = write_provider_execution_attestation(destination, attestation)
    loaded = load_provider_execution_attestation(destination)

    assert plain_json(written) == plain_json(repeated) == plain_json(loaded)
    changed = _spec()
    changed["completed_at_utc"] = "2026-08-18T01:04:06Z"
    with pytest.raises(FileExistsError):
        write_provider_execution_attestation(
            destination,
            build_provider_execution_attestation(changed),
        )


def test_validation_rejects_mutation_and_noncanonical_order() -> None:
    attestation = plain_json(build_provider_execution_attestation(_spec()))
    mutated = copy.deepcopy(attestation)
    mutated["terminal_status"] = "failed"
    with pytest.raises(ValueError, match="ID does not match"):
        validate_provider_execution_attestation(mutated)

    noncanonical = copy.deepcopy(attestation)
    noncanonical["input_artifacts"].reverse()
    unsigned = copy.deepcopy(noncanonical)
    unsigned.pop("provider_execution_attestation_id")
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    noncanonical["provider_execution_attestation_id"] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError, match="canonical form"):
        validate_provider_execution_attestation(noncanonical)


def test_strict_contract_rejects_missing_evidence_duplicates_and_time_reversal() -> None:
    specification = _spec()
    specification["execution_evidence_sha256"] = None
    with pytest.raises(ValueError, match="requires its exact SHA-256"):
        build_provider_execution_attestation(specification)

    specification = _spec()
    specification["environment_variables"] = [
        {"name": "A", "value_sha256": DIGEST_A},
        {"name": "A", "value_sha256": DIGEST_B},
    ]
    with pytest.raises(ValueError, match="duplicate environment variable"):
        build_provider_execution_attestation(specification)

    specification = _spec()
    specification["completed_at_utc"] = "2026-08-17T23:59:59Z"
    with pytest.raises(ValueError, match="must not precede"):
        build_provider_execution_attestation(specification)


def test_validation_rejects_coercive_schema_and_boolean_aliases() -> None:
    attestation = plain_json(build_provider_execution_attestation(_spec()))
    attestation["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        validate_provider_execution_attestation(attestation)

    attestation = plain_json(build_provider_execution_attestation(_spec()))
    attestation["execution_evidence_complete"] = 1
    with pytest.raises(ValueError, match="must be a Boolean"):
        validate_provider_execution_attestation(attestation)


def test_declarative_failure_is_valid_but_incomplete() -> None:
    specification = _spec()
    specification.update(
        {
            "terminal_status": "failed",
            "execution_evidence_mode": "declarative-only-v1",
            "execution_evidence_sha256": None,
            "prediction_provider_manifest_id": None,
        }
    )
    attestation = build_provider_execution_attestation(specification)
    assert attestation["execution_evidence_complete"] is False


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    destination = tmp_path / "duplicate.json"
    destination.write_text('{"schema": "a", "schema": "b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_provider_execution_attestation(destination)


def test_cli_create_and_verify_complete(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    specification = tmp_path / "specification.json"
    output = tmp_path / "attestation.json"
    specification.write_text(json.dumps(_spec()), encoding="utf-8")

    assert main(["create", str(specification), "--output", str(output)]) == 0
    created_id = capsys.readouterr().out.strip()
    assert main(["verify", str(output), "--require-complete"]) == 0
    verified_id = capsys.readouterr().out.strip()
    assert created_id == verified_id
