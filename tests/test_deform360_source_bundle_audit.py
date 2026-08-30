from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "audit_deform360_source_bundle.py"
PROTOCOL = ROOT / "protocols" / "deform360-source-bundle-audit-v1.json"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("deform360_source_bundle_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    return _load_module()


def _canonical_id(record: dict[str, object], identity_field: str) -> str:
    identity = dict(record)
    identity.pop(identity_field, None)
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_checked_protocol_is_canonical_and_target_closed(audit: ModuleType) -> None:
    record = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert audit.validate_protocol_record(record) == record
    assert record["protocol_id"] == _canonical_id(record, "protocol_id")
    assert record["metadata_access_authorized"] is True
    for key in (
        "file_content_reads_authorized",
        "prediction_execution_authorized",
        "provider_residuals_authorized",
        "target_payloads_authorized",
        "target_outcomes_authorized",
        "dataset_mutation_authorized",
    ):
        assert record[key] is False


def test_request_binds_the_exact_protocol_blob(audit: ModuleType) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    blob = "a" * 40
    request: dict[str, object] = {
        "schema": audit.REQUEST_SCHEMA,
        "schema_version": 1,
        "profile": audit.PROFILE,
        "issue_number": audit.ISSUE_NUMBER,
        "source_protocol_path": audit.EXPECTED_PROTOCOL_PATH,
        "source_protocol_git_blob_sha": blob,
        "protocol_id": protocol["protocol_id"],
        "execution_authorized": True,
        "metadata_access_authorized": True,
        "file_content_reads_authorized": False,
        "prediction_execution_authorized": False,
        "provider_residuals_authorized": False,
        "target_payloads_authorized": False,
        "target_outcomes_authorized": False,
        "dataset_mutation_authorized": False,
        "claim_boundary": audit.CLAIM_BOUNDARY,
    }
    request["request_id"] = _canonical_id(request, "request_id")

    assert (
        audit.validate_request_record(
            request,
            protocol=protocol,
            source_protocol_git_blob_sha=blob,
        )
        == request
    )
    with pytest.raises(audit.AuditContractError, match="merged protocol blob"):
        audit.validate_request_record(
            request,
            protocol=protocol,
            source_protocol_git_blob_sha="b" * 40,
        )


def test_scan_reads_metadata_only_skips_forbidden_paths_and_symlinks(
    audit: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "object-a").mkdir()
    (source / "object-a" / "prediction.npz").write_bytes(b"12345")
    (source / "object-b.json").write_text("{}", encoding="utf-8")
    (source / "target-secret").mkdir()
    (source / "target-secret" / "must-not-appear.bin").write_bytes(b"secret")
    (source / "shadow_payload.dat").write_bytes(b"secret")
    (source / "link-to-object").symlink_to(source / "object-a", target_is_directory=True)

    original_open = builtins.open

    def guarded_open(file: object, *args: object, **kwargs: object):
        path = Path(file) if isinstance(file, (str, os.PathLike)) else None
        if path is not None and path.is_relative_to(source):
            raise AssertionError(f"dataset content open attempted: {path}")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    decision, inventory = audit.scan_source_root(
        source,
        forbidden_tokens=tuple(audit.EXPECTED_FORBIDDEN_TOKENS),
        max_entries=100,
        max_depth=10,
        largest_file_limit=10,
        sample_path_limit=20,
    )

    assert decision == "source-bundle-present"
    counts = inventory["counts_including_root"]
    assert counts["directory"] == 2
    assert counts["file"] == 2
    assert counts["symlink"] == 1
    assert counts["forbidden-subtree-skipped"] == 2
    assert inventory["total_regular_file_bytes"] == 7
    assert inventory["forbidden_token_counts"] == {"shadow": 1, "target": 1}
    serialized = json.dumps(inventory, sort_keys=True)
    assert "must-not-appear" not in serialized
    assert "target-secret" not in serialized
    assert "shadow_payload" not in serialized
    assert "prediction.npz" in serialized


def test_scan_is_deterministic_under_mtime_changes(audit: ModuleType, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "object" / "prediction.npz"
    payload.parent.mkdir()
    payload.write_bytes(b"payload")

    arguments = {
        "forbidden_tokens": tuple(audit.EXPECTED_FORBIDDEN_TOKENS),
        "max_entries": 100,
        "max_depth": 10,
        "largest_file_limit": 10,
        "sample_path_limit": 20,
    }
    first_decision, first = audit.scan_source_root(source, **arguments)
    os.utime(payload, (1_000_000_000, 1_000_000_000))
    second_decision, second = audit.scan_source_root(source, **arguments)

    assert first_decision == second_decision == "source-bundle-present"
    assert first == second


def test_missing_root_and_entry_limit_are_bounded(audit: ModuleType, tmp_path: Path) -> None:
    missing_decision, missing = audit.scan_source_root(
        tmp_path / "missing",
        forbidden_tokens=tuple(audit.EXPECTED_FORBIDDEN_TOKENS),
        max_entries=10,
        max_depth=10,
        largest_file_limit=10,
        sample_path_limit=10,
    )
    assert missing_decision == "source-root-missing"
    assert "does not exist" in missing["error"]

    source = tmp_path / "source"
    source.mkdir()
    for index in range(3):
        (source / f"file-{index}.bin").write_bytes(b"x")
    limited_decision, limited = audit.scan_source_root(
        source,
        forbidden_tokens=tuple(audit.EXPECTED_FORBIDDEN_TOKENS),
        max_entries=2,
        max_depth=10,
        largest_file_limit=10,
        sample_path_limit=10,
    )
    assert limited_decision == "entry-limit-exceeded"
    assert limited["entry_limit_exceeded"] is True
    assert limited["entry_count_excluding_root"] == 2


def test_tiny_limits_and_wrong_contracts_fail_closed(audit: ModuleType) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["metadata_access_authorized"] = False
    protocol["protocol_id"] = _canonical_id(protocol, "protocol_id")
    with pytest.raises(audit.AuditContractError, match="metadata_access_authorized"):
        audit.validate_protocol_record(protocol)
