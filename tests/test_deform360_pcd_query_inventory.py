from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "inventory_deform360_pcd_archives.py"
PROTOCOL = ROOT / "protocols" / "deform360-pcd-query-inventory-v1.json"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("deform360_pcd_inventory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inventory() -> ModuleType:
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


def _write_tar(path: Path, frame_count: int, *, unsafe: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as bundle:
        for index in range(frame_count):
            payload = f"frame-{index}".encode()
            member = tarfile.TarInfo(f"pcd_clean/{index:06d}.npz")
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
        if unsafe:
            payload = b"bad"
            member = tarfile.TarInfo("../escape")
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))


def test_checked_protocol_is_canonical_and_payload_closed(inventory: ModuleType) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert inventory.validate_protocol_record(protocol) == protocol
    assert protocol["protocol_id"] == _canonical_id(protocol, "protocol_id")
    assert protocol["archive_header_reads_authorized"] is True
    for key in (
        "archive_member_payload_reads_authorized",
        "numeric_array_reads_authorized",
        "provider_predictions_authorized",
        "physical_query_scoring_authorized",
        "target_outcomes_authorized",
        "dataset_mutation_authorized",
    ):
        assert protocol[key] is False


def test_request_binds_protocol_blob_and_canonical_identity(inventory: ModuleType) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    blob = "a" * 40
    request: dict[str, object] = {
        "schema": inventory.REQUEST_SCHEMA,
        "schema_version": 1,
        "profile": inventory.PROFILE,
        "issue_number": inventory.ISSUE_NUMBER,
        "source_protocol_path": inventory.EXPECTED_PROTOCOL_PATH,
        "source_protocol_git_blob_sha": blob,
        "protocol_id": protocol["protocol_id"],
        "execution_authorized": True,
        "metadata_access_authorized": True,
        "archive_header_reads_authorized": True,
        "archive_member_payload_reads_authorized": False,
        "numeric_array_reads_authorized": False,
        "provider_predictions_authorized": False,
        "physical_query_scoring_authorized": False,
        "target_outcomes_authorized": False,
        "dataset_mutation_authorized": False,
        "claim_boundary": inventory.CLAIM_BOUNDARY,
    }
    request["request_id"] = _canonical_id(request, "request_id")
    assert (
        inventory.validate_request_record(
            request,
            protocol=protocol,
            source_protocol_git_blob_sha=blob,
        )
        == request
    )
    with pytest.raises(inventory.InventoryContractError, match="merged protocol blob"):
        inventory.validate_request_record(
            request,
            protocol=protocol,
            source_protocol_git_blob_sha="b" * 40,
        )


def test_archive_header_inspection_finds_persistent_frames(
    inventory: ModuleType, tmp_path: Path
) -> None:
    archive = tmp_path / "pcd_clean.tar"
    _write_tar(archive, 25)
    result = inventory.inspect_archive_headers(
        archive,
        max_tar_members=100,
        minimum_frame_count=24,
    )
    assert result["eligible"] is True
    assert result["reason"] == "eligible"
    assert result["frame_member_count"] == 25
    assert result["frame_index_min"] == 0
    assert result["frame_index_max"] == 24
    assert result["frame_indices_contiguous"] is True
    assert result["duplicate_frame_member_count"] == 0


def test_archive_inspection_rejects_unsafe_members_without_extracting(
    inventory: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "pcd_clean.tar"
    _write_tar(archive, 24, unsafe=True)

    def forbidden_extractfile(*args: object, **kwargs: object) -> None:
        raise AssertionError("archive payload extraction was attempted")

    monkeypatch.setattr(tarfile.TarFile, "extractfile", forbidden_extractfile)
    result = inventory.inspect_archive_headers(
        archive,
        max_tar_members=100,
        minimum_frame_count=24,
    )
    assert result["eligible"] is False
    assert result["reason"] == "unsafe-member-path"


def test_inventory_enforces_exclusion_union_and_object_units(
    inventory: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "processed"
    for object_id in ("001-rope", "101-test-object", "102-other-object"):
        for episode in range(3):
            _write_tar(root / object_id / f"episode_{episode}" / "pcd_clean.tar", 24)

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["source_root"] = str(root)
    protocol["excluded_object_ids"] = ["001-rope"]
    protocol["minimum_eligible_object_count"] = 2
    protocol["protocol_id"] = _canonical_id(protocol, "protocol_id")
    monkeypatch.setattr(inventory, "EXPECTED_SOURCE_ROOT", str(root))

    decision, result = inventory.inventory_source_root(root, protocol)
    assert decision == "inventory-ready"
    assert result["eligible_objects"] == ["101-test-object", "102-other-object"]
    assert result["eligible_episode_counts"] == {
        "101-test-object": 3,
        "102-other-object": 3,
    }
    excluded = [
        record for record in result["records"] if record["object_id"] == "001-rope"
    ]
    assert len(excluded) == 3
    assert {record["reason"] for record in excluded} == {
        "excluded-by-prior-evidence-union"
    }


def test_missing_archive_is_a_retained_ineligible_record(
    inventory: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "processed"
    (root / "101-test-object" / "episode_0").mkdir(parents=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["source_root"] = str(root)
    protocol["excluded_object_ids"] = []
    protocol["minimum_eligible_object_count"] = 1
    protocol["minimum_eligible_episodes_per_object"] = 1
    protocol["protocol_id"] = _canonical_id(protocol, "protocol_id")
    monkeypatch.setattr(inventory, "EXPECTED_SOURCE_ROOT", str(root))

    decision, result = inventory.inventory_source_root(root, protocol)
    assert decision == "inventory-insufficient"
    assert result["records"][0]["reason"] == "archive-missing"
    assert result["eligible_object_count"] == 0
