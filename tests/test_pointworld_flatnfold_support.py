from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d.pointworld_flatnfold_support import (
    POINTWORLD_FLATNFOLD_INVENTORY_SCHEMA,
    build_pointworld_flatnfold_support_request,
    scaffold_pointworld_flatnfold_support_inventory,
)
from prob4d.provider_support_feasibility import (
    evaluate_provider_support_feasibility,
)


def _inventory() -> dict[str, object]:
    frames = list(range(10))
    streams = []
    for index, camera_id in enumerate(("camera-0", "camera-1", "camera-2")):
        streams.append(
            {
                "garment_id": "garment-001",
                "demonstration_id": "robot-demo-004",
                "camera_id": camera_id,
                "causal_frame_start": 0,
                "causal_frame_stop_exclusive": 10,
                "required_frame_ids": frames,
                "available_frame_ids": frames,
                "geometry_supported_frame_ids": frames,
                "minimum_geometry_support_fraction": 1.0,
                "intrinsics_id": str(index + 1) * 64,
                "extrinsics_id": str(index + 4) * 64,
                "metric_anchor_id": "7" * 64,
                "action_sequence_id": "8" * 64,
                "technical_failure_code": None,
                "metadata": {"camera_index": index},
            }
        )
    return {
        "schema_name": POINTWORLD_FLATNFOLD_INVENTORY_SCHEMA,
        "schema_version": 1,
        "protocol_id": "pointworld-flatnfold-source-qualification-v1",
        "source_revision": "a" * 40,
        "pointworld_revision": "b" * 40,
        "checkpoint_sha256": "c" * 64,
        "model_set_id": "d" * 64,
        "loader_id": "e" * 64,
        "cohort_binding_id": "f" * 64,
        "promotion_lock_id": "0" * 64,
        "flatnfold_revision": "1" * 40,
        "dataset_bytes_id": "2" * 64,
        "coordinate_semantics": "metric-baxter-base",
        "required_camera_ids": ["camera-0", "camera-1", "camera-2"],
        "admission_rule": "all-streams",
        "minimum_supported_fraction": 1.0,
        "permitted_technical_exclusion_codes": [],
        "maximum_technical_exclusions": 0,
        "prediction_payloads_opened": False,
        "residuals_used": False,
        "target_outcomes_used": False,
        "streams": streams,
        "metadata": {"split": "inventory-only"},
    }


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_inventory_builds_three_camera_garment_grouped_support_request(
    tmp_path: Path,
) -> None:
    path = tmp_path / "inventory.json"
    _write(path, _inventory())
    request = build_pointworld_flatnfold_support_request(path)
    result = evaluate_provider_support_feasibility(request)

    assert request.provider_family == "pointworld"
    assert request.coordinate_semantics == "metric-baxter-base"
    assert len(request.streams) == 3
    assert {stream.group_id for stream in request.streams} == {"garment-001"}
    assert {stream.stream_id for stream in request.streams} == {
        "robot-demo-004:camera-0",
        "robot-demo-004:camera-1",
        "robot-demo-004:camera-2",
    }
    assert request.metadata["statistical_unit"] == "complete-physical-garment"
    assert request.metadata["garment_count"] == 1
    assert request.metadata["demonstration_count"] == 1
    assert result.support_feasible is True
    assert result.stream_count == 3
    assert result.supported_stream_count == 3


def test_inventory_requires_all_three_cameras_per_demonstration(tmp_path: Path) -> None:
    inventory = _inventory()
    streams = list(inventory["streams"])  # type: ignore[arg-type]
    inventory["streams"] = streams[:2]
    path = tmp_path / "inventory.json"
    _write(path, inventory)

    with pytest.raises(ValueError, match="all required cameras"):
        build_pointworld_flatnfold_support_request(path)


def test_inventory_requires_one_action_and_frame_schedule_per_demo(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    streams = list(inventory["streams"])  # type: ignore[arg-type]
    streams[1] = dict(streams[1])
    streams[1]["action_sequence_id"] = "9" * 64
    inventory["streams"] = streams
    path = tmp_path / "action.json"
    _write(path, inventory)
    with pytest.raises(ValueError, match="one action sequence"):
        build_pointworld_flatnfold_support_request(path)

    inventory = _inventory()
    streams = list(inventory["streams"])  # type: ignore[arg-type]
    streams[2] = dict(streams[2])
    streams[2]["required_frame_ids"] = list(range(9))
    inventory["streams"] = streams
    path = tmp_path / "frames.json"
    _write(path, inventory)
    with pytest.raises(ValueError, match="one causal frame schedule"):
        build_pointworld_flatnfold_support_request(path)


def test_inventory_fails_before_constructing_request_after_any_outcome_access(
    tmp_path: Path,
) -> None:
    for field in (
        "prediction_payloads_opened",
        "residuals_used",
        "target_outcomes_used",
    ):
        inventory = _inventory()
        inventory[field] = True
        path = tmp_path / f"{field}.json"
        _write(path, inventory)
        with pytest.raises(ValueError, match="before payload"):
            build_pointworld_flatnfold_support_request(path)


def test_inventory_retains_support_negative_missing_frame(tmp_path: Path) -> None:
    inventory = _inventory()
    streams = list(inventory["streams"])  # type: ignore[arg-type]
    streams[0] = dict(streams[0])
    streams[0]["available_frame_ids"] = list(range(9))
    streams[0]["geometry_supported_frame_ids"] = list(range(9))
    inventory["streams"] = streams
    path = tmp_path / "inventory.json"
    _write(path, inventory)

    request = build_pointworld_flatnfold_support_request(path)
    result = evaluate_provider_support_feasibility(request)
    assert result.support_feasible is False
    assert result.supported_stream_count == 2


def test_support_inventory_scaffold_is_incomplete_and_no_clobber(
    tmp_path: Path,
) -> None:
    path = tmp_path / "inventory.json"
    scaffold_pointworld_flatnfold_support_inventory(path)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["schema_name"] == POINTWORLD_FLATNFOLD_INVENTORY_SCHEMA
    assert record["metadata"]["ready_for_evaluation"] is False
    assert len(record["streams"]) == 3
    assert record["source_revision"].startswith("REPLACE_WITH_")
    with pytest.raises(FileExistsError):
        scaffold_pointworld_flatnfold_support_inventory(path)
