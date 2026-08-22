from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import prob4d.cut3r_source_freeze_verification as verifier


def _module() -> ModuleType:
    return verifier


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return payload


def _source_groups() -> list[dict[str, Any]]:
    return [
        {
            "group_id": "001-alpha-episode-0000",
            "object_id": "001-alpha",
            "episode_id": 0,
            "stratum": "sheet",
            "role": "development",
        },
        {
            "group_id": "002-beta-episode-0001",
            "object_id": "002-beta",
            "episode_id": 1,
            "stratum": "volumetric",
            "role": "calibration",
        },
        {
            "group_id": "003-gamma-episode-0002",
            "object_id": "003-gamma",
            "episode_id": 2,
            "stratum": "sheet",
            "role": "source_evaluation",
        },
    ]


def _target_groups() -> list[dict[str, Any]]:
    return [{"object_id": "100-target", "episode_id": 3, "stratum": "volumetric"}]


def _selection(source_groups: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selection_artifact_sha256": _digest("selection-artifact"),
        "selection_sha256": _digest("selection-semantics"),
        "selection": {
            "calibration": [
                {
                    "object_id": group["object_id"],
                    "episode_id": group["episode_id"],
                    "stratum": group["stratum"],
                }
                for group in source_groups
            ],
            "confirmation": _target_groups(),
        },
    }


def _protocol(source_groups: list[dict[str, Any]], selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "prob4d.cut3r-deform360-source-freeze-protocol",
        "schema_version": 1,
        "protocol_name": "cut3r-test-source-v1",
        "claim_boundary": "Synthetic source-freeze verifier test only.",
        "information_boundary": {
            "camera_panel_change_after_freeze_allowed": False,
            "downstream_physical_innovations_opened": False,
            "replacement_after_freeze_allowed": False,
            "source_future_geometry_opened": False,
            "source_prediction_payloads_opened": False,
            "source_residuals_or_truth_opened": False,
            "source_rgb_frames_decoded": False,
            "source_rgb_video_bytes_hashed": True,
            "target_outcomes_opened": False,
            "target_payloads_opened": False,
        },
        "source_groups": source_groups,
        "forbidden_target_groups": _target_groups(),
        "source_dataset": {
            "selection_artifact_sha256": selection["selection_artifact_sha256"],
            "selection_sha256": selection["selection_sha256"],
        },
        "provider": {
            "repository": "CUT3R/CUT3R",
            "revision": "1" * 40,
            "checkpoint_filename": "cut3r-test.pth",
            "execution_mode": "recurrent-online",
            "revisit_count": 1,
            "global_alignment": False,
            "second_pass_allowed": False,
            "confidence_threshold": 1.5,
        },
        "camera_panel": {
            "minimum_common_supported_cameras": 2,
            "panel_size": 2,
            "selection_rule": "deterministic-test-panel",
            "first_camera_rule": "lexicographic-test-first",
        },
        "windowing": {
            "frame_start": 0,
            "frame_stop_exclusive": 10,
            "evaluation_frame_start": 4,
            "evaluation_frame_stop_exclusive": 10,
            "window_size": 6,
            "overlap": 2,
            "storage_dtype": "float32",
            "random_seeds": [7, 11],
            "include_revisit_diagnostic": False,
        },
    }


def _case(group: dict[str, Any], camera: str) -> dict[str, Any]:
    group_id = group["group_id"]
    object_id = group["object_id"]
    episode_id = group["episode_id"]
    case_id = f"{group_id}-{camera}"
    episode_path = f"{object_id}/episode_{episode_id:04d}"
    payload: dict[str, Any] = {
        "case_id": case_id,
        "group_id": group_id,
        "object_id": object_id,
        "episode_id": episode_id,
        "camera": camera,
        "relative_episode_path": episode_path,
        "relative_camera_path": f"{episode_path}/{camera}",
        "input_video_sha256": _digest(f"video:{case_id}"),
        "input_video_byte_count": 100 + len(case_id),
        "aligned_timestamp_count": 12,
        "sidecar_sha256": {
            "aligned_timestamps.txt": _digest(f"timestamps:{case_id}"),
            "alignment.json": _digest(f"alignment:{case_id}"),
            "metadata.json": _digest(f"metadata:{case_id}"),
        },
        "sidecar_byte_count": {
            "aligned_timestamps.txt": 40,
            "alignment.json": 50,
            "metadata.json": 60,
        },
    }
    payload["source_case_id"] = _canonical_sha(payload)
    return payload


def _comparison(
    source_groups: list[dict[str, Any]],
    source_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    cases_by_group = {group["group_id"]: [] for group in source_groups}
    for case in source_cases:
        cases_by_group[case["group_id"]].append(
            {
                "case_id": case["case_id"],
                "input_video_sha256": case["input_video_sha256"],
                "input_video_byte_count": case["input_video_byte_count"],
                "frame_start": 0,
                "frame_stop_exclusive": 10,
                "evaluation_frame_start": 4,
                "evaluation_frame_stop_exclusive": 10,
            }
        )
    return {
        "protocol_name": "cut3r-test-source-v1",
        "provider_revision": "1" * 40,
        "checkpoint_sha256": _digest("checkpoint"),
        "prob4d_revision": "2" * 40,
        "prob4d_distribution_sha256": _digest("wheel"),
        "window_size": 6,
        "overlap": 2,
        "confidence_threshold": 1.5,
        "storage_dtype": "float32",
        "random_seeds": [7, 11],
        "groups": [
            {
                "group_id": group_id,
                "cases": sorted(cases, key=lambda record: record["case_id"]),
            }
            for group_id, cases in sorted(cases_by_group.items())
        ],
        "group_roles": {
            role: sorted(
                group["group_id"] for group in source_groups if group["role"] == role
            )
            for role in ("development", "calibration", "source_evaluation")
        },
        "include_revisit_diagnostic": False,
    }


def _bundle(tmp_path: Path) -> dict[str, Path]:
    source_groups = _source_groups()
    selection = _selection(source_groups)
    selection_path = tmp_path / "selection.json"
    selection_bytes = _write_json(selection_path, selection)
    protocol = _protocol(source_groups, selection)
    protocol_path = tmp_path / "protocol.json"
    protocol_bytes = _write_json(protocol_path, protocol)

    cameras = ["cam-a", "cam-b"]
    stream_rows = [
        {
            "camera": camera,
            "missing_required_members": [],
            "aligned_timestamp_count": 12,
            "required_frame_count": 10,
            "supported": True,
            "group_id": group["group_id"],
            "object_id": group["object_id"],
            "episode_id": group["episode_id"],
        }
        for group in source_groups
        for camera in cameras
    ]
    source_cases = sorted(
        [_case(group, camera) for group in source_groups for camera in cameras],
        key=lambda record: record["case_id"],
    )
    comparison = _comparison(source_groups, source_cases)
    comparison_path = tmp_path / "comparison.json"
    _write_json(comparison_path, comparison)

    freeze: dict[str, Any] = {
        "schema": "prob4d.cut3r-deform360-source-freeze",
        "schema_version": 1,
        "protocol_name": protocol["protocol_name"],
        "decision": "source-support-freeze-ready",
        "source_protocol": {
            "sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "byte_count": len(protocol_bytes),
        },
        "deform360_selection": {
            "sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "byte_count": len(selection_bytes),
            "selection_artifact_sha256": selection["selection_artifact_sha256"],
            "selection_sha256": selection["selection_sha256"],
        },
        "provider": {
            "repository": "CUT3R/CUT3R",
            "revision": "1" * 40,
            "checkpoint_filename": "cut3r-test.pth",
            "checkpoint_sha256": _digest("checkpoint"),
            "checkpoint_byte_count": 1000,
            "execution_mode": "recurrent-online",
            "revisit_count": 1,
            "global_alignment": False,
            "second_pass_allowed": False,
        },
        "prob4d": {
            "revision": "2" * 40,
            "distribution_filename": "prob4d-0.5.0-py3-none-any.whl",
            "distribution_sha256": _digest("wheel"),
            "distribution_byte_count": 2000,
        },
        "source_group_count": len(source_groups),
        "source_groups": source_groups,
        "forbidden_target_group_count": len(_target_groups()),
        "forbidden_target_groups": _target_groups(),
        "support": {
            "required_frame_interval": [0, 10],
            "common_supported_camera_count": 2,
            "common_supported_cameras": cameras,
            "minimum_common_supported_cameras": 2,
            "stream_rows": stream_rows,
        },
        "camera_calibration_inputs": [
            {
                "group_id": group["group_id"],
                "object_id": group["object_id"],
                "episode_id": group["episode_id"],
                "intrinsics": {
                    "relative_path": (
                        f"{group['object_id']}/episode_{group['episode_id']:04d}/"
                        "undistorted_intrinsics.npy"
                    ),
                    "sha256": _digest(f"intrinsics:{group['group_id']}"),
                    "byte_count": 300,
                },
                "extrinsics": {
                    "relative_path": (
                        f"{group['object_id']}/episode_{group['episode_id']:04d}/extrinsics.npy"
                    ),
                    "sha256": _digest(f"extrinsics:{group['group_id']}"),
                    "byte_count": 400,
                },
            }
            for group in source_groups
        ],
        "camera_panel": {
            "selected_cameras": cameras,
            "panel_size": 2,
            "selection_rule": "deterministic-test-panel",
            "first_camera_rule": "lexicographic-test-first",
            "camera_center_maximum_deviation_m": {"cam-a": 0.001, "cam-b": 0.001},
            "camera_direction": {"cam-a": [1.0, 0.0, 0.0], "cam-b": [0.0, 1.0, 0.0]},
        },
        "source_cases": source_cases,
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
        "comparison_spec_sha256": _canonical_sha(comparison),
    }
    freeze["source_freeze_id"] = _canonical_sha(freeze)
    freeze_path = tmp_path / "freeze.json"
    _write_json(freeze_path, freeze)
    return {
        "freeze": freeze_path,
        "comparison": comparison_path,
        "protocol": protocol_path,
        "selection": selection_path,
    }


def test_verifies_complete_support_positive_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    bundle = _bundle(tmp_path)
    status = module.main(
        [
            str(bundle["freeze"]),
            "--comparison-spec",
            str(bundle["comparison"]),
            "--protocol",
            str(bundle["protocol"]),
            "--selection",
            str(bundle["selection"]),
            "--require-complete-bindings",
            "--require-support-pass",
            "--json",
        ]
    )
    assert status == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["decision"] == "source-support-freeze-ready"
    assert summary["source_group_count"] == 3
    assert summary["source_case_count"] == 6
    assert summary["complete_bindings_verified"] is True


def test_rejects_tampered_source_case_content(tmp_path: Path) -> None:
    module = _module()
    bundle = _bundle(tmp_path)
    freeze = json.loads(bundle["freeze"].read_text(encoding="utf-8"))
    freeze["source_cases"][0]["input_video_byte_count"] += 1
    freeze["source_freeze_id"] = _canonical_sha(
        {key: value for key, value in freeze.items() if key != "source_freeze_id"}
    )
    _write_json(bundle["freeze"], freeze)
    with pytest.raises(ValueError, match="source_case_id"):
        module.validate_source_freeze(freeze)


def test_rejects_incomplete_source_case_panel(tmp_path: Path) -> None:
    module = _module()
    bundle = _bundle(tmp_path)
    freeze = json.loads(bundle["freeze"].read_text(encoding="utf-8"))
    freeze["source_cases"].pop()
    freeze["source_freeze_id"] = _canonical_sha(
        {key: value for key, value in freeze.items() if key != "source_freeze_id"}
    )
    with pytest.raises(ValueError, match="full source-group/camera panel"):
        module.validate_source_freeze(freeze)


def test_protocol_reconstruction_rejects_rebound_evaluation_window(tmp_path: Path) -> None:
    module = _module()
    bundle = _bundle(tmp_path)
    freeze = json.loads(bundle["freeze"].read_text(encoding="utf-8"))
    comparison = json.loads(bundle["comparison"].read_text(encoding="utf-8"))
    comparison["groups"][0]["cases"][0]["evaluation_frame_start"] = 3
    freeze["comparison_spec_sha256"] = _canonical_sha(comparison)
    freeze["source_freeze_id"] = _canonical_sha(
        {key: value for key, value in freeze.items() if key != "source_freeze_id"}
    )
    module.validate_source_freeze(freeze, comparison_spec=comparison)

    protocol, protocol_bytes = module._load_json_object(bundle["protocol"], name="protocol")
    with pytest.raises(ValueError, match="independently reconstructed"):
        module.validate_source_freeze(
            freeze,
            comparison_spec=comparison,
            protocol=protocol,
            protocol_bytes=protocol_bytes,
        )


def test_selection_binding_rejects_roster_drift_even_after_rehash(tmp_path: Path) -> None:
    module = _module()
    bundle = _bundle(tmp_path)
    freeze = json.loads(bundle["freeze"].read_text(encoding="utf-8"))
    selection = json.loads(bundle["selection"].read_text(encoding="utf-8"))
    selection["selection"]["calibration"][0]["episode_id"] = 9
    selection_bytes = _write_json(bundle["selection"], selection)
    freeze["deform360_selection"]["sha256"] = hashlib.sha256(selection_bytes).hexdigest()
    freeze["deform360_selection"]["byte_count"] = len(selection_bytes)
    freeze["source_freeze_id"] = _canonical_sha(
        {key: value for key, value in freeze.items() if key != "source_freeze_id"}
    )
    with pytest.raises(ValueError, match="calibration selection differs"):
        module.validate_source_freeze(
            freeze,
            selection=selection,
            selection_bytes=selection_bytes,
        )


def test_valid_support_negative_returns_three_only_when_required(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    bundle = _bundle(tmp_path)
    freeze = json.loads(bundle["freeze"].read_text(encoding="utf-8"))
    for row in freeze["support"]["stream_rows"]:
        if row["group_id"] == "003-gamma-episode-0002":
            row["missing_required_members"] = ["video"]
            row["aligned_timestamp_count"] = 0
            row["supported"] = False
    freeze["support"]["common_supported_camera_count"] = 0
    freeze["support"]["common_supported_cameras"] = []
    freeze["decision"] = "insufficient-common-camera-support"
    freeze["camera_panel"] = None
    freeze["source_cases"] = []
    freeze.pop("comparison_spec_sha256")
    freeze["source_freeze_id"] = _canonical_sha(
        {key: value for key, value in freeze.items() if key != "source_freeze_id"}
    )
    _write_json(bundle["freeze"], freeze)

    base_arguments = [
        str(bundle["freeze"]),
        "--protocol",
        str(bundle["protocol"]),
        "--selection",
        str(bundle["selection"]),
        "--require-complete-bindings",
    ]
    assert module.main(base_arguments) == 0
    capsys.readouterr()
    assert module.main([*base_arguments, "--require-support-pass"]) == 3


def test_strict_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": 1, "schema": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        module._load_json_object(path, name="duplicate test")
