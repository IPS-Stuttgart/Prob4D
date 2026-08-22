from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/build_cut3r_source_comparison_preflight.py"
REQUEST = (
    ROOT
    / "protocols"
    / "execution_requests"
    / "cut3r_deform360_source_comparison_preflight_v1.json"
)


def _module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "cut3r_source_comparison_preflight",
        SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_checked_in_request_preserves_outcome_blind_boundary() -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))

    assert request["source_group_count"] == 10
    assert request["forbidden_target_group_count"] == 12
    assert request["expected_case_count"] == 40
    assert request["source_rgb_frames_decoded"] is False
    assert request["source_prediction_payloads_opened"] is False
    assert request["source_residuals_or_truth_opened"] is False
    assert request["target_payloads_opened"] is False
    assert request["target_outcomes_opened"] is False
    assert request["comparison_execution_authorized"] is False


def test_video_descriptor_collection_preserves_context_and_deduplicates() -> None:
    module = _module()
    digest = "a" * 64
    value = {
        "groups": [
            {
                "group_id": "group-a",
                "role": "source-evaluation",
                "cases": [
                    {
                        "case_id": "case-a",
                        "view_id": "camera-0",
                        "video": {
                            "path": "group-a/camera-0/undistorted.mp4",
                            "sha256": digest,
                            "byte_count": 123,
                        },
                    }
                ],
            }
        ]
    }

    records = module._collect_video_descriptors(value)

    assert records == [
        {
            "group_id": "group-a",
            "role": "source-evaluation",
            "case_id": "case-a",
            "view_id": "camera-0",
            "relative_video_path": "group-a/camera-0/undistorted.mp4",
            "video_sha256": digest,
            "video_byte_count": 123,
        }
    ]


def test_video_descriptor_rejects_path_escape() -> None:
    module = _module()
    with pytest.raises(ValueError, match="confined relative path"):
        module._collect_video_descriptors(
            {
                "path": "../undistorted.mp4",
                "sha256": "a" * 64,
                "byte_count": 1,
            }
        )


def test_candidate_reference_inventory_does_not_open_content(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "processed"
    video = root / "object" / "camera" / "undistorted.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    point_map = root / "object" / "point_map.npz"
    point_map.write_bytes(b"not-opened")
    unrelated = root / "object" / "notes.bin"
    unrelated.write_bytes(b"ignored")

    records = module._candidate_reference_files(video, root=root)

    assert records == [
        {
            "relative_path": "object/point_map.npz",
            "byte_count": len(b"not-opened"),
            "suffix": ".npz",
            "content_opened": False,
        }
    ]


def test_file_hash_is_streaming_and_exact(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "video.mp4"
    path.write_bytes(b"abc" * 1000)

    assert module._file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_request_rejects_missing_merged_lock(tmp_path: Path) -> None:
    module = _module()
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError, match="required merged lock is missing"):
        module.validate_request(request_path, repository=tmp_path)
