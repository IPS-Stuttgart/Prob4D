from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from prob4d import _cut3r_source_preflight_cases as cases
from prob4d import _cut3r_source_preflight_common as common
from prob4d import _cut3r_source_preflight_environment as environment
from prob4d import _cut3r_source_preflight_freeze as freeze_contract
from prob4d import _cut3r_source_preflight_runtime as runtime
from prob4d.cut3r_comparison import build_cut3r_comparison_lock

ROOT = Path(__file__).resolve().parents[1]
REQUEST = (
    ROOT
    / "protocols"
    / "execution_requests"
    / "cut3r_deform360_source_comparison_preflight_v1.json"
)

REQUEST_VALUE = {
    "claim_boundary": (
        "This request authorizes one retained-runner metadata preflight for the frozen "
        "CUT3R Deform360 source comparison. It may resolve exact source video and "
        "sidecar paths, verify hashes, inspect CUT3R's installed callable surface, and "
        "enumerate source-only candidate reference files. It may not decode RGB frames, "
        "execute CUT3R, open source residuals or truth values, open confirmation or "
        "target payloads, run BayesianPhysTwin, or run Causal4D."
    ),
    "comparison_execution_authorized": False,
    "comparison_lock_path": "protocols/locks/cut3r_deform360_comparison_lock_v2.json",
    "comparison_spec_path": "protocols/locks/cut3r_deform360_comparison_spec_v2.json",
    "expected_case_count": 40,
    "forbidden_target_group_count": 12,
    "issue_number": 49,
    "schema": "prob4d.cut3r-deform360-source-comparison-preflight-request",
    "schema_version": 1,
    "source_freeze_path": "protocols/locks/cut3r_deform360_source_freeze_v2.json",
    "source_group_count": 10,
    "source_prediction_payloads_opened": False,
    "source_residuals_or_truth_opened": False,
    "source_rgb_frames_decoded": False,
    "target_outcomes_opened": False,
    "target_payloads_opened": False,
}

INFORMATION_BOUNDARY = {
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
}


def _request() -> dict[str, object]:
    request = dict(REQUEST_VALUE)
    request["preflight_request_id"] = common._record_id(request)
    return request


def _synthetic_contract():
    provider_revision = "a" * 40
    checkpoint_sha = "b" * 64
    prob4d_revision = "c" * 40
    distribution_sha = "d" * 64
    cameras = ["cam0", "cam1", "cam2", "cam3"]
    roles = {
        "development": [],
        "calibration": [],
        "source_evaluation": [],
    }
    source_groups = []
    source_cases = []
    lock_groups = []
    for group_index in range(10):
        group_id = f"group-{group_index:02d}"
        object_id = f"{group_index:03d}-object"
        episode_id = group_index + 1
        if group_index < 4:
            role = "development"
        elif group_index < 7:
            role = "calibration"
        else:
            role = "source_evaluation"
        roles[role].append(group_id)
        source_groups.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": f"stratum-{group_index % 2}",
                "group_id": group_id,
                "role": role,
            }
        )
        lock_cases = []
        for camera_index, camera in enumerate(cameras):
            case_id = f"{group_id}-{camera}"
            digest = f"{group_index:02x}{camera_index:02x}".ljust(64, "e")
            byte_count = 1000 + group_index * 10 + camera_index
            locator = {
                "case_id": case_id,
                "group_id": group_id,
                "object_id": object_id,
                "episode_id": episode_id,
                "camera": camera,
                "relative_episode_path": f"{object_id}/episode_{episode_id:04d}",
                "relative_camera_path": (f"{object_id}/episode_{episode_id:04d}/{camera}"),
                "input_video_sha256": digest,
                "input_video_byte_count": byte_count,
                "aligned_timestamp_count": 64,
                "sidecar_sha256": {
                    "aligned_timestamps.txt": "1" * 64,
                    "alignment.json": "2" * 64,
                    "metadata.json": "3" * 64,
                },
                "sidecar_byte_count": {
                    "aligned_timestamps.txt": 10,
                    "alignment.json": 11,
                    "metadata.json": 12,
                },
            }
            locator["source_case_id"] = common._record_id(locator)
            source_cases.append(locator)
            lock_cases.append(
                {
                    "case_id": case_id,
                    "input_video_sha256": digest,
                    "input_video_byte_count": byte_count,
                    "frame_start": 0,
                    "frame_stop_exclusive": 64,
                    "evaluation_frame_start": 16,
                    "evaluation_frame_stop_exclusive": 64,
                }
            )
        lock_groups.append({"group_id": group_id, "cases": lock_cases})
    spec = {
        "protocol_name": "synthetic",
        "provider_revision": provider_revision,
        "checkpoint_sha256": checkpoint_sha,
        "prob4d_revision": prob4d_revision,
        "prob4d_distribution_sha256": distribution_sha,
        "window_size": 16,
        "overlap": 8,
        "confidence_threshold": 0.0,
        "storage_dtype": "float32",
        "random_seeds": [0],
        "groups": lock_groups,
        "group_roles": roles,
        "include_revisit_diagnostic": False,
    }
    lock = {
        "lock_id": "f" * 64,
        "provider": {
            "repository": "CUT3R/CUT3R",
            "revision": provider_revision,
            "checkpoint_sha256": checkpoint_sha,
        },
        "prob4d": {
            "project_id": "prob4d",
            "revision": prob4d_revision,
            "distribution_sha256": distribution_sha,
        },
        "groups": lock_groups,
        "group_roles": roles,
    }
    freeze = {
        "schema": "prob4d.cut3r-deform360-source-freeze",
        "schema_version": 1,
        "decision": "source-support-freeze-ready",
        "source_group_count": 10,
        "source_groups": source_groups,
        "forbidden_target_group_count": 12,
        "forbidden_target_groups": [
            {
                "object_id": f"target-{index:02d}",
                "episode_id": 100 + index,
                "stratum": "target",
            }
            for index in range(12)
        ],
        "provider": {
            "repository": "CUT3R/CUT3R",
            "revision": provider_revision,
            "checkpoint_filename": "model.pth",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_byte_count": 1234,
            "execution_mode": "recurrent-online",
            "revisit_count": 1,
            "global_alignment": False,
            "second_pass_allowed": False,
        },
        "prob4d": {
            "revision": prob4d_revision,
            "distribution_filename": "prob4d.whl",
            "distribution_sha256": distribution_sha,
            "distribution_byte_count": 4321,
        },
        "comparison_spec_sha256": common._record_id(spec),
        "camera_panel": {
            "selected_cameras": cameras,
        },
        "source_cases": source_cases,
        "information_boundary": dict(INFORMATION_BOUNDARY),
        "claim_boundary": "synthetic source-only boundary",
    }
    freeze["source_freeze_id"] = common._record_id(freeze)
    return _request(), spec, lock, freeze


def test_checked_in_request_matches_canonical_request() -> None:
    checked_in = json.loads(REQUEST.read_text(encoding="utf-8"))

    assert checked_in == _request()
    assert runtime._file_sha256 is common._file_sha256


def test_checked_in_request_preserves_outcome_blind_boundary() -> None:
    request = _request()

    assert request["source_group_count"] == 10
    assert request["forbidden_target_group_count"] == 12
    assert request["expected_case_count"] == 40
    assert request["source_rgb_frames_decoded"] is False
    assert request["source_prediction_payloads_opened"] is False
    assert request["source_residuals_or_truth_opened"] is False
    assert request["target_payloads_opened"] is False
    assert request["target_outcomes_opened"] is False
    assert request["comparison_execution_authorized"] is False


def test_request_id_is_canonical_and_content_addressed() -> None:
    request = _request()
    recorded = request.pop("preflight_request_id")

    assert recorded == hashlib.sha256(common._canonical_json_bytes(request)).hexdigest()


def test_validate_request_recomputes_identity(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    for relative in (
        REQUEST_VALUE["source_freeze_path"],
        REQUEST_VALUE["comparison_spec_path"],
        REQUEST_VALUE["comparison_lock_path"],
    ):
        path = repository / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    request = _request()
    request["claim_boundary"] = str(request["claim_boundary"]) + " changed"
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        common.validate_request(request_path, repository=repository)


def test_validate_source_freeze_matches_real_locator_shape() -> None:
    request, spec, lock, freeze = _synthetic_contract()

    contract = freeze_contract._validate_source_freeze(
        freeze,
        request=request,
        spec=spec,
        lock=lock,
    )

    assert contract["provider_revision"] == "a" * 40
    assert contract["checkpoint_sha256"] == "b" * 64
    assert contract["prob4d_distribution_filename"] == "prob4d.whl"
    assert len(contract["descriptors"]) == 40
    first = contract["descriptors"][0]
    assert first["relative_video_path"].endswith("/cam0/undistorted.mp4")
    assert first["sidecars"]["alignment.json"]["relative_path"].endswith("/cam0/alignment.json")


def test_validate_source_freeze_rejects_information_boundary_drift() -> None:
    request, spec, lock, freeze = _synthetic_contract()
    freeze["information_boundary"]["target_payloads_opened"] = True
    freeze.pop("source_freeze_id")
    freeze["source_freeze_id"] = common._record_id(freeze)

    with pytest.raises(ValueError, match="information boundary"):
        freeze_contract._validate_source_freeze(
            freeze,
            request=request,
            spec=spec,
            lock=lock,
        )


def test_load_comparison_lock_rejects_noncanonical_retained_bytes(
    tmp_path: Path,
) -> None:
    request, spec, lock, freeze = _synthetic_contract()
    del request, lock, freeze
    canonical = build_cut3r_comparison_lock(spec)
    path = tmp_path / "comparison-lock.json"
    path.write_text(json.dumps(canonical), encoding="utf-8")

    assert freeze_contract._load_comparison_lock(path, spec) == canonical

    canonical["claim_boundary"] = str(canonical["claim_boundary"]) + " changed"
    path.write_text(json.dumps(canonical), encoding="utf-8")
    with pytest.raises(ValueError):
        freeze_contract._load_comparison_lock(path, spec)


def test_validate_source_freeze_rejects_provider_drift() -> None:
    request, spec, lock, freeze = _synthetic_contract()
    freeze["provider"]["revision"] = "9" * 40
    freeze.pop("source_freeze_id")
    freeze["source_freeze_id"] = common._record_id(freeze)

    with pytest.raises(ValueError, match="different provider revisions"):
        freeze_contract._validate_source_freeze(
            freeze,
            request=request,
            spec=spec,
            lock=lock,
        )


def test_source_case_rejects_noncanonical_camera_path() -> None:
    request, spec, lock, freeze = _synthetic_contract()
    del request, spec, lock
    case = freeze["source_cases"][0]
    case["relative_camera_path"] = "../target"
    case.pop("source_case_id")
    case["source_case_id"] = common._record_id(case)
    groups = cases._validate_source_groups(freeze["source_groups"], expected_count=10)

    with pytest.raises(ValueError, match="confined relative path"):
        cases._collect_source_case_descriptors(
            freeze,
            source_groups=groups,
            expected_case_count=40,
        )


def test_candidate_reference_inventory_is_confined_to_source_episode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processed"
    episode = root / "source" / "episode_0001"
    camera = episode / "cam0"
    camera.mkdir(parents=True)
    (camera / "undistorted.mp4").write_bytes(b"video")
    source_map = episode / "point_map.npz"
    source_map.write_bytes(b"not-opened")
    target = root / "target" / "episode_0002"
    target.mkdir(parents=True)
    (target / "truth.npy").write_bytes(b"forbidden")

    records = environment._candidate_reference_files(episode, root=root)

    assert records == [
        {
            "relative_path": "source/episode_0001/point_map.npz",
            "byte_count": len(b"not-opened"),
            "suffix": ".npz",
            "content_opened": False,
        }
    ]


def test_file_hash_is_streaming_and_exact(tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"abc" * 1000)

    assert common._file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_request_rejects_missing_merged_lock(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request()), encoding="utf-8")

    with pytest.raises(ValueError, match="required merged lock"):
        common.validate_request(request_path, repository=tmp_path)


def test_diagnostics_are_content_addressed_without_retaining_paths() -> None:
    text = "/retained/cut3r/checkpoint /retained/cut3r"
    replacements = {
        "/retained/cut3r": "<CUT3R_CHECKOUT>",
        "/retained/cut3r/checkpoint": "<CUT3R_CHECKPOINT>",
    }

    sanitized = environment._sanitize_text(text, replacements)
    evidence = environment._text_evidence(text, replacements)

    assert sanitized == "<CUT3R_CHECKPOINT> <CUT3R_CHECKOUT>"
    assert evidence["content_retained"] is False
    assert set(evidence) == {"sha256", "byte_count", "content_retained"}


def test_cut3r_surface_refuses_untracked_demo(tmp_path: Path) -> None:
    checkout = tmp_path / "cut3r"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Prob4D tests"],
        cwd=checkout,
        check=True,
    )
    (checkout / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=checkout, check=True)
    (checkout / "demo.py").write_text(
        "raise SystemExit('untracked demo must not execute')\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")

    surface = environment._cut3r_surface(checkout, checkpoint)

    assert surface["demo_relative_path"] is None
    assert surface["demo_resolution"]["tracked_candidate_count"] == 0
    assert surface["demo_help_status"] == 127
    assert "origin_url" not in surface


def test_github_remote_normalization_is_exact() -> None:
    assert (
        environment._github_repository_from_remote("https://github.com/CUT3R/CUT3R.git")
        == "CUT3R/CUT3R"
    )
    assert (
        environment._github_repository_from_remote("git@github.com:CUT3R/CUT3R.git")
        == "CUT3R/CUT3R"
    )
    assert environment._github_repository_from_remote("/tmp/local-cut3r") is None
    assert (
        environment._github_repository_from_remote(
            "https://token@example.invalid@github.com/CUT3R/CUT3R.git"
        )
        is None
    )
