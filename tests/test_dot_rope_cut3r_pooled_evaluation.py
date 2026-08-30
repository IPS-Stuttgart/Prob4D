from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from prob4d.dot_rope_cut3r_study import content_id

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "evaluate_dot_rope_cut3r_pooled.py"
WORKFLOW = ROOT / ".github" / "workflows" / "dot-rope-marker-support-audit-v1.yml"


def _load_module():
    spec = importlib.util.spec_from_file_location("dot_pooled", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(module):
    request = {
        "archive": module.ARCHIVE,
        "base_evaluator_git_blob_sha1": module.BASE_EVALUATOR_GIT_BLOB_SHA1,
        "camera": module.CAMERA,
        "claim_boundary": "source-only evaluation",
        "coordinate_columns": [0, 1],
        "coordinate_mode": "pixel-zero-based",
        "execution_nonce": "test-1",
        "frames": module.FRAMES,
        "marker_payloads_opened": True,
        "marker_support_audit_decision": ("current-convention-pooled-support-feasible"),
        "marker_support_audit_id": "a" * 64,
        "marker_support_audit_run_id": 1,
        "normal_view_pixels_opened": False,
        "performance_metrics_authorized": True,
        "preprocessing_transform": module.PREPROCESSING_TRANSFORM,
        "provider_artifact_name": module.PROVIDER_ARTIFACT_NAME,
        "provider_bundle_id": module.PROVIDER_BUNDLE_ID,
        "provider_request_id": module.PROVIDER_REQUEST_ID,
        "provider_revision": module.PROVIDER_REVISION,
        "provider_run_id": module.PROVIDER_RUN_ID,
        "reserved_sequences": module.RESERVED,
        "schema": module.REQUEST_SCHEMA,
        "schema_version": module.SCHEMA_VERSION,
        "selected_coordinate_candidate": "columns-0-1:pixel-zero-based",
        "source_protocol_git_blob_sha1": module.SOURCE_PROTOCOL_GIT_BLOB_SHA1,
        "source_protocol_id": module.SOURCE_PROTOCOL_ID,
        "source_protocol_path": module.SOURCE_PROTOCOL_PATH,
        "source_sequences": module.SEQUENCES,
        "support_rule": module.SUPPORT_RULE,
        "target_payloads_opened": False,
    }
    request["request_id"] = content_id(request)
    return request


def test_cut3r_transform_matches_resize_and_crop() -> None:
    module = _load_module()
    coordinates = np.asarray([[611.5, 511.5], [0.0, 0.0], [1223.0, 1023.0]])
    mapped, valid, metadata = module.cut3r_output_coordinates(
        coordinates,
        original_width=1224,
        original_height=1024,
        output_width=512,
        output_height=416,
    )
    assert metadata == {
        "resized_width": 512,
        "resized_height": 428,
        "crop_left": 0,
        "crop_top": 6,
        "output_width": 512,
        "output_height": 416,
    }
    np.testing.assert_allclose(mapped[0], [255.5, 207.5], atol=1.0e-12)
    assert valid.tolist() == [True, True, True]
    assert mapped[1, 1] < 0.0
    assert mapped[2, 1] > 415.0


def test_coordinate_parser_preserves_registered_numeric_columns() -> None:
    module = _load_module()
    module._ACTIVE_COORDINATE_COLUMNS = (1, 2)
    points = module._parse_coordinate_text(
        "marker 7: 120.5, 40.25\nmarker 8: 130, 50\nmarker 9: 140, 60\n",
        2,
    )
    np.testing.assert_allclose(
        points,
        [[120.5, 40.25], [130.0, 50.0], [140.0, 60.0]],
    )


def _fake_run(sequence: str, run_name: str, frames: list[int]):
    height, width = 16, 16
    points = np.zeros((len(frames), height, width, 3), dtype=np.float64)
    confidence = np.ones((len(frames), height, width), dtype=np.float64)
    for index in range(len(frames)):
        yy, xx = np.mgrid[:height, :width]
        points[index, ..., 0] = xx
        points[index, ..., 1] = yy
        points[index, ..., 2] = index
    return {
        "frames": np.asarray(frames, dtype=np.int64),
        "points": points,
        "confidence": confidence,
        "original_sizes": np.asarray([[16, 16]] * len(frames), dtype=np.int64),
        "_sequence": sequence,
        "_run_name": run_name,
    }


def test_pooled_pair_accepts_five_per_frame_across_three_frames(monkeypatch) -> None:
    module = _load_module()
    first = _fake_run("R01", "window_a", [3, 4, 5])
    second = _fake_run("R01", "window_b", [3, 4, 5])
    frame_payloads = {
        frame: (
            np.asarray([[2.0 + index, 3.0] for index in range(5)]),
            np.asarray([[index, 0.0, float(frame)] for index in range(5)]),
        )
        for frame in [3, 4, 5]
    }

    def sample(run, frame, coordinates_2d, coordinates_3d):
        del run, frame, coordinates_2d
        return coordinates_3d.copy(), coordinates_3d.copy(), np.arange(5)

    monkeypatch.setattr(module, "_sample_markers", sample)
    source, target, groups = module._collect_pair(
        first,
        second,
        frame_payloads,
        [3, 4, 5],
    )
    assert source.shape == target.shape == (15, 3)
    assert groups.tolist() == [3] * 5 + [4] * 5 + [5] * 5


def test_pooled_pair_rejects_one_nonempty_frame(monkeypatch) -> None:
    module = _load_module()
    first = _fake_run("R01", "window_a", [3, 4, 5])
    second = _fake_run("R01", "window_b", [3, 4, 5])
    frame_payloads = {frame: (np.zeros((6, 2)), np.zeros((6, 3))) for frame in [3, 4, 5]}

    def sample(run, frame, coordinates_2d, coordinates_3d):
        del run, coordinates_2d
        count = 6 if frame == 3 else 0
        return (
            coordinates_3d[:count].copy(),
            coordinates_3d[:count].copy(),
            np.arange(count),
        )

    monkeypatch.setattr(module, "_sample_markers", sample)
    try:
        module._collect_pair(first, second, frame_payloads, [3, 4, 5])
    except ValueError as error:
        assert "two nonempty frames" in str(error)
    else:
        raise AssertionError("single-frame pooled support unexpectedly passed")


def test_request_is_content_addressed(tmp_path: Path) -> None:
    module = _load_module()
    request = _request(module)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    assert module.validate_request(path)["request_id"] == request["request_id"]


def test_request_rejects_ambiguous_audit(tmp_path: Path) -> None:
    module = _load_module()
    request = _request(module)
    request["marker_support_audit_decision"] = "coordinate-convention-ambiguous"
    unsigned = {key: value for key, value in request.items() if key != "request_id"}
    request["request_id"] = content_id(unsigned)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    try:
        module.validate_request(path)
    except ValueError as error:
        assert "did not authorize" in str(error)
    else:
        raise AssertionError("ambiguous coordinate audit unexpectedly authorized scoring")


def test_workflow_is_request_bound_and_reuses_sealed_provider() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "protocols/execution_requests/dot_rope_cut3r_pooled_evaluation_v2.json" in text
    assert "branches: [main]" in text
    assert "runs-on: [self-hosted, Linux, X64, gpuserver4090]" in text
    assert 'test "$RUNNER_NAME" = "workstation1"' in text
    assert "actions: read" in text
    assert "github-token: ${{ github.token }}" in text
    assert "Download exact previously sealed provider bundle" in text
    assert "evaluate_dot_rope_cut3r_pooled.py" in text
    assert "normal-view pixels" in text
    assert "R04-R70 remained unopened" in text
    assert "secrets." not in text
    assert "git push" not in text
