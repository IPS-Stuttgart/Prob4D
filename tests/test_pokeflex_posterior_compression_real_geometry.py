from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_pokeflex_posterior_compression_real_geometry.py"
SPEC = importlib.util.spec_from_file_location("pokeflex_real_geometry_study", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PROTOCOL = json.loads(
    (
        ROOT
        / "protocols/pokeflex-posterior-compression-real-geometry-v1.json"
    ).read_text(encoding="utf-8")
)


def _trajectory() -> np.ndarray:
    rng = np.random.default_rng(20260831)
    base = rng.normal(size=(96, 3)) * np.asarray([0.20, 0.08, 0.04])
    frames = []
    for index in range(10):
        time = index / 9.0
        frame = base.copy()
        frame[:, 0] += 0.025 * time
        frame[:, 1] += 0.04 * time * np.sin(4.0 * base[:, 0])
        frame[:, 2] += 0.03 * time**2 * (base[:, 0] / 0.20) ** 2
        frames.append(frame)
    return np.stack(frames)


def _obj_bytes(points: np.ndarray) -> bytes:
    return "".join(
        f"v {point[0]:.12g} {point[1]:.12g} {point[2]:.12g}\n"
        for point in points
    ).encode()


def test_obj_sequence_discovery_and_loading(tmp_path: Path) -> None:
    archive_path = tmp_path / "take.zip"
    trajectory = _trajectory()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for frame_index, frame in enumerate(trajectory):
            archive.writestr(
                f"object/mesh_{frame_index:05d}.obj",
                _obj_bytes(frame),
            )
        archive.writestr("object/readme.txt", "not geometry")
    with zipfile.ZipFile(archive_path) as archive:
        candidates = MODULE.discover_candidates(
            archive.infolist(),
            supported_suffixes=set(PROTOCOL["supported_geometry_suffixes"]),
            minimum_frames=PROTOCOL["minimum_sequence_frames"],
            maximum_member_bytes=PROTOCOL["maximum_geometry_member_bytes"],
        )
        assert candidates
        assert candidates[0].kind == "files"
        loaded = MODULE.load_candidate_trajectory(
            archive,
            candidates[0],
            maximum_frames=PROTOCOL["maximum_sequence_frames"],
            maximum_member_bytes=PROTOCOL["maximum_geometry_member_bytes"],
            maximum_vertices=PROTOCOL["maximum_vertices_per_frame"],
        )
    np.testing.assert_allclose(loaded, trajectory, rtol=1e-10, atol=1e-12)


def test_array_trajectory_is_loaded_without_pickle() -> None:
    trajectory = _trajectory()
    payload = io.BytesIO()
    np.savez_compressed(payload, metadata=np.arange(4), vertices=trajectory)
    loaded = MODULE.load_array_trajectory(payload.getvalue(), ".npz")
    np.testing.assert_array_equal(loaded, trajectory)


def test_real_geometry_diagnostic_preserves_full_query_posterior() -> None:
    result = MODULE.evaluate_trajectory(_trajectory(), protocol=PROTOCOL)
    assert result["query_dimension"] == 3
    assert result["full_shared_rank"] == 56
    assert result["retained_shared_rank"] <= 3
    assert not result["exact_factor_fallback"]
    assert result["full_vs_compressed_relative_gain_error"] < 1e-8
    assert result["full_vs_compressed_relative_covariance_error"] < 1e-8
    assert result["full_vs_compressed_mean_difference"] < 1e-8
    assert result["shared_factor_payload_reduction_ratio"] > 10.0
    assert (
        result["full_vs_equal_rank_pca_relative_covariance_error"]
        > result["full_vs_compressed_relative_covariance_error"]
    )


def test_end_to_end_run_retains_incomplete_inventory_state(tmp_path: Path) -> None:
    dataset = tmp_path / "pokeflex"
    dataset.mkdir()
    trajectory = _trajectory()
    with zipfile.ZipFile(dataset / "available_take.zip", "w") as archive:
        for frame_index, frame in enumerate(trajectory):
            archive.writestr(
                f"meshes/frame_{frame_index:04d}.obj",
                _obj_bytes(frame),
            )
    output = tmp_path / "output"
    result = MODULE.run(dataset, PROTOCOL, output)
    assert result["status"] == "evaluated-real-geometry"
    assert result["dataset_state"] == "known-incomplete-local-mirror"
    assert result["inventory_summary"]["observed_zip_count"] == 1
    assert not result["inventory_summary"]["release_completeness_asserted"]
    assert result["summary"]["evaluated_archive_count"] == 1
    assert not result["prior_causal4d_pokeflex_target_opened_or_repaired"]


def test_unsupported_and_corrupt_archives_are_retained_not_silently_dropped(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "pokeflex"
    dataset.mkdir()
    with zipfile.ZipFile(dataset / "rgb_only.zip", "w") as archive:
        archive.writestr("rgb/frame_0001.png", b"not-a-real-png")
    (dataset / "truncated.zip").write_bytes(b"PK\x03\x04truncated")
    result = MODULE.run(dataset, PROTOCOL, tmp_path / "output")
    assert result["status"] == "no-supported-geometry-candidates"
    assert result["inventory_summary"]["observed_zip_count"] == 2
    assert result["inventory_summary"]["unreadable_central_directories"] == 1
    assert result["inventory_summary"]["archives_with_supported_candidate"] == 0


def test_variable_vertex_counts_fail_closed(tmp_path: Path) -> None:
    archive_path = tmp_path / "take.zip"
    trajectory = _trajectory()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for frame_index, frame in enumerate(trajectory[:6]):
            if frame_index == 5:
                frame = frame[:-1]
            archive.writestr(
                f"mesh_{frame_index:03d}.obj",
                _obj_bytes(frame),
            )
    with zipfile.ZipFile(archive_path) as archive:
        candidate = MODULE.discover_candidates(
            archive.infolist(),
            supported_suffixes={".obj"},
            minimum_frames=6,
            maximum_member_bytes=10_000_000,
        )[0]
        try:
            MODULE.load_candidate_trajectory(
                archive,
                candidate,
                maximum_frames=6,
                maximum_member_bytes=10_000_000,
                maximum_vertices=1_000,
            )
        except ValueError as error:
            assert "material identity" in str(error)
        else:
            raise AssertionError("variable vertex counts must fail closed")
