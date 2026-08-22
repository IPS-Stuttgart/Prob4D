from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prob4d.cut3r_camera_geometry import CameraRelativeDepthDisagreementModel
from prob4d.cut3r_direct_provider_adapter import (
    CUT3R_DIRECT_RAY_FRAME_SEMANTICS,
    CUT3R_DIRECT_RAY_SEMANTICS,
    import_cut3r_direct_prediction_manifest,
)
from prob4d.cut3r_pointmap_fidelity import (
    build_cut3r_pointmap_fidelity_report,
    load_cut3r_pointmap_fidelity_report,
    save_cut3r_pointmap_fidelity_report,
    verify_cut3r_pointmap_fidelity_report,
)
from prob4d.data import PredictionWindow
from prob4d.prediction_cli import main as prediction_main


def _write_direct_frame(
    root: Path,
    index: int,
    *,
    point_offset_x: float = 0.1,
    common_translation: np.ndarray | None = None,
) -> np.ndarray:
    stem = f"{index:06d}"
    for directory in ("points", "depth", "conf", "camera"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    depth = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    confidence = np.full((2, 2), 2.0, dtype=np.float32)
    intrinsics = np.asarray(
        [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    pixels = np.stack((columns, rows, np.ones_like(rows)), axis=-1)
    rays = np.linalg.solve(intrinsics, pixels.reshape(-1, 3).T).T.reshape(2, 2, 3)
    points = rays * depth[..., None]
    points[..., 0] += point_offset_x
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(index)
    if common_translation is not None:
        pose[:3, 3] += np.asarray(common_translation, dtype=np.float64)

    np.save(root / "points" / f"{stem}.npy", points.astype(np.float32))
    np.save(root / "depth" / f"{stem}.npy", depth)
    np.save(root / "conf" / f"{stem}.npy", confidence)
    np.savez(
        root / "camera" / f"{stem}.npz",
        pose=pose,
        intrinsics=intrinsics,
    )
    return points


def _import(root: Path, output: Path):
    return import_cut3r_direct_prediction_manifest(
        root,
        output,
        sequence_id="direct-sequence",
        cut3r_revision="a" * 40,
        checkpoint_sha256="b" * 64,
        input_video_sha256="c" * 64,
        input_video_byte_count=1234,
        frame_start=10,
        confidence_threshold=1.5,
    )


def test_direct_import_preserves_original_xyz_and_camera_rays(tmp_path: Path) -> None:
    source = tmp_path / "cut3r"
    first_points = _write_direct_frame(source, 0)
    second_points = _write_direct_frame(source, 1)
    output = tmp_path / "bundle" / "provider.json"

    manifest = _import(source, output)

    assert manifest.provider_family == "CUT3R-online-direct"
    assert manifest.ray_semantics == CUT3R_DIRECT_RAY_SEMANTICS
    assert manifest.metadata["direct_pointmap_preserved"] is True
    assert manifest.metadata["depth_reprojection_used"] is False
    assert manifest.metadata["geometry_source"] == "pts3d-in-self-view-direct-v1"
    assert manifest.metadata["raw_confidence_source_bound"] is True
    assert (
        manifest.metadata["camera_ray_frame_semantics"]
        == CUT3R_DIRECT_RAY_FRAME_SEMANTICS
    )
    assert manifest.metadata["world_origin_ray_fallback_allowed"] is False
    assert manifest.metadata["dense_array_byte_count"] == 120
    assert manifest.metadata["ray_direction_array_byte_count"] == 96
    assert manifest.metadata["total_dense_array_byte_count"] == 216
    payload = manifest.payloads[0]
    assert payload.has_ray_directions is True
    window = PredictionWindow.from_npz(
        output.parent / payload.path,
        dense_storage_dtype="float32",
    )
    np.testing.assert_allclose(window.point_map[0], first_points, atol=1e-6)
    expected_second = second_points.copy()
    expected_second[..., 0] += 1.0
    np.testing.assert_allclose(window.point_map[1], expected_second, atol=1e-6)
    assert window.ray_directions is not None
    expected_rays = first_points / np.linalg.norm(first_points, axis=-1, keepdims=True)
    np.testing.assert_allclose(window.ray_directions[0], expected_rays, atol=1e-6)
    np.testing.assert_allclose(window.ray_directions[1], expected_rays, atol=1e-6)
    assert window.frame_indices.tolist() == [10, 11]
    assert np.all(window.valid_mask)


def test_direct_camera_relative_covariance_is_common_translation_invariant(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first"
    second_source = tmp_path / "second"
    _write_direct_frame(first_source, 0)
    _write_direct_frame(
        second_source,
        0,
        common_translation=np.asarray([100.0, -40.0, 25.0]),
    )
    first_output = tmp_path / "first-bundle" / "provider.json"
    second_output = tmp_path / "second-bundle" / "provider.json"
    first_manifest = _import(first_source, first_output)
    second_manifest = _import(second_source, second_output)
    first = PredictionWindow.from_npz(
        first_output.parent / first_manifest.payloads[0].path,
        dense_storage_dtype="float32",
    )
    second = PredictionWindow.from_npz(
        second_output.parent / second_manifest.payloads[0].path,
        dense_storage_dtype="float32",
    )

    model = CameraRelativeDepthDisagreementModel(
        parallel_floor=0.1,
        parallel_depth_coefficient=0.2,
        lateral_floor=0.05,
        lateral_depth_coefficient=0.1,
    )
    first_covariance = model.predict(first)
    second_covariance = model.predict(second)

    np.testing.assert_allclose(
        first_covariance.parallel_variance,
        second_covariance.parallel_variance,
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        first_covariance.lateral_variance,
        second_covariance.lateral_variance,
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        first_covariance.ray_directions,
        second_covariance.ray_directions,
        atol=1e-6,
        rtol=1e-6,
    )


def test_direct_import_requires_direct_point_members(tmp_path: Path) -> None:
    source = tmp_path / "cut3r"
    _write_direct_frame(source, 0)
    for member in (source / "points").iterdir():
        member.unlink()

    with pytest.raises(ValueError, match="points.*empty"):
        _import(source, tmp_path / "bundle/provider.json")


def test_fidelity_audit_localizes_reprojection_and_passes_prefix_closure(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "prefix"
    extended = tmp_path / "extended"
    for index in range(2):
        _write_direct_frame(prefix, index)
        _write_direct_frame(extended, index)
    _write_direct_frame(extended, 2)

    report = build_cut3r_pointmap_fidelity_report(
        prefix,
        extended,
        prefix_frame_count=2,
        confidence_threshold=1.5,
        maximum_rms_error_m=1e-3,
        maximum_frame_p95_error_m=1e-3,
    )

    assert report["geometry_classification"] == "direct-pointmap-required"
    assert report["direct_route_ready"] is True
    assert report["depth_reprojection_compatibility_admissible"] is False
    fidelity = report["geometry_fidelity"]
    assert fidelity["evaluated_point_count"] == 8
    assert fidelity["point_weighted_rms_error_m"] == pytest.approx(0.1)
    assert len(report["source_bundles"]["prefix"]["members"]) == 8
    closure = report["causal_prefix_closure"]
    assert closure["status"] == "pass"

    output = tmp_path / "fidelity.json"
    save_cut3r_pointmap_fidelity_report(output, report)
    loaded = load_cut3r_pointmap_fidelity_report(output)
    assert loaded == report
    assert verify_cut3r_pointmap_fidelity_report(output, prefix, extended) == report


def test_fidelity_audit_detects_noncausal_prefix_change(tmp_path: Path) -> None:
    prefix = tmp_path / "prefix"
    extended = tmp_path / "extended"
    _write_direct_frame(prefix, 0)
    _write_direct_frame(extended, 0)
    _write_direct_frame(extended, 1)
    path = extended / "points" / "000000.npy"
    changed = np.load(path, allow_pickle=False)
    changed[0, 0, 0] += 1e-2
    np.save(path, changed)

    report = build_cut3r_pointmap_fidelity_report(
        prefix,
        extended,
        prefix_frame_count=1,
        confidence_threshold=1.5,
        maximum_rms_error_m=1.0,
        maximum_frame_p95_error_m=1.0,
        point_closure_tolerance_m=1e-6,
    )

    assert report["direct_route_ready"] is False
    closure = report["causal_prefix_closure"]
    assert closure["status"] == "fail"
    assert closure["point_maximum_difference_m"] == pytest.approx(1e-2)


def test_prediction_cli_exposes_direct_import_and_fidelity_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as direct_exit:
        prediction_main(["import-cut3r-direct", "--help"])
    assert direct_exit.value.code == 0
    direct_help = capsys.readouterr().out
    assert "direct XYZ point maps" in direct_help
    assert "--checkpoint-sha256" in direct_help

    with pytest.raises(SystemExit) as fidelity_exit:
        prediction_main(["cut3r-fidelity", "--help"])
    assert fidelity_exit.value.code == 0
    fidelity_help = capsys.readouterr().out
    assert "causal-prefix closure" in fidelity_help
    assert "{build,verify}" in fidelity_help
