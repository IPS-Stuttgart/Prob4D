from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prob4d.cut3r_camera_geometry import (
    CameraRelativeDepthDisagreementModel,
    recover_camera_relative_geometry,
)
from prob4d.cut3r_provider_adapter import (
    CUT3R_CAMERA_RAY_FRAME_SEMANTICS,
    CUT3R_CAMERA_RAY_SEMANTICS,
    import_cut3r_online_prediction_manifest,
)
from prob4d.data import PredictionWindow
from prob4d.uncertainty import DepthDisagreementModel


def _normalized(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def _geometry_window(
    camera_center: np.ndarray,
    *,
    translation: np.ndarray | None = None,
) -> PredictionWindow:
    rays = _normalized(
        np.asarray(
            [
                [-0.25, -0.25, 1.0],
                [0.25, -0.25, 1.0],
                [-0.25, 0.25, 1.0],
                [0.25, 0.25, 1.0],
            ],
            dtype=np.float64,
        )
    )
    ranges = np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    offset = np.zeros(3) if translation is None else np.asarray(translation)
    points = camera_center + offset + ranges[:, None] * rays
    return PredictionWindow(
        window_id="cut3r-online",
        frame_indices=np.asarray([0], dtype=np.int64),
        point_map=points.reshape(1, 2, 2, 3),
        valid_mask=np.ones((1, 2, 2), dtype=bool),
        ray_directions=rays.reshape(1, 2, 2, 3),
        dense_storage_dtype="float64",
    )


def _write_cut3r_frame(
    root: Path,
    index: int,
    *,
    translation: np.ndarray,
) -> None:
    stem = f"{index:06d}"
    (root / "depth").mkdir(parents=True, exist_ok=True)
    (root / "conf").mkdir(parents=True, exist_ok=True)
    (root / "camera").mkdir(parents=True, exist_ok=True)
    depth = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    confidence = np.full((2, 2), 3.0, dtype=np.float32)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = translation
    intrinsics = np.asarray(
        [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    np.save(root / "depth" / f"{stem}.npy", depth)
    np.save(root / "conf" / f"{stem}.npy", confidence)
    np.savez_compressed(
        root / "camera" / f"{stem}.npz",
        pose=pose,
        intrinsics=intrinsics,
    )


def test_camera_relative_geometry_recovers_centres_and_ranges() -> None:
    camera_center = np.asarray([10.0, -3.0, 2.0])
    window = _geometry_window(camera_center)

    geometry = recover_camera_relative_geometry(window)

    np.testing.assert_allclose(
        geometry.camera_centers_local,
        camera_center[None, :],
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        geometry.range_local,
        np.asarray([[[2.0, 3.0], [4.0, 5.0]]]),
        atol=1e-12,
        rtol=1e-12,
    )
    assert geometry.maximum_perpendicular_residual_local < 1e-12
    assert geometry.summary()["semantics"].endswith("-v1")


def test_camera_relative_covariance_is_common_frame_translation_invariant() -> None:
    camera_center = np.asarray([1.0, 2.0, 3.0])
    translated = np.asarray([100.0, -40.0, 25.0])
    first = _geometry_window(camera_center)
    second = _geometry_window(camera_center, translation=translated)
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
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        first_covariance.lateral_variance,
        second_covariance.lateral_variance,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        first_covariance.ray_directions,
        second_covariance.ray_directions,
        atol=1e-12,
        rtol=1e-12,
    )

    legacy_first = DepthDisagreementModel(
        parallel_floor=0.1,
        parallel_depth_coefficient=0.2,
        lateral_floor=0.05,
        lateral_depth_coefficient=0.1,
    ).predict(first)
    legacy_second = DepthDisagreementModel(
        parallel_floor=0.1,
        parallel_depth_coefficient=0.2,
        lateral_floor=0.05,
        lateral_depth_coefficient=0.1,
    ).predict(second)
    assert not np.allclose(
        legacy_first.parallel_variance,
        legacy_second.parallel_variance,
    )


def test_camera_relative_geometry_fails_closed_without_explicit_rays() -> None:
    with_rays = _geometry_window(np.asarray([1.0, 2.0, 3.0]))
    without_rays = PredictionWindow(
        window_id=with_rays.window_id,
        frame_indices=with_rays.frame_indices,
        point_map=with_rays.point_map,
        valid_mask=with_rays.valid_mask,
        dense_storage_dtype="float64",
    )

    with pytest.raises(ValueError, match="explicit camera-origin ray_directions"):
        recover_camera_relative_geometry(without_rays)


def test_cut3r_import_preserves_true_camera_rays(tmp_path: Path) -> None:
    source = tmp_path / "cut3r"
    _write_cut3r_frame(
        source,
        0,
        translation=np.asarray([10.0, -4.0, 2.0]),
    )
    output = tmp_path / "bundle" / "provider.json"

    manifest = import_cut3r_online_prediction_manifest(
        source,
        output,
        sequence_id="sequence-a",
        cut3r_revision="a" * 40,
        checkpoint_sha256="b" * 64,
        input_video_sha256="c" * 64,
        input_video_byte_count=1234,
        confidence_threshold=1.5,
    )

    payload = manifest.payloads[0]
    assert payload.has_ray_directions is True
    assert manifest.ray_semantics == CUT3R_CAMERA_RAY_SEMANTICS
    assert (
        manifest.metadata["camera_ray_frame_semantics"]
        == CUT3R_CAMERA_RAY_FRAME_SEMANTICS
    )
    assert manifest.metadata["camera_ray_translation_invariant"] is True
    assert manifest.metadata["world_origin_ray_fallback_allowed"] is False
    assert manifest.metadata["dense_array_byte_count"] == 60
    assert manifest.metadata["ray_direction_array_byte_count"] == 48
    assert manifest.metadata["total_dense_array_byte_count"] == 108

    window = PredictionWindow.from_npz(
        output.parent / payload.path,
        dense_storage_dtype="float32",
    )
    assert window.ray_directions is not None
    expected_ray = _normalized(np.asarray([-0.25, -0.25, 1.0]))
    np.testing.assert_allclose(
        window.ray_directions[0, 0, 0],
        expected_ray,
        atol=1e-6,
        rtol=1e-6,
    )
    world_origin_direction = _normalized(window.point_map[0, 0, 0])
    assert not np.allclose(
        window.ray_directions[0, 0, 0],
        world_origin_direction,
    )

    geometry = recover_camera_relative_geometry(window)
    np.testing.assert_allclose(
        geometry.camera_centers_local[0],
        np.asarray([10.0, -4.0, 2.0]),
        atol=2e-5,
        rtol=2e-5,
    )
