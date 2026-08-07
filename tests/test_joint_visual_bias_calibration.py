from __future__ import annotations

import copy

import numpy as np
import pytest

from prob4d.joint_visual_bias_calibration import (
    JOINT_VISUAL_BIAS_METADATA_KEY,
    JointVisualBiasCalibrationGroupV1,
    JointVisualBiasLayoutV1,
    build_joint_visual_bias_nuisance_from_calibration,
    expand_joint_visual_bias_jacobian,
    fit_joint_visual_bias_calibration,
    joint_visual_bias_layout_from_calibration,
    joint_visual_bias_selection_summary,
)
from prob4d.visual_bias_calibration import (
    load_visual_bias_calibration,
    write_visual_bias_calibration,
)


def _full_layout() -> JointVisualBiasLayoutV1:
    return JointVisualBiasLayoutV1(
        camera_ids=("camera-0", "camera-1"),
        shared_basis_names=("shared-depth",),
        camera_basis_names=("camera-tangent",),
    )


def _designs(
    layout: JointVisualBiasLayoutV1,
    row_camera_indices: np.ndarray,
    *,
    camera_one_active: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = row_camera_indices.size
    shared = np.zeros(
        (row_count, 3, len(layout.shared_basis_names)),
        dtype=np.float64,
    )
    if layout.shared_basis_names:
        shared[:, 0, 0] = 1.0
    camera = np.zeros(
        (row_count, 3, len(layout.camera_basis_names)),
        dtype=np.float64,
    )
    if layout.camera_basis_names:
        camera[row_camera_indices == 0, 1, 0] = 1.0
        if camera_one_active:
            camera[row_camera_indices == 1, 1, 0] = 1.0
    return shared, camera


def _full_groups() -> tuple[JointVisualBiasCalibrationGroupV1, ...]:
    layout = _full_layout()
    generator = np.random.default_rng(4)
    groups: list[JointVisualBiasCalibrationGroupV1] = []
    for group_index in range(10):
        row_camera_indices = np.repeat(np.arange(2, dtype=np.int64), 16)
        shared, camera = _designs(layout, row_camera_indices)
        latent = generator.normal(size=3)
        coefficients = np.asarray(
            [
                0.35 * latent[0],
                0.25 * latent[1],
                0.18 * latent[1] + 0.12 * latent[2],
            ],
            dtype=np.float64,
        )
        expanded = expand_joint_visual_bias_jacobian(
            layout,
            row_camera_indices,
            shared,
            camera,
            require_all_cameras=True,
        )
        residual = np.einsum("ncr,r->nc", expanded, coefficients)
        residual += generator.normal(scale=0.01, size=residual.shape)
        covariance = np.repeat(
            (0.01**2 * np.eye(3, dtype=np.float64))[None],
            residual.shape[0],
            axis=0,
        )
        groups.append(
            JointVisualBiasCalibrationGroupV1(
                group_id=f"object-{group_index:02d}",
                layout=layout,
                row_camera_indices=row_camera_indices,
                residual=np.asarray(residual, dtype=np.float64),
                shared_bias_jacobian=shared,
                camera_bias_jacobian=camera,
                conditional_covariance=covariance,
                metadata={"episode_count": 2},
            )
        )
    return tuple(groups)


def _partial_groups() -> tuple[JointVisualBiasCalibrationGroupV1, ...]:
    layout = JointVisualBiasLayoutV1(
        camera_ids=("camera-0", "camera-1"),
        shared_basis_names=(),
        camera_basis_names=("camera-local",),
    )
    generator = np.random.default_rng(8)
    groups: list[JointVisualBiasCalibrationGroupV1] = []
    for group_index in range(8):
        row_camera_indices = np.repeat(np.arange(2, dtype=np.int64), 12)
        shared, camera = _designs(
            layout,
            row_camera_indices,
            camera_one_active=False,
        )
        coefficients = np.asarray(
            [0.3 * generator.normal(), 0.0],
            dtype=np.float64,
        )
        expanded = expand_joint_visual_bias_jacobian(
            layout,
            row_camera_indices,
            shared,
            camera,
            require_all_cameras=True,
        )
        residual = np.einsum("ncr,r->nc", expanded, coefficients)
        residual += generator.normal(scale=0.01, size=residual.shape)
        covariance = np.repeat(
            (0.01**2 * np.eye(3, dtype=np.float64))[None],
            residual.shape[0],
            axis=0,
        )
        groups.append(
            JointVisualBiasCalibrationGroupV1(
                group_id=f"partial-{group_index:02d}",
                layout=layout,
                row_camera_indices=row_camera_indices,
                residual=np.asarray(residual, dtype=np.float64),
                shared_bias_jacobian=shared,
                camera_bias_jacobian=camera,
                conditional_covariance=covariance,
            )
        )
    return tuple(groups)


def _fit_full():
    return fit_joint_visual_bias_calibration(
        _full_groups(),
        provider_manifest_id="a" * 64,
        calibration_source_id="b" * 64,
        group_definition="complete physical object",
        residual_definition="metric source minus provider point",
        uses_truth=True,
        covariance_shrinkage=0.0,
        minimum_nll_improvement=0.0,
    )


def test_layout_round_trip_and_tamper_detection() -> None:
    layout = _full_layout()

    loaded = JointVisualBiasLayoutV1.from_mapping(layout.to_dict())

    assert loaded == layout
    assert layout.basis_names == (
        "shared::shared-depth",
        "camera::camera-tangent::camera-0",
        "camera::camera-tangent::camera-1",
    )
    assert len(layout.layout_id or "") == 64

    tampered = copy.deepcopy(layout.to_dict())
    tampered["expanded_basis_names"][1] = "camera::changed::camera-0"
    with pytest.raises(ValueError, match="expanded basis names changed"):
        JointVisualBiasLayoutV1.from_mapping(tampered)

    coercive = copy.deepcopy(layout.to_dict())
    coercive["camera_ids"] = "camera-0"
    with pytest.raises(ValueError, match="JSON array"):
        JointVisualBiasLayoutV1.from_mapping(coercive)


def test_joint_design_expands_shared_and_camera_specific_columns_exactly() -> None:
    layout = _full_layout()
    row_camera_indices = np.asarray([0, 1, 0, 1], dtype=np.int64)
    shared, camera = _designs(layout, row_camera_indices)

    expanded = expand_joint_visual_bias_jacobian(
        layout,
        row_camera_indices,
        shared,
        camera,
        require_all_cameras=True,
    )

    assert expanded.shape == (4, 3, 3)
    np.testing.assert_array_equal(expanded[:, 0, 0], np.ones(4))
    np.testing.assert_array_equal(expanded[:, 1, 1], [1.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(expanded[:, 1, 2], [0.0, 1.0, 0.0, 1.0])
    assert not expanded.flags.writeable
    with pytest.raises(ValueError):
        expanded.setflags(write=True)

    shared[:, :, :] = 7.0
    assert float(expanded[0, 0, 0]) == 1.0


def test_calibration_group_requires_every_camera_and_exact_float64() -> None:
    layout = _full_layout()
    row_camera_indices = np.zeros(4, dtype=np.int64)
    shared, camera = _designs(layout, row_camera_indices)
    covariance = np.repeat(np.eye(3, dtype=np.float64)[None], 4, axis=0)

    with pytest.raises(ValueError, match="every calibration group"):
        JointVisualBiasCalibrationGroupV1(
            group_id="object",
            layout=layout,
            row_camera_indices=row_camera_indices,
            residual=np.zeros((4, 3), dtype=np.float64),
            shared_bias_jacobian=shared,
            camera_bias_jacobian=camera,
            conditional_covariance=covariance,
        )

    complete_indices = np.asarray([0, 1, 0, 1], dtype=np.int64)
    complete_shared, complete_camera = _designs(layout, complete_indices)
    with pytest.raises(ValueError, match="dtype float64"):
        JointVisualBiasCalibrationGroupV1(
            group_id="object",
            layout=layout,
            row_camera_indices=complete_indices,
            residual=np.zeros((4, 3), dtype=np.float32),
            shared_bias_jacobian=complete_shared,
            camera_bias_jacobian=complete_camera,
            conditional_covariance=covariance,
        )


def test_group_artifact_identity_binds_camera_row_assignment() -> None:
    layout = _full_layout()
    row_camera_indices = np.asarray([0, 1, 0, 1], dtype=np.int64)
    shared, camera = _designs(layout, row_camera_indices)
    covariance = np.repeat(np.eye(3, dtype=np.float64)[None], 4, axis=0)
    first = JointVisualBiasCalibrationGroupV1(
        group_id="object",
        layout=layout,
        row_camera_indices=row_camera_indices,
        residual=np.zeros((4, 3), dtype=np.float64),
        shared_bias_jacobian=shared,
        camera_bias_jacobian=camera,
        conditional_covariance=covariance,
    )
    reversed_indices = np.asarray([1, 0, 1, 0], dtype=np.int64)
    reversed_shared, reversed_camera = _designs(layout, reversed_indices)
    second = JointVisualBiasCalibrationGroupV1(
        group_id="object",
        layout=layout,
        row_camera_indices=reversed_indices,
        residual=np.zeros((4, 3), dtype=np.float64),
        shared_bias_jacobian=reversed_shared,
        camera_bias_jacobian=reversed_camera,
        conditional_covariance=covariance,
    )

    assert first.group_artifact_id != second.group_artifact_id


def test_joint_fit_selects_complete_shared_and_camera_covariance() -> None:
    calibration = _fit_full()

    assert calibration.selected_rank == 3
    assert calibration.selected_basis_names == _full_layout().basis_names
    assert calibration.selected_covariance.shape == (3, 3)
    assert abs(float(calibration.selected_covariance[1, 2])) > 0.01
    assert joint_visual_bias_layout_from_calibration(calibration) == _full_layout()
    summary = joint_visual_bias_selection_summary(calibration)
    assert summary["complete_camera_mode_boundary"] is True
    assert summary["complete_camera_basis_names"] == ["camera-tangent"]
    assert tuple(calibration.group_ids) == tuple(
        f"object-{group_index:02d}" for group_index in range(10)
    )


def test_joint_nuisance_instantiates_one_latent_across_all_cameras() -> None:
    calibration = _fit_full()
    group = _full_groups()[0]

    nuisance = build_joint_visual_bias_nuisance_from_calibration(
        calibration,
        observation_artifact_id="c" * 64,
        observation_identity_sha256="d" * 64,
        row_camera_indices=group.row_camera_indices,
        shared_bias_jacobian=group.shared_bias_jacobian,
        camera_bias_jacobian=group.camera_bias_jacobian,
        metadata={"case_id": "target-free-example"},
    )

    assert nuisance.bias_ids == ("joint-visual-cameras",)
    assert nuisance.basis_names == calibration.selected_basis_names
    assert nuisance.latent_dimension == 3
    np.testing.assert_array_equal(
        nuisance.joint_bias_covariance,
        calibration.selected_covariance,
    )
    np.testing.assert_array_equal(
        nuisance.row_bias_indices,
        np.zeros(group.row_camera_indices.size, dtype=np.int64),
    )
    joint_metadata = nuisance.metadata[JOINT_VISUAL_BIAS_METADATA_KEY]
    assert joint_metadata["layout_id"] == _full_layout().layout_id
    assert joint_metadata["camera_row_counts"] == {
        "camera-0": 16,
        "camera-1": 16,
    }


def test_joint_calibration_round_trips_through_existing_artifact_path(tmp_path) -> None:
    calibration = _fit_full()
    manifest = tmp_path / "joint-visual-bias.json"

    write_visual_bias_calibration(calibration, manifest)
    loaded = load_visual_bias_calibration(manifest)

    assert loaded.artifact_id == calibration.artifact_id
    assert joint_visual_bias_layout_from_calibration(loaded) == _full_layout()
    np.testing.assert_array_equal(
        loaded.selected_covariance,
        calibration.selected_covariance,
    )


def test_partial_camera_mode_is_fail_closed_by_default() -> None:
    arguments = {
        "provider_manifest_id": "a" * 64,
        "calibration_source_id": "b" * 64,
        "group_definition": "complete physical object",
        "residual_definition": "metric source minus provider point",
        "uses_truth": True,
        "covariance_shrinkage": 0.0,
        "minimum_nll_improvement": 0.0,
    }

    with pytest.raises(ValueError, match="cuts through a complete camera mode"):
        fit_joint_visual_bias_calibration(_partial_groups(), **arguments)

    calibration = fit_joint_visual_bias_calibration(
        _partial_groups(),
        allow_partial_camera_mode=True,
        **arguments,
    )
    summary = joint_visual_bias_selection_summary(calibration)
    assert calibration.selected_rank == 1
    assert summary["complete_camera_mode_boundary"] is False
    assert summary["partial_camera_basis_name"] == "camera-local"
    assert summary["partial_camera_ids"] == ["camera-0"]


def test_joint_fit_rejects_layout_mismatch_and_reserved_metadata() -> None:
    groups = list(_full_groups())
    alternate = JointVisualBiasLayoutV1(
        camera_ids=("camera-0", "camera-1"),
        shared_basis_names=("other-shared",),
        camera_basis_names=("camera-tangent",),
    )
    source = groups[-1]
    groups[-1] = JointVisualBiasCalibrationGroupV1(
        group_id=source.group_id,
        layout=alternate,
        row_camera_indices=source.row_camera_indices,
        residual=source.residual,
        shared_bias_jacobian=source.shared_bias_jacobian,
        camera_bias_jacobian=source.camera_bias_jacobian,
        conditional_covariance=source.conditional_covariance,
    )
    with pytest.raises(ValueError, match="same layout"):
        fit_joint_visual_bias_calibration(
            groups,
            provider_manifest_id="a" * 64,
            calibration_source_id="b" * 64,
            group_definition="complete physical object",
            residual_definition="metric source minus provider point",
            uses_truth=True,
        )

    with pytest.raises(ValueError, match="reserved keys"):
        fit_joint_visual_bias_calibration(
            _full_groups(),
            provider_manifest_id="a" * 64,
            calibration_source_id="b" * 64,
            group_definition="complete physical object",
            residual_definition="metric source minus provider point",
            uses_truth=True,
            metadata={"nested": {"uses_target_outcomes": False}},
        )
