"""Fitting and sidecar construction for joint cross-camera visual bias."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._immutable_array import immutable_integer_array
from ._immutable_json import plain_json
from ._joint_visual_bias_common import (
    _CALIBRATION_METADATA_FIELDS,
    JOINT_VISUAL_BIAS_CLAIM_BOUNDARY,
    JOINT_VISUAL_BIAS_COVARIANCE_SEMANTICS,
    JOINT_VISUAL_BIAS_METADATA_KEY,
    FloatArray,
    _metadata,
    _nonempty_string,
    _sha256,
)
from ._joint_visual_bias_group import JointVisualBiasCalibrationGroupV1
from ._joint_visual_bias_layout import (
    JointVisualBiasLayoutV1,
    expand_joint_visual_bias_jacobian,
)
from .visual_bias import VisualBiasNuisanceV1
from .visual_bias_calibration import (
    VisualBiasCalibrationV1,
    build_visual_bias_nuisance_from_calibration,
    fit_visual_bias_calibration,
)


def joint_visual_bias_layout_from_calibration(
    calibration: VisualBiasCalibrationV1,
) -> JointVisualBiasLayoutV1:
    """Recover and validate the joint layout bound into one calibration artifact."""

    record = calibration.metadata.get(JOINT_VISUAL_BIAS_METADATA_KEY)
    if not isinstance(record, Mapping):
        raise ValueError("calibration has no joint visual-bias layout metadata")
    missing = sorted(_CALIBRATION_METADATA_FIELDS - set(record))
    extra = sorted(set(record) - _CALIBRATION_METADATA_FIELDS)
    if missing or extra:
        raise ValueError(
            f"joint visual-bias metadata fields changed: missing={missing}, extra={extra}"
        )
    if record["schema"] != "prob4d.joint-visual-bias-calibration":
        raise ValueError("unsupported joint visual-bias calibration schema")
    if record["schema_version"] != 1:
        raise ValueError("unsupported joint visual-bias calibration version")
    if record["uses_target_outcomes"] is not False:
        raise ValueError("joint visual-bias calibration may not use target outcomes")
    if record["uses_downstream_physical_innovation"] is not False:
        raise ValueError("joint visual-bias calibration may not use physical innovations")
    if record["covariance_semantics"] != JOINT_VISUAL_BIAS_COVARIANCE_SEMANTICS:
        raise ValueError("joint visual-bias covariance semantics changed")
    if record["claim_boundary"] != JOINT_VISUAL_BIAS_CLAIM_BOUNDARY:
        raise ValueError("joint visual-bias calibration claim boundary changed")
    layout = JointVisualBiasLayoutV1.from_mapping(record["layout"])
    if record["layout_id"] != layout.layout_id:
        raise ValueError("joint visual-bias calibration layout ID mismatch")
    if calibration.basis_names != layout.basis_names:
        raise ValueError("calibration basis names differ from the joint layout")
    group_ids = record["group_artifact_ids"]
    if not isinstance(group_ids, Mapping) or not group_ids:
        raise ValueError("joint visual-bias calibration has no group artifact identities")
    normalized_group_ids: list[str] = []
    for group_id, artifact_id in group_ids.items():
        normalized_group_ids.append(
            _nonempty_string(group_id, name="joint visual-bias group ID")
        )
        _sha256(
            artifact_id,
            name=f"joint visual-bias group artifact ID for {group_id!r}",
        )
    canonical_group_ids = tuple(sorted(calibration.group_ids))
    if tuple(sorted(normalized_group_ids)) != canonical_group_ids:
        raise ValueError("joint visual-bias group artifact identities differ from calibration")
    if type(record["allow_partial_camera_mode"]) is not bool:
        raise ValueError("allow_partial_camera_mode must be a JSON boolean")
    return layout


def joint_visual_bias_selection_summary(
    calibration: VisualBiasCalibrationV1,
) -> dict[str, object]:
    """Describe how the nested selected rank intersects shared and camera blocks."""

    layout = joint_visual_bias_layout_from_calibration(calibration)
    shared_count = len(layout.shared_basis_names)
    selected_shared = min(calibration.selected_rank, shared_count)
    remaining = max(0, calibration.selected_rank - shared_count)
    complete_modes, partial_count = divmod(remaining, len(layout.camera_ids))
    partial_mode = None
    partial_cameras: tuple[str, ...] = ()
    if partial_count:
        partial_mode = layout.camera_basis_names[complete_modes]
        partial_cameras = layout.camera_ids[:partial_count]
    return {
        "layout_id": layout.layout_id,
        "selected_rank": calibration.selected_rank,
        "selected_basis_names": list(calibration.selected_basis_names),
        "selected_shared_basis_names": list(
            layout.shared_basis_names[:selected_shared]
        ),
        "complete_camera_basis_names": list(
            layout.camera_basis_names[:complete_modes]
        ),
        "partial_camera_basis_name": partial_mode,
        "partial_camera_ids": list(partial_cameras),
        "complete_camera_mode_boundary": partial_mode is None,
    }


def fit_joint_visual_bias_calibration(
    groups: Sequence[JointVisualBiasCalibrationGroupV1],
    *,
    provider_manifest_id: str,
    calibration_source_id: str,
    group_definition: str,
    residual_definition: str,
    uses_truth: bool,
    covariance_shrinkage: float = 0.25,
    minimum_nll_improvement: float = 0.0,
    gauge_projection_tolerance: float = 1e-8,
    allow_partial_camera_mode: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> VisualBiasCalibrationV1:
    """Fit one joint latent across shared and camera-specific source modes."""

    if type(allow_partial_camera_mode) is not bool:
        raise ValueError("allow_partial_camera_mode must be a boolean")
    values = tuple(groups)
    if len(values) < 3 or not all(
        isinstance(group, JointVisualBiasCalibrationGroupV1) for group in values
    ):
        raise ValueError("joint visual-bias calibration requires at least three groups")
    ordered = tuple(sorted(values, key=lambda group: group.group_id))
    if len({group.group_id for group in ordered}) != len(ordered):
        raise ValueError("joint visual-bias group IDs must be unique")
    layout = ordered[0].layout
    if any(group.layout.layout_id != layout.layout_id for group in ordered[1:]):
        raise ValueError("all joint visual-bias groups must use the same layout")
    supplied_metadata = {} if metadata is None else dict(metadata)
    frozen_metadata = _metadata(
        supplied_metadata,
        name="joint visual-bias calibration metadata",
    )
    joint_metadata = {
        "schema": "prob4d.joint-visual-bias-calibration",
        "schema_version": 1,
        "layout": layout.to_dict(),
        "layout_id": layout.layout_id,
        "group_artifact_ids": {
            group.group_id: group.group_artifact_id for group in ordered
        },
        "allow_partial_camera_mode": allow_partial_camera_mode,
        "uses_target_outcomes": False,
        "uses_downstream_physical_innovation": False,
        "covariance_semantics": JOINT_VISUAL_BIAS_COVARIANCE_SEMANTICS,
        "claim_boundary": JOINT_VISUAL_BIAS_CLAIM_BOUNDARY,
    }
    calibration = fit_visual_bias_calibration(
        tuple(group.to_visual_bias_calibration_group() for group in ordered),
        basis_names=layout.basis_names,
        provider_manifest_id=provider_manifest_id,
        calibration_source_id=calibration_source_id,
        group_definition=group_definition,
        residual_definition=residual_definition,
        uses_truth=uses_truth,
        covariance_shrinkage=covariance_shrinkage,
        minimum_nll_improvement=minimum_nll_improvement,
        gauge_projection_tolerance=gauge_projection_tolerance,
        metadata={
            **plain_json(frozen_metadata),
            JOINT_VISUAL_BIAS_METADATA_KEY: joint_metadata,
        },
    )
    summary = joint_visual_bias_selection_summary(calibration)
    if not allow_partial_camera_mode and not summary["complete_camera_mode_boundary"]:
        raise ValueError(
            "selected nested rank cuts through a complete camera mode; "
            "reorder or revise the source-frozen basis, or explicitly permit the "
            "asymmetric exploratory selection"
        )
    return calibration


def build_joint_visual_bias_nuisance_from_calibration(
    calibration: VisualBiasCalibrationV1,
    *,
    observation_artifact_id: str,
    observation_identity_sha256: str,
    row_camera_indices: object,
    shared_bias_jacobian: object,
    camera_bias_jacobian: object,
    bias_id: str = "joint-visual-cameras",
    conditional_covariance: FloatArray | None = None,
    gauge_design: FloatArray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> VisualBiasNuisanceV1:
    """Instantiate one joint cross-camera bias latent for a new observation."""

    layout = joint_visual_bias_layout_from_calibration(calibration)
    indices = immutable_integer_array(row_camera_indices, name="row_camera_indices")
    expanded = expand_joint_visual_bias_jacobian(
        layout,
        indices,
        shared_bias_jacobian,
        camera_bias_jacobian,
        require_all_cameras=False,
    )
    supplied_metadata = {} if metadata is None else dict(metadata)
    frozen_metadata = _metadata(
        supplied_metadata,
        name="joint visual-bias nuisance metadata",
    )
    counts = {
        camera_id: int(np.count_nonzero(indices == camera_index))
        for camera_index, camera_id in enumerate(layout.camera_ids)
    }
    return build_visual_bias_nuisance_from_calibration(
        calibration,
        observation_artifact_id=observation_artifact_id,
        observation_identity_sha256=observation_identity_sha256,
        bias_id=bias_id,
        bias_jacobian=expanded,
        conditional_covariance=conditional_covariance,
        gauge_design=gauge_design,
        metadata={
            **plain_json(frozen_metadata),
            JOINT_VISUAL_BIAS_METADATA_KEY: {
                "layout_id": layout.layout_id,
                "camera_ids": list(layout.camera_ids),
                "camera_row_counts": counts,
                "selection": joint_visual_bias_selection_summary(calibration),
            },
        },
    )


__all__ = [
    "build_joint_visual_bias_nuisance_from_calibration",
    "fit_joint_visual_bias_calibration",
    "joint_visual_bias_layout_from_calibration",
    "joint_visual_bias_selection_summary",
]
