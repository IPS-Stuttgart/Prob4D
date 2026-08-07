"""Immutable complete-group inputs for joint visual-bias calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._immutable_array import immutable_integer_array
from ._immutable_json import plain_json
from ._joint_visual_bias_common import (
    FloatArray,
    IntArray,
    _array_descriptor,
    _float64_array,
    _metadata,
    _nonempty_string,
    _sha256_json,
)
from ._joint_visual_bias_layout import (
    JointVisualBiasLayoutV1,
    expand_joint_visual_bias_jacobian,
)
from .visual_bias_calibration import VisualBiasCalibrationGroup

@dataclass(frozen=True, slots=True)
class JointVisualBiasCalibrationGroupV1:
    """One complete source/calibration object with a multiview joint-bias design."""

    group_id: str
    layout: JointVisualBiasLayoutV1
    row_camera_indices: IntArray
    residual: FloatArray
    shared_bias_jacobian: FloatArray
    camera_bias_jacobian: FloatArray
    conditional_covariance: FloatArray
    gauge_design: FloatArray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    group_artifact_id: str | None = None

    def __post_init__(self) -> None:
        group_id = _nonempty_string(self.group_id, name="group_id")
        if not isinstance(self.layout, JointVisualBiasLayoutV1):
            raise TypeError("layout must be JointVisualBiasLayoutV1")
        residual = _float64_array(self.residual, name="residual")
        if residual.ndim != 2 or residual.shape[1] != 3 or residual.shape[0] < 1:
            raise ValueError("residual must have shape (N, 3)")
        row_count = residual.shape[0]
        indices = immutable_integer_array(
            self.row_camera_indices,
            name="row_camera_indices",
        )
        if indices.shape != (row_count,):
            raise ValueError("row_camera_indices must have one entry per residual row")
        expanded = expand_joint_visual_bias_jacobian(
            self.layout,
            indices,
            self.shared_bias_jacobian,
            self.camera_bias_jacobian,
            require_all_cameras=True,
        )
        shared = _float64_array(
            self.shared_bias_jacobian,
            name="shared_bias_jacobian",
            shape=(row_count, 3, len(self.layout.shared_basis_names)),
        )
        camera = _float64_array(
            self.camera_bias_jacobian,
            name="camera_bias_jacobian",
            shape=(row_count, 3, len(self.layout.camera_basis_names)),
        )
        covariance = _float64_array(
            self.conditional_covariance,
            name="conditional_covariance",
            shape=(row_count, 3, 3),
        )
        gauge = None
        if self.gauge_design is not None:
            gauge = _float64_array(self.gauge_design, name="gauge_design")
            if gauge.ndim != 3 or gauge.shape[:2] != (row_count, 3) or gauge.shape[2] < 1:
                raise ValueError("gauge_design must have shape (N, 3, K) with K positive")
        metadata = _metadata(self.metadata, name="joint visual-bias group metadata")
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "row_camera_indices", indices)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "shared_bias_jacobian", shared)
        object.__setattr__(self, "camera_bias_jacobian", camera)
        object.__setattr__(self, "conditional_covariance", covariance)
        object.__setattr__(self, "gauge_design", gauge)
        object.__setattr__(self, "metadata", metadata)
        # Reuse the established SPD, row, and gauge-design contract.
        VisualBiasCalibrationGroup(
            group_id=group_id,
            residual=residual,
            bias_jacobian=expanded,
            conditional_covariance=covariance,
            gauge_design=gauge,
            metadata=self._v1_metadata(include_artifact_id=False),
        )
        expected = _sha256_json(self.identity_record())
        if self.group_artifact_id is not None and self.group_artifact_id != expected:
            raise ValueError("joint visual-bias group artifact ID mismatch")
        object.__setattr__(self, "group_artifact_id", expected)

    def camera_row_counts(self) -> dict[str, int]:
        return {
            camera_id: int(np.count_nonzero(self.row_camera_indices == camera_index))
            for camera_index, camera_id in enumerate(self.layout.camera_ids)
        }

    def expanded_bias_jacobian(self) -> FloatArray:
        return expand_joint_visual_bias_jacobian(
            self.layout,
            self.row_camera_indices,
            self.shared_bias_jacobian,
            self.camera_bias_jacobian,
            require_all_cameras=True,
        )

    def _v1_metadata(self, *, include_artifact_id: bool) -> dict[str, Any]:
        result = plain_json(self.metadata)
        result["joint_visual_bias_layout_id"] = self.layout.layout_id
        result["camera_row_counts"] = self.camera_row_counts()
        if include_artifact_id:
            result["joint_visual_bias_group_artifact_id"] = self.group_artifact_id
        return result

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": "prob4d.joint-visual-bias-calibration-group",
            "schema_version": 1,
            "group_id": self.group_id,
            "layout_id": self.layout.layout_id,
            "arrays": {
                "row_camera_indices": _array_descriptor(self.row_camera_indices),
                "residual": _array_descriptor(self.residual),
                "shared_bias_jacobian": _array_descriptor(self.shared_bias_jacobian),
                "camera_bias_jacobian": _array_descriptor(self.camera_bias_jacobian),
                "conditional_covariance": _array_descriptor(self.conditional_covariance),
                "gauge_design": (
                    None if self.gauge_design is None else _array_descriptor(self.gauge_design)
                ),
            },
            "metadata": plain_json(self.metadata),
        }

    def to_visual_bias_calibration_group(self) -> VisualBiasCalibrationGroup:
        return VisualBiasCalibrationGroup(
            group_id=self.group_id,
            residual=self.residual,
            bias_jacobian=self.expanded_bias_jacobian(),
            conditional_covariance=self.conditional_covariance,
            gauge_design=self.gauge_design,
            metadata=self._v1_metadata(include_artifact_id=True),
        )



__all__ = ["JointVisualBiasCalibrationGroupV1"]
