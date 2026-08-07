"""Canonical joint visual-bias latent layout and row-design expansion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from ._immutable_array import immutable_array, immutable_integer_array
from ._joint_visual_bias_common import (
    FloatArray,
    JOINT_VISUAL_BIAS_BASIS_ORDER,
    JOINT_VISUAL_BIAS_CLAIM_BOUNDARY,
    JOINT_VISUAL_BIAS_LAYOUT_SCHEMA,
    JOINT_VISUAL_BIAS_LAYOUT_VERSION,
    _LAYOUT_FIELDS,
    _float64_array,
    _json_nonempty_string_tuple,
    _json_string_tuple,
    _sha256,
    _sha256_json,
    _string_tuple,
)


@dataclass(frozen=True, slots=True)
class JointVisualBiasLayoutV1:
    """Canonical latent-column layout shared by calibration and observations."""

    camera_ids: tuple[str, ...]
    shared_basis_names: tuple[str, ...]
    camera_basis_names: tuple[str, ...]
    layout_id: str | None = None

    def __post_init__(self) -> None:
        camera_ids = _string_tuple(
            self.camera_ids,
            name="camera_ids",
            minimum=2,
            require_sorted=True,
        )
        shared = _string_tuple(
            self.shared_basis_names,
            name="shared_basis_names",
        )
        camera = _string_tuple(
            self.camera_basis_names,
            name="camera_basis_names",
        )
        if not shared and not camera:
            raise ValueError("joint visual-bias layout requires at least one basis mode")
        if set(shared) & set(camera):
            raise ValueError("shared and camera-specific basis names must be disjoint")
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "shared_basis_names", shared)
        object.__setattr__(self, "camera_basis_names", camera)
        expected = _sha256_json(self.descriptor())
        if self.layout_id is not None and self.layout_id != expected:
            raise ValueError("joint visual-bias layout ID mismatch")
        object.__setattr__(self, "layout_id", expected)

    @property
    def basis_names(self) -> tuple[str, ...]:
        shared = tuple(f"shared::{name}" for name in self.shared_basis_names)
        camera = tuple(
            f"camera::{basis_name}::{camera_id}"
            for basis_name in self.camera_basis_names
            for camera_id in self.camera_ids
        )
        return shared + camera

    @property
    def basis_dimension(self) -> int:
        return len(self.basis_names)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": JOINT_VISUAL_BIAS_LAYOUT_SCHEMA,
            "schema_version": JOINT_VISUAL_BIAS_LAYOUT_VERSION,
            "camera_ids": list(self.camera_ids),
            "shared_basis_names": list(self.shared_basis_names),
            "camera_basis_names": list(self.camera_basis_names),
            "expanded_basis_names": list(self.basis_names),
            "basis_order_semantics": JOINT_VISUAL_BIAS_BASIS_ORDER,
            "claim_boundary": JOINT_VISUAL_BIAS_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "layout_id": self.layout_id}

    @classmethod
    def from_mapping(cls, value: object) -> JointVisualBiasLayoutV1:
        if not isinstance(value, Mapping):
            raise ValueError("joint visual-bias layout must be a JSON object")
        missing = sorted(_LAYOUT_FIELDS - set(value))
        extra = sorted(set(value) - _LAYOUT_FIELDS)
        if missing or extra:
            raise ValueError(
                f"joint visual-bias layout fields changed: missing={missing}, extra={extra}"
            )
        if value["schema"] != JOINT_VISUAL_BIAS_LAYOUT_SCHEMA:
            raise ValueError("unsupported joint visual-bias layout schema")
        if value["schema_version"] != JOINT_VISUAL_BIAS_LAYOUT_VERSION:
            raise ValueError("unsupported joint visual-bias layout version")
        if value["basis_order_semantics"] != JOINT_VISUAL_BIAS_BASIS_ORDER:
            raise ValueError("joint visual-bias basis ordering changed")
        if value["claim_boundary"] != JOINT_VISUAL_BIAS_CLAIM_BOUNDARY:
            raise ValueError("joint visual-bias claim boundary changed")
        layout = cls(
            camera_ids=_json_string_tuple(
                value["camera_ids"],
                name="camera_ids",
                minimum=2,
                require_sorted=True,
            ),
            shared_basis_names=_json_string_tuple(
                value["shared_basis_names"],
                name="shared_basis_names",
            ),
            camera_basis_names=_json_string_tuple(
                value["camera_basis_names"],
                name="camera_basis_names",
            ),
            layout_id=_sha256(value["layout_id"], name="layout_id"),
        )
        expanded = _json_nonempty_string_tuple(
            value["expanded_basis_names"],
            name="expanded_basis_names",
        )
        if layout.basis_names != expanded:
            raise ValueError("joint visual-bias expanded basis names changed")
        return layout


def expand_joint_visual_bias_jacobian(
    layout: JointVisualBiasLayoutV1,
    row_camera_indices: object,
    shared_bias_jacobian: object,
    camera_bias_jacobian: object,
    *,
    require_all_cameras: bool,
) -> FloatArray:
    """Expand shared and camera-local row bases into one complete joint design."""

    if not isinstance(layout, JointVisualBiasLayoutV1):
        raise TypeError("layout must be JointVisualBiasLayoutV1")
    indices = immutable_integer_array(row_camera_indices, name="row_camera_indices")
    if indices.ndim != 1 or indices.size < 1:
        raise ValueError("row_camera_indices must be a non-empty vector")
    camera_count = len(layout.camera_ids)
    if np.any(indices < 0) or np.any(indices >= camera_count):
        raise ValueError("row_camera_indices refer to an unknown camera")
    if require_all_cameras and set(int(item) for item in indices) != set(range(camera_count)):
        raise ValueError("every calibration group must contain rows from every camera")
    row_count = int(indices.size)
    shared = _float64_array(
        shared_bias_jacobian,
        name="shared_bias_jacobian",
        shape=(row_count, 3, len(layout.shared_basis_names)),
    )
    camera = _float64_array(
        camera_bias_jacobian,
        name="camera_bias_jacobian",
        shape=(row_count, 3, len(layout.camera_basis_names)),
    )
    expanded = np.zeros((row_count, 3, layout.basis_dimension), dtype=np.float64)
    shared_count = len(layout.shared_basis_names)
    if shared_count:
        expanded[:, :, :shared_count] = shared
    for mode_index in range(len(layout.camera_basis_names)):
        for camera_index in range(camera_count):
            column = shared_count + mode_index * camera_count + camera_index
            mask = indices == camera_index
            expanded[mask, :, column] = camera[mask, :, mode_index]
    return cast(FloatArray, immutable_array(expanded, dtype=np.float64))

__all__ = ["JointVisualBiasLayoutV1", "expand_joint_visual_bias_jacobian"]
