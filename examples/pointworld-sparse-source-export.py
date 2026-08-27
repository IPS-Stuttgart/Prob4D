"""Minimal integration sketch for a frozen external PointWorld runtime.

This example intentionally does not import PointWorld or Flat'n'Fold. Execute it
inside the frozen PointWorld environment after replacing the synthetic tensors
with the exact model/batch arrays and all identity placeholders with SHA-256
values derived from executed bytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prob4d.pointworld_sparse_adapter import (
    write_pointworld_sparse_source_export,
)


def export_exact_pointworld_window(
    *,
    output: Path,
    frame_indices: np.ndarray,
    scene_coord0: np.ndarray,
    predicted_displacement_from_context: np.ndarray,
    scene_exists: np.ndarray,
    prediction_valid_mask: np.ndarray,
    provider_log_variance: np.ndarray,
    provider_revision: str,
    checkpoint_sha256: str,
    loader_id: str,
    camera_geometry_id: str,
    action_sequence_id: str,
) -> None:
    """Write one strict source artifact without interpolation or truth access."""

    write_pointworld_sparse_source_export(
        output,
        window_id="REPLACE_WITH_WINDOW_ID",
        sequence_id="REPLACE_WITH_FLATNFOLD_SEQUENCE_ID",
        provider_revision=provider_revision,
        checkpoint_sha256=checkpoint_sha256,
        loader_id=loader_id,
        camera_geometry_id=camera_geometry_id,
        action_sequence_id=action_sequence_id,
        coordinate_semantics="metric-baxter-base",
        frame_indices=frame_indices,
        scene_coord0=scene_coord0,
        predicted_displacement_from_context=(
            predicted_displacement_from_context
        ),
        scene_exists=scene_exists,
        prediction_valid_mask=prediction_valid_mask,
        provider_log_variance=provider_log_variance,
        dense_storage_dtype=(
            "float32" if scene_coord0.dtype == np.float32 else "float64"
        ),
        metadata={
            "uses_target_truth": False,
            "uses_downstream_physical_innovation": False,
            "rasterization": "none",
            "identity_scope": "within-window-context-point-order",
        },
    )
