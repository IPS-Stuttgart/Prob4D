from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prob4d.calibration import (
    GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1,
    load_gauge_covariance_calibration,
    load_point_uncertainty_calibration,
    save_gauge_covariance_calibration,
    save_point_uncertainty_calibration,
)
from prob4d.observation_export import (
    MetricGaugeAnchor,
    load_metric_gauge_anchor,
    save_metric_gauge_anchor,
)
from prob4d.sim3 import Sim3


def _anchor(
    *,
    source_artifact_sha256: str = "a" * 64,
    metadata: dict[str, Any] | None = None,
) -> MetricGaugeAnchor:
    return MetricGaugeAnchor(
        window_id="window_0000",
        global_from_local=Sim3.identity(),
        covariance=np.eye(7) * 1e-6,
        coordinate_frame="phystwin-world",
        source_kind="prefix_registration",
        source_artifact_sha256=source_artifact_sha256,
        metadata=(
            {
                "calibration_artifact_sha256": "b" * 64,
                "nested": {"values": [1, 2]},
            }
            if metadata is None
            else metadata
        ),
    )


def _common_provenance() -> dict[str, Any]:
    return {
        "calibration_case_ids": ("case-a",),
        "source_repository": "IPS-Stuttgart/Prob4D",
        "source_revision": "1" * 40,
        "motioncrafter_revision": "2" * 40,
        "model_identifier": "unit-test-model",
        "covariance_method": "unit-test-covariance",
        "image_resolution": (2, 2),
        "window_size": 4,
        "window_overlap": 2,
        "covariance_cluster_size": 1,
        "input_artifact_sha256": ("3" * 64,),
        "metadata": {"nested": {"values": [1, 2]}},
    }


def _gauge_calibration() -> GaugeCovarianceCalibrationV1:
    return GaugeCovarianceCalibrationV1(
        scale=1.0,
        rotation=1.0,
        translation=1.0,
        count=1,
        trim_quantile=0.99,
        **_common_provenance(),
    )


def _point_calibration() -> PointUncertaintyCalibrationV1:
    return PointUncertaintyCalibrationV1(
        parallel_floor=0.0,
        parallel_depth_coefficient=0.0,
        lateral_floor=0.0,
        lateral_depth_coefficient=0.0,
        disagreement_gain=0.0,
        parallel_scale=1.0,
        lateral_scale=1.0,
        count=1,
        trim_quantile=0.99,
        parallel_scale_update=1.0,
        lateral_scale_update=1.0,
        parallel_normalized_mse=1.0,
        lateral_normalized_mse=1.0,
        **_common_provenance(),
    )


def _anchor_payload(anchor: MetricGaugeAnchor) -> dict[str, Any]:
    return {"artifact_id": anchor.artifact_id, **anchor.descriptor()}


def test_metric_anchor_metadata_is_recursively_immutable() -> None:
    metadata = {
        "calibration_artifact_sha256": "b" * 64,
        "nested": {"values": [1, 2]},
    }
    anchor = _anchor(metadata=metadata)
    artifact_id = anchor.artifact_id

    metadata["calibration_artifact_sha256"] = "c" * 64
    metadata["nested"]["values"].append(3)

    assert anchor.artifact_id == artifact_id
    assert anchor.descriptor()["metadata"]["nested"]["values"] == [1, 2]
    json.dumps(anchor.descriptor(), sort_keys=True, allow_nan=False)

    with pytest.raises(TypeError, match="immutable"):
        anchor.metadata["nested"]["values"].append(3)
    with pytest.raises(TypeError, match="immutable"):
        anchor.metadata["new"] = "value"


def test_metric_anchor_loader_rejects_duplicate_and_extra_fields(
    tmp_path: Path,
) -> None:
    anchor = _anchor()
    payload = _anchor_payload(anchor)

    duplicate = tmp_path / "duplicate.json"
    serialized = json.dumps(payload, sort_keys=True)
    duplicate.write_text(
        serialized.replace("{", '{"schema_name":"duplicate",', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_metric_gauge_anchor(duplicate)

    extra = tmp_path / "extra.json"
    payload["unexpected"] = True
    extra.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields changed"):
        load_metric_gauge_anchor(extra)


def test_metric_anchor_loader_rejects_scalar_coercion(tmp_path: Path) -> None:
    anchor = _anchor()
    payload = _anchor_payload(anchor)
    payload["schema_version"] = True
    path = tmp_path / "coercive.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be an integer"):
        load_metric_gauge_anchor(path)


def test_metric_anchor_loader_preserves_legacy_missing_artifact_id(
    tmp_path: Path,
) -> None:
    anchor = _anchor()
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(anchor.descriptor()), encoding="utf-8")

    restored = load_metric_gauge_anchor(path)
    assert restored.artifact_id == anchor.artifact_id


def test_metric_anchor_publication_is_no_clobber_by_default(
    tmp_path: Path,
) -> None:
    path = tmp_path / "anchor.json"
    first = _anchor()
    replacement = _anchor(source_artifact_sha256="c" * 64)

    save_metric_gauge_anchor(path, first)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_metric_gauge_anchor(path, replacement)
    assert path.read_bytes() == original

    save_metric_gauge_anchor(path, replacement, overwrite=True)
    assert load_metric_gauge_anchor(path).artifact_id == replacement.artifact_id


@pytest.mark.parametrize(
    ("artifact", "save", "load", "filename"),
    [
        pytest.param(
            _gauge_calibration(),
            save_gauge_covariance_calibration,
            load_gauge_covariance_calibration,
            "gauge.json",
            id="gauge",
        ),
        pytest.param(
            _point_calibration(),
            save_point_uncertainty_calibration,
            load_point_uncertainty_calibration,
            "point.json",
            id="point",
        ),
    ],
)
def test_calibration_publication_is_atomic_no_clobber_and_verified(
    tmp_path: Path,
    artifact: GaugeCovarianceCalibrationV1 | PointUncertaintyCalibrationV1,
    save: Any,
    load: Any,
    filename: str,
) -> None:
    path = tmp_path / filename
    save(artifact, path)
    original = path.read_bytes()
    assert load(path).artifact_id == artifact.artifact_id

    with pytest.raises(FileExistsError):
        save(artifact, path)
    assert path.read_bytes() == original

    save(artifact, path, overwrite=True)
    assert load(path).to_dict() == artifact.to_dict()
