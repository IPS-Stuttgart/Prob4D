from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.query_projection_binding import (
    BOUND_QUERY_COVARIANCE_PROJECTION_CLAIM_BOUNDARY,
    OBSERVATION_ROW_BINDING_SCHEMA,
    OBSERVATION_ROW_BINDING_VERSION,
    QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY,
    QUERY_JACOBIAN_BINDING_SCHEMA,
    QUERY_JACOBIAN_BINDING_VERSION,
    project_bound_joint_covariance_to_query,
    validate_query_jacobian_binding,
    write_bound_query_covariance_projection,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _content_id(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding_record(
    jacobian: np.ndarray,
    row_ids: tuple[str, ...],
) -> dict[str, object]:
    canonical = np.ascontiguousarray(jacobian, dtype=np.dtype("<f8"))
    unsigned: dict[str, object] = {
        "schema": QUERY_JACOBIAN_BINDING_SCHEMA,
        "schema_version": QUERY_JACOBIAN_BINDING_VERSION,
        "query_name": "endpoint-displacement",
        "component_order": ["x", "z"],
        "physical_unit": "m",
        "coordinate_frame": "registered-world",
        "source_observation_artifact_id": _sha256("observation"),
        "provider_manifest_id": _sha256("provider"),
        "causal_frame_stop": 18,
        "query_jacobian": {
            "dtype": "<f8",
            "shape": list(canonical.shape),
            "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
        },
        "observation_rows": {
            "schema": OBSERVATION_ROW_BINDING_SCHEMA,
            "schema_version": OBSERVATION_ROW_BINDING_VERSION,
            "count": len(row_ids),
            "sha256": _content_id({"row_ids": list(row_ids)}),
        },
        "target_outcomes_used": False,
        "future_frames_used": False,
        "claim_boundary": QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY,
        "metadata": {},
    }
    return {"artifact_id": _content_id(unsigned), **unsigned}


def _inputs() -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    jacobian = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, -1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    rows = ("factor-a/point-0", "factor-b/point-4")
    local = np.repeat(np.eye(3, dtype=np.float64)[None, ...], 2, axis=0)
    factor = np.zeros((2, 3, 1), dtype=np.float64)
    factor[:, 0, 0] = 0.5
    return jacobian, rows, local, factor


def test_bound_projection_validates_query_lineage_and_covariance_inputs() -> None:
    jacobian, rows, local, factor = _inputs()
    binding = _binding_record(jacobian, rows)

    result = project_bound_joint_covariance_to_query(
        binding,
        jacobian,
        rows,
        local,
        factor,
    )

    assert result.binding.artifact_id == binding["artifact_id"]
    assert result.projection.observation_count == 2
    assert result.projection.query_dimension == 2
    assert result.to_record()["claim_boundary"] == (
        BOUND_QUERY_COVARIANCE_PROJECTION_CLAIM_BOUNDARY
    )
    assert result.to_record()["local_covariance_m2"]["shape"] == [2, 3, 3]
    assert result.to_record()["low_rank_factor_m"]["shape"] == [2, 3, 1]


def test_bound_projection_rejects_changed_jacobian_or_row_order() -> None:
    jacobian, rows, local, factor = _inputs()
    binding = _binding_record(jacobian, rows)
    changed = jacobian.copy()
    changed[0, 0, 0] += 1e-12

    with pytest.raises(ValueError, match="bytes differ"):
        project_bound_joint_covariance_to_query(
            binding,
            changed,
            rows,
            local,
            factor,
        )
    with pytest.raises(ValueError, match="row_ids differ"):
        project_bound_joint_covariance_to_query(
            binding,
            jacobian,
            tuple(reversed(rows)),
            local,
            factor,
        )


def test_bound_projection_identity_binds_covariance_bytes() -> None:
    jacobian, rows, local, factor = _inputs()
    binding = _binding_record(jacobian, rows)
    original = project_bound_joint_covariance_to_query(
        binding,
        jacobian,
        rows,
        local,
        factor,
    )
    changed_local = local.copy()
    changed_local[0, 0, 0] = 2.0
    changed = project_bound_joint_covariance_to_query(
        binding,
        jacobian,
        rows,
        changed_local,
        factor,
    )

    assert changed.artifact_id != original.artifact_id
    assert (
        changed.local_covariance_descriptor["sha256"]
        != original.local_covariance_descriptor["sha256"]
    )


def test_independent_binding_validator_rejects_tampered_identity() -> None:
    jacobian, rows, _, _ = _inputs()
    record = _binding_record(jacobian, rows)
    record["artifact_id"] = _sha256("tampered")

    with pytest.raises(ValueError, match="artifact_id"):
        validate_query_jacobian_binding(record)


def test_bound_projection_writes_atomically_without_clobber(tmp_path: Path) -> None:
    jacobian, rows, local, factor = _inputs()
    result = project_bound_joint_covariance_to_query(
        _binding_record(jacobian, rows),
        jacobian,
        rows,
        local,
        factor,
    )
    path = tmp_path / "bound-query-projection.json"

    write_bound_query_covariance_projection(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == result.to_record()
    with pytest.raises(FileExistsError):
        write_bound_query_covariance_projection(result, path)


def test_scalar_query_and_rank_zero_are_supported() -> None:
    jacobian = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rows = ("row-0", "row-1")
    local = np.repeat(np.eye(3)[None, ...], 2, axis=0)
    factor = np.empty((2, 3, 0), dtype=np.float64)
    record = _binding_record(jacobian[None, ...], rows)
    record["component_order"] = ["distance"]
    unsigned = dict(record)
    unsigned.pop("artifact_id")
    record["artifact_id"] = _content_id(unsigned)

    result = project_bound_joint_covariance_to_query(
        record,
        jacobian,
        rows,
        local,
        factor,
    )

    assert result.projection.query_dimension == 1
    assert result.low_rank_factor_descriptor["shape"] == [2, 3, 0]


def test_preview_covariance_facade_exposes_bound_projection() -> None:
    from prob4d.api import covariance_v1

    assert covariance_v1.project_bound_joint_covariance_to_query is (
        project_bound_joint_covariance_to_query
    )
    assert callable(covariance_v1.validate_query_jacobian_binding)
