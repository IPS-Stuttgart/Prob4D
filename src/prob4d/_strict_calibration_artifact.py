"""Strict scalar and field validation for calibration artifact schema v1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._strict_json import (
    require_exact_fields,
    require_exact_integer,
    require_finite_json_mapping,
    require_json_number,
    require_mapping,
    require_nonempty_string,
    require_revision,
    require_sha256,
    require_string_sequence,
)

_COMMON_PROVENANCE_FIELDS = frozenset(
    {
        "calibration_case_ids",
        "source_repository",
        "source_revision",
        "motioncrafter_revision",
        "model_identifier",
        "covariance_method",
        "image_resolution",
        "window_size",
        "window_overlap",
        "covariance_cluster_size",
        "input_artifact_sha256",
        "metadata",
    }
)
_GAUGE_CALIBRATION_FIELDS = frozenset(
    {"scale", "rotation", "translation", "count", "trim_quantile"}
)
_POINT_CALIBRATION_FIELDS = frozenset(
    {
        "parallel_floor",
        "parallel_depth_coefficient",
        "lateral_floor",
        "lateral_depth_coefficient",
        "disagreement_gain",
        "parallel_scale",
        "lateral_scale",
        "count",
        "trim_quantile",
        "parallel_scale_update",
        "lateral_scale_update",
        "parallel_normalized_mse",
        "lateral_normalized_mse",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {"artifact_id", "schema", "version", "calibration", "provenance"}
)


def _require_optional_integer(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> None:
    if value is not None:
        require_exact_integer(value, name=name, minimum=minimum)


def _validate_resolution(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("image_resolution must contain two positive integers")
    resolution = tuple(value)
    if len(resolution) != 2:
        raise ValueError("image_resolution must contain two positive integers")
    for item in resolution:
        require_exact_integer(item, name="image_resolution item", minimum=1)


def _validate_common_provenance_values(value: Mapping[str, Any]) -> None:
    require_exact_fields(
        value,
        _COMMON_PROVENANCE_FIELDS,
        name="calibration provenance",
    )
    case_ids = require_string_sequence(
        value["calibration_case_ids"],
        name="calibration_case_ids",
    )
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("calibration_case_ids must be unique")
    require_nonempty_string(value["source_repository"], name="source_repository")
    require_revision(value["source_revision"], name="source_revision")
    require_revision(value["motioncrafter_revision"], name="motioncrafter_revision")
    require_nonempty_string(value["model_identifier"], name="model_identifier")
    require_nonempty_string(value["covariance_method"], name="covariance_method")
    _validate_resolution(value["image_resolution"])
    _require_optional_integer(value["window_size"], name="window_size", minimum=1)
    _require_optional_integer(
        value["window_overlap"],
        name="window_overlap",
        minimum=0,
    )
    _require_optional_integer(
        value["covariance_cluster_size"],
        name="covariance_cluster_size",
        minimum=1,
    )
    digests = require_string_sequence(
        value["input_artifact_sha256"],
        name="input_artifact_sha256",
    )
    for digest in digests:
        require_sha256(digest, name="input_artifact_sha256")
    if len(set(digests)) != len(digests):
        raise ValueError("input_artifact_sha256 values must be unique")
    require_finite_json_mapping(value["metadata"], name="metadata")


def _common_values_from_artifact(artifact: Any) -> dict[str, Any]:
    return {
        field: getattr(artifact, field)
        for field in _COMMON_PROVENANCE_FIELDS
    }


def _validate_number_fields(
    value: Mapping[str, Any],
    *,
    fields: frozenset[str],
) -> None:
    for field in fields - {"count"}:
        require_json_number(value[field], name=field)
    require_exact_integer(value["count"], name="count", minimum=1)


def validate_gauge_calibration_values(artifact: Any) -> None:
    calibration = {
        field: getattr(artifact, field)
        for field in _GAUGE_CALIBRATION_FIELDS
    }
    _validate_number_fields(calibration, fields=_GAUGE_CALIBRATION_FIELDS)
    _validate_common_provenance_values(_common_values_from_artifact(artifact))


def validate_point_calibration_values(artifact: Any) -> None:
    calibration = {
        field: getattr(artifact, field)
        for field in _POINT_CALIBRATION_FIELDS
    }
    _validate_number_fields(calibration, fields=_POINT_CALIBRATION_FIELDS)
    _validate_common_provenance_values(_common_values_from_artifact(artifact))


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    calibration_fields: frozenset[str],
) -> None:
    require_exact_fields(payload, _ARTIFACT_FIELDS, name="calibration artifact")
    require_sha256(payload["artifact_id"], name="artifact_id")
    require_nonempty_string(payload["schema"], name="schema")
    require_exact_integer(payload["version"], name="version")
    calibration = require_mapping(payload["calibration"], name="calibration")
    require_exact_fields(calibration, calibration_fields, name="calibration")
    _validate_number_fields(calibration, fields=calibration_fields)
    provenance = require_mapping(payload["provenance"], name="provenance")
    _validate_common_provenance_values(provenance)


def validate_gauge_calibration_payload(payload: Mapping[str, Any]) -> None:
    _validate_payload(
        payload,
        calibration_fields=_GAUGE_CALIBRATION_FIELDS,
    )


def validate_point_calibration_payload(payload: Mapping[str, Any]) -> None:
    _validate_payload(
        payload,
        calibration_fields=_POINT_CALIBRATION_FIELDS,
    )


__all__ = [
    "validate_gauge_calibration_payload",
    "validate_gauge_calibration_values",
    "validate_point_calibration_payload",
    "validate_point_calibration_values",
]
