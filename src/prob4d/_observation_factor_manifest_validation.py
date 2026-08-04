"""Strict schema-v4 observation-factor manifest type validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._observation_factor_bundle import (
    OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION,
)
from ._strict_json import (
    require_exact_fields,
    require_exact_integer,
    require_finite_json_mapping,
    require_json_number,
    require_mapping,
    require_nonempty_string,
    require_sha256,
)

_ROOT_FIELDS_V4 = frozenset(
    {
        "schema",
        "schema_version",
        "gauge_parameterization",
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
        "causal_frame_stop",
        "causal_frame_stop_convention",
        "metadata",
        "payload",
        "gauges",
        "factors",
        "gauge_covariance",
    }
)
_PAYLOAD_FIELDS = frozenset({"path", "sha256", "allow_pickle"})
_GAUGE_FIELDS = frozenset({"gauge_id", "mean_key", "covariance_key"})
_FACTOR_FIELDS = frozenset(
    {
        "factor_id",
        "frame_index",
        "view_id",
        "window_id",
        "gauge_id",
        "correlation_group_id",
        "causal_frame_stop",
        "prior_nominal_probability",
        "composite_weight",
        "arrays",
        "ray_directions_local_key",
    }
)
_FACTOR_ARRAY_FIELDS = frozenset(
    {
        "point_ids",
        "points_local_m",
        "valid_mask",
        "local_covariance_m2",
        "association_probability",
        "prior_reliability",
    }
)
_GAUGE_COVARIANCE_FIELDS = frozenset(
    {
        "semantics",
        "joint_covariance_key",
        "ordered_gauge_ids",
        "cross_window_covariance_preserved",
        "diagonal_blocks_match_gauge_marginals",
    }
)


def _require_json_list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _validate_gauges(value: Any) -> None:
    gauges = _require_json_list(value, name="gauges")
    if not gauges:
        raise ValueError("gauges must not be empty")
    gauge_ids: list[str] = []
    for index, item in enumerate(gauges):
        gauge = require_mapping(item, name=f"gauges[{index}]")
        require_exact_fields(gauge, _GAUGE_FIELDS, name=f"gauges[{index}]")
        gauge_ids.append(
            require_nonempty_string(
                gauge["gauge_id"],
                name=f"gauges[{index}].gauge_id",
            )
        )
        require_nonempty_string(
            gauge["mean_key"],
            name=f"gauges[{index}].mean_key",
        )
        require_nonempty_string(
            gauge["covariance_key"],
            name=f"gauges[{index}].covariance_key",
        )
    if len(set(gauge_ids)) != len(gauge_ids):
        raise ValueError("gauge IDs must be unique")


def _validate_factors(value: Any) -> None:
    factors = _require_json_list(value, name="factors")
    if not factors:
        raise ValueError("factors must not be empty")
    factor_ids: list[str] = []
    for index, item in enumerate(factors):
        factor = require_mapping(item, name=f"factors[{index}]")
        require_exact_fields(factor, _FACTOR_FIELDS, name=f"factors[{index}]")
        for field in (
            "factor_id",
            "view_id",
            "window_id",
            "gauge_id",
            "correlation_group_id",
        ):
            value_string = require_nonempty_string(
                factor[field],
                name=f"factors[{index}].{field}",
            )
            if field == "factor_id":
                factor_ids.append(value_string)
        require_exact_integer(
            factor["frame_index"],
            name=f"factors[{index}].frame_index",
            minimum=0,
        )
        require_exact_integer(
            factor["causal_frame_stop"],
            name=f"factors[{index}].causal_frame_stop",
            minimum=1,
        )
        require_json_number(
            factor["prior_nominal_probability"],
            name=f"factors[{index}].prior_nominal_probability",
        )
        require_json_number(
            factor["composite_weight"],
            name=f"factors[{index}].composite_weight",
        )
        arrays = require_mapping(
            factor["arrays"],
            name=f"factors[{index}].arrays",
        )
        require_exact_fields(
            arrays,
            _FACTOR_ARRAY_FIELDS,
            name=f"factors[{index}].arrays",
        )
        for field in sorted(_FACTOR_ARRAY_FIELDS):
            require_nonempty_string(
                arrays[field],
                name=f"factors[{index}].arrays.{field}",
            )
        ray_key = factor["ray_directions_local_key"]
        if ray_key is not None:
            require_nonempty_string(
                ray_key,
                name=f"factors[{index}].ray_directions_local_key",
            )
    if len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor IDs must be unique")


def _validate_gauge_covariance(value: Any) -> None:
    covariance = require_mapping(value, name="gauge_covariance")
    require_exact_fields(
        covariance,
        _GAUGE_COVARIANCE_FIELDS,
        name="gauge_covariance",
    )
    require_nonempty_string(
        covariance["semantics"],
        name="gauge_covariance.semantics",
    )
    require_nonempty_string(
        covariance["joint_covariance_key"],
        name="gauge_covariance.joint_covariance_key",
    )
    ordered_ids = _require_json_list(
        covariance["ordered_gauge_ids"],
        name="gauge_covariance.ordered_gauge_ids",
    )
    if not ordered_ids or any(
        not isinstance(item, str) or not item for item in ordered_ids
    ):
        raise ValueError(
            "gauge_covariance.ordered_gauge_ids must contain nonempty strings"
        )
    for field in (
        "cross_window_covariance_preserved",
        "diagonal_blocks_match_gauge_marginals",
    ):
        if not isinstance(covariance[field], bool):
            raise ValueError(f"gauge_covariance.{field} must be Boolean")


def validate_observation_factor_manifest_types(record: Mapping[str, Any]) -> None:
    """Reject scalar coercions in current manifests before legacy upgrading."""

    require_finite_json_mapping(record, name="observation-factor manifest")
    if record.get("schema") != OBSERVATION_FACTOR_SCHEMA:
        raise ValueError("manifest is not a Prob4D observation-factor bundle")
    schema_version = require_exact_integer(
        record.get("schema_version"),
        name="schema_version",
    )
    if schema_version != OBSERVATION_FACTOR_SCHEMA_VERSION:
        return

    require_exact_fields(record, _ROOT_FIELDS_V4, name="schema-v4 manifest")
    for field in (
        "gauge_parameterization",
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
    ):
        require_nonempty_string(record[field], name=field)
    require_exact_integer(
        record["causal_frame_stop"],
        name="causal_frame_stop",
        minimum=1,
    )
    if record["causal_frame_stop_convention"] != "exclusive":
        raise ValueError("schema-v4 causal frame stop must be exclusive")
    require_finite_json_mapping(record["metadata"], name="metadata")

    payload = require_mapping(record["payload"], name="payload")
    require_exact_fields(payload, _PAYLOAD_FIELDS, name="payload")
    require_nonempty_string(payload["path"], name="payload.path")
    require_sha256(payload["sha256"], name="payload.sha256")
    if payload["allow_pickle"] is not False:
        raise ValueError("payload.allow_pickle must be the literal Boolean false")

    _validate_gauges(record["gauges"])
    _validate_factors(record["factors"])
    _validate_gauge_covariance(record["gauge_covariance"])


__all__ = ["validate_observation_factor_manifest_types"]
