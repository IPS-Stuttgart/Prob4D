"""Strict loading and verification for Sim(3) linearization certificates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .._immutable_json import plain_json
from .._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_json_number,
    require_mapping,
    require_sha256,
)
from .sim3_linearization import (
    SIM3_LINEARIZATION_CLAIM_BOUNDARY,
    SIM3_LINEARIZATION_SCHEMA,
    SIM3_LINEARIZATION_VERSION,
    GaussianLinearizationAdequacyV1,
    LinearizationAdequacyThresholdsV1,
)

_CERTIFICATE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "parameterization",
        "parameter_order",
        "parameter_dimension",
        "output_shape",
        "sample_count",
        "batch_size",
        "seed",
        "finite_difference_step",
        "jacobian_validated",
        "thresholds",
        "point_diagnostics",
        "query_diagnostics",
        "adequate",
        "failure_reasons",
        "metadata",
        "claim_boundary",
        "gaussian_linearization_adequacy_id",
    }
)
_THRESHOLD_FIELDS = frozenset(
    {
        "maximum_relative_trace_error",
        "maximum_relative_frobenius_error",
        "maximum_mean_shift_standard_deviations",
        "maximum_principal_axis_angle_degrees",
        "minimum_principal_axis_anisotropy",
        "maximum_query_relative_trace_error",
        "maximum_query_relative_frobenius_error",
        "maximum_query_mean_shift_standard_deviations",
    }
)
_POINT_FIELDS = frozenset(
    {
        "item_index",
        "relative_trace_error",
        "relative_frobenius_error",
        "mean_shift_standard_deviations",
        "nonlinear_trace",
        "linearized_trace",
        "principal_axis_anisotropy",
        "principal_axis_angle_degrees",
    }
)
_QUERY_FIELDS = frozenset(
    {
        "query_dimension",
        "relative_trace_error",
        "relative_frobenius_error",
        "mean_shift_standard_deviations",
        "nonlinear_trace",
        "linearized_trace",
    }
)


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _finite_number(value: object, *, name: str) -> float:
    return require_json_number(value, name=name)


def _nonnegative_number(value: object, *, name: str) -> float:
    result = _finite_number(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_nonnegative_number(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_number(value, name=name)


def _string_tuple(
    value: object,
    *,
    name: str,
    allow_empty: bool,
    require_unique: bool,
) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    result = tuple(
        require_exact_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if require_unique and len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _output_shape(value: object) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2:
        raise ValueError("output_shape must be a two-element JSON array")
    return (
        require_exact_integer(value[0], name="output_shape[0]", minimum=1),
        require_exact_integer(value[1], name="output_shape[1]", minimum=1),
    )


def _thresholds(value: object) -> LinearizationAdequacyThresholdsV1:
    mapping = require_mapping(value, name="linearization thresholds")
    require_exact_fields(mapping, _THRESHOLD_FIELDS, name="linearization thresholds")
    return LinearizationAdequacyThresholdsV1(
        maximum_relative_trace_error=_nonnegative_number(
            mapping["maximum_relative_trace_error"],
            name="maximum_relative_trace_error",
        ),
        maximum_relative_frobenius_error=_nonnegative_number(
            mapping["maximum_relative_frobenius_error"],
            name="maximum_relative_frobenius_error",
        ),
        maximum_mean_shift_standard_deviations=_nonnegative_number(
            mapping["maximum_mean_shift_standard_deviations"],
            name="maximum_mean_shift_standard_deviations",
        ),
        maximum_principal_axis_angle_degrees=_nonnegative_number(
            mapping["maximum_principal_axis_angle_degrees"],
            name="maximum_principal_axis_angle_degrees",
        ),
        minimum_principal_axis_anisotropy=_nonnegative_number(
            mapping["minimum_principal_axis_anisotropy"],
            name="minimum_principal_axis_anisotropy",
        ),
        maximum_query_relative_trace_error=_nonnegative_number(
            mapping["maximum_query_relative_trace_error"],
            name="maximum_query_relative_trace_error",
        ),
        maximum_query_relative_frobenius_error=_nonnegative_number(
            mapping["maximum_query_relative_frobenius_error"],
            name="maximum_query_relative_frobenius_error",
        ),
        maximum_query_mean_shift_standard_deviations=_nonnegative_number(
            mapping["maximum_query_mean_shift_standard_deviations"],
            name="maximum_query_mean_shift_standard_deviations",
        ),
    )


def _point_diagnostic(value: object, *, index: int) -> Mapping[str, Any]:
    mapping = require_mapping(value, name=f"point_diagnostics[{index}]")
    require_exact_fields(mapping, _POINT_FIELDS, name=f"point_diagnostics[{index}]")
    item_index = require_exact_integer(
        mapping["item_index"],
        name=f"point_diagnostics[{index}].item_index",
        minimum=0,
    )
    if item_index != index:
        raise ValueError("point_diagnostics item_index values must be contiguous and ordered")
    normalized: dict[str, Any] = {
        "item_index": item_index,
        "relative_trace_error": _nonnegative_number(
            mapping["relative_trace_error"],
            name=f"point_diagnostics[{index}].relative_trace_error",
        ),
        "relative_frobenius_error": _nonnegative_number(
            mapping["relative_frobenius_error"],
            name=f"point_diagnostics[{index}].relative_frobenius_error",
        ),
        "mean_shift_standard_deviations": _nonnegative_number(
            mapping["mean_shift_standard_deviations"],
            name=f"point_diagnostics[{index}].mean_shift_standard_deviations",
        ),
        "nonlinear_trace": _nonnegative_number(
            mapping["nonlinear_trace"],
            name=f"point_diagnostics[{index}].nonlinear_trace",
        ),
        "linearized_trace": _nonnegative_number(
            mapping["linearized_trace"],
            name=f"point_diagnostics[{index}].linearized_trace",
        ),
        "principal_axis_anisotropy": _nonnegative_number(
            mapping["principal_axis_anisotropy"],
            name=f"point_diagnostics[{index}].principal_axis_anisotropy",
        ),
        "principal_axis_angle_degrees": _optional_nonnegative_number(
            mapping["principal_axis_angle_degrees"],
            name=f"point_diagnostics[{index}].principal_axis_angle_degrees",
        ),
    }
    return require_finite_json_mapping(
        normalized,
        name=f"point_diagnostics[{index}]",
    )


def _query_diagnostic(value: object) -> Mapping[str, Any] | None:
    if value is None:
        return None
    mapping = require_mapping(value, name="query_diagnostics")
    require_exact_fields(mapping, _QUERY_FIELDS, name="query_diagnostics")
    normalized = {
        "query_dimension": require_exact_integer(
            mapping["query_dimension"],
            name="query_diagnostics.query_dimension",
            minimum=1,
        ),
        "relative_trace_error": _nonnegative_number(
            mapping["relative_trace_error"],
            name="query_diagnostics.relative_trace_error",
        ),
        "relative_frobenius_error": _nonnegative_number(
            mapping["relative_frobenius_error"],
            name="query_diagnostics.relative_frobenius_error",
        ),
        "mean_shift_standard_deviations": _nonnegative_number(
            mapping["mean_shift_standard_deviations"],
            name="query_diagnostics.mean_shift_standard_deviations",
        ),
        "nonlinear_trace": _nonnegative_number(
            mapping["nonlinear_trace"],
            name="query_diagnostics.nonlinear_trace",
        ),
        "linearized_trace": _nonnegative_number(
            mapping["linearized_trace"],
            name="query_diagnostics.linearized_trace",
        ),
    }
    return require_finite_json_mapping(normalized, name="query_diagnostics")


def _replayed_failure_reasons(
    points: tuple[Mapping[str, Any], ...],
    query: Mapping[str, Any] | None,
    thresholds: LinearizationAdequacyThresholdsV1,
    *,
    output_dimension: int,
) -> tuple[str, ...]:
    """Replay the diagnostic decision from persisted metrics and frozen thresholds."""

    maximum_trace_error = max(
        float(point["relative_trace_error"]) for point in points
    )
    maximum_frobenius_error = max(
        float(point["relative_frobenius_error"]) for point in points
    )
    maximum_mean_shift = max(
        float(point["mean_shift_standard_deviations"]) for point in points
    )
    axis_angles: list[float] = []
    for index, point in enumerate(points):
        anisotropy = float(point["principal_axis_anisotropy"])
        raw_angle = point["principal_axis_angle_degrees"]
        angle = None if raw_angle is None else float(raw_angle)
        should_have_angle = (
            output_dimension >= 2
            and anisotropy >= thresholds.minimum_principal_axis_anisotropy
        )
        if should_have_angle != (angle is not None):
            raise ValueError(
                "point diagnostic principal-axis angle presence changed at "
                f"item {index}"
            )
        if angle is not None:
            if angle > 90.0:
                raise ValueError(
                    "point diagnostic principal-axis angle must not exceed 90 degrees"
                )
            axis_angles.append(angle)

    reasons: list[str] = []
    if maximum_trace_error > thresholds.maximum_relative_trace_error:
        reasons.append("point-trace-distortion")
    if maximum_frobenius_error > thresholds.maximum_relative_frobenius_error:
        reasons.append("point-frobenius-distortion")
    if maximum_mean_shift > thresholds.maximum_mean_shift_standard_deviations:
        reasons.append("point-mean-shift")
    if axis_angles and max(axis_angles) > thresholds.maximum_principal_axis_angle_degrees:
        reasons.append("point-principal-axis-rotation")
    if query is not None:
        if float(query["relative_trace_error"]) > (
            thresholds.maximum_query_relative_trace_error
        ):
            reasons.append("query-trace-distortion")
        if float(query["relative_frobenius_error"]) > (
            thresholds.maximum_query_relative_frobenius_error
        ):
            reasons.append("query-frobenius-distortion")
        if float(query["mean_shift_standard_deviations"]) > (
            thresholds.maximum_query_mean_shift_standard_deviations
        ):
            reasons.append("query-mean-shift")
    return tuple(reasons)


def gaussian_linearization_adequacy_from_dict(
    value: object,
) -> GaussianLinearizationAdequacyV1:
    """Reconstruct one certificate and replay every derived field and identity."""

    mapping = require_mapping(value, name="Gaussian linearization certificate")
    require_exact_fields(
        mapping,
        _CERTIFICATE_FIELDS,
        name="Gaussian linearization certificate",
    )
    if mapping["schema"] != SIM3_LINEARIZATION_SCHEMA:
        raise ValueError("Gaussian linearization certificate schema changed")
    if mapping["schema_version"] != SIM3_LINEARIZATION_VERSION:
        raise ValueError("Gaussian linearization certificate version changed")
    if mapping["claim_boundary"] != SIM3_LINEARIZATION_CLAIM_BOUNDARY:
        raise ValueError("Gaussian linearization certificate claim boundary changed")

    shape = _output_shape(mapping["output_shape"])
    raw_points = mapping["point_diagnostics"]
    if type(raw_points) is not list:
        raise ValueError("point_diagnostics must be a JSON array")
    if len(raw_points) != shape[0]:
        raise ValueError("point_diagnostics count must match output_shape[0]")
    points = tuple(
        _point_diagnostic(item, index=index)
        for index, item in enumerate(raw_points)
    )
    failure_reasons = _string_tuple(
        mapping["failure_reasons"],
        name="failure_reasons",
        allow_empty=True,
        require_unique=True,
    )
    threshold_set = _thresholds(mapping["thresholds"])
    query = _query_diagnostic(mapping["query_diagnostics"])
    replayed_reasons = _replayed_failure_reasons(
        points,
        query,
        threshold_set,
        output_dimension=shape[1],
    )
    if failure_reasons != replayed_reasons:
        raise ValueError(
            "Gaussian linearization certificate decision does not match diagnostics"
        )
    adequate = _strict_bool(mapping["adequate"], name="adequate")
    if adequate != (not replayed_reasons):
        raise ValueError(
            "Gaussian linearization certificate adequacy does not match diagnostics"
        )
    result = GaussianLinearizationAdequacyV1(
        parameterization=require_exact_string(
            mapping["parameterization"],
            name="parameterization",
        ),
        parameter_order=_string_tuple(
            mapping["parameter_order"],
            name="parameter_order",
            allow_empty=True,
            require_unique=True,
        ),
        parameter_dimension=require_exact_integer(
            mapping["parameter_dimension"],
            name="parameter_dimension",
            minimum=1,
        ),
        output_shape=shape,
        sample_count=require_exact_integer(
            mapping["sample_count"],
            name="sample_count",
            minimum=2,
        ),
        batch_size=require_exact_integer(
            mapping["batch_size"],
            name="batch_size",
            minimum=1,
        ),
        seed=require_exact_integer(mapping["seed"], name="seed", minimum=0),
        finite_difference_step=_nonnegative_number(
            mapping["finite_difference_step"],
            name="finite_difference_step",
        ),
        jacobian_validated=_strict_bool(
            mapping["jacobian_validated"],
            name="jacobian_validated",
        ),
        thresholds=threshold_set,
        point_diagnostics=points,
        query_diagnostics=query,
        adequate=adequate,
        failure_reasons=failure_reasons,
        metadata=require_finite_json_mapping(mapping["metadata"], name="metadata"),
    )
    supplied_id = require_sha256(
        mapping["gaussian_linearization_adequacy_id"],
        name="gaussian_linearization_adequacy_id",
    )
    if supplied_id != result.gaussian_linearization_adequacy_id:
        raise ValueError("Gaussian linearization certificate identity mismatch")
    if plain_json(mapping) != plain_json(result.to_dict()):
        raise ValueError("Gaussian linearization certificate derived fields changed")
    return result


def load_gaussian_linearization_adequacy(
    path: str | Path,
) -> GaussianLinearizationAdequacyV1:
    """Load one strict certificate while rejecting duplicate keys and tampering."""

    return gaussian_linearization_adequacy_from_dict(
        load_json_object(path, name="Gaussian linearization certificate")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--fail-on-inadequate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    certificate = load_gaussian_linearization_adequacy(arguments.artifact)
    print(
        json.dumps(
            {
                "gaussian_linearization_adequacy_id": (
                    certificate.gaussian_linearization_adequacy_id
                ),
                "adequate": certificate.adequate,
                "parameterization": certificate.parameterization,
                "sample_count": certificate.sample_count,
                "query_projection_evaluated": certificate.query_diagnostics is not None,
            },
            sort_keys=True,
        )
    )
    if arguments.fail_on_inadequate and not certificate.adequate:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "gaussian_linearization_adequacy_from_dict",
    "load_gaussian_linearization_adequacy",
]
