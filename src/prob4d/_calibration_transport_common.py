"""Shared types and validation for calibration-transport certificates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_json_number,
    require_mapping,
    require_sha256,
)

FloatArray: TypeAlias = NDArray[np.floating[Any]]
CALIBRATION_TRANSPORT_MODEL_SCHEMA = "prob4d.calibration-transport-model"
CALIBRATION_TRANSPORT_EVIDENCE_SCHEMA = "prob4d.calibration-transport-evidence"
CALIBRATION_TRANSPORT_FEATURE_CONTRACT_SCHEMA = (
    "prob4d.calibration-transport-feature-contract"
)
CALIBRATION_TRANSPORT_VERSION = 1
CALIBRATION_TRANSPORT_CLAIM_BOUNDARY = (
    "This source-only certificate measures whether target-prefix feature summaries "
    "are supported by complete calibration objects or acquisition sessions. It does "
    "not inspect target truth, downstream physical innovations, or posterior outcomes; "
    "it does not prove calibration transfer, authorize a BayesianPhysTwin update, or "
    "establish Prob4D, BayesianPhysTwin, or Causal4D benefit. A rejected certificate "
    "must be handled by the downstream exact physical fallback."
)

_MODEL_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "model_id",
        "feature_contract_id",
        "feature_names",
        "policy",
        "source_unit_ids",
        "source_row_counts",
        "source_embeddings",
        "source_metadata",
        "embedding_center",
        "embedding_scale",
        "source_nonconformity_scores",
        "support_threshold",
        "threshold_rank",
        "metadata",
        "claim_boundary",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "evidence_id",
        "model_id",
        "feature_contract_id",
        "feature_names",
        "target_unit_ids",
        "target_row_counts",
        "target_embeddings",
        "target_metadata",
        "group_results",
        "target_group_count",
        "target_row_count",
        "supported_group_count",
        "supported_row_count",
        "unsupported_group_fraction",
        "unsupported_row_fraction",
        "accepted",
        "decision_reasons",
        "worst_target_unit_id",
        "worst_feature_name",
        "metadata",
        "claim_boundary",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "quantile_levels",
        "miscoverage_rate",
        "minimum_source_units",
        "neighbor_count",
        "maximum_unsupported_group_fraction",
        "maximum_unsupported_row_fraction",
        "absolute_scale_floor",
        "relative_scale_floor",
        "distance_semantics",
        "threshold_semantics",
    }
)

_DISTANCE_SEMANTICS = "mean-k-nearest-rms-robust-standardized-quantile-distance-v1"
_THRESHOLD_SEMANTICS = "finite-source-upper-rank-of-leave-one-unit-out-scores-v1"
_MAD_NORMAL_SCALE = 1.482602218505602
_IQR_NORMAL_SCALE = 1.3489795003921634


class SourceFeatureGrid(Protocol):
    """Structural interface implemented by source-only feature grids."""

    feature_names: tuple[str, ...]

    def flattened(self) -> FloatArray:
        """Return valid feature rows with shape ``(N, F)``."""

        ...


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _readonly_float(value: object, *, ndim: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _strict_feature_names(value: object, *, name: str = "feature_names") -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a canonical tuple")
    names = tuple(
        require_exact_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not names:
        raise ValueError(f"{name} must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must be unique")
    return names


def _strict_identifier_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a canonical tuple")
    identifiers = tuple(
        require_exact_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not identifiers:
        raise ValueError(f"{name} must not be empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must be unique")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError(f"{name} must be sorted")
    return identifiers


def _strict_integer_tuple(value: object, *, name: str, minimum: int) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a canonical tuple")
    result = tuple(
        require_exact_integer(item, name=f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _strict_probability(
    value: object,
    *,
    name: str,
    lower_open: bool,
    upper_open: bool,
) -> float:
    result = require_json_number(value, name=name)
    lower_valid = result > 0.0 if lower_open else result >= 0.0
    upper_valid = result < 1.0 if upper_open else result <= 1.0
    if not lower_valid or not upper_valid:
        lower = "(" if lower_open else "["
        upper = ")" if upper_open else "]"
        raise ValueError(f"{name} must lie in {lower}0, 1{upper}")
    return result


def _strict_nonnegative_real(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _strict_quantile_levels(value: object) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise TypeError("quantile_levels must be a canonical tuple")
    levels = tuple(
        _strict_probability(
            item,
            name=f"quantile_levels[{index}]",
            lower_open=True,
            upper_open=True,
        )
        for index, item in enumerate(value)
    )
    if not levels:
        raise ValueError("quantile_levels must not be empty")
    if levels != tuple(sorted(levels)) or len(set(levels)) != len(levels):
        raise ValueError("quantile_levels must be strictly increasing")
    return levels


def _strict_metadata_tuple(
    value: object,
    *,
    name: str,
    expected_length: int,
) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not tuple or len(value) != expected_length:
        raise ValueError(f"{name} must be a tuple with one entry per unit")
    return tuple(
        frozen_finite_json_mapping(
            require_finite_json_mapping(item, name=f"{name}[{index}]"),
            name=f"{name}[{index}]",
        )
        for index, item in enumerate(value)
    )


def calibration_transport_feature_contract_id(
    feature_names: tuple[str, ...],
    *,
    semantics: str,
    configuration: Mapping[str, Any],
) -> str:
    """Return a stable digest for one source-only feature definition."""

    names = _strict_feature_names(feature_names)
    semantics_value = require_exact_string(semantics, name="semantics")
    normalized_configuration = frozen_finite_json_mapping(
        require_finite_json_mapping(configuration, name="configuration"),
        name="configuration",
    )
    payload = {
        "schema": CALIBRATION_TRANSPORT_FEATURE_CONTRACT_SCHEMA,
        "schema_version": CALIBRATION_TRANSPORT_VERSION,
        "feature_names": list(names),
        "semantics": semantics_value,
        "configuration": plain_json(normalized_configuration),
    }
    return _sha256_json(payload)


@dataclass(frozen=True, slots=True)
class CalibrationTransportUnitV1:
    """One complete object/session or predeclared target-prefix group."""

    unit_id: str
    feature_contract_id: str
    feature_names: tuple[str, ...]
    feature_values: FloatArray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unit_id = require_exact_string(self.unit_id, name="unit_id")
        contract_id = require_sha256(
            self.feature_contract_id,
            name="feature_contract_id",
        )
        names = _strict_feature_names(self.feature_names)
        values = _readonly_float(
            self.feature_values,
            ndim=2,
            name="feature_values",
        )
        if values.shape[0] < 1:
            raise ValueError("feature_values must contain at least one row")
        if values.shape[1] != len(names):
            raise ValueError("feature_values column count differs from feature_names")
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(self.metadata, name="metadata"),
            name="metadata",
        )
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "feature_contract_id", contract_id)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "metadata", metadata)

    @property
    def row_count(self) -> int:
        return int(self.feature_values.shape[0])

    @classmethod
    def from_feature_grid(
        cls,
        unit_id: str,
        feature_grid: SourceFeatureGrid,
        *,
        feature_contract_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> CalibrationTransportUnitV1:
        """Build a unit from a ``SourceReliabilityFeatures``-like grid."""

        grid_metadata = getattr(feature_grid, "metadata", {})
        combined_metadata: dict[str, Any] = {
            "feature_grid_metadata": plain_json(grid_metadata),
        }
        if metadata is not None:
            combined_metadata["unit_metadata"] = plain_json(metadata)
        return cls(
            unit_id=unit_id,
            feature_contract_id=feature_contract_id,
            feature_names=tuple(feature_grid.feature_names),
            feature_values=feature_grid.flattened(),
            metadata=combined_metadata,
        )


@dataclass(frozen=True, slots=True)
class CalibrationTransportPolicyV1:
    """Frozen source-only support and target-rejection policy."""

    quantile_levels: tuple[float, ...]
    miscoverage_rate: float
    minimum_source_units: int
    neighbor_count: int
    maximum_unsupported_group_fraction: float
    maximum_unsupported_row_fraction: float
    absolute_scale_floor: float
    relative_scale_floor: float
    distance_semantics: str = _DISTANCE_SEMANTICS
    threshold_semantics: str = _THRESHOLD_SEMANTICS

    def __post_init__(self) -> None:
        levels = _strict_quantile_levels(self.quantile_levels)
        miscoverage = _strict_probability(
            self.miscoverage_rate,
            name="miscoverage_rate",
            lower_open=True,
            upper_open=True,
        )
        minimum_source_units = require_exact_integer(
            self.minimum_source_units,
            name="minimum_source_units",
            minimum=3,
        )
        neighbor_count = require_exact_integer(
            self.neighbor_count,
            name="neighbor_count",
            minimum=1,
        )
        if neighbor_count >= minimum_source_units:
            raise ValueError("neighbor_count must be smaller than minimum_source_units")
        maximum_group = _strict_probability(
            self.maximum_unsupported_group_fraction,
            name="maximum_unsupported_group_fraction",
            lower_open=False,
            upper_open=False,
        )
        maximum_row = _strict_probability(
            self.maximum_unsupported_row_fraction,
            name="maximum_unsupported_row_fraction",
            lower_open=False,
            upper_open=False,
        )
        absolute_floor = _strict_nonnegative_real(
            self.absolute_scale_floor,
            name="absolute_scale_floor",
        )
        relative_floor = _strict_nonnegative_real(
            self.relative_scale_floor,
            name="relative_scale_floor",
        )
        if absolute_floor == 0.0 and relative_floor == 0.0:
            raise ValueError("at least one scale floor must be positive")
        if self.distance_semantics != _DISTANCE_SEMANTICS:
            raise ValueError(f"distance_semantics must equal {_DISTANCE_SEMANTICS!r}")
        if self.threshold_semantics != _THRESHOLD_SEMANTICS:
            raise ValueError(f"threshold_semantics must equal {_THRESHOLD_SEMANTICS!r}")
        object.__setattr__(self, "quantile_levels", levels)
        object.__setattr__(self, "miscoverage_rate", miscoverage)
        object.__setattr__(self, "minimum_source_units", minimum_source_units)
        object.__setattr__(self, "neighbor_count", neighbor_count)
        object.__setattr__(
            self,
            "maximum_unsupported_group_fraction",
            maximum_group,
        )
        object.__setattr__(
            self,
            "maximum_unsupported_row_fraction",
            maximum_row,
        )
        object.__setattr__(self, "absolute_scale_floor", absolute_floor)
        object.__setattr__(self, "relative_scale_floor", relative_floor)

    def to_dict(self) -> dict[str, object]:
        return {
            "quantile_levels": list(self.quantile_levels),
            "miscoverage_rate": self.miscoverage_rate,
            "minimum_source_units": self.minimum_source_units,
            "neighbor_count": self.neighbor_count,
            "maximum_unsupported_group_fraction": (
                self.maximum_unsupported_group_fraction
            ),
            "maximum_unsupported_row_fraction": self.maximum_unsupported_row_fraction,
            "absolute_scale_floor": self.absolute_scale_floor,
            "relative_scale_floor": self.relative_scale_floor,
            "distance_semantics": self.distance_semantics,
            "threshold_semantics": self.threshold_semantics,
        }

    @classmethod
    def from_dict(cls, value: object) -> CalibrationTransportPolicyV1:
        mapping = require_mapping(value, name="calibration transport policy")
        require_exact_fields(
            mapping,
            _POLICY_FIELDS,
            name="calibration transport policy",
        )
        raw_levels = mapping["quantile_levels"]
        if type(raw_levels) is not list:
            raise ValueError("quantile_levels must be a JSON array")
        return cls(
            quantile_levels=tuple(raw_levels),
            miscoverage_rate=mapping["miscoverage_rate"],
            minimum_source_units=mapping["minimum_source_units"],
            neighbor_count=mapping["neighbor_count"],
            maximum_unsupported_group_fraction=mapping[
                "maximum_unsupported_group_fraction"
            ],
            maximum_unsupported_row_fraction=mapping[
                "maximum_unsupported_row_fraction"
            ],
            absolute_scale_floor=mapping["absolute_scale_floor"],
            relative_scale_floor=mapping["relative_scale_floor"],
            distance_semantics=mapping["distance_semantics"],
            threshold_semantics=mapping["threshold_semantics"],
        )


def _strict_json_matrix(value: object, *, name: str) -> FloatArray:
    if type(value) is not list or not value:
        raise ValueError(f"{name} must be a non-empty JSON matrix")
    rows: list[list[float]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(value):
        if type(raw_row) is not list or not raw_row:
            raise ValueError(f"{name}[{row_index}] must be a non-empty JSON array")
        row = [
            require_json_number(item, name=f"{name}[{row_index}][{column_index}]")
            for column_index, item in enumerate(raw_row)
        ]
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{name} rows must have equal length")
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)
