"""Strict contracts for the source-only joint ``Sim(3)`` closure diagnostic."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._immutable_array import immutable_array
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import (
    _exact_keys,
    _strict_integer,
    _strict_list,
    _strict_mapping,
    _strict_real,
    _strict_string,
)
from .sim3 import Sim3

FloatArray: TypeAlias = NDArray[np.float64]
JsonReport: TypeAlias = Mapping[str, Any]

GAUGE_LINEARIZATION_CLOSURE_SCHEMA: Final = "prob4d.gauge-linearization-closure"
GAUGE_LINEARIZATION_CLOSURE_VERSION: Final = 1
GAUGE_LINEARIZATION_SIGMA_POINT_RULE: Final = (
    "pivoted-psd-spherical-radial-cubature-v1"
)
GAUGE_LINEARIZATION_EVIDENCE_PARTITION: Final = "source-diagnostic"
GAUGE_LINEARIZATION_CLOSURE_CLAIM_BOUNDARY: Final = (
    "This source-only artifact evaluates first-order Sim(3) composition and point/query "
    "covariance closure under its declared Gaussian coordinate model. A passing result "
    "does not establish provider competence, empirical covariance calibration, "
    "BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or state of the art. "
    "A negative result redirects gauge propagation or nonlinear query projection and "
    "does not authorize a richer conditional point-uncertainty model."
)
_HARD_BRANCH_CUT_TOLERANCE: Final = 1e-7


def _nonnegative_real(value: Any, *, name: str) -> float:
    result = _strict_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_real(value: Any, *, name: str) -> float:
    result = _strict_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _probability(value: Any, *, name: str) -> float:
    result = _nonnegative_real(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: str | Path, *, name: str) -> Mapping[str, Any]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable or invalid JSON: {source}") from error
    mapping = _strict_mapping(value, name=name)
    if any(type(key) is not str for key in mapping):
        raise ValueError(f"{name} keys must be strings")
    return mapping


def _strict_numeric_tree(value: Any, *, name: str) -> Any:
    if type(value) is list:
        return [
            _strict_numeric_tree(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    return _strict_real(value, name=name)


def _numeric_array(value: Any, *, name: str) -> FloatArray:
    normalized = _strict_numeric_tree(_strict_list(value, name=name), name=name)
    try:
        array = np.asarray(normalized, dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error
    if array.dtype == np.dtype(object) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite rectangular numeric array")
    return cast(FloatArray, array)


def _require_symmetric_psd(value: FloatArray, *, name: str) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    scale = max(float(np.max(np.abs(symmetric), initial=0.0)), 1.0)
    if not np.allclose(matrix, symmetric, atol=1e-12 * scale, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if float(np.min(np.linalg.eigvalsh(symmetric), initial=0.0)) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetric


def _canonical_transform_vector(value: FloatArray, *, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (7,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain seven finite values")
    if not np.allclose(
        vector,
        Sim3.from_vector(vector).as_vector(),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError(
            f"{name} must use the canonical shortest-rotation Sim(3) coordinates"
        )
    return vector


@dataclass(frozen=True, slots=True)
class GaugeLinearizationCaseV1:
    """One joint Gaussian transform chain and optional downstream linear query."""

    case_id: str
    group_id: str
    transform_vectors: FloatArray
    joint_covariance: FloatArray
    points_local_m: FloatArray
    query_jacobian: FloatArray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vectors = np.asarray(self.transform_vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != 7 or vectors.shape[0] < 1:
            raise ValueError("transform_vectors must have shape (K, 7) with K >= 1")
        vectors = np.stack(
            [
                _canonical_transform_vector(vector, name=f"transform_vectors[{index}]")
                for index, vector in enumerate(vectors)
            ]
        )
        dimension = 7 * len(vectors)
        covariance = _require_symmetric_psd(
            np.asarray(self.joint_covariance, dtype=np.float64),
            name="joint_covariance",
        )
        if covariance.shape != (dimension, dimension):
            raise ValueError(
                f"joint_covariance must have shape ({dimension}, {dimension})"
            )
        points = np.asarray(self.points_local_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
            raise ValueError("points_local_m must have shape (N, 3) with N >= 1")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_local_m must be finite")
        query = None if self.query_jacobian is None else np.asarray(
            self.query_jacobian,
            dtype=np.float64,
        )
        if query is not None and (
            query.ndim != 3 or query.shape[0] < 1 or query.shape[1:] != points.shape
        ):
            raise ValueError("query_jacobian must have shape (Q, N, 3) matching points")
        if query is not None and not np.all(np.isfinite(query)):
            raise ValueError("query_jacobian must be finite")
        object.__setattr__(self, "case_id", _strict_string(self.case_id, name="case_id"))
        object.__setattr__(self, "group_id", _strict_string(self.group_id, name="group_id"))
        object.__setattr__(
            self,
            "transform_vectors",
            immutable_array(vectors, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "joint_covariance",
            immutable_array(covariance, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "points_local_m",
            immutable_array(points, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "query_jacobian",
            None if query is None else immutable_array(query, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="gauge linearization case metadata",
            ),
        )

    @property
    def transform_count(self) -> int:
        return int(self.transform_vectors.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.points_local_m.shape[0])

    @property
    def query_dimension(self) -> int:
        return 0 if self.query_jacobian is None else int(self.query_jacobian.shape[0])

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "group_id": self.group_id,
            "transform_vectors": self.transform_vectors.tolist(),
            "joint_covariance": self.joint_covariance.tolist(),
            "points_local_m": self.points_local_m.tolist(),
            "query_jacobian": (
                None if self.query_jacobian is None else self.query_jacobian.tolist()
            ),
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> GaugeLinearizationCaseV1:
        mapping = _strict_mapping(value, name="gauge linearization case")
        _exact_keys(
            mapping,
            {
                "case_id",
                "group_id",
                "transform_vectors",
                "joint_covariance",
                "points_local_m",
                "query_jacobian",
                "metadata",
            },
            name="gauge linearization case",
        )
        query = mapping["query_jacobian"]
        return cls(
            case_id=mapping["case_id"],
            group_id=mapping["group_id"],
            transform_vectors=_numeric_array(
                mapping["transform_vectors"],
                name="transform_vectors",
            ),
            joint_covariance=_numeric_array(
                mapping["joint_covariance"],
                name="joint_covariance",
            ),
            points_local_m=_numeric_array(
                mapping["points_local_m"],
                name="points_local_m",
            ),
            query_jacobian=(
                None if query is None else _numeric_array(query, name="query_jacobian")
            ),
            metadata=_strict_mapping(mapping["metadata"], name="case metadata"),
        )


@dataclass(frozen=True, slots=True)
class GaugeLinearizationPolicyV1:
    """Frozen source-only closure thresholds and covariance-rank convention."""

    minimum_group_count: int
    minimum_group_pass_fraction: float
    minimum_branch_cut_clearance_radians: float
    maximum_normalized_mean_shift: float
    maximum_relative_covariance_frobenius_error: float
    maximum_directional_variance_ratio_deviation: float
    maximum_variance_outside_linear_support_fraction: float
    covariance_rank_relative_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_group_count",
            _strict_integer(
                self.minimum_group_count,
                name="minimum_group_count",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "minimum_group_pass_fraction",
            _probability(
                self.minimum_group_pass_fraction,
                name="minimum_group_pass_fraction",
            ),
        )
        for name in (
            "minimum_branch_cut_clearance_radians",
            "maximum_normalized_mean_shift",
            "maximum_relative_covariance_frobenius_error",
            "maximum_directional_variance_ratio_deviation",
            "maximum_variance_outside_linear_support_fraction",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_real(getattr(self, name), name=name),
            )
        if self.minimum_branch_cut_clearance_radians > math.pi:
            raise ValueError("minimum_branch_cut_clearance_radians must not exceed pi")
        if self.maximum_variance_outside_linear_support_fraction > 1.0:
            raise ValueError(
                "maximum_variance_outside_linear_support_fraction must lie in [0, 1]"
            )
        object.__setattr__(
            self,
            "covariance_rank_relative_tolerance",
            _positive_real(
                self.covariance_rank_relative_tolerance,
                name="covariance_rank_relative_tolerance",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Any) -> GaugeLinearizationPolicyV1:
        mapping = _strict_mapping(value, name="gauge linearization policy")
        _exact_keys(mapping, set(cls.__dataclass_fields__), name="gauge linearization policy")
        return cls(**mapping)
