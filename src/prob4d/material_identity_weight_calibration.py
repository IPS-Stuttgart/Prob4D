"""Group-balanced source calibration for material-identity mixture weights.

The portable material-identity mixture requires one calibrated log weight for
its mandatory null hypothesis and every linked source hypothesis.  This module
fits those weights from complete source/calibration objects or acquisition
sessions with deterministic group cross-fitting.  It never consumes target
outcomes and does not decide whether BayesianPhysTwin accepts an update.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import (
    _SHA256,
    _exact_keys,
    _sha256_json,
    _strict_bool,
    _strict_digest,
    _strict_integer,
    _strict_list,
    _strict_mapping,
    _strict_real,
    _strict_string,
)
from .material_identity_mixture import (
    LocalTrackEndpoint,
    MaterialIdentityCandidateV1,
    MaterialIdentityMixtureV1,
)

FloatArray: TypeAlias = NDArray[np.floating[Any]]

MATERIAL_IDENTITY_WEIGHT_CALIBRATION_SCHEMA = (
    "prob4d.material-identity-weight-calibration"
)
MATERIAL_IDENTITY_WEIGHT_CALIBRATION_VERSION = 1
MATERIAL_IDENTITY_WEIGHT_CALIBRATION_DATA_SCHEMA = (
    "prob4d.material-identity-weight-calibration-data"
)
MATERIAL_IDENTITY_WEIGHT_CALIBRATION_DATA_VERSION = 1
MATERIAL_IDENTITY_WEIGHT_SEMANTICS: Literal[
    "group-cross-fitted-conditional-logit-v1"
] = "group-cross-fitted-conditional-logit-v1"
MATERIAL_IDENTITY_WEIGHT_EVIDENCE_PARTITION = "source-calibration"
MATERIAL_IDENTITY_WEIGHT_USES_TARGET_OUTCOMES = False
MATERIAL_IDENTITY_WEIGHT_CLAIM_BOUNDARY = (
    "This artifact fits source-side material-identity weights with complete "
    "physical objects or acquisition sessions as independent groups. It does "
    "not establish target association calibration, BayesianPhysTwin benefit, "
    "Causal4D intervention benefit, deployment safety, or state of the art."
)

_CANDIDATE_KINDS = frozenset({"null", "linked"})
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def _probability(value: Any, *, name: str) -> float:
    result = _strict_real(value, name=name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _positive_real(value: Any, *, name: str) -> float:
    result = _strict_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _canonical_string_tuple(
    values: Sequence[Any],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    result = tuple(
        _strict_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _strict_real_vector(value: Any, *, name: str) -> FloatArray:
    items = _strict_list(value, name=name)
    if not items:
        raise ValueError(f"{name} must not be empty")
    result = np.asarray(
        [
            _strict_real(item, name=f"{name}[{index}]")
            for index, item in enumerate(items)
        ],
        dtype=np.float64,
    )
    result.setflags(write=False)
    return cast(FloatArray, result)


def _strict_real_matrix(value: Any, *, name: str) -> FloatArray:
    rows = _strict_list(value, name=name)
    if not rows:
        raise ValueError(f"{name} must not be empty")
    normalized: list[list[float]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(rows):
        row = _strict_list(raw_row, name=f"{name}[{row_index}]")
        if width is None:
            width = len(row)
            if width < 1:
                raise ValueError(f"{name} rows must not be empty")
        if len(row) != width:
            raise ValueError(f"{name} rows must have equal length")
        normalized.append(
            [
                _strict_real(item, name=f"{name}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            ]
        )
    result = np.asarray(normalized, dtype=np.float64)
    result.setflags(write=False)
    return cast(FloatArray, result)


def _readonly(value: Any, *, shape: tuple[int, ...], name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have finite shape {shape}")
    result.setflags(write=False)
    return cast(FloatArray, result)


def _softmax(logits: FloatArray) -> FloatArray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = float(np.max(values))
    shifted = np.exp(values - maximum)
    probability = shifted / float(np.sum(shifted))
    return cast(FloatArray, probability)


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


@dataclass(frozen=True, slots=True)
class MaterialIdentityCalibrationExampleV1:
    """One labelled candidate set from one independent source group."""

    example_id: str
    group_id: str
    candidate_ids: tuple[str, ...]
    candidate_kinds: tuple[Literal["null", "linked"], ...]
    features: FloatArray
    true_candidate_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        example_id = _strict_string(self.example_id, name="example_id")
        group_id = _strict_string(self.group_id, name="group_id")
        candidate_ids = tuple(
            _strict_digest(
                value,
                name=f"candidate_ids[{index}]",
                pattern=_SHA256,
            )
            for index, value in enumerate(self.candidate_ids)
        )
        if len(candidate_ids) < 2 or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_ids must contain at least two unique digests")
        kinds: list[Literal["null", "linked"]] = []
        for index, value in enumerate(self.candidate_kinds):
            kind = _strict_string(value, name=f"candidate_kinds[{index}]")
            if kind not in _CANDIDATE_KINDS:
                raise ValueError("candidate kinds must be 'null' or 'linked'")
            kinds.append(cast(Literal["null", "linked"], kind))
        candidate_kinds = tuple(kinds)
        if len(candidate_kinds) != len(candidate_ids):
            raise ValueError("candidate_kinds must align with candidate_ids")
        if candidate_kinds.count("null") != 1:
            raise ValueError("each calibration example requires exactly one null candidate")
        feature_values = np.asarray(self.features, dtype=np.float64).copy()
        if feature_values.ndim != 2 or feature_values.shape[0] != len(candidate_ids):
            raise ValueError("features must have one row per candidate")
        if feature_values.shape[1] < 1 or not np.all(np.isfinite(feature_values)):
            raise ValueError("features must be a finite non-empty matrix")
        true_candidate_id = _strict_digest(
            self.true_candidate_id,
            name="true_candidate_id",
            pattern=_SHA256,
        )
        if true_candidate_id not in candidate_ids:
            raise ValueError("true_candidate_id must occur in candidate_ids")

        order = tuple(
            sorted(
                range(len(candidate_ids)),
                key=lambda index: (
                    0 if candidate_kinds[index] == "null" else 1,
                    candidate_ids[index],
                ),
            )
        )
        candidate_ids = tuple(candidate_ids[index] for index in order)
        candidate_kinds = tuple(candidate_kinds[index] for index in order)
        feature_values = feature_values[np.asarray(order, dtype=np.int64)]
        feature_values.setflags(write=False)

        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "candidate_kinds", candidate_kinds)
        object.__setattr__(self, "features", feature_values)
        object.__setattr__(self, "true_candidate_id", true_candidate_id)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="material-identity calibration example metadata",
            ),
        )

    @property
    def true_index(self) -> int:
        return self.candidate_ids.index(self.true_candidate_id)

    @property
    def null_index(self) -> int:
        return self.candidate_kinds.index("null")

    @property
    def true_kind(self) -> Literal["null", "linked"]:
        return self.candidate_kinds[self.true_index]

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "group_id": self.group_id,
            "candidate_ids": list(self.candidate_ids),
            "candidate_kinds": list(self.candidate_kinds),
            "features": self.features.tolist(),
            "true_candidate_id": self.true_candidate_id,
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> MaterialIdentityCalibrationExampleV1:
        mapping = _strict_mapping(value, name="material-identity calibration example")
        _exact_keys(
            mapping,
            {
                "example_id",
                "group_id",
                "candidate_ids",
                "candidate_kinds",
                "features",
                "true_candidate_id",
                "metadata",
            },
            name="material-identity calibration example",
        )
        return cls(
            example_id=mapping["example_id"],
            group_id=mapping["group_id"],
            candidate_ids=tuple(
                _strict_list(mapping["candidate_ids"], name="candidate_ids")
            ),
            candidate_kinds=tuple(
                _strict_list(mapping["candidate_kinds"], name="candidate_kinds")
            ),
            features=_strict_real_matrix(mapping["features"], name="features"),
            true_candidate_id=mapping["true_candidate_id"],
            metadata=_strict_mapping(mapping["metadata"], name="example metadata"),
        )


@dataclass(frozen=True, slots=True)
class MaterialIdentityCalibrationReportV1:
    """Cross-fitted source diagnostics and final-fit convergence evidence."""

    example_count: int
    candidate_count: int
    group_count: int
    fold_count: int
    feature_count: int
    cross_fit_iterations: tuple[int, ...]
    final_fit_iterations: int
    final_fit_converged: bool
    cross_fitted_log_loss: float
    uniform_log_loss: float
    log_loss_advantage_vs_uniform: float
    cross_fitted_brier_score: float
    cross_fitted_top1_accuracy: float
    cross_fitted_mean_true_probability: float
    observed_null_fraction: float
    mean_predicted_null_probability: float
    top_choice_ece: float
    worst_group_log_loss: float
    ridge: float
    maximum_iterations: int
    convergence_tolerance: float

    def __post_init__(self) -> None:
        for name in (
            "example_count",
            "candidate_count",
            "group_count",
            "fold_count",
            "feature_count",
            "final_fit_iterations",
            "maximum_iterations",
        ):
            object.__setattr__(
                self,
                name,
                _strict_integer(getattr(self, name), name=name, minimum=1),
            )
        iterations = tuple(
            _strict_integer(value, name=f"cross_fit_iterations[{index}]", minimum=1)
            for index, value in enumerate(self.cross_fit_iterations)
        )
        if len(iterations) != self.fold_count:
            raise ValueError("cross_fit_iterations must have one value per fold")
        object.__setattr__(self, "cross_fit_iterations", iterations)
        object.__setattr__(
            self,
            "final_fit_converged",
            _strict_bool(self.final_fit_converged, name="final_fit_converged"),
        )
        for name in (
            "cross_fitted_log_loss",
            "uniform_log_loss",
            "cross_fitted_brier_score",
            "worst_group_log_loss",
            "ridge",
            "convergence_tolerance",
        ):
            value = _strict_real(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "log_loss_advantage_vs_uniform",
            _strict_real(
                self.log_loss_advantage_vs_uniform,
                name="log_loss_advantage_vs_uniform",
            ),
        )
        for name in (
            "cross_fitted_top1_accuracy",
            "cross_fitted_mean_true_probability",
            "observed_null_fraction",
            "mean_predicted_null_probability",
            "top_choice_ece",
        ):
            object.__setattr__(
                self,
                name,
                _probability(getattr(self, name), name=name),
            )
        if self.group_count < 2:
            raise ValueError("group_count must contain at least two independent groups")
        if self.fold_count > self.group_count:
            raise ValueError("fold_count cannot exceed group_count")
        if self.candidate_count < 2 * self.example_count:
            raise ValueError("candidate_count must retain at least two candidates per example")
        if any(value > self.maximum_iterations for value in self.cross_fit_iterations):
            raise ValueError("cross_fit_iterations exceed maximum_iterations")
        if self.final_fit_iterations > self.maximum_iterations:
            raise ValueError("final_fit_iterations exceed maximum_iterations")
        expected_advantage = self.uniform_log_loss - self.cross_fitted_log_loss
        if not math.isclose(
            self.log_loss_advantage_vs_uniform,
            expected_advantage,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("log_loss_advantage_vs_uniform is inconsistent")
        if self.cross_fitted_brier_score > 2.0:
            raise ValueError("cross_fitted_brier_score cannot exceed 2")

    def to_dict(self) -> dict[str, object]:
        return {
            "example_count": self.example_count,
            "candidate_count": self.candidate_count,
            "group_count": self.group_count,
            "fold_count": self.fold_count,
            "feature_count": self.feature_count,
            "cross_fit_iterations": list(self.cross_fit_iterations),
            "final_fit_iterations": self.final_fit_iterations,
            "final_fit_converged": self.final_fit_converged,
            "cross_fitted_log_loss": self.cross_fitted_log_loss,
            "uniform_log_loss": self.uniform_log_loss,
            "log_loss_advantage_vs_uniform": self.log_loss_advantage_vs_uniform,
            "cross_fitted_brier_score": self.cross_fitted_brier_score,
            "cross_fitted_top1_accuracy": self.cross_fitted_top1_accuracy,
            "cross_fitted_mean_true_probability": (
                self.cross_fitted_mean_true_probability
            ),
            "observed_null_fraction": self.observed_null_fraction,
            "mean_predicted_null_probability": self.mean_predicted_null_probability,
            "top_choice_ece": self.top_choice_ece,
            "worst_group_log_loss": self.worst_group_log_loss,
            "ridge": self.ridge,
            "maximum_iterations": self.maximum_iterations,
            "convergence_tolerance": self.convergence_tolerance,
        }

    @classmethod
    def from_dict(cls, value: Any) -> MaterialIdentityCalibrationReportV1:
        mapping = _strict_mapping(value, name="material-identity calibration report")
        expected = {
            "example_count",
            "candidate_count",
            "group_count",
            "fold_count",
            "feature_count",
            "cross_fit_iterations",
            "final_fit_iterations",
            "maximum_iterations",
            "final_fit_converged",
            "cross_fitted_log_loss",
            "uniform_log_loss",
            "log_loss_advantage_vs_uniform",
            "cross_fitted_brier_score",
            "cross_fitted_top1_accuracy",
            "cross_fitted_mean_true_probability",
            "observed_null_fraction",
            "mean_predicted_null_probability",
            "top_choice_ece",
            "worst_group_log_loss",
            "ridge",
            "convergence_tolerance",
        }
        _exact_keys(mapping, expected, name="material-identity calibration report")
        return cls(
            example_count=mapping["example_count"],
            candidate_count=mapping["candidate_count"],
            group_count=mapping["group_count"],
            fold_count=mapping["fold_count"],
            feature_count=mapping["feature_count"],
            cross_fit_iterations=tuple(
                _strict_list(
                    mapping["cross_fit_iterations"],
                    name="cross_fit_iterations",
                )
            ),
            final_fit_iterations=mapping["final_fit_iterations"],
            final_fit_converged=mapping["final_fit_converged"],
            cross_fitted_log_loss=mapping["cross_fitted_log_loss"],
            uniform_log_loss=mapping["uniform_log_loss"],
            log_loss_advantage_vs_uniform=(
                mapping["log_loss_advantage_vs_uniform"]
            ),
            cross_fitted_brier_score=mapping["cross_fitted_brier_score"],
            cross_fitted_top1_accuracy=mapping["cross_fitted_top1_accuracy"],
            cross_fitted_mean_true_probability=(
                mapping["cross_fitted_mean_true_probability"]
            ),
            observed_null_fraction=mapping["observed_null_fraction"],
            mean_predicted_null_probability=(
                mapping["mean_predicted_null_probability"]
            ),
            top_choice_ece=mapping["top_choice_ece"],
            worst_group_log_loss=mapping["worst_group_log_loss"],
            ridge=mapping["ridge"],
            maximum_iterations=mapping["maximum_iterations"],
            convergence_tolerance=mapping["convergence_tolerance"],
        )


@dataclass(frozen=True, slots=True)
class MaterialIdentityWeightCalibrationV1:
    """Content-addressed conditional-logit calibration for identity weights."""

    feature_names: tuple[str, ...]
    feature_center: FloatArray
    feature_scale: FloatArray
    coefficients: FloatArray
    null_bias: float
    calibration_data_id: str
    feature_schema_id: str
    association_rule_id: str
    tracklet_producer_revision: str
    association_revision: str
    label_definition: str
    group_definition: str
    calibration_group_ids: tuple[str, ...]
    report: MaterialIdentityCalibrationReportV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    weight_semantics: Literal[
        "group-cross-fitted-conditional-logit-v1"
    ] = MATERIAL_IDENTITY_WEIGHT_SEMANTICS
    evidence_partition: str = MATERIAL_IDENTITY_WEIGHT_EVIDENCE_PARTITION
    uses_target_outcomes: bool = MATERIAL_IDENTITY_WEIGHT_USES_TARGET_OUTCOMES

    def __post_init__(self) -> None:
        names = tuple(
            _strict_string(value, name=f"feature_names[{index}]")
            for index, value in enumerate(self.feature_names)
        )
        if not names or len(names) != len(set(names)):
            raise ValueError("feature_names must be non-empty and unique")
        feature_count = len(names)
        center = _readonly(
            self.feature_center,
            shape=(feature_count,),
            name="feature_center",
        )
        scale = _readonly(
            self.feature_scale,
            shape=(feature_count,),
            name="feature_scale",
        )
        if np.any(scale <= 0.0):
            raise ValueError("feature_scale must be positive")
        coefficients = _readonly(
            self.coefficients,
            shape=(feature_count,),
            name="coefficients",
        )
        null_bias = _strict_real(self.null_bias, name="null_bias")
        calibration_data_id = _strict_digest(
            self.calibration_data_id,
            name="calibration_data_id",
            pattern=_SHA256,
        )
        feature_schema_id = _strict_digest(
            self.feature_schema_id,
            name="feature_schema_id",
            pattern=_SHA256,
        )
        association_rule_id = _strict_digest(
            self.association_rule_id,
            name="association_rule_id",
            pattern=_SHA256,
        )
        tracklet_revision = _strict_digest(
            self.tracklet_producer_revision,
            name="tracklet_producer_revision",
            pattern=_REVISION,
        )
        association_revision = _strict_digest(
            self.association_revision,
            name="association_revision",
            pattern=_REVISION,
        )
        label_definition = _strict_string(
            self.label_definition,
            name="label_definition",
        )
        group_definition = _strict_string(
            self.group_definition,
            name="group_definition",
        )
        groups = _canonical_string_tuple(
            self.calibration_group_ids,
            name="calibration_group_ids",
        )
        if self.report.feature_count != feature_count:
            raise ValueError("calibration report feature count changed")
        if self.report.group_count != len(groups):
            raise ValueError("calibration report group count changed")
        if self.weight_semantics != MATERIAL_IDENTITY_WEIGHT_SEMANTICS:
            raise ValueError("unsupported material-identity weight semantics")
        if self.evidence_partition != MATERIAL_IDENTITY_WEIGHT_EVIDENCE_PARTITION:
            raise ValueError("material-identity calibration must be source-calibration")
        if self.uses_target_outcomes is not False:
            raise ValueError("material-identity calibration may not use target outcomes")

        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "null_bias", null_bias)
        object.__setattr__(self, "calibration_data_id", calibration_data_id)
        object.__setattr__(self, "feature_schema_id", feature_schema_id)
        object.__setattr__(self, "association_rule_id", association_rule_id)
        object.__setattr__(
            self,
            "tracklet_producer_revision",
            tracklet_revision,
        )
        object.__setattr__(self, "association_revision", association_revision)
        object.__setattr__(self, "label_definition", label_definition)
        object.__setattr__(self, "group_definition", group_definition)
        object.__setattr__(self, "calibration_group_ids", groups)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="material-identity weight calibration metadata",
            ),
        )

    def predict_log_weights(
        self,
        features: FloatArray,
        candidate_kinds: Sequence[str],
        *,
        feature_names: Sequence[str] | None = None,
    ) -> FloatArray:
        """Return normalized candidate log weights in the supplied candidate order."""

        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("features must match the calibrated feature dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("material-identity prediction features must be finite")
        if not self.report.final_fit_converged:
            raise ValueError("material-identity calibration final fit did not converge")
        if feature_names is not None and tuple(feature_names) != self.feature_names:
            raise ValueError("material-identity feature names changed")
        kinds = tuple(candidate_kinds)
        if len(kinds) != values.shape[0]:
            raise ValueError("candidate_kinds must align with feature rows")
        if kinds.count("null") != 1 or any(kind not in _CANDIDATE_KINDS for kind in kinds):
            raise ValueError("candidate_kinds require exactly one null and linked values")
        standardized = (values - self.feature_center) / self.feature_scale
        logits = standardized @ self.coefficients
        logits = np.asarray(logits, dtype=np.float64)
        logits[np.asarray([kind == "null" for kind in kinds], dtype=bool)] += self.null_bias
        maximum = float(np.max(logits))
        log_normalizer = maximum + math.log(float(np.sum(np.exp(logits - maximum))))
        result = logits - log_normalizer
        result.setflags(write=False)
        return cast(FloatArray, result)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": MATERIAL_IDENTITY_WEIGHT_CALIBRATION_SCHEMA,
            "schema_version": MATERIAL_IDENTITY_WEIGHT_CALIBRATION_VERSION,
            "weight_semantics": self.weight_semantics,
            "evidence_partition": self.evidence_partition,
            "uses_target_outcomes": self.uses_target_outcomes,
            "feature_names": list(self.feature_names),
            "feature_center": self.feature_center.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "null_bias": self.null_bias,
            "calibration_data_id": self.calibration_data_id,
            "feature_schema_id": self.feature_schema_id,
            "association_rule_id": self.association_rule_id,
            "tracklet_producer_revision": self.tracklet_producer_revision,
            "association_revision": self.association_revision,
            "label_definition": self.label_definition,
            "group_definition": self.group_definition,
            "calibration_group_ids": list(self.calibration_group_ids),
            "report": self.report.to_dict(),
            "metadata": plain_json(self.metadata),
            "claim_boundary": MATERIAL_IDENTITY_WEIGHT_CLAIM_BOUNDARY,
        }

    @property
    def artifact_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    @classmethod
    def from_dict(cls, value: Any) -> MaterialIdentityWeightCalibrationV1:
        mapping = _strict_mapping(value, name="material-identity weight calibration")
        expected = {
            "artifact_id",
            "schema",
            "schema_version",
            "weight_semantics",
            "evidence_partition",
            "uses_target_outcomes",
            "feature_names",
            "feature_center",
            "feature_scale",
            "coefficients",
            "null_bias",
            "calibration_data_id",
            "feature_schema_id",
            "association_rule_id",
            "tracklet_producer_revision",
            "association_revision",
            "label_definition",
            "group_definition",
            "calibration_group_ids",
            "report",
            "metadata",
            "claim_boundary",
        }
        _exact_keys(mapping, expected, name="material-identity weight calibration")
        artifact_id = _strict_digest(
            mapping["artifact_id"],
            name="artifact_id",
            pattern=_SHA256,
        )
        if mapping["schema"] != MATERIAL_IDENTITY_WEIGHT_CALIBRATION_SCHEMA:
            raise ValueError("unsupported material-identity calibration schema")
        version = _strict_integer(
            mapping["schema_version"],
            name="schema_version",
            minimum=1,
        )
        if version != MATERIAL_IDENTITY_WEIGHT_CALIBRATION_VERSION:
            raise ValueError("unsupported material-identity calibration version")
        if mapping["claim_boundary"] != MATERIAL_IDENTITY_WEIGHT_CLAIM_BOUNDARY:
            raise ValueError("material-identity calibration claim boundary changed")
        names = tuple(_strict_list(mapping["feature_names"], name="feature_names"))
        model = cls(
            feature_names=names,
            feature_center=_strict_real_vector(
                mapping["feature_center"],
                name="feature_center",
            ),
            feature_scale=_strict_real_vector(
                mapping["feature_scale"],
                name="feature_scale",
            ),
            coefficients=_strict_real_vector(
                mapping["coefficients"],
                name="coefficients",
            ),
            null_bias=mapping["null_bias"],
            calibration_data_id=mapping["calibration_data_id"],
            feature_schema_id=mapping["feature_schema_id"],
            association_rule_id=mapping["association_rule_id"],
            tracklet_producer_revision=mapping["tracklet_producer_revision"],
            association_revision=mapping["association_revision"],
            label_definition=mapping["label_definition"],
            group_definition=mapping["group_definition"],
            calibration_group_ids=tuple(
                _strict_list(mapping["calibration_group_ids"], name="group IDs")
            ),
            report=MaterialIdentityCalibrationReportV1.from_dict(mapping["report"]),
            metadata=_strict_mapping(mapping["metadata"], name="metadata"),
            weight_semantics=mapping["weight_semantics"],
            evidence_partition=mapping["evidence_partition"],
            uses_target_outcomes=mapping["uses_target_outcomes"],
        )
        if model.artifact_id != artifact_id:
            raise ValueError("material-identity calibration artifact_id mismatch")
        return model


@dataclass(frozen=True, slots=True)
class _FitResult:
    center: FloatArray
    scale: FloatArray
    parameters: FloatArray
    iterations: int
    converged: bool


def _canonical_examples(
    examples: Sequence[MaterialIdentityCalibrationExampleV1],
    *,
    feature_count: int,
) -> tuple[MaterialIdentityCalibrationExampleV1, ...]:
    result = tuple(sorted(examples, key=lambda item: (item.group_id, item.example_id)))
    if not result:
        raise ValueError("material-identity calibration requires examples")
    if len({item.example_id for item in result}) != len(result):
        raise ValueError("material-identity calibration example IDs must be unique")
    if any(item.features.shape[1] != feature_count for item in result):
        raise ValueError("calibration example feature dimensions changed")
    groups = {item.group_id for item in result}
    if len(groups) < 2:
        raise ValueError("material-identity calibration requires at least two groups")
    null_groups = {item.group_id for item in result if item.true_kind == "null"}
    linked_groups = {item.group_id for item in result if item.true_kind == "linked"}
    if not null_groups or not linked_groups:
        raise ValueError("calibration requires both true null and true linked examples")
    return result


def _example_weights(
    examples: Sequence[MaterialIdentityCalibrationExampleV1],
) -> FloatArray:
    groups = tuple(sorted({item.group_id for item in examples}))
    counts = {
        group_id: sum(item.group_id == group_id for item in examples)
        for group_id in groups
    }
    result = np.asarray(
        [1.0 / (len(groups) * counts[item.group_id]) for item in examples],
        dtype=np.float64,
    )
    if not np.isclose(float(np.sum(result)), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("material-identity group-balanced weights changed")
    result.setflags(write=False)
    return cast(FloatArray, result)


def _feature_location_scale(
    examples: Sequence[MaterialIdentityCalibrationExampleV1],
    example_weights: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    feature_count = examples[0].features.shape[1]
    center = np.zeros(feature_count, dtype=np.float64)
    for example, weight in zip(examples, example_weights, strict=True):
        center += weight * np.mean(example.features, axis=0)
    variance = np.zeros(feature_count, dtype=np.float64)
    for example, weight in zip(examples, example_weights, strict=True):
        centered = example.features - center
        variance += weight * np.mean(centered**2, axis=0)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    center.setflags(write=False)
    scale.setflags(write=False)
    return cast(FloatArray, center), cast(FloatArray, scale)


def _example_design(
    example: MaterialIdentityCalibrationExampleV1,
    center: FloatArray,
    scale: FloatArray,
) -> FloatArray:
    standardized = (example.features - center) / scale
    null_indicator = np.asarray(
        [kind == "null" for kind in example.candidate_kinds],
        dtype=np.float64,
    )[:, None]
    return cast(FloatArray, np.column_stack((null_indicator, standardized)))


def _objective_gradient_hessian(
    examples: Sequence[MaterialIdentityCalibrationExampleV1],
    weights: FloatArray,
    center: FloatArray,
    scale: FloatArray,
    parameters: FloatArray,
    ridge: float,
) -> tuple[float, FloatArray, FloatArray]:
    dimension = len(parameters)
    objective = 0.5 * ridge * float(parameters @ parameters)
    gradient = ridge * np.asarray(parameters, dtype=np.float64)
    hessian = ridge * np.eye(dimension, dtype=np.float64)
    for example, weight in zip(examples, weights, strict=True):
        design = _example_design(example, center, scale)
        logits = design @ parameters
        probability = _softmax(logits)
        true_index = example.true_index
        objective += weight * (
            -float(logits[true_index])
            + float(np.max(logits))
            + math.log(float(np.sum(np.exp(logits - np.max(logits)))))
        )
        expected = probability @ design
        gradient += weight * (expected - design[true_index])
        covariance = np.diag(probability) - np.outer(probability, probability)
        hessian += weight * (design.T @ covariance @ design)
    return (
        objective,
        cast(FloatArray, gradient),
        cast(FloatArray, hessian),
    )


def _fit_conditional_logit(
    examples: Sequence[MaterialIdentityCalibrationExampleV1],
    *,
    ridge: float,
    maximum_iterations: int,
    convergence_tolerance: float,
) -> _FitResult:
    weights = _example_weights(examples)
    center, scale = _feature_location_scale(examples, weights)
    parameters = np.zeros(examples[0].features.shape[1] + 1, dtype=np.float64)
    converged = False
    iterations = 0
    for iteration in range(1, maximum_iterations + 1):
        iterations = iteration
        objective, gradient, hessian = _objective_gradient_hessian(
            examples,
            weights,
            center,
            scale,
            parameters,
            ridge,
        )
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= convergence_tolerance * (1.0 + abs(objective)):
            converged = True
            break
        step = np.linalg.solve(hessian, gradient)
        step_norm = float(np.linalg.norm(step))
        if step_norm <= convergence_tolerance * (1.0 + np.linalg.norm(parameters)):
            converged = True
            break
        fraction = 1.0
        accepted = False
        for _ in range(24):
            candidate = parameters - fraction * step
            candidate_objective, _, _ = _objective_gradient_hessian(
                examples,
                weights,
                center,
                scale,
                candidate,
                ridge,
            )
            objective_tolerance = 1e-14 * (1.0 + abs(objective))
            if candidate_objective <= objective + objective_tolerance:
                parameters = candidate
                accepted = True
                if fraction * step_norm <= convergence_tolerance * (
                    1.0 + np.linalg.norm(parameters)
                ):
                    converged = True
                break
            fraction *= 0.5
        if not accepted or converged:
            break
    parameters.setflags(write=False)
    return _FitResult(
        center=center,
        scale=scale,
        parameters=cast(FloatArray, parameters),
        iterations=iterations,
        converged=converged,
    )


def _predict_example(
    example: MaterialIdentityCalibrationExampleV1,
    fit: _FitResult,
) -> FloatArray:
    design = _example_design(example, fit.center, fit.scale)
    return _softmax(design @ fit.parameters)


def _fold_groups(groups: Sequence[str], fold_count: int) -> tuple[tuple[str, ...], ...]:
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    for index, group_id in enumerate(sorted(groups)):
        folds[index % fold_count].append(group_id)
    return tuple(tuple(fold) for fold in folds)


def _weighted_ece(
    confidences: FloatArray,
    correctness: FloatArray,
    weights: FloatArray,
    *,
    bin_count: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    result = 0.0
    for index in range(bin_count):
        if index + 1 == bin_count:
            selected = (confidences >= edges[index]) & (confidences <= edges[index + 1])
        else:
            selected = (confidences >= edges[index]) & (confidences < edges[index + 1])
        mass = float(np.sum(weights[selected]))
        if mass <= 0.0:
            continue
        confidence = float(np.sum(weights[selected] * confidences[selected]) / mass)
        accuracy = float(np.sum(weights[selected] * correctness[selected]) / mass)
        result += mass * abs(confidence - accuracy)
    return result


def _calibration_data_id(
    examples: Sequence[MaterialIdentityCalibrationExampleV1],
    feature_names: Sequence[str],
) -> str:
    return _sha256_json(
        {
            "schema": MATERIAL_IDENTITY_WEIGHT_CALIBRATION_DATA_SCHEMA,
            "schema_version": MATERIAL_IDENTITY_WEIGHT_CALIBRATION_DATA_VERSION,
            "feature_names": list(feature_names),
            "examples": [example.to_dict() for example in examples],
        }
    )


def fit_material_identity_weight_calibration(
    examples: Sequence[MaterialIdentityCalibrationExampleV1],
    *,
    feature_names: Sequence[str],
    feature_schema_id: str,
    association_rule_id: str,
    tracklet_producer_revision: str,
    association_revision: str,
    label_definition: str,
    group_definition: str,
    cross_fit_fold_count: int,
    ridge: float = 1e-2,
    maximum_iterations: int = 100,
    convergence_tolerance: float = 1e-10,
    metadata: Mapping[str, Any] | None = None,
) -> MaterialIdentityWeightCalibrationV1:
    """Fit one deterministic group-cross-fitted conditional-logit model."""

    names = tuple(
        _strict_string(value, name=f"feature_names[{index}]")
        for index, value in enumerate(feature_names)
    )
    if not names or len(names) != len(set(names)):
        raise ValueError("feature_names must be non-empty and unique")
    canonical = _canonical_examples(examples, feature_count=len(names))
    group_ids = tuple(sorted({item.group_id for item in canonical}))
    fold_count = _strict_integer(
        cross_fit_fold_count,
        name="cross_fit_fold_count",
        minimum=2,
    )
    if fold_count > len(group_ids):
        raise ValueError("cross_fit_fold_count cannot exceed calibration groups")
    ridge_value = _positive_real(ridge, name="ridge")
    maximum_iterations_value = _strict_integer(
        maximum_iterations,
        name="maximum_iterations",
        minimum=1,
    )
    tolerance = _positive_real(
        convergence_tolerance,
        name="convergence_tolerance",
    )

    predictions: dict[str, FloatArray] = {}
    cross_fit_iterations: list[int] = []
    for validation_groups in _fold_groups(group_ids, fold_count):
        validation_set = frozenset(validation_groups)
        training = tuple(item for item in canonical if item.group_id not in validation_set)
        validation = tuple(item for item in canonical if item.group_id in validation_set)
        if not training or not validation:
            raise RuntimeError("material-identity cross-fit fold is empty")
        training_kinds = {item.true_kind for item in training}
        if training_kinds != _CANDIDATE_KINDS:
            raise ValueError(
                "every material-identity cross-fit training fold must contain "
                "true null and true linked examples"
            )
        fit = _fit_conditional_logit(
            training,
            ridge=ridge_value,
            maximum_iterations=maximum_iterations_value,
            convergence_tolerance=tolerance,
        )
        if not fit.converged:
            raise RuntimeError("material-identity cross-fit optimization did not converge")
        cross_fit_iterations.append(fit.iterations)
        for example in validation:
            predictions[example.example_id] = _predict_example(example, fit)
    if set(predictions) != {item.example_id for item in canonical}:
        raise RuntimeError("material-identity cross-fit predictions are incomplete")

    weights = _example_weights(canonical)
    log_losses: FloatArray = np.empty(len(canonical), dtype=np.float64)
    uniform_losses: FloatArray = np.empty(len(canonical), dtype=np.float64)
    brier_scores: FloatArray = np.empty(len(canonical), dtype=np.float64)
    top1: FloatArray = np.empty(len(canonical), dtype=np.float64)
    true_probabilities: FloatArray = np.empty(len(canonical), dtype=np.float64)
    predicted_null: FloatArray = np.empty(len(canonical), dtype=np.float64)
    observed_null: FloatArray = np.empty(len(canonical), dtype=np.float64)
    confidences: FloatArray = np.empty(len(canonical), dtype=np.float64)
    group_losses: dict[str, list[float]] = {group_id: [] for group_id in group_ids}
    for index, example in enumerate(canonical):
        probability = predictions[example.example_id]
        true_index = example.true_index
        true_probability = max(float(probability[true_index]), float(np.finfo(np.float64).tiny))
        log_losses[index] = -math.log(true_probability)
        uniform_losses[index] = math.log(len(example.candidate_ids))
        target: FloatArray = np.zeros(len(probability), dtype=np.float64)
        target[true_index] = 1.0
        brier_scores[index] = float(np.sum((probability - target) ** 2))
        selected = int(np.argmax(probability))
        top1[index] = float(selected == true_index)
        true_probabilities[index] = true_probability
        predicted_null[index] = float(probability[example.null_index])
        observed_null[index] = float(example.true_kind == "null")
        confidences[index] = float(probability[selected])
        group_losses[example.group_id].append(float(log_losses[index]))

    final_fit = _fit_conditional_logit(
        canonical,
        ridge=ridge_value,
        maximum_iterations=maximum_iterations_value,
        convergence_tolerance=tolerance,
    )
    if not final_fit.converged:
        raise RuntimeError("material-identity final optimization did not converge")
    log_loss = float(np.sum(weights * log_losses))
    uniform_log_loss = float(np.sum(weights * uniform_losses))
    report = MaterialIdentityCalibrationReportV1(
        example_count=len(canonical),
        candidate_count=sum(len(item.candidate_ids) for item in canonical),
        group_count=len(group_ids),
        fold_count=fold_count,
        feature_count=len(names),
        cross_fit_iterations=tuple(cross_fit_iterations),
        final_fit_iterations=final_fit.iterations,
        final_fit_converged=final_fit.converged,
        cross_fitted_log_loss=log_loss,
        uniform_log_loss=uniform_log_loss,
        log_loss_advantage_vs_uniform=uniform_log_loss - log_loss,
        cross_fitted_brier_score=float(np.sum(weights * brier_scores)),
        cross_fitted_top1_accuracy=float(np.sum(weights * top1)),
        cross_fitted_mean_true_probability=float(
            np.sum(weights * true_probabilities)
        ),
        observed_null_fraction=float(np.sum(weights * observed_null)),
        mean_predicted_null_probability=float(np.sum(weights * predicted_null)),
        top_choice_ece=_weighted_ece(confidences, top1, weights),
        worst_group_log_loss=max(
            float(np.mean(values)) for values in group_losses.values()
        ),
        ridge=ridge_value,
        maximum_iterations=maximum_iterations_value,
        convergence_tolerance=tolerance,
    )
    return MaterialIdentityWeightCalibrationV1(
        feature_names=names,
        feature_center=final_fit.center,
        feature_scale=final_fit.scale,
        coefficients=final_fit.parameters[1:],
        null_bias=float(final_fit.parameters[0]),
        calibration_data_id=_calibration_data_id(canonical, names),
        feature_schema_id=feature_schema_id,
        association_rule_id=association_rule_id,
        tracklet_producer_revision=tracklet_producer_revision,
        association_revision=association_revision,
        label_definition=label_definition,
        group_definition=group_definition,
        calibration_group_ids=group_ids,
        report=report,
        metadata={} if metadata is None else metadata,
    )


def write_material_identity_weight_calibration(
    path: str | Path,
    model: MaterialIdentityWeightCalibrationV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish and reload one calibration artifact."""

    destination = Path(path)
    atomic_write_text(
        destination,
        json.dumps(model.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        overwrite=overwrite,
    )
    restored = load_material_identity_weight_calibration(destination)
    if restored.artifact_id != model.artifact_id:
        raise RuntimeError("published material-identity calibration changed")


def load_material_identity_weight_calibration(
    path: str | Path,
) -> MaterialIdentityWeightCalibrationV1:
    """Load and content-validate one strict calibration artifact."""

    return MaterialIdentityWeightCalibrationV1.from_dict(
        _load_json(path, name="material-identity weight calibration")
    )


def calibration_from_config(value: Any) -> MaterialIdentityWeightCalibrationV1:
    """Fit a calibration artifact from the strict raw JSON configuration form."""

    mapping = _strict_mapping(value, name="material-identity calibration configuration")
    expected = {
        "feature_names",
        "feature_schema_id",
        "association_rule_id",
        "tracklet_producer_revision",
        "association_revision",
        "label_definition",
        "group_definition",
        "cross_fit_fold_count",
        "ridge",
        "maximum_iterations",
        "convergence_tolerance",
        "examples",
        "metadata",
        "uses_target_outcomes",
    }
    _exact_keys(mapping, expected, name="material-identity calibration configuration")
    if _strict_bool(mapping["uses_target_outcomes"], name="uses_target_outcomes"):
        raise ValueError("material-identity source calibration may not use target outcomes")
    examples = tuple(
        MaterialIdentityCalibrationExampleV1.from_dict(item)
        for item in _strict_list(mapping["examples"], name="examples")
    )
    return fit_material_identity_weight_calibration(
        examples,
        feature_names=tuple(
            _strict_list(mapping["feature_names"], name="feature_names")
        ),
        feature_schema_id=mapping["feature_schema_id"],
        association_rule_id=mapping["association_rule_id"],
        tracklet_producer_revision=mapping["tracklet_producer_revision"],
        association_revision=mapping["association_revision"],
        label_definition=mapping["label_definition"],
        group_definition=mapping["group_definition"],
        cross_fit_fold_count=mapping["cross_fit_fold_count"],
        ridge=mapping["ridge"],
        maximum_iterations=mapping["maximum_iterations"],
        convergence_tolerance=mapping["convergence_tolerance"],
        metadata=_strict_mapping(mapping["metadata"], name="metadata"),
    )


def _candidate_from_mapping(
    value: Any,
    *,
    index: int,
) -> tuple[MaterialIdentityCandidateV1, FloatArray]:
    name = f"candidates[{index}]"
    mapping = _strict_mapping(value, name=name)
    _exact_keys(
        mapping,
        {
            "source_endpoint",
            "association_result_id",
            "source_score",
            "features",
            "metadata",
        },
        name=name,
    )
    source_raw = mapping["source_endpoint"]
    source = (
        None
        if source_raw is None
        else LocalTrackEndpoint.from_mapping(source_raw, name=f"{name}.source_endpoint")
    )
    candidate = MaterialIdentityCandidateV1(
        source_endpoint=source,
        association_result_id=mapping["association_result_id"],
        source_score=mapping["source_score"],
        calibrated_log_weight=0.0,
        metadata=_strict_mapping(mapping["metadata"], name=f"{name}.metadata"),
    )
    raw_features = _strict_list(mapping["features"], name=f"{name}.features")
    features = np.asarray(
        [
            _strict_real(item, name=f"{name}.features[{feature_index}]")
            for feature_index, item in enumerate(raw_features)
        ],
        dtype=np.float64,
    )
    features.setflags(write=False)
    return candidate, cast(FloatArray, features)


def calibrated_mixture_from_config(
    model: MaterialIdentityWeightCalibrationV1,
    value: Any,
) -> MaterialIdentityMixtureV1:
    """Build one mixture whose log weights come only from the calibrated model."""

    mapping = _strict_mapping(value, name="calibrated material-identity mixture config")
    expected = {
        "target_endpoint",
        "window_order",
        "causal_frame_stop",
        "association_rule_id",
        "feature_schema_id",
        "tracklet_producer_revision",
        "association_revision",
        "feature_names",
        "candidates",
        "metadata",
    }
    _exact_keys(mapping, expected, name="calibrated material-identity mixture config")
    association_rule_id = _strict_digest(
        mapping["association_rule_id"],
        name="association_rule_id",
        pattern=_SHA256,
    )
    feature_schema_id = _strict_digest(
        mapping["feature_schema_id"],
        name="feature_schema_id",
        pattern=_SHA256,
    )
    tracklet_revision = _strict_digest(
        mapping["tracklet_producer_revision"],
        name="tracklet_producer_revision",
        pattern=_REVISION,
    )
    association_revision = _strict_digest(
        mapping["association_revision"],
        name="association_revision",
        pattern=_REVISION,
    )
    if association_rule_id != model.association_rule_id:
        raise ValueError("association_rule_id differs from calibration")
    if feature_schema_id != model.feature_schema_id:
        raise ValueError("feature_schema_id differs from calibration")
    if tracklet_revision != model.tracklet_producer_revision:
        raise ValueError("tracklet producer revision differs from calibration")
    if association_revision != model.association_revision:
        raise ValueError("association revision differs from calibration")
    feature_names = tuple(
        _strict_list(mapping["feature_names"], name="feature_names")
    )
    if feature_names != model.feature_names:
        raise ValueError("feature names differ from calibration")

    parsed = [
        _candidate_from_mapping(item, index=index)
        for index, item in enumerate(_strict_list(mapping["candidates"], name="candidates"))
    ]
    if not parsed:
        raise ValueError("calibrated mixture requires candidates")
    parsed.sort(key=lambda item: item[0].ordering_key())
    candidates = tuple(item[0] for item in parsed)
    features = np.stack([item[1] for item in parsed], axis=0)
    if features.shape[1] != len(model.feature_names):
        raise ValueError("candidate feature dimension differs from calibration")
    kinds = tuple(candidate.kind for candidate in candidates)
    log_weights = model.predict_log_weights(
        features,
        kinds,
        feature_names=feature_names,
    )
    weighted_candidates = tuple(
        MaterialIdentityCandidateV1(
            source_endpoint=candidate.source_endpoint,
            association_result_id=candidate.association_result_id,
            source_score=candidate.source_score,
            calibrated_log_weight=float(log_weight),
            metadata=candidate.metadata,
        )
        for candidate, log_weight in zip(candidates, log_weights, strict=True)
    )
    return MaterialIdentityMixtureV1(
        target_endpoint=LocalTrackEndpoint.from_mapping(
            mapping["target_endpoint"],
            name="target_endpoint",
        ),
        window_order=tuple(_strict_list(mapping["window_order"], name="window_order")),
        causal_frame_stop=mapping["causal_frame_stop"],
        association_rule_id=association_rule_id,
        calibration_id=model.artifact_id,
        tracklet_producer_revision=tracklet_revision,
        association_revision=association_revision,
        candidates=weighted_candidates,
        metadata={
            **plain_json(_strict_mapping(mapping["metadata"], name="metadata")),
            "material_identity_weight_calibration_id": model.artifact_id,
            "material_identity_feature_schema_id": model.feature_schema_id,
        },
    )


def calibration_summary(
    model: MaterialIdentityWeightCalibrationV1,
) -> dict[str, object]:
    """Return one concise deterministic JSON summary."""

    return {
        "artifact_id": model.artifact_id,
        "weight_semantics": model.weight_semantics,
        "calibration_data_id": model.calibration_data_id,
        "feature_names": list(model.feature_names),
        "calibration_group_ids": list(model.calibration_group_ids),
        "report": model.report.to_dict(),
    }


__all__ = [
    "MATERIAL_IDENTITY_WEIGHT_CALIBRATION_DATA_SCHEMA",
    "MATERIAL_IDENTITY_WEIGHT_CALIBRATION_DATA_VERSION",
    "MATERIAL_IDENTITY_WEIGHT_CALIBRATION_SCHEMA",
    "MATERIAL_IDENTITY_WEIGHT_CALIBRATION_VERSION",
    "MATERIAL_IDENTITY_WEIGHT_CLAIM_BOUNDARY",
    "MATERIAL_IDENTITY_WEIGHT_SEMANTICS",
    "MaterialIdentityCalibrationExampleV1",
    "MaterialIdentityCalibrationReportV1",
    "MaterialIdentityWeightCalibrationV1",
    "calibrated_mixture_from_config",
    "calibration_from_config",
    "calibration_summary",
    "fit_material_identity_weight_calibration",
    "load_material_identity_weight_calibration",
    "write_material_identity_weight_calibration",
]
