"""Calibration-separated corroboration guard for distinct 4-D providers.

The guard evaluates matched point predictions from two provider-neutral manifests
without assuming independent provider errors.  It is deliberately upstream of a
BayesianPhysTwin update: an admitted case means only that the registered providers
corroborate one another under the frozen source-side score.  A downstream physical
regret guard and exact physical fallback remain mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_revision,
    require_sha256,
    require_string_sequence,
)
from .finite_sample_threshold import (
    FiniteSampleUpperThreshold,
    fit_finite_sample_upper_threshold,
)
from .prediction_provider_manifest import (
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    verify_prediction_provider_manifest,
)

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

CROSS_PROVIDER_PANEL_SCHEMA: Final = "prob4d.cross-provider-panel"
CROSS_PROVIDER_PANEL_VERSION: Final = 1
CROSS_PROVIDER_CALIBRATION_SCHEMA: Final = (
    "prob4d.cross-provider-corroboration-calibration"
)
CROSS_PROVIDER_CALIBRATION_VERSION: Final = 1
CROSS_PROVIDER_DECISION_SCHEMA: Final = "prob4d.cross-provider-corroboration-decision"
CROSS_PROVIDER_DECISION_VERSION: Final = 1
CROSS_PROVIDER_SCORE_SEMANTICS: Final = (
    "rowwise-normalized-mahalanobis-higher-quantile-v1"
)
UNKNOWN_DEPENDENCE_COVARIANCE_SEMANTICS: Final = (
    "young-inequality-difference-covariance-upper-bound-v1"
)
EXPLICIT_CROSS_COVARIANCE_SEMANTICS: Final = (
    "explicit-cross-covariance-difference-v1"
)
COVARIANCE_MODES: Final = (
    UNKNOWN_DEPENDENCE_COVARIANCE_SEMANTICS,
    EXPLICIT_CROSS_COVARIANCE_SEMANTICS,
)
CROSS_PROVIDER_CLAIM_BOUNDARY: Final = (
    "Source-side cross-provider corroboration only. Admission does not establish "
    "absolute geometric correctness, shared-common-bias detection, calibrated "
    "deployment uncertainty, BayesianPhysTwin benefit, Causal4D benefit, safety, "
    "or state of the art. A downstream physical regret guard and exact fallback "
    "remain separate requirements."
)

_PANEL_FIELDS: Final = frozenset(
    {"schema", "schema_version", "purpose", "cases", "metadata"}
)
_CASE_FIELDS: Final = frozenset(
    {
        "case_id",
        "first_manifest",
        "second_manifest",
        "first_payload_ids",
        "second_payload_ids",
        "matched_observations",
        "matched_observations_sha256",
        "alignment_artifact_id",
        "row_identity_sha256",
        "coordinate_frame_id",
    }
)
_PROVIDER_FIELDS: Final = frozenset(
    {
        "contract_id",
        "provider_family",
        "provider_repository",
        "provider_revision",
        "model_set_id",
        "loader_id",
        "coordinate_semantics",
        "point_semantics",
    }
)
_CASE_SCORE_FIELDS: Final = frozenset(
    {
        "case_id",
        "sequence_id",
        "first_provider_contract_id",
        "second_provider_contract_id",
        "first_manifest_artifact_id",
        "second_manifest_artifact_id",
        "first_manifest_sha256",
        "second_manifest_sha256",
        "first_provider_run_id",
        "second_provider_run_id",
        "first_payload_ids",
        "second_payload_ids",
        "matched_observations_sha256",
        "alignment_artifact_id",
        "row_identity_sha256",
        "coordinate_frame_id",
        "shared_input_dependence_group_id",
        "covariance_mode",
        "row_count",
        "valid_count",
        "support_fraction",
        "mean_row_score",
        "median_row_score",
        "case_score",
        "maximum_row_score",
    }
)
_THRESHOLD_FIELDS: Final = frozenset(
    {
        "semantics",
        "miscoverage",
        "calibration_count",
        "order_statistic_rank",
        "threshold",
        "guaranteed_miscoverage_upper_bound",
        "canonical_scores_sha256",
        "exchangeability_boundary",
    }
)
_CALIBRATION_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "first_provider",
        "second_provider",
        "covariance_mode",
        "score_semantics",
        "row_quantile",
        "minimum_support_fraction",
        "calibration_panel_source_sha256",
        "calibration_panel_semantic_id",
        "calibration_cases",
        "finite_sample_threshold",
        "metadata",
        "claim_boundary",
    }
)
_DECISION_CASE_FIELDS: Final = frozenset(
    {"score", "admitted", "rejection_reasons"}
)
_DECISION_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "calibration_artifact_id",
        "first_provider",
        "second_provider",
        "covariance_mode",
        "score_semantics",
        "row_quantile",
        "minimum_support_fraction",
        "threshold",
        "target_panel_source_sha256",
        "target_panel_semantic_id",
        "cases",
        "accepted_count",
        "rejected_count",
        "all_cases_admitted",
        "metadata",
        "claim_boundary",
    }
)
_MATCHED_REQUIRED_FIELDS: Final = frozenset(
    {
        "first_points_m",
        "second_points_m",
        "first_covariance_m2",
        "second_covariance_m2",
        "valid_mask",
        "alignment_artifact_id",
        "row_identity_sha256",
        "coordinate_frame_id",
    }
)
_MATCHED_OPTIONAL_FIELDS: Final = frozenset({"cross_covariance_m2"})


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json_bytes_object(payload: bytes, *, name: str) -> dict[str, Any]:
    """Parse the exact JSON bytes whose content digest is retained."""

    class _DuplicateKeyError(ValueError):
        pass

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise _DuplicateKeyError(
                    f"{name} contains duplicate JSON object key {key!r}"
                )
            result[key] = item
        return result

    def reject_constant(token: str) -> Any:
        raise ValueError(f"{name} contains non-finite JSON number {token!r}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must contain UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must contain valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as error:
        raise ValueError(f"cannot read {path.name!r}") from error


def _exact_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be Boolean")
    return bool(value)


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw.item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _probability(value: object, *, name: str, open_interval: bool = False) -> float:
    result = _finite_real(value, name=name, minimum=0.0, maximum=1.0)
    if open_interval and not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


def _safe_relative_path(value: object, *, name: str) -> str:
    path = require_exact_string(value, name=name)
    if "\\" in path:
        raise ValueError(f"{name} must be a safe POSIX relative path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    return pure.as_posix()


def _resolved_member(root: Path, relative_path: object, *, name: str) -> Path:
    safe = _safe_relative_path(relative_path, name=name)
    root_resolved = root.resolve()
    current = root_resolved
    for part in PurePosixPath(safe).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symbolic link")
    candidate = current.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} escapes the panel directory") from error
    if not candidate.is_file():
        raise ValueError(f"{name} does not identify a regular file")
    return candidate


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_json_once(path: Path, record: Mapping[str, Any]) -> Path:
    content = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if load_json_object(path, name=path.name) != plain_json(record):
            raise ValueError(f"refusing to replace different artifact {path.name!r}")
        return path
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if load_json_object(path, name=path.name) != plain_json(record):
                raise ValueError(f"concurrent writer published different {path.name!r}")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _strict_sha_sequence(value: object, *, name: str) -> tuple[str, ...]:
    values = require_string_sequence(value, name=name)
    result = tuple(
        require_sha256(item, name=f"{name}[{index}]")
        for index, item in enumerate(values)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


@dataclass(frozen=True)
class ProviderContractV1:
    """Provider identity that must remain fixed across calibration and target cases."""

    provider_family: str
    provider_repository: str
    provider_revision: str
    model_set_id: str
    loader_id: str
    coordinate_semantics: str
    point_semantics: str
    contract_id: str | None = None

    def __post_init__(self) -> None:
        family = require_exact_string(self.provider_family, name="provider_family")
        repository = require_exact_string(
            self.provider_repository,
            name="provider_repository",
        )
        revision = require_revision(self.provider_revision, name="provider_revision")
        model_set_id = require_sha256(self.model_set_id, name="model_set_id")
        loader_id = require_sha256(self.loader_id, name="loader_id")
        coordinate = require_exact_string(
            self.coordinate_semantics,
            name="coordinate_semantics",
        )
        point = require_exact_string(self.point_semantics, name="point_semantics")
        object.__setattr__(self, "provider_family", family)
        object.__setattr__(self, "provider_repository", repository)
        object.__setattr__(self, "provider_revision", revision)
        object.__setattr__(self, "model_set_id", model_set_id)
        object.__setattr__(self, "loader_id", loader_id)
        object.__setattr__(self, "coordinate_semantics", coordinate)
        object.__setattr__(self, "point_semantics", point)
        expected = _sha256_json(self.identity_record())
        if self.contract_id is not None and require_sha256(
            self.contract_id,
            name="contract_id",
        ) != expected:
            raise ValueError("provider contract ID mismatch")
        object.__setattr__(self, "contract_id", expected)

    @classmethod
    def from_manifest(cls, manifest: PredictionProviderManifestV1) -> ProviderContractV1:
        return cls(
            provider_family=manifest.provider_family,
            provider_repository=manifest.provider_repository,
            provider_revision=manifest.provider_revision,
            model_set_id=manifest.model_set_id,
            loader_id=manifest.loader_id,
            coordinate_semantics=manifest.coordinate_semantics,
            point_semantics=manifest.point_semantics,
        )

    def identity_record(self) -> dict[str, object]:
        return {
            "provider_family": self.provider_family,
            "provider_repository": self.provider_repository,
            "provider_revision": self.provider_revision,
            "model_set_id": self.model_set_id,
            "loader_id": self.loader_id,
            "coordinate_semantics": self.coordinate_semantics,
            "point_semantics": self.point_semantics,
        }

    def to_record(self) -> dict[str, object]:
        return {"contract_id": self.contract_id, **self.identity_record()}

    @classmethod
    def from_record(cls, value: object) -> ProviderContractV1:
        mapping = require_mapping(value, name="provider contract")
        require_exact_fields(mapping, _PROVIDER_FIELDS, name="provider contract")
        return cls(
            provider_family=mapping["provider_family"],
            provider_repository=mapping["provider_repository"],
            provider_revision=mapping["provider_revision"],
            model_set_id=mapping["model_set_id"],
            loader_id=mapping["loader_id"],
            coordinate_semantics=mapping["coordinate_semantics"],
            point_semantics=mapping["point_semantics"],
            contract_id=mapping["contract_id"],
        )


@dataclass(frozen=True)
class CrossProviderScoreSummary:
    """Source-only normalized disagreement summary for one independent case."""

    covariance_mode: str
    row_count: int
    valid_count: int
    support_fraction: float
    mean_row_score: float
    median_row_score: float
    case_score: float
    maximum_row_score: float

    def __post_init__(self) -> None:
        if self.covariance_mode not in COVARIANCE_MODES:
            raise ValueError("unsupported cross-provider covariance mode")
        row_count = require_exact_integer(self.row_count, name="row_count", minimum=1)
        valid_count = require_exact_integer(
            self.valid_count,
            name="valid_count",
            minimum=1,
        )
        if valid_count > row_count:
            raise ValueError("valid_count exceeds row_count")
        support = _probability(self.support_fraction, name="support_fraction")
        expected_support = valid_count / row_count
        if not math.isclose(support, expected_support, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("support_fraction differs from row counts")
        values = (
            _finite_real(self.mean_row_score, name="mean_row_score", minimum=0.0),
            _finite_real(self.median_row_score, name="median_row_score", minimum=0.0),
            _finite_real(self.case_score, name="case_score", minimum=0.0),
            _finite_real(self.maximum_row_score, name="maximum_row_score", minimum=0.0),
        )
        if values[3] + 1e-15 < max(values[:3]):
            raise ValueError("maximum_row_score is smaller than a score summary")
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "valid_count", valid_count)
        object.__setattr__(self, "support_fraction", support)
        object.__setattr__(self, "mean_row_score", values[0])
        object.__setattr__(self, "median_row_score", values[1])
        object.__setattr__(self, "case_score", values[2])
        object.__setattr__(self, "maximum_row_score", values[3])


def _validated_matched_arrays(
    first_points_m: np.ndarray,
    second_points_m: np.ndarray,
    first_covariance_m2: np.ndarray,
    second_covariance_m2: np.ndarray,
    valid_mask: np.ndarray,
    cross_covariance_m2: np.ndarray | None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray, FloatArray | None]:
    first_points = np.asarray(first_points_m)
    second_points = np.asarray(second_points_m)
    first_covariance = np.asarray(first_covariance_m2)
    second_covariance = np.asarray(second_covariance_m2)
    valid = np.asarray(valid_mask)
    row_count = first_points.shape[0] if first_points.ndim == 2 else -1
    if first_points.dtype != np.dtype(np.float64) or first_points.shape != (row_count, 3):
        raise ValueError("first_points_m must be float64 with shape (N, 3)")
    if second_points.dtype != np.dtype(np.float64) or second_points.shape != (
        row_count,
        3,
    ):
        raise ValueError("second_points_m must be float64 with shape (N, 3)")
    expected_covariance_shape = (row_count, 3, 3)
    if first_covariance.dtype != np.dtype(np.float64) or first_covariance.shape != (
        expected_covariance_shape
    ):
        raise ValueError("first_covariance_m2 must be float64 with shape (N, 3, 3)")
    if second_covariance.dtype != np.dtype(np.float64) or second_covariance.shape != (
        expected_covariance_shape
    ):
        raise ValueError("second_covariance_m2 must be float64 with shape (N, 3, 3)")
    if valid.dtype != np.dtype(np.bool_) or valid.shape != (row_count,):
        raise ValueError("valid_mask must be Boolean with shape (N,)")
    if row_count < 1 or not np.any(valid):
        raise ValueError("matched provider input requires at least one valid row")
    arrays = (first_points, second_points, first_covariance, second_covariance)
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("matched provider arrays must be finite")
    cross: FloatArray | None = None
    if cross_covariance_m2 is not None:
        raw_cross = np.asarray(cross_covariance_m2)
        if raw_cross.dtype != np.dtype(np.float64) or raw_cross.shape != (
            expected_covariance_shape
        ):
            raise ValueError(
                "cross_covariance_m2 must be float64 with shape (N, 3, 3)"
            )
        if not np.all(np.isfinite(raw_cross)):
            raise ValueError("cross_covariance_m2 must be finite")
        cross = raw_cross
    return (
        first_points,
        second_points,
        first_covariance,
        second_covariance,
        valid,
        cross,
    )


def _difference_covariance_unknown_dependence(
    first_covariance: np.ndarray,
    second_covariance: np.ndarray,
) -> np.ndarray:
    first_trace = np.trace(first_covariance, axis1=-2, axis2=-1)
    second_trace = np.trace(second_covariance, axis1=-2, axis2=-1)
    if np.any(first_trace <= 0.0) or np.any(second_trace <= 0.0):
        raise ValueError("provider covariance traces must be positive")
    beta = np.sqrt(second_trace / first_trace)
    return (
        (1.0 + beta)[..., None, None] * first_covariance
        + (1.0 + 1.0 / beta)[..., None, None] * second_covariance
    )


def compute_cross_provider_score(
    first_points_m: np.ndarray,
    second_points_m: np.ndarray,
    first_covariance_m2: np.ndarray,
    second_covariance_m2: np.ndarray,
    valid_mask: np.ndarray,
    *,
    row_quantile: float,
    cross_covariance_m2: np.ndarray | None = None,
) -> CrossProviderScoreSummary:
    """Compute one case-level corroboration score without independence assumptions.

    When cross-provider covariance is unavailable, the score uses a PSD upper
    bound on ``Cov(e_first - e_second)`` obtained from Young's inequality.  This
    is conservative for unknown provider dependence and never substitutes an
    independence assumption merely because the provider implementations differ.
    """

    quantile = _probability(row_quantile, name="row_quantile", open_interval=True)
    (
        first_points,
        second_points,
        first_covariance,
        second_covariance,
        valid,
        cross_covariance,
    ) = _validated_matched_arrays(
        first_points_m,
        second_points_m,
        first_covariance_m2,
        second_covariance_m2,
        valid_mask,
        cross_covariance_m2,
    )
    mode = (
        EXPLICIT_CROSS_COVARIANCE_SEMANTICS
        if cross_covariance is not None
        else UNKNOWN_DEPENDENCE_COVARIANCE_SEMANTICS
    )
    if not np.allclose(
        first_covariance,
        np.swapaxes(first_covariance, -1, -2),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("first provider covariance must be symmetric")
    if not np.allclose(
        second_covariance,
        np.swapaxes(second_covariance, -1, -2),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("second provider covariance must be symmetric")
    try:
        np.linalg.cholesky(first_covariance)
        np.linalg.cholesky(second_covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("provider covariance must be positive definite") from error
    if cross_covariance is None:
        difference_covariance = _difference_covariance_unknown_dependence(
            first_covariance,
            second_covariance,
        )
    else:
        difference_covariance = (
            first_covariance
            + second_covariance
            - cross_covariance
            - np.swapaxes(cross_covariance, -1, -2)
        )
    difference_covariance = 0.5 * (
        difference_covariance + np.swapaxes(difference_covariance, -1, -2)
    )
    try:
        cholesky = np.linalg.cholesky(difference_covariance[valid])
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "cross-provider difference covariance must be positive definite"
        ) from error
    residual = (first_points - second_points)[valid]
    whitened = np.linalg.solve(cholesky, residual[..., None])[..., 0]
    squared = np.maximum(np.einsum("...i,...i->...", whitened, whitened), 0.0)
    row_scores = np.sqrt(squared / 3.0)
    valid_count = int(row_scores.size)
    return CrossProviderScoreSummary(
        covariance_mode=mode,
        row_count=int(valid.size),
        valid_count=valid_count,
        support_fraction=float(valid_count / valid.size),
        mean_row_score=float(np.mean(row_scores)),
        median_row_score=float(np.median(row_scores)),
        case_score=float(np.quantile(row_scores, quantile, method="higher")),
        maximum_row_score=float(np.max(row_scores)),
    )


@dataclass(frozen=True)
class CrossProviderCaseScoreV1:
    """One independent object/session score with complete provider provenance."""

    case_id: str
    sequence_id: str
    first_provider_contract_id: str
    second_provider_contract_id: str
    first_manifest_artifact_id: str
    second_manifest_artifact_id: str
    first_manifest_sha256: str
    second_manifest_sha256: str
    first_provider_run_id: str
    second_provider_run_id: str
    first_payload_ids: tuple[str, ...]
    second_payload_ids: tuple[str, ...]
    matched_observations_sha256: str
    alignment_artifact_id: str
    row_identity_sha256: str
    coordinate_frame_id: str
    shared_input_dependence_group_id: str
    covariance_mode: str
    row_count: int
    valid_count: int
    support_fraction: float
    mean_row_score: float
    median_row_score: float
    case_score: float
    maximum_row_score: float

    def __post_init__(self) -> None:
        for name in ("case_id", "sequence_id", "coordinate_frame_id"):
            object.__setattr__(
                self,
                name,
                require_exact_string(getattr(self, name), name=name),
            )
        for name in (
            "first_provider_contract_id",
            "second_provider_contract_id",
            "first_manifest_artifact_id",
            "second_manifest_artifact_id",
            "first_manifest_sha256",
            "second_manifest_sha256",
            "first_provider_run_id",
            "second_provider_run_id",
            "matched_observations_sha256",
            "alignment_artifact_id",
            "row_identity_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=name),
            )
        if type(self.first_payload_ids) is not tuple:
            raise TypeError("first_payload_ids must be a canonical tuple")
        if type(self.second_payload_ids) is not tuple:
            raise TypeError("second_payload_ids must be a canonical tuple")
        first_payloads = _strict_sha_sequence(
            self.first_payload_ids,
            name="first_payload_ids",
        )
        second_payloads = _strict_sha_sequence(
            self.second_payload_ids,
            name="second_payload_ids",
        )
        shared_group = require_exact_string(
            self.shared_input_dependence_group_id,
            name="shared_input_dependence_group_id",
        )
        if not shared_group.startswith("input-video:"):
            raise ValueError("shared dependence group must be an input-video identity")
        summary = CrossProviderScoreSummary(
            covariance_mode=self.covariance_mode,
            row_count=self.row_count,
            valid_count=self.valid_count,
            support_fraction=self.support_fraction,
            mean_row_score=self.mean_row_score,
            median_row_score=self.median_row_score,
            case_score=self.case_score,
            maximum_row_score=self.maximum_row_score,
        )
        object.__setattr__(self, "first_payload_ids", first_payloads)
        object.__setattr__(self, "second_payload_ids", second_payloads)
        object.__setattr__(self, "shared_input_dependence_group_id", shared_group)
        for name in (
            "covariance_mode",
            "row_count",
            "valid_count",
            "support_fraction",
            "mean_row_score",
            "median_row_score",
            "case_score",
            "maximum_row_score",
        ):
            object.__setattr__(self, name, getattr(summary, name))

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "sequence_id": self.sequence_id,
            "first_provider_contract_id": self.first_provider_contract_id,
            "second_provider_contract_id": self.second_provider_contract_id,
            "first_manifest_artifact_id": self.first_manifest_artifact_id,
            "second_manifest_artifact_id": self.second_manifest_artifact_id,
            "first_manifest_sha256": self.first_manifest_sha256,
            "second_manifest_sha256": self.second_manifest_sha256,
            "first_provider_run_id": self.first_provider_run_id,
            "second_provider_run_id": self.second_provider_run_id,
            "first_payload_ids": list(self.first_payload_ids),
            "second_payload_ids": list(self.second_payload_ids),
            "matched_observations_sha256": self.matched_observations_sha256,
            "alignment_artifact_id": self.alignment_artifact_id,
            "row_identity_sha256": self.row_identity_sha256,
            "coordinate_frame_id": self.coordinate_frame_id,
            "shared_input_dependence_group_id": (
                self.shared_input_dependence_group_id
            ),
            "covariance_mode": self.covariance_mode,
            "row_count": self.row_count,
            "valid_count": self.valid_count,
            "support_fraction": self.support_fraction,
            "mean_row_score": self.mean_row_score,
            "median_row_score": self.median_row_score,
            "case_score": self.case_score,
            "maximum_row_score": self.maximum_row_score,
        }

    @classmethod
    def from_record(cls, value: object) -> CrossProviderCaseScoreV1:
        mapping = require_mapping(value, name="cross-provider case score")
        require_exact_fields(mapping, _CASE_SCORE_FIELDS, name="case score")
        return cls(
            case_id=mapping["case_id"],
            sequence_id=mapping["sequence_id"],
            first_provider_contract_id=mapping["first_provider_contract_id"],
            second_provider_contract_id=mapping["second_provider_contract_id"],
            first_manifest_artifact_id=mapping["first_manifest_artifact_id"],
            second_manifest_artifact_id=mapping["second_manifest_artifact_id"],
            first_manifest_sha256=mapping["first_manifest_sha256"],
            second_manifest_sha256=mapping["second_manifest_sha256"],
            first_provider_run_id=mapping["first_provider_run_id"],
            second_provider_run_id=mapping["second_provider_run_id"],
            first_payload_ids=tuple(
                require_string_sequence(
                    mapping["first_payload_ids"],
                    name="first_payload_ids",
                )
            ),
            second_payload_ids=tuple(
                require_string_sequence(
                    mapping["second_payload_ids"],
                    name="second_payload_ids",
                )
            ),
            matched_observations_sha256=mapping["matched_observations_sha256"],
            alignment_artifact_id=mapping["alignment_artifact_id"],
            row_identity_sha256=mapping["row_identity_sha256"],
            coordinate_frame_id=mapping["coordinate_frame_id"],
            shared_input_dependence_group_id=mapping[
                "shared_input_dependence_group_id"
            ],
            covariance_mode=mapping["covariance_mode"],
            row_count=mapping["row_count"],
            valid_count=mapping["valid_count"],
            support_fraction=mapping["support_fraction"],
            mean_row_score=mapping["mean_row_score"],
            median_row_score=mapping["median_row_score"],
            case_score=mapping["case_score"],
            maximum_row_score=mapping["maximum_row_score"],
        )


@dataclass(frozen=True)
class EvaluatedCrossProviderPanel:
    """Internal verified panel with one score per independent case."""

    purpose: str
    source_sha256: str
    semantic_id: str
    first_provider: ProviderContractV1
    second_provider: ProviderContractV1
    covariance_mode: str
    row_quantile: float
    cases: tuple[CrossProviderCaseScoreV1, ...]
    metadata: Mapping[str, Any]


def _selected_payloads(
    manifest: PredictionProviderManifestV1,
    payload_ids: tuple[str, ...],
    *,
    name: str,
) -> tuple[PredictionPayloadDescriptorV1, ...]:
    by_id: dict[str, PredictionPayloadDescriptorV1] = {}
    for item in manifest.payloads:
        if item.payload_id is None:
            raise ValueError("selected provider payload ID is not materialized")
        by_id[item.payload_id] = item
    missing = sorted(set(payload_ids) - by_id.keys())
    if missing:
        raise ValueError(f"{name} refers to unknown payload IDs: {missing}")
    return tuple(by_id[item] for item in payload_ids)


def _common_dependence_groups(
    payloads: Sequence[PredictionPayloadDescriptorV1],
) -> set[str]:
    iterator = iter(payloads)
    first = next(iterator)
    common = set(first.dependence_group_ids)
    for payload in iterator:
        common.intersection_update(payload.dependence_group_ids)
    return common


def _shared_input_group(
    first_payloads: Sequence[PredictionPayloadDescriptorV1],
    second_payloads: Sequence[PredictionPayloadDescriptorV1],
) -> str:
    shared = _common_dependence_groups(first_payloads) & _common_dependence_groups(
        second_payloads
    )
    input_groups = sorted(group for group in shared if group.startswith("input-video:"))
    if len(input_groups) != 1:
        raise ValueError(
            "selected provider payloads must share exactly one input-video dependence group"
        )
    return input_groups[0]


def _selected_frame_ids(
    payloads: Sequence[PredictionPayloadDescriptorV1],
) -> tuple[int, ...]:
    return tuple(sorted({frame for payload in payloads for frame in payload.output_frame_ids}))


def _selected_view_ids(
    payloads: Sequence[PredictionPayloadDescriptorV1],
) -> tuple[str, ...]:
    return tuple(sorted({payload.view_id for payload in payloads}))


def _load_matched_observations(
    path: Path,
    *,
    expected_sha256: str,
    alignment_artifact_id: str,
    row_identity_sha256: str,
    coordinate_frame_id: str,
    row_quantile: float,
) -> CrossProviderScoreSummary:
    payload = path.read_bytes()
    if _sha256_bytes(payload) != expected_sha256:
        raise ValueError("matched-observation SHA-256 mismatch")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            fields = set(archive.files)
            missing = sorted(_MATCHED_REQUIRED_FIELDS - fields)
            extra = sorted(fields - _MATCHED_REQUIRED_FIELDS - _MATCHED_OPTIONAL_FIELDS)
            if missing or extra:
                raise ValueError(
                    "matched-observation fields changed; "
                    f"missing={missing}, extra={extra}"
                )
            embedded_alignment = np.asarray(archive["alignment_artifact_id"])
            embedded_rows = np.asarray(archive["row_identity_sha256"])
            embedded_frame = np.asarray(archive["coordinate_frame_id"])
            for value, expected, name in (
                (
                    embedded_alignment,
                    alignment_artifact_id,
                    "alignment_artifact_id",
                ),
                (embedded_rows, row_identity_sha256, "row_identity_sha256"),
                (embedded_frame, coordinate_frame_id, "coordinate_frame_id"),
            ):
                if value.shape != () or value.dtype.kind != "U":
                    raise ValueError(
                        f"matched-observation {name} must be a Unicode scalar"
                    )
                if str(value.item()) != expected:
                    raise ValueError(f"matched-observation {name} mismatch")
            cross = (
                archive["cross_covariance_m2"]
                if "cross_covariance_m2" in archive.files
                else None
            )
            return compute_cross_provider_score(
                archive["first_points_m"],
                archive["second_points_m"],
                archive["first_covariance_m2"],
                archive["second_covariance_m2"],
                archive["valid_mask"],
                row_quantile=row_quantile,
                cross_covariance_m2=cross,
            )
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"cannot load matched observations {path.name!r}") from error


def _required_source_only_metadata(value: object, *, name: str) -> Mapping[str, Any]:
    mapping = require_finite_json_mapping(value, name=name)
    for field_name in (
        "uses_truth",
        "uses_target_outcomes",
        "uses_downstream_physical_innovation",
        "alignment_uses_truth",
        "alignment_uses_downstream_physical_innovation",
    ):
        if mapping.get(field_name) is not False:
            raise ValueError(f"{name} must declare {field_name}=false")
    return mapping


def evaluate_cross_provider_panel(
    panel_path: str | Path,
    *,
    row_quantile: float,
    expected_purpose: str,
) -> EvaluatedCrossProviderPanel:
    """Verify manifests, payloads, matched bytes, and one score per panel case."""

    quantile = _probability(row_quantile, name="row_quantile", open_interval=True)
    if expected_purpose not in {"calibration", "target"}:
        raise ValueError("expected_purpose must be calibration or target")
    source = Path(panel_path).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError("cross-provider panel must be a regular non-symlink file")
    source_bytes = source.read_bytes()
    record = _load_json_bytes_object(
        source_bytes,
        name="cross-provider panel",
    )
    require_exact_fields(record, _PANEL_FIELDS, name="cross-provider panel")
    if record["schema"] != CROSS_PROVIDER_PANEL_SCHEMA:
        raise ValueError("unsupported cross-provider panel schema")
    if record["schema_version"] != CROSS_PROVIDER_PANEL_VERSION:
        raise ValueError("unsupported cross-provider panel version")
    purpose = require_exact_string(record["purpose"], name="panel purpose")
    if purpose != expected_purpose:
        raise ValueError(f"expected a {expected_purpose} panel")
    metadata = _required_source_only_metadata(
        record["metadata"],
        name="cross-provider panel metadata",
    )
    raw_cases = record["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cross-provider panel requires a nonempty cases array")

    root = source.parent
    first_contract: ProviderContractV1 | None = None
    second_contract: ProviderContractV1 | None = None
    covariance_mode: str | None = None
    evaluated: list[CrossProviderCaseScoreV1] = []
    seen_case_ids: set[str] = set()
    for case_index, raw_case in enumerate(raw_cases):
        case = require_mapping(raw_case, name=f"panel case {case_index}")
        require_exact_fields(case, _CASE_FIELDS, name=f"panel case {case_index}")
        case_id = require_exact_string(case["case_id"], name="case_id")
        if case_id in seen_case_ids:
            raise ValueError("cross-provider panel case IDs must be unique")
        seen_case_ids.add(case_id)
        first_manifest_path = _resolved_member(
            root,
            case["first_manifest"],
            name=f"case {case_id!r} first manifest",
        )
        second_manifest_path = _resolved_member(
            root,
            case["second_manifest"],
            name=f"case {case_id!r} second manifest",
        )
        first_manifest, _ = verify_prediction_provider_manifest(
            first_manifest_path,
            verify_payloads=True,
        )
        second_manifest, _ = verify_prediction_provider_manifest(
            second_manifest_path,
            verify_payloads=True,
        )
        if first_manifest.sequence_id != second_manifest.sequence_id:
            raise ValueError("provider manifests refer to different sequences")
        current_first = ProviderContractV1.from_manifest(first_manifest)
        current_second = ProviderContractV1.from_manifest(second_manifest)
        if current_first.contract_id == current_second.contract_id:
            raise ValueError(
                "alternative constructions or stochastic members from one provider "
                "must not be presented as distinct providers"
            )
        if first_contract is None:
            first_contract = current_first
            second_contract = current_second
        elif current_first != first_contract or current_second != second_contract:
            raise ValueError("provider contracts changed across panel cases")

        first_payload_ids = _strict_sha_sequence(
            case["first_payload_ids"],
            name="first_payload_ids",
        )
        second_payload_ids = _strict_sha_sequence(
            case["second_payload_ids"],
            name="second_payload_ids",
        )
        first_payloads = _selected_payloads(
            first_manifest,
            first_payload_ids,
            name="first_payload_ids",
        )
        second_payloads = _selected_payloads(
            second_manifest,
            second_payload_ids,
            name="second_payload_ids",
        )
        if _selected_frame_ids(first_payloads) != _selected_frame_ids(second_payloads):
            raise ValueError("selected provider payloads have different output-frame support")
        if _selected_view_ids(first_payloads) != _selected_view_ids(second_payloads):
            raise ValueError("selected provider payloads have different view identities")
        shared_input = _shared_input_group(first_payloads, second_payloads)
        matched_sha = require_sha256(
            case["matched_observations_sha256"],
            name="matched_observations_sha256",
        )
        alignment_artifact_id = require_sha256(
            case["alignment_artifact_id"],
            name="alignment_artifact_id",
        )
        row_identity_sha256 = require_sha256(
            case["row_identity_sha256"],
            name="row_identity_sha256",
        )
        coordinate_frame_id = require_exact_string(
            case["coordinate_frame_id"],
            name="coordinate_frame_id",
        )
        matched_path = _resolved_member(
            root,
            case["matched_observations"],
            name=f"case {case_id!r} matched observations",
        )
        score = _load_matched_observations(
            matched_path,
            expected_sha256=matched_sha,
            alignment_artifact_id=alignment_artifact_id,
            row_identity_sha256=row_identity_sha256,
            coordinate_frame_id=coordinate_frame_id,
            row_quantile=quantile,
        )
        if covariance_mode is None:
            covariance_mode = score.covariance_mode
        elif score.covariance_mode != covariance_mode:
            raise ValueError("cross-provider covariance mode changed across panel cases")
        first_artifact_id = first_manifest.artifact_id
        second_artifact_id = second_manifest.artifact_id
        if first_artifact_id is None or second_artifact_id is None:
            raise ValueError("provider manifest artifact IDs must be materialized")
        evaluated.append(
            CrossProviderCaseScoreV1(
                case_id=case_id,
                sequence_id=first_manifest.sequence_id,
                first_provider_contract_id=str(current_first.contract_id),
                second_provider_contract_id=str(current_second.contract_id),
                first_manifest_artifact_id=first_artifact_id,
                second_manifest_artifact_id=second_artifact_id,
                first_manifest_sha256=_sha256_file(first_manifest_path),
                second_manifest_sha256=_sha256_file(second_manifest_path),
                first_provider_run_id=first_manifest.provider_run_id,
                second_provider_run_id=second_manifest.provider_run_id,
                first_payload_ids=first_payload_ids,
                second_payload_ids=second_payload_ids,
                matched_observations_sha256=matched_sha,
                alignment_artifact_id=alignment_artifact_id,
                row_identity_sha256=row_identity_sha256,
                coordinate_frame_id=coordinate_frame_id,
                shared_input_dependence_group_id=shared_input,
                covariance_mode=score.covariance_mode,
                row_count=score.row_count,
                valid_count=score.valid_count,
                support_fraction=score.support_fraction,
                mean_row_score=score.mean_row_score,
                median_row_score=score.median_row_score,
                case_score=score.case_score,
                maximum_row_score=score.maximum_row_score,
            )
        )
    if first_contract is None or second_contract is None or covariance_mode is None:
        raise RuntimeError("cross-provider panel evaluation produced no cases")
    ordered_cases = tuple(sorted(evaluated, key=lambda item: item.case_id))
    semantic_record = {
        "schema": CROSS_PROVIDER_PANEL_SCHEMA,
        "schema_version": CROSS_PROVIDER_PANEL_VERSION,
        "purpose": purpose,
        "first_provider": first_contract.to_record(),
        "second_provider": second_contract.to_record(),
        "covariance_mode": covariance_mode,
        "score_semantics": CROSS_PROVIDER_SCORE_SEMANTICS,
        "row_quantile": quantile,
        "cases": [item.to_record() for item in ordered_cases],
        "metadata": plain_json(metadata),
    }
    return EvaluatedCrossProviderPanel(
        purpose=purpose,
        source_sha256=_sha256_bytes(source_bytes),
        semantic_id=_sha256_json(semantic_record),
        first_provider=first_contract,
        second_provider=second_contract,
        covariance_mode=covariance_mode,
        row_quantile=quantile,
        cases=ordered_cases,
        metadata=frozen_finite_json_mapping(
            metadata,
            name="cross-provider panel metadata",
        ),
    )


def _threshold_from_record(value: object) -> FiniteSampleUpperThreshold:
    mapping = require_mapping(value, name="finite-sample threshold")
    require_exact_fields(mapping, _THRESHOLD_FIELDS, name="finite-sample threshold")
    threshold = FiniteSampleUpperThreshold(
        semantics=mapping["semantics"],
        miscoverage=mapping["miscoverage"],
        calibration_count=mapping["calibration_count"],
        order_statistic_rank=mapping["order_statistic_rank"],
        threshold=mapping["threshold"],
        guaranteed_miscoverage_upper_bound=mapping[
            "guaranteed_miscoverage_upper_bound"
        ],
        canonical_scores_sha256=mapping["canonical_scores_sha256"],
    )
    if mapping["exchangeability_boundary"] != threshold.to_dict()[
        "exchangeability_boundary"
    ]:
        raise ValueError("finite-sample exchangeability boundary changed")
    return threshold


@dataclass(frozen=True)
class CrossProviderCalibrationV1:
    """Frozen clean-panel threshold for a fixed ordered provider pair."""

    first_provider: ProviderContractV1
    second_provider: ProviderContractV1
    covariance_mode: str
    row_quantile: float
    minimum_support_fraction: float
    calibration_panel_source_sha256: str
    calibration_panel_semantic_id: str
    calibration_cases: tuple[CrossProviderCaseScoreV1, ...]
    finite_sample_threshold: FiniteSampleUpperThreshold
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None
    score_semantics: str = CROSS_PROVIDER_SCORE_SEMANTICS
    claim_boundary: str = CROSS_PROVIDER_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if not isinstance(self.first_provider, ProviderContractV1) or not isinstance(
            self.second_provider,
            ProviderContractV1,
        ):
            raise TypeError("calibration provider contracts are invalid")
        if self.first_provider.contract_id == self.second_provider.contract_id:
            raise ValueError("cross-provider calibration requires distinct providers")
        if self.covariance_mode not in COVARIANCE_MODES:
            raise ValueError("unsupported cross-provider covariance mode")
        quantile = _probability(
            self.row_quantile,
            name="row_quantile",
            open_interval=True,
        )
        minimum_support = _probability(
            self.minimum_support_fraction,
            name="minimum_support_fraction",
        )
        source_sha = require_sha256(
            self.calibration_panel_source_sha256,
            name="calibration_panel_source_sha256",
        )
        semantic_id = require_sha256(
            self.calibration_panel_semantic_id,
            name="calibration_panel_semantic_id",
        )
        if type(self.calibration_cases) is not tuple or not self.calibration_cases:
            raise TypeError("calibration_cases must be a nonempty canonical tuple")
        cases = tuple(self.calibration_cases)
        if any(not isinstance(item, CrossProviderCaseScoreV1) for item in cases):
            raise TypeError("calibration_cases contain an invalid score")
        if tuple(sorted(item.case_id for item in cases)) != tuple(
            item.case_id for item in cases
        ):
            raise ValueError("calibration cases must be ordered by case_id")
        if len({item.case_id for item in cases}) != len(cases):
            raise ValueError("calibration case IDs must be unique")
        first_contract_id = self.first_provider.contract_id
        second_contract_id = self.second_provider.contract_id
        if first_contract_id is None or second_contract_id is None:
            raise ValueError("provider contract IDs must be materialized")
        if any(item.covariance_mode != self.covariance_mode for item in cases):
            raise ValueError("calibration cases mix covariance modes")
        if any(
            item.first_provider_contract_id != first_contract_id
            or item.second_provider_contract_id != second_contract_id
            for item in cases
        ):
            raise ValueError("calibration cases differ from the provider contracts")
        if any(item.support_fraction < minimum_support for item in cases):
            raise ValueError(
                "clean calibration case falls below minimum common support"
            )
        refit = fit_finite_sample_upper_threshold(
            np.asarray([item.case_score for item in cases], dtype=np.float64),
            miscoverage=self.finite_sample_threshold.miscoverage,
        )
        if refit.to_dict() != self.finite_sample_threshold.to_dict():
            raise ValueError("finite-sample threshold differs from calibration cases")
        if self.score_semantics != CROSS_PROVIDER_SCORE_SEMANTICS:
            raise ValueError("cross-provider score semantics changed")
        if self.claim_boundary != CROSS_PROVIDER_CLAIM_BOUNDARY:
            raise ValueError("cross-provider claim boundary changed")
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(
                self.metadata,
                name="cross-provider calibration metadata",
            ),
            name="cross-provider calibration metadata",
        )
        object.__setattr__(self, "row_quantile", quantile)
        object.__setattr__(self, "minimum_support_fraction", minimum_support)
        object.__setattr__(self, "calibration_panel_source_sha256", source_sha)
        object.__setattr__(self, "calibration_panel_semantic_id", semantic_id)
        object.__setattr__(self, "calibration_cases", cases)
        object.__setattr__(self, "metadata", metadata)
        expected = _sha256_json(self.identity_record())
        if self.artifact_id is not None and require_sha256(
            self.artifact_id,
            name="artifact_id",
        ) != expected:
            raise ValueError("cross-provider calibration artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": CROSS_PROVIDER_CALIBRATION_SCHEMA,
            "schema_version": CROSS_PROVIDER_CALIBRATION_VERSION,
            "first_provider": self.first_provider.to_record(),
            "second_provider": self.second_provider.to_record(),
            "covariance_mode": self.covariance_mode,
            "score_semantics": self.score_semantics,
            "row_quantile": self.row_quantile,
            "minimum_support_fraction": self.minimum_support_fraction,
            "calibration_panel_source_sha256": (
                self.calibration_panel_source_sha256
            ),
            "calibration_panel_semantic_id": self.calibration_panel_semantic_id,
            "calibration_cases": [item.to_record() for item in self.calibration_cases],
            "finite_sample_threshold": self.finite_sample_threshold.to_dict(),
            "metadata": plain_json(self.metadata),
            "claim_boundary": self.claim_boundary,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "artifact_id": self.artifact_id}

    @classmethod
    def from_record(cls, value: object) -> CrossProviderCalibrationV1:
        mapping = require_mapping(value, name="cross-provider calibration")
        require_exact_fields(mapping, _CALIBRATION_FIELDS, name="calibration")
        if mapping["schema"] != CROSS_PROVIDER_CALIBRATION_SCHEMA:
            raise ValueError("unsupported cross-provider calibration schema")
        if mapping["schema_version"] != CROSS_PROVIDER_CALIBRATION_VERSION:
            raise ValueError("unsupported cross-provider calibration version")
        raw_cases = mapping["calibration_cases"]
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("calibration_cases must be a nonempty JSON array")
        return cls(
            first_provider=ProviderContractV1.from_record(mapping["first_provider"]),
            second_provider=ProviderContractV1.from_record(mapping["second_provider"]),
            covariance_mode=mapping["covariance_mode"],
            score_semantics=mapping["score_semantics"],
            row_quantile=mapping["row_quantile"],
            minimum_support_fraction=mapping["minimum_support_fraction"],
            calibration_panel_source_sha256=mapping[
                "calibration_panel_source_sha256"
            ],
            calibration_panel_semantic_id=mapping[
                "calibration_panel_semantic_id"
            ],
            calibration_cases=tuple(
                CrossProviderCaseScoreV1.from_record(item) for item in raw_cases
            ),
            finite_sample_threshold=_threshold_from_record(
                mapping["finite_sample_threshold"]
            ),
            metadata=require_finite_json_mapping(
                mapping["metadata"],
                name="cross-provider calibration metadata",
            ),
            claim_boundary=mapping["claim_boundary"],
            artifact_id=mapping["artifact_id"],
        )


@dataclass(frozen=True)
class CrossProviderDecisionCaseV1:
    """One target case and the exact corroboration admission reasons."""

    score: CrossProviderCaseScoreV1
    admitted: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.score, CrossProviderCaseScoreV1):
            raise TypeError("decision case score is invalid")
        admitted = _exact_boolean(self.admitted, name="admitted")
        if type(self.rejection_reasons) is not tuple:
            raise TypeError("rejection_reasons must be a canonical tuple")
        reasons = require_string_sequence(
            self.rejection_reasons,
            name="rejection_reasons",
            allow_empty=True,
        )
        if len(set(reasons)) != len(reasons):
            raise ValueError("rejection_reasons must be unique")
        if admitted != (not reasons):
            raise ValueError("admitted must be true exactly when no rejection reason exists")
        object.__setattr__(self, "admitted", admitted)
        object.__setattr__(self, "rejection_reasons", reasons)

    def to_record(self) -> dict[str, object]:
        return {
            "score": self.score.to_record(),
            "admitted": self.admitted,
            "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_record(cls, value: object) -> CrossProviderDecisionCaseV1:
        mapping = require_mapping(value, name="cross-provider decision case")
        require_exact_fields(mapping, _DECISION_CASE_FIELDS, name="decision case")
        return cls(
            score=CrossProviderCaseScoreV1.from_record(mapping["score"]),
            admitted=mapping["admitted"],
            rejection_reasons=tuple(
                require_string_sequence(
                    mapping["rejection_reasons"],
                    name="rejection_reasons",
                    allow_empty=True,
                )
            ),
        )


def _rejection_reasons(
    score: CrossProviderCaseScoreV1,
    *,
    threshold: float,
    minimum_support_fraction: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if score.support_fraction < minimum_support_fraction:
        reasons.append("insufficient-common-support")
    if score.case_score > threshold:
        reasons.append("cross-provider-disagreement")
    return tuple(reasons)


@dataclass(frozen=True)
class CrossProviderDecisionV1:
    """Target-panel corroboration decisions under one frozen calibration."""

    calibration_artifact_id: str
    first_provider: ProviderContractV1
    second_provider: ProviderContractV1
    covariance_mode: str
    row_quantile: float
    minimum_support_fraction: float
    threshold: float
    target_panel_source_sha256: str
    target_panel_semantic_id: str
    cases: tuple[CrossProviderDecisionCaseV1, ...]
    accepted_count: int
    rejected_count: int
    all_cases_admitted: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None
    score_semantics: str = CROSS_PROVIDER_SCORE_SEMANTICS
    claim_boundary: str = CROSS_PROVIDER_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        calibration_id = require_sha256(
            self.calibration_artifact_id,
            name="calibration_artifact_id",
        )
        if not isinstance(self.first_provider, ProviderContractV1) or not isinstance(
            self.second_provider,
            ProviderContractV1,
        ):
            raise TypeError("decision provider contracts are invalid")
        if self.first_provider.contract_id == self.second_provider.contract_id:
            raise ValueError("cross-provider decision requires distinct providers")
        if self.covariance_mode not in COVARIANCE_MODES:
            raise ValueError("unsupported cross-provider covariance mode")
        quantile = _probability(
            self.row_quantile,
            name="row_quantile",
            open_interval=True,
        )
        minimum_support = _probability(
            self.minimum_support_fraction,
            name="minimum_support_fraction",
        )
        threshold = _finite_real(self.threshold, name="threshold", minimum=0.0)
        source_sha = require_sha256(
            self.target_panel_source_sha256,
            name="target_panel_source_sha256",
        )
        semantic_id = require_sha256(
            self.target_panel_semantic_id,
            name="target_panel_semantic_id",
        )
        if type(self.cases) is not tuple or not self.cases:
            raise TypeError("decision cases must be a nonempty canonical tuple")
        cases = tuple(self.cases)
        if any(not isinstance(item, CrossProviderDecisionCaseV1) for item in cases):
            raise TypeError("decision cases contain an invalid value")
        if tuple(sorted(item.score.case_id for item in cases)) != tuple(
            item.score.case_id for item in cases
        ):
            raise ValueError("decision cases must be ordered by case_id")
        if len({item.score.case_id for item in cases}) != len(cases):
            raise ValueError("decision case IDs must be unique")
        first_contract_id = self.first_provider.contract_id
        second_contract_id = self.second_provider.contract_id
        if first_contract_id is None or second_contract_id is None:
            raise ValueError("provider contract IDs must be materialized")
        for item in cases:
            if item.score.covariance_mode != self.covariance_mode:
                raise ValueError("decision cases mix covariance modes")
            if (
                item.score.first_provider_contract_id != first_contract_id
                or item.score.second_provider_contract_id != second_contract_id
            ):
                raise ValueError("decision cases differ from the provider contracts")
            expected_reasons = _rejection_reasons(
                item.score,
                threshold=threshold,
                minimum_support_fraction=minimum_support,
            )
            if item.rejection_reasons != expected_reasons:
                raise ValueError("decision rejection reasons contradict the frozen guard")
        accepted = sum(item.admitted for item in cases)
        rejected = len(cases) - accepted
        if require_exact_integer(
            self.accepted_count,
            name="accepted_count",
            minimum=0,
        ) != accepted:
            raise ValueError("accepted_count differs from decision cases")
        if require_exact_integer(
            self.rejected_count,
            name="rejected_count",
            minimum=0,
        ) != rejected:
            raise ValueError("rejected_count differs from decision cases")
        all_admitted = _exact_boolean(
            self.all_cases_admitted,
            name="all_cases_admitted",
        )
        if all_admitted != (rejected == 0):
            raise ValueError("all_cases_admitted contradicts the decision cases")
        if self.score_semantics != CROSS_PROVIDER_SCORE_SEMANTICS:
            raise ValueError("cross-provider score semantics changed")
        if self.claim_boundary != CROSS_PROVIDER_CLAIM_BOUNDARY:
            raise ValueError("cross-provider claim boundary changed")
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(
                self.metadata,
                name="cross-provider decision metadata",
            ),
            name="cross-provider decision metadata",
        )
        object.__setattr__(self, "calibration_artifact_id", calibration_id)
        object.__setattr__(self, "row_quantile", quantile)
        object.__setattr__(self, "minimum_support_fraction", minimum_support)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "target_panel_source_sha256", source_sha)
        object.__setattr__(self, "target_panel_semantic_id", semantic_id)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "accepted_count", accepted)
        object.__setattr__(self, "rejected_count", rejected)
        object.__setattr__(self, "all_cases_admitted", all_admitted)
        object.__setattr__(self, "metadata", metadata)
        expected = _sha256_json(self.identity_record())
        if self.artifact_id is not None and require_sha256(
            self.artifact_id,
            name="artifact_id",
        ) != expected:
            raise ValueError("cross-provider decision artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": CROSS_PROVIDER_DECISION_SCHEMA,
            "schema_version": CROSS_PROVIDER_DECISION_VERSION,
            "calibration_artifact_id": self.calibration_artifact_id,
            "first_provider": self.first_provider.to_record(),
            "second_provider": self.second_provider.to_record(),
            "covariance_mode": self.covariance_mode,
            "score_semantics": self.score_semantics,
            "row_quantile": self.row_quantile,
            "minimum_support_fraction": self.minimum_support_fraction,
            "threshold": self.threshold,
            "target_panel_source_sha256": self.target_panel_source_sha256,
            "target_panel_semantic_id": self.target_panel_semantic_id,
            "cases": [item.to_record() for item in self.cases],
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "all_cases_admitted": self.all_cases_admitted,
            "metadata": plain_json(self.metadata),
            "claim_boundary": self.claim_boundary,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "artifact_id": self.artifact_id}

    @classmethod
    def from_record(cls, value: object) -> CrossProviderDecisionV1:
        mapping = require_mapping(value, name="cross-provider decision")
        require_exact_fields(mapping, _DECISION_FIELDS, name="decision")
        if mapping["schema"] != CROSS_PROVIDER_DECISION_SCHEMA:
            raise ValueError("unsupported cross-provider decision schema")
        if mapping["schema_version"] != CROSS_PROVIDER_DECISION_VERSION:
            raise ValueError("unsupported cross-provider decision version")
        raw_cases = mapping["cases"]
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("decision cases must be a nonempty JSON array")
        return cls(
            calibration_artifact_id=mapping["calibration_artifact_id"],
            first_provider=ProviderContractV1.from_record(mapping["first_provider"]),
            second_provider=ProviderContractV1.from_record(mapping["second_provider"]),
            covariance_mode=mapping["covariance_mode"],
            score_semantics=mapping["score_semantics"],
            row_quantile=mapping["row_quantile"],
            minimum_support_fraction=mapping["minimum_support_fraction"],
            threshold=mapping["threshold"],
            target_panel_source_sha256=mapping["target_panel_source_sha256"],
            target_panel_semantic_id=mapping["target_panel_semantic_id"],
            cases=tuple(CrossProviderDecisionCaseV1.from_record(item) for item in raw_cases),
            accepted_count=mapping["accepted_count"],
            rejected_count=mapping["rejected_count"],
            all_cases_admitted=mapping["all_cases_admitted"],
            metadata=require_finite_json_mapping(
                mapping["metadata"],
                name="cross-provider decision metadata",
            ),
            claim_boundary=mapping["claim_boundary"],
            artifact_id=mapping["artifact_id"],
        )


def fit_cross_provider_calibration(
    panel: EvaluatedCrossProviderPanel,
    *,
    miscoverage: float,
    row_quantile: float,
    minimum_support_fraction: float,
) -> CrossProviderCalibrationV1:
    if panel.purpose != "calibration":
        raise ValueError("cross-provider calibration requires a calibration panel")
    quantile = _probability(row_quantile, name="row_quantile", open_interval=True)
    if not math.isclose(
        panel.row_quantile,
        quantile,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("calibration row_quantile differs from panel evaluation")
    minimum_support = _probability(
        minimum_support_fraction,
        name="minimum_support_fraction",
    )
    if any(item.support_fraction < minimum_support for item in panel.cases):
        raise ValueError("clean calibration panel contains insufficient-support cases")
    threshold = fit_finite_sample_upper_threshold(
        np.asarray([item.case_score for item in panel.cases], dtype=np.float64),
        miscoverage=miscoverage,
    )
    return CrossProviderCalibrationV1(
        first_provider=panel.first_provider,
        second_provider=panel.second_provider,
        covariance_mode=panel.covariance_mode,
        row_quantile=quantile,
        minimum_support_fraction=minimum_support,
        calibration_panel_source_sha256=panel.source_sha256,
        calibration_panel_semantic_id=panel.semantic_id,
        calibration_cases=panel.cases,
        finite_sample_threshold=threshold,
        metadata={
            "panel_metadata": plain_json(panel.metadata),
            "payloads_verified": True,
            "uses_truth": False,
            "uses_target_outcomes": False,
            "uses_downstream_physical_innovation": False,
            "alignment_uses_truth": False,
            "alignment_uses_downstream_physical_innovation": False,
        },
    )


def apply_cross_provider_calibration(
    calibration: CrossProviderCalibrationV1,
    panel: EvaluatedCrossProviderPanel,
) -> CrossProviderDecisionV1:
    if panel.purpose != "target":
        raise ValueError("cross-provider evaluation requires a target panel")
    if panel.first_provider != calibration.first_provider or panel.second_provider != (
        calibration.second_provider
    ):
        raise ValueError("target provider contract differs from calibration")
    if panel.covariance_mode != calibration.covariance_mode:
        raise ValueError("target covariance mode differs from calibration")
    if not math.isclose(
        panel.row_quantile,
        calibration.row_quantile,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("target row_quantile differs from calibration")
    threshold = calibration.finite_sample_threshold.threshold
    decision_cases: list[CrossProviderDecisionCaseV1] = []
    for score in panel.cases:
        reasons = _rejection_reasons(
            score,
            threshold=threshold,
            minimum_support_fraction=calibration.minimum_support_fraction,
        )
        decision_cases.append(
            CrossProviderDecisionCaseV1(
                score=score,
                admitted=not reasons,
                rejection_reasons=reasons,
            )
        )
    cases = tuple(decision_cases)
    accepted = sum(item.admitted for item in cases)
    calibration_artifact_id = calibration.artifact_id
    if calibration_artifact_id is None:
        raise ValueError("calibration artifact ID is not materialized")
    return CrossProviderDecisionV1(
        calibration_artifact_id=calibration_artifact_id,
        first_provider=calibration.first_provider,
        second_provider=calibration.second_provider,
        covariance_mode=calibration.covariance_mode,
        row_quantile=calibration.row_quantile,
        minimum_support_fraction=calibration.minimum_support_fraction,
        threshold=threshold,
        target_panel_source_sha256=panel.source_sha256,
        target_panel_semantic_id=panel.semantic_id,
        cases=cases,
        accepted_count=accepted,
        rejected_count=len(cases) - accepted,
        all_cases_admitted=accepted == len(cases),
        metadata={
            "panel_metadata": plain_json(panel.metadata),
            "payloads_verified": True,
            "uses_truth": False,
            "uses_target_outcomes": False,
            "uses_downstream_physical_innovation": False,
            "alignment_uses_truth": False,
            "alignment_uses_downstream_physical_innovation": False,
        },
    )


def save_cross_provider_calibration(
    path: str | Path,
    calibration: CrossProviderCalibrationV1,
) -> Path:
    return _write_json_once(Path(path), calibration.to_record())


def load_cross_provider_calibration(path: str | Path) -> CrossProviderCalibrationV1:
    return CrossProviderCalibrationV1.from_record(
        load_json_object(path, name="cross-provider calibration")
    )


def save_cross_provider_decision(
    path: str | Path,
    decision: CrossProviderDecisionV1,
) -> Path:
    return _write_json_once(Path(path), decision.to_record())


def load_cross_provider_decision(path: str | Path) -> CrossProviderDecisionV1:
    return CrossProviderDecisionV1.from_record(
        load_json_object(path, name="cross-provider decision")
    )


def _simulate_case_score(
    generator: np.random.Generator,
    *,
    rows: int,
    noise_std_m: float,
    provider_specific_bias_std_m: float,
    shared_bias_std_m: float,
    row_quantile: float,
) -> float:
    truth = generator.normal(0.0, 0.05, size=(rows, 3))
    shared_bias = generator.normal(0.0, shared_bias_std_m, size=(1, 3))
    first = truth + shared_bias + generator.normal(0.0, noise_std_m, size=(rows, 3))
    provider_bias = generator.normal(0.0, provider_specific_bias_std_m, size=(1, 3))
    second = (
        truth
        + shared_bias
        + provider_bias
        + generator.normal(0.0, noise_std_m, size=(rows, 3))
    )
    covariance = np.repeat(
        (noise_std_m**2 * np.eye(3, dtype=np.float64))[None],
        rows,
        axis=0,
    )
    score = compute_cross_provider_score(
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
        covariance,
        covariance,
        np.ones(rows, dtype=bool),
        row_quantile=row_quantile,
    )
    return score.case_score


def run_cross_provider_guard_stress(
    *,
    calibration_cases: int = 800,
    clean_target_cases: int = 1000,
    corrupted_target_cases: int = 1000,
    shared_bias_target_cases: int = 1000,
    rows_per_case: int = 256,
    noise_std_m: float = 0.003,
    provider_specific_bias_std_m: float = 0.015,
    shared_bias_std_m: float = 0.015,
    row_quantile: float = 0.95,
    miscoverage: float = 0.05,
    seed: int = 20260806,
) -> dict[str, object]:
    """Run a fresh-seed mechanism stress with an explicit common-bias limitation."""

    for name, value in {
        "calibration_cases": calibration_cases,
        "clean_target_cases": clean_target_cases,
        "corrupted_target_cases": corrupted_target_cases,
        "shared_bias_target_cases": shared_bias_target_cases,
        "rows_per_case": rows_per_case,
    }.items():
        require_exact_integer(value, name=name, minimum=1)
    require_exact_integer(seed, name="seed", minimum=0)
    noise = _finite_real(noise_std_m, name="noise_std_m", strictly_positive=True)
    provider_bias = _finite_real(
        provider_specific_bias_std_m,
        name="provider_specific_bias_std_m",
        strictly_positive=True,
    )
    shared_bias = _finite_real(
        shared_bias_std_m,
        name="shared_bias_std_m",
        strictly_positive=True,
    )
    quantile = _probability(row_quantile, name="row_quantile", open_interval=True)
    alpha = _probability(miscoverage, name="miscoverage", open_interval=True)
    generator = np.random.default_rng(seed)

    def panel(count: int, *, provider_bias_std: float, common_bias_std: float) -> np.ndarray:
        return np.asarray(
            [
                _simulate_case_score(
                    generator,
                    rows=rows_per_case,
                    noise_std_m=noise,
                    provider_specific_bias_std_m=provider_bias_std,
                    shared_bias_std_m=common_bias_std,
                    row_quantile=quantile,
                )
                for _ in range(count)
            ],
            dtype=np.float64,
        )

    calibration_scores = panel(
        calibration_cases,
        provider_bias_std=0.0,
        common_bias_std=shared_bias,
    )
    threshold = fit_finite_sample_upper_threshold(
        calibration_scores,
        miscoverage=alpha,
    )
    clean_scores = panel(
        clean_target_cases,
        provider_bias_std=0.0,
        common_bias_std=shared_bias,
    )
    corrupted_scores = panel(
        corrupted_target_cases,
        provider_bias_std=provider_bias,
        common_bias_std=shared_bias,
    )
    common_bias_scores = panel(
        shared_bias_target_cases,
        provider_bias_std=0.0,
        common_bias_std=shared_bias * 3.0,
    )
    limit = threshold.threshold
    clean_rejection = float(np.mean(clean_scores > limit))
    corrupted_detection = float(np.mean(corrupted_scores > limit))
    common_bias_rejection = float(np.mean(common_bias_scores > limit))
    tolerance = 0.03
    gates = {
        "clean_false_rejection_at_most_alpha_plus_0_03": (
            clean_rejection <= alpha + tolerance
        ),
        "provider_specific_corruption_detection_at_least_0_95": (
            corrupted_detection >= 0.95
        ),
        "shared_common_bias_not_misrepresented_as_detected": (
            common_bias_rejection <= alpha + tolerance
        ),
    }
    report: dict[str, object] = {
        "schema": "prob4d.cross-provider-corroboration-stress.v1",
        "configuration": {
            "calibration_cases": calibration_cases,
            "clean_target_cases": clean_target_cases,
            "corrupted_target_cases": corrupted_target_cases,
            "shared_bias_target_cases": shared_bias_target_cases,
            "rows_per_case": rows_per_case,
            "noise_std_m": noise,
            "provider_specific_bias_std_m": provider_bias,
            "shared_bias_std_m": shared_bias,
            "row_quantile": quantile,
            "miscoverage": alpha,
            "seed": seed,
        },
        "finite_sample_threshold": threshold.to_dict(),
        "results": {
            "clean_false_rejection_rate": clean_rejection,
            "provider_specific_corruption_detection_rate": corrupted_detection,
            "shared_common_bias_rejection_rate": common_bias_rejection,
            "calibration_score_mean": float(np.mean(calibration_scores)),
            "clean_score_mean": float(np.mean(clean_scores)),
            "corrupted_score_mean": float(np.mean(corrupted_scores)),
            "shared_common_bias_score_mean": float(np.mean(common_bias_scores)),
        },
        "gates": gates,
        "decision": (
            "pass-controlled-cross-provider-corroboration-mechanism"
            if all(gates.values())
            else "controlled-cross-provider-mechanism-gate-failed"
        ),
        "claim_boundary": CROSS_PROVIDER_CLAIM_BOUNDARY,
    }
    report["artifact_id"] = _sha256_json(report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d diagnostic cross-provider-guard",
        description=(
            "calibrate and evaluate source-only corroboration for two distinct "
            "provider-neutral prediction contracts"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("panel")
    calibrate.add_argument("output")
    calibrate.add_argument("--miscoverage", type=float, default=0.05)
    calibrate.add_argument("--row-quantile", type=float, default=0.95)
    calibrate.add_argument("--minimum-support-fraction", type=float, default=0.8)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("calibration")
    evaluate.add_argument("panel")
    evaluate.add_argument("output")

    verify_calibration = subparsers.add_parser("verify-calibration")
    verify_calibration.add_argument("calibration")

    verify_decision = subparsers.add_parser("verify-decision")
    verify_decision.add_argument("decision")
    verify_decision.add_argument("--calibration", required=True)

    stress = subparsers.add_parser("stress")
    stress.add_argument("--output", required=True)
    stress.add_argument("--calibration-cases", type=int, default=800)
    stress.add_argument("--clean-target-cases", type=int, default=1000)
    stress.add_argument("--corrupted-target-cases", type=int, default=1000)
    stress.add_argument("--shared-bias-target-cases", type=int, default=1000)
    stress.add_argument("--rows-per-case", type=int, default=256)
    stress.add_argument("--seed", type=int, default=20260806)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    if arguments.command == "calibrate":
        panel = evaluate_cross_provider_panel(
            arguments.panel,
            row_quantile=arguments.row_quantile,
            expected_purpose="calibration",
        )
        calibration = fit_cross_provider_calibration(
            panel,
            miscoverage=arguments.miscoverage,
            row_quantile=arguments.row_quantile,
            minimum_support_fraction=arguments.minimum_support_fraction,
        )
        save_cross_provider_calibration(arguments.output, calibration)
        print(
            json.dumps(
                {
                    "artifact_id": calibration.artifact_id,
                    "calibration_case_count": len(calibration.calibration_cases),
                    "threshold": calibration.finite_sample_threshold.threshold,
                    "guaranteed_miscoverage_upper_bound": (
                        calibration.finite_sample_threshold.guaranteed_miscoverage_upper_bound
                    ),
                    "covariance_mode": calibration.covariance_mode,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "evaluate":
        calibration = load_cross_provider_calibration(arguments.calibration)
        panel = evaluate_cross_provider_panel(
            arguments.panel,
            row_quantile=calibration.row_quantile,
            expected_purpose="target",
        )
        decision = apply_cross_provider_calibration(calibration, panel)
        save_cross_provider_decision(arguments.output, decision)
        print(
            json.dumps(
                {
                    "artifact_id": decision.artifact_id,
                    "accepted_count": decision.accepted_count,
                    "rejected_count": decision.rejected_count,
                    "all_cases_admitted": decision.all_cases_admitted,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "verify-calibration":
        calibration = load_cross_provider_calibration(arguments.calibration)
        print(
            json.dumps(
                {
                    "artifact_id": calibration.artifact_id,
                    "calibration_case_count": len(calibration.calibration_cases),
                    "threshold": calibration.finite_sample_threshold.threshold,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "verify-decision":
        decision = load_cross_provider_decision(arguments.decision)
        if arguments.calibration is not None:
            calibration = load_cross_provider_calibration(arguments.calibration)
            if decision.calibration_artifact_id != calibration.artifact_id:
                raise ValueError("decision references a different calibration artifact")
            if decision.first_provider != calibration.first_provider or (
                decision.second_provider != calibration.second_provider
            ):
                raise ValueError("decision provider contracts differ from calibration")
            if not math.isclose(
                decision.threshold,
                calibration.finite_sample_threshold.threshold,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError("decision threshold differs from calibration")
        print(
            json.dumps(
                {
                    "artifact_id": decision.artifact_id,
                    "accepted_count": decision.accepted_count,
                    "rejected_count": decision.rejected_count,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "stress":
        report = run_cross_provider_guard_stress(
            calibration_cases=arguments.calibration_cases,
            clean_target_cases=arguments.clean_target_cases,
            corrupted_target_cases=arguments.corrupted_target_cases,
            shared_bias_target_cases=arguments.shared_bias_target_cases,
            rows_per_case=arguments.rows_per_case,
            seed=arguments.seed,
        )
        _write_json_once(Path(arguments.output), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        gates = require_mapping(report["gates"], name="stress gates")
        return 0 if all(bool(value) for value in gates.values()) else 3
    raise RuntimeError("unsupported cross-provider guard command")


__all__ = [
    "COVARIANCE_MODES",
    "CROSS_PROVIDER_CALIBRATION_SCHEMA",
    "CROSS_PROVIDER_CLAIM_BOUNDARY",
    "CROSS_PROVIDER_DECISION_SCHEMA",
    "CROSS_PROVIDER_PANEL_SCHEMA",
    "CROSS_PROVIDER_SCORE_SEMANTICS",
    "CrossProviderCalibrationV1",
    "CrossProviderCaseScoreV1",
    "CrossProviderDecisionCaseV1",
    "CrossProviderDecisionV1",
    "CrossProviderScoreSummary",
    "EvaluatedCrossProviderPanel",
    "ProviderContractV1",
    "apply_cross_provider_calibration",
    "compute_cross_provider_score",
    "evaluate_cross_provider_panel",
    "fit_cross_provider_calibration",
    "load_cross_provider_calibration",
    "load_cross_provider_decision",
    "main",
    "run_cross_provider_guard_stress",
    "save_cross_provider_calibration",
    "save_cross_provider_decision",
]


if __name__ == "__main__":
    raise SystemExit(main())
