"""Causal, content-addressed preflight for provider prediction batches.

The preflight inspects only payloads admitted by an exclusive causal cutoff.  It
turns ordinary batch incompatibilities into a structured artifact before a
source or target scorer attempts to stack arrays.  Integrity failures remain
exceptions: malformed or changed bytes are not scientific negative results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from .data import PredictionWindow
from .prediction_provider_manifest import load_prediction_provider_manifest

PREDICTION_BATCH_PREFLIGHT_SCHEMA: Final = "prob4d.prediction-batch-preflight"
PREDICTION_BATCH_PREFLIGHT_VERSION: Final = 1
PREDICTION_BATCH_POLICY_SCHEMA: Final = "prob4d.prediction-batch-policy"
PREDICTION_BATCH_POLICY_VERSION: Final = 1

_STATUS_PASS: Final = "pass"
_STATUS_INCOMPATIBLE: Final = "batch-incompatible"
_VALID_STATUSES: Final = frozenset({_STATUS_PASS, _STATUS_INCOMPATIBLE})


class PredictionBatchIntegrityError(ValueError):
    """Raised when identity-bearing input bytes are missing, changed, or malformed."""


def _canonical_json_bytes(record: Mapping[str, object]) -> bytes:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact_id(record: Mapping[str, object]) -> str:
    payload = dict(record)
    payload.pop("artifact_id", None)
    return _sha256_bytes(_canonical_json_bytes(payload))


def _strict_json_object(path: Path, *, name: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise ValueError(f"{name} contains non-finite number {token!r}")

    if path.is_symlink():
        raise ValueError(f"{name} is a symbolic link")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {name}") from error
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must contain valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _require_exact_fields(
    record: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    actual = frozenset(record)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields differ: missing={missing}, extra={extra}")


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_integer(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    result = tuple(_require_string(item, name=f"{name} item") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _require_shape(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return tuple(_require_integer(item, name=f"{name} item", minimum=0) for item in value)


@dataclass(frozen=True)
class PredictionBatchPolicyV1:
    """Prospectively declared compatibility requirements for one scorer batch."""

    require_nonempty: bool = True
    require_common_frame_count: bool = True
    require_common_spatial_shape: bool = True
    require_common_point_dtype: bool = True
    require_common_optional_fields: bool = True

    def to_record(self) -> dict[str, object]:
        return {
            "schema": PREDICTION_BATCH_POLICY_SCHEMA,
            "schema_version": PREDICTION_BATCH_POLICY_VERSION,
            "require_nonempty": self.require_nonempty,
            "require_common_frame_count": self.require_common_frame_count,
            "require_common_spatial_shape": self.require_common_spatial_shape,
            "require_common_point_dtype": self.require_common_point_dtype,
            "require_common_optional_fields": self.require_common_optional_fields,
        }

    @classmethod
    def from_record(cls, value: object) -> PredictionBatchPolicyV1:
        if not isinstance(value, Mapping):
            raise ValueError("prediction batch policy must be an object")
        expected = frozenset(
            {
                "schema",
                "schema_version",
                "require_nonempty",
                "require_common_frame_count",
                "require_common_spatial_shape",
                "require_common_point_dtype",
                "require_common_optional_fields",
            }
        )
        _require_exact_fields(value, expected, name="prediction batch policy")
        if value["schema"] != PREDICTION_BATCH_POLICY_SCHEMA:
            raise ValueError("unsupported prediction batch policy schema")
        if value["schema_version"] != PREDICTION_BATCH_POLICY_VERSION:
            raise ValueError("unsupported prediction batch policy version")
        return cls(
            require_nonempty=_require_bool(
                value["require_nonempty"], name="require_nonempty"
            ),
            require_common_frame_count=_require_bool(
                value["require_common_frame_count"],
                name="require_common_frame_count",
            ),
            require_common_spatial_shape=_require_bool(
                value["require_common_spatial_shape"],
                name="require_common_spatial_shape",
            ),
            require_common_point_dtype=_require_bool(
                value["require_common_point_dtype"],
                name="require_common_point_dtype",
            ),
            require_common_optional_fields=_require_bool(
                value["require_common_optional_fields"],
                name="require_common_optional_fields",
            ),
        )


@dataclass(frozen=True)
class PredictionBatchEntryV1:
    """Exact payload signature opened during causal preflight."""

    payload_id: str
    window_id: str
    relative_path: str
    output_frame_ids: tuple[int, ...]
    point_shape: tuple[int, ...]
    point_dtype: str
    scene_flow_shape: tuple[int, ...] | None
    ray_shape: tuple[int, ...] | None
    dense_storage_dtype: str
    payload_sha256: str
    payload_byte_count: int

    def __post_init__(self) -> None:
        _require_string(self.payload_id, name="payload_id")
        _require_string(self.window_id, name="window_id")
        _require_string(self.relative_path, name="relative_path")
        _require_string(self.point_dtype, name="point_dtype")
        _require_string(self.dense_storage_dtype, name="dense_storage_dtype")
        if len(self.point_shape) < 2 or self.point_shape[-1] != 3:
            raise ValueError("point_shape must end in coordinate dimension 3")
        if len(self.output_frame_ids) != self.point_shape[0]:
            raise ValueError("output frame count differs from point tensor")
        if len(set(self.output_frame_ids)) != len(self.output_frame_ids):
            raise ValueError("output_frame_ids must be unique")
        if self.scene_flow_shape is not None and self.scene_flow_shape != self.point_shape:
            raise ValueError("scene-flow shape must equal point shape")
        if self.ray_shape is not None and self.ray_shape != self.point_shape:
            raise ValueError("ray shape must equal point shape")
        if len(self.payload_sha256) != 64:
            raise ValueError("payload_sha256 must have 64 hexadecimal characters")
        try:
            int(self.payload_sha256, 16)
        except ValueError as error:
            raise ValueError("payload_sha256 must be hexadecimal") from error
        if self.payload_byte_count < 0:
            raise ValueError("payload_byte_count must be nonnegative")

    @property
    def frame_count(self) -> int:
        return self.point_shape[0]

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self.point_shape[1:-1]

    @property
    def optional_field_signature(self) -> tuple[bool, bool]:
        return self.scene_flow_shape is not None, self.ray_shape is not None

    def to_record(self) -> dict[str, object]:
        return {
            "payload_id": self.payload_id,
            "window_id": self.window_id,
            "relative_path": self.relative_path,
            "output_frame_ids": list(self.output_frame_ids),
            "point_shape": list(self.point_shape),
            "point_dtype": self.point_dtype,
            "scene_flow_shape": (
                None if self.scene_flow_shape is None else list(self.scene_flow_shape)
            ),
            "ray_shape": None if self.ray_shape is None else list(self.ray_shape),
            "dense_storage_dtype": self.dense_storage_dtype,
            "payload_sha256": self.payload_sha256,
            "payload_byte_count": self.payload_byte_count,
        }

    @classmethod
    def from_record(cls, value: object) -> PredictionBatchEntryV1:
        if not isinstance(value, Mapping):
            raise ValueError("prediction batch entry must be an object")
        expected = frozenset(
            {
                "payload_id",
                "window_id",
                "relative_path",
                "output_frame_ids",
                "point_shape",
                "point_dtype",
                "scene_flow_shape",
                "ray_shape",
                "dense_storage_dtype",
                "payload_sha256",
                "payload_byte_count",
            }
        )
        _require_exact_fields(value, expected, name="prediction batch entry")
        output_frame_ids = _require_shape(
            value["output_frame_ids"], name="output_frame_ids"
        )
        scene_shape_value = value["scene_flow_shape"]
        ray_shape_value = value["ray_shape"]
        return cls(
            payload_id=_require_string(value["payload_id"], name="payload_id"),
            window_id=_require_string(value["window_id"], name="window_id"),
            relative_path=_require_string(value["relative_path"], name="relative_path"),
            output_frame_ids=output_frame_ids,
            point_shape=_require_shape(value["point_shape"], name="point_shape"),
            point_dtype=_require_string(value["point_dtype"], name="point_dtype"),
            scene_flow_shape=(
                None
                if scene_shape_value is None
                else _require_shape(scene_shape_value, name="scene_flow_shape")
            ),
            ray_shape=(
                None
                if ray_shape_value is None
                else _require_shape(ray_shape_value, name="ray_shape")
            ),
            dense_storage_dtype=_require_string(
                value["dense_storage_dtype"], name="dense_storage_dtype"
            ),
            payload_sha256=_require_string(
                value["payload_sha256"], name="payload_sha256"
            ),
            payload_byte_count=_require_integer(
                value["payload_byte_count"], name="payload_byte_count", minimum=0
            ),
        )


@dataclass(frozen=True)
class PredictionBatchViolationV1:
    """One deterministic scorer-compatibility violation."""

    code: str
    field_name: str
    reference_payload_id: str | None
    payload_id: str | None
    expected: object
    observed: object
    message: str

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code,
            "field_name": self.field_name,
            "reference_payload_id": self.reference_payload_id,
            "payload_id": self.payload_id,
            "expected": self.expected,
            "observed": self.observed,
            "message": self.message,
        }

    @classmethod
    def from_record(cls, value: object) -> PredictionBatchViolationV1:
        if not isinstance(value, Mapping):
            raise ValueError("prediction batch violation must be an object")
        expected_fields = frozenset(
            {
                "code",
                "field_name",
                "reference_payload_id",
                "payload_id",
                "expected",
                "observed",
                "message",
            }
        )
        _require_exact_fields(value, expected_fields, name="prediction batch violation")
        reference = value["reference_payload_id"]
        payload = value["payload_id"]
        if reference is not None:
            reference = _require_string(reference, name="reference_payload_id")
        if payload is not None:
            payload = _require_string(payload, name="payload_id")
        return cls(
            code=_require_string(value["code"], name="violation code"),
            field_name=_require_string(value["field_name"], name="field_name"),
            reference_payload_id=reference,
            payload_id=payload,
            expected=value["expected"],
            observed=value["observed"],
            message=_require_string(value["message"], name="violation message"),
        )


@dataclass(frozen=True)
class PredictionBatchPreflightV1:
    """Replayable result for one causally selected provider batch."""

    manifest_artifact_id: str
    causal_frame_stop: int | None
    policy: PredictionBatchPolicyV1
    selected_payload_ids: tuple[str, ...]
    excluded_future_payload_ids: tuple[str, ...]
    entries: tuple[PredictionBatchEntryV1, ...]
    violations: tuple[PredictionBatchViolationV1, ...]
    status: str
    future_prediction_payloads_opened: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        _require_string(self.manifest_artifact_id, name="manifest_artifact_id")
        if self.causal_frame_stop is not None and self.causal_frame_stop < 0:
            raise ValueError("causal_frame_stop must be nonnegative")
        if self.status not in _VALID_STATUSES:
            raise ValueError("unsupported prediction batch preflight status")
        if self.future_prediction_payloads_opened != 0:
            raise ValueError("causal preflight must not open future payloads")
        if len(set(self.selected_payload_ids)) != len(self.selected_payload_ids):
            raise ValueError("selected_payload_ids must be unique")
        if len(set(self.excluded_future_payload_ids)) != len(
            self.excluded_future_payload_ids
        ):
            raise ValueError("excluded_future_payload_ids must be unique")
        if set(self.selected_payload_ids) & set(self.excluded_future_payload_ids):
            raise ValueError("selected and excluded payload rosters overlap")
        if tuple(entry.payload_id for entry in self.entries) != self.selected_payload_ids:
            raise ValueError("entry order differs from selected payload order")
        if self.status == _STATUS_PASS and self.violations:
            raise ValueError("passing preflight cannot retain violations")
        if self.status == _STATUS_INCOMPATIBLE and not self.violations:
            raise ValueError("incompatible preflight requires at least one violation")
        try:
            _canonical_json_bytes(dict(self.metadata))
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be finite JSON") from error
        expected_id = _artifact_id(self.to_record(include_artifact_id=False))
        if self.artifact_id is None:
            object.__setattr__(self, "artifact_id", expected_id)
        elif self.artifact_id != expected_id:
            raise ValueError("prediction batch preflight artifact ID mismatch")

    @property
    def compatible(self) -> bool:
        return self.status == _STATUS_PASS

    def to_record(self, *, include_artifact_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema": PREDICTION_BATCH_PREFLIGHT_SCHEMA,
            "schema_version": PREDICTION_BATCH_PREFLIGHT_VERSION,
            "manifest_artifact_id": self.manifest_artifact_id,
            "causal_frame_stop": self.causal_frame_stop,
            "policy": self.policy.to_record(),
            "selected_payload_ids": list(self.selected_payload_ids),
            "excluded_future_payload_ids": list(self.excluded_future_payload_ids),
            "entries": [entry.to_record() for entry in self.entries],
            "violations": [violation.to_record() for violation in self.violations],
            "status": self.status,
            "future_prediction_payloads_opened": self.future_prediction_payloads_opened,
            "metadata": dict(self.metadata),
        }
        if include_artifact_id:
            record["artifact_id"] = self.artifact_id
        return record

    @classmethod
    def from_record(cls, value: object) -> PredictionBatchPreflightV1:
        if not isinstance(value, Mapping):
            raise ValueError("prediction batch preflight must be an object")
        expected = frozenset(
            {
                "schema",
                "schema_version",
                "manifest_artifact_id",
                "causal_frame_stop",
                "policy",
                "selected_payload_ids",
                "excluded_future_payload_ids",
                "entries",
                "violations",
                "status",
                "future_prediction_payloads_opened",
                "metadata",
                "artifact_id",
            }
        )
        _require_exact_fields(value, expected, name="prediction batch preflight")
        if value["schema"] != PREDICTION_BATCH_PREFLIGHT_SCHEMA:
            raise ValueError("unsupported prediction batch preflight schema")
        if value["schema_version"] != PREDICTION_BATCH_PREFLIGHT_VERSION:
            raise ValueError("unsupported prediction batch preflight version")
        cutoff_value = value["causal_frame_stop"]
        cutoff = (
            None
            if cutoff_value is None
            else _require_integer(cutoff_value, name="causal_frame_stop", minimum=0)
        )
        raw_entries = value["entries"]
        raw_violations = value["violations"]
        raw_metadata = value["metadata"]
        if not isinstance(raw_entries, list):
            raise ValueError("entries must be a JSON array")
        if not isinstance(raw_violations, list):
            raise ValueError("violations must be a JSON array")
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            manifest_artifact_id=_require_string(
                value["manifest_artifact_id"], name="manifest_artifact_id"
            ),
            causal_frame_stop=cutoff,
            policy=PredictionBatchPolicyV1.from_record(value["policy"]),
            selected_payload_ids=_require_string_tuple(
                value["selected_payload_ids"], name="selected_payload_ids"
            ),
            excluded_future_payload_ids=_require_string_tuple(
                value["excluded_future_payload_ids"],
                name="excluded_future_payload_ids",
            ),
            entries=tuple(PredictionBatchEntryV1.from_record(item) for item in raw_entries),
            violations=tuple(
                PredictionBatchViolationV1.from_record(item) for item in raw_violations
            ),
            status=_require_string(value["status"], name="status"),
            future_prediction_payloads_opened=_require_integer(
                value["future_prediction_payloads_opened"],
                name="future_prediction_payloads_opened",
                minimum=0,
            ),
            metadata=dict(raw_metadata),
            artifact_id=_require_string(value["artifact_id"], name="artifact_id"),
        )


def _safe_payload_path(manifest_path: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PredictionBatchIntegrityError("prediction payload path escapes manifest root")
    root = manifest_path.parent.resolve()
    candidate = manifest_path.parent / relative
    if candidate.is_symlink():
        raise PredictionBatchIntegrityError("prediction payload is a symbolic link")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PredictionBatchIntegrityError(
            "prediction payload path escapes manifest root"
        ) from error
    return resolved


def _snapshot_window(
    payload_path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    expected_window_id: str,
) -> tuple[PredictionWindow, bytes]:
    try:
        payload = payload_path.read_bytes()
    except OSError as error:
        raise PredictionBatchIntegrityError(
            f"cannot read selected prediction payload {payload_path.name!r}"
        ) from error
    if len(payload) != expected_byte_count:
        raise PredictionBatchIntegrityError("selected prediction payload byte count mismatch")
    if _sha256_bytes(payload) != expected_sha256:
        raise PredictionBatchIntegrityError("selected prediction payload SHA-256 mismatch")
    with tempfile.TemporaryDirectory(prefix="prob4d-batch-preflight-") as directory:
        snapshot = Path(directory) / "prediction-window.npz"
        snapshot.write_bytes(payload)
        try:
            window = PredictionWindow.from_npz(snapshot)
        except (OSError, KeyError, ValueError) as error:
            raise PredictionBatchIntegrityError(
                "selected payload is not a canonical PredictionWindow"
            ) from error
    try:
        unchanged = payload_path.read_bytes()
    except OSError as error:
        raise PredictionBatchIntegrityError(
            "selected prediction payload became unreadable during preflight"
        ) from error
    if unchanged != payload:
        raise PredictionBatchIntegrityError(
            "selected prediction payload changed during preflight"
        )
    if window.window_id != expected_window_id:
        raise PredictionBatchIntegrityError(
            "selected descriptor and PredictionWindow IDs differ"
        )
    return window, payload


def _entry_from_descriptor(manifest_path: Path, descriptor: Any) -> PredictionBatchEntryV1:
    payload_path = _safe_payload_path(manifest_path, descriptor.path)
    window, payload = _snapshot_window(
        payload_path,
        expected_sha256=descriptor.sha256,
        expected_byte_count=descriptor.byte_count,
        expected_window_id=descriptor.window_id,
    )
    output_frame_ids = tuple(int(value) for value in window.frame_indices)
    if output_frame_ids != tuple(descriptor.output_frame_ids):
        raise PredictionBatchIntegrityError(
            "selected descriptor and PredictionWindow frame identities differ"
        )
    if (window.scene_flow is not None) != bool(descriptor.has_scene_flow):
        raise PredictionBatchIntegrityError(
            "selected descriptor scene-flow declaration differs from payload"
        )
    if (window.ray_directions is not None) != bool(descriptor.has_ray_directions):
        raise PredictionBatchIntegrityError(
            "selected descriptor ray declaration differs from payload"
        )
    if window.dense_storage_dtype != descriptor.dense_storage_dtype:
        raise PredictionBatchIntegrityError(
            "selected descriptor storage dtype differs from payload"
        )
    point_shape = tuple(int(value) for value in window.point_map.shape)
    scene_shape = (
        None
        if window.scene_flow is None
        else tuple(int(value) for value in window.scene_flow.shape)
    )
    ray_shape = (
        None
        if window.ray_directions is None
        else tuple(int(value) for value in window.ray_directions.shape)
    )
    return PredictionBatchEntryV1(
        payload_id=descriptor.payload_id,
        window_id=descriptor.window_id,
        relative_path=descriptor.path,
        output_frame_ids=output_frame_ids,
        point_shape=point_shape,
        point_dtype=np.dtype(window.point_map.dtype).str,
        scene_flow_shape=scene_shape,
        ray_shape=ray_shape,
        dense_storage_dtype=window.dense_storage_dtype,
        payload_sha256=_sha256_bytes(payload),
        payload_byte_count=len(payload),
    )


def _violation(
    *,
    code: str,
    field_name: str,
    reference: PredictionBatchEntryV1,
    entry: PredictionBatchEntryV1,
    expected: object,
    observed: object,
) -> PredictionBatchViolationV1:
    return PredictionBatchViolationV1(
        code=code,
        field_name=field_name,
        reference_payload_id=reference.payload_id,
        payload_id=entry.payload_id,
        expected=expected,
        observed=observed,
        message=(
            f"payload {entry.payload_id!r} has incompatible {field_name}; "
            f"reference payload is {reference.payload_id!r}"
        ),
    )


def _batch_violations(
    entries: tuple[PredictionBatchEntryV1, ...],
    policy: PredictionBatchPolicyV1,
) -> tuple[PredictionBatchViolationV1, ...]:
    if not entries:
        if not policy.require_nonempty:
            return ()
        return (
            PredictionBatchViolationV1(
                code="no-causally-admitted-payloads",
                field_name="payload_roster",
                reference_payload_id=None,
                payload_id=None,
                expected="at least one causally admitted payload",
                observed=0,
                message="no prediction payload is admitted by the causal cutoff",
            ),
        )
    reference = entries[0]
    violations: list[PredictionBatchViolationV1] = []
    for entry in entries[1:]:
        if policy.require_common_frame_count and entry.frame_count != reference.frame_count:
            violations.append(
                _violation(
                    code="frame-count-mismatch",
                    field_name="frame_count",
                    reference=reference,
                    entry=entry,
                    expected=reference.frame_count,
                    observed=entry.frame_count,
                )
            )
        if (
            policy.require_common_spatial_shape
            and entry.spatial_shape != reference.spatial_shape
        ):
            violations.append(
                _violation(
                    code="spatial-shape-mismatch",
                    field_name="spatial_shape",
                    reference=reference,
                    entry=entry,
                    expected=list(reference.spatial_shape),
                    observed=list(entry.spatial_shape),
                )
            )
        if policy.require_common_point_dtype and entry.point_dtype != reference.point_dtype:
            violations.append(
                _violation(
                    code="point-dtype-mismatch",
                    field_name="point_dtype",
                    reference=reference,
                    entry=entry,
                    expected=reference.point_dtype,
                    observed=entry.point_dtype,
                )
            )
        if (
            policy.require_common_optional_fields
            and entry.optional_field_signature != reference.optional_field_signature
        ):
            violations.append(
                _violation(
                    code="optional-field-mismatch",
                    field_name="optional_field_signature",
                    reference=reference,
                    entry=entry,
                    expected=list(reference.optional_field_signature),
                    observed=list(entry.optional_field_signature),
                )
            )
    return tuple(violations)


def preflight_prediction_batch(
    manifest_path: str | Path,
    *,
    causal_frame_stop: int | None = None,
    policy: PredictionBatchPolicyV1 | None = None,
    metadata: Mapping[str, object] | None = None,
) -> PredictionBatchPreflightV1:
    """Inspect only the causally admitted payloads and classify batch compatibility."""

    path_input = Path(manifest_path)
    if path_input.is_symlink():
        raise PredictionBatchIntegrityError("prediction-provider manifest is a symbolic link")
    path = path_input.resolve()
    if causal_frame_stop is not None and causal_frame_stop < 0:
        raise ValueError("causal_frame_stop must be nonnegative")
    manifest = load_prediction_provider_manifest(path)
    selected = tuple(
        descriptor
        for descriptor in manifest.payloads
        if causal_frame_stop is None
        or descriptor.is_causally_admitted(causal_frame_stop)
    )
    excluded = tuple(
        descriptor
        for descriptor in manifest.payloads
        if descriptor not in selected
    )
    entries = tuple(_entry_from_descriptor(path, descriptor) for descriptor in selected)
    active_policy = policy or PredictionBatchPolicyV1()
    violations = _batch_violations(entries, active_policy)
    return PredictionBatchPreflightV1(
        manifest_artifact_id=manifest.artifact_id,
        causal_frame_stop=causal_frame_stop,
        policy=active_policy,
        selected_payload_ids=tuple(entry.payload_id for entry in entries),
        excluded_future_payload_ids=tuple(
            descriptor.payload_id for descriptor in excluded
        ),
        entries=entries,
        violations=violations,
        status=_STATUS_PASS if not violations else _STATUS_INCOMPATIBLE,
        future_prediction_payloads_opened=0,
        metadata={} if metadata is None else dict(metadata),
    )


def write_prediction_batch_preflight(
    path: str | Path,
    artifact: PredictionBatchPreflightV1,
) -> Path:
    """Persist one artifact atomically, allowing only idempotent repetition."""

    destination_input = Path(path)
    if destination_input.is_symlink():
        raise ValueError("prediction batch preflight output is a symbolic link")
    destination = destination_input.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        artifact.to_record(),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") == content:
            return destination
        raise FileExistsError("refusing to replace a different preflight artifact")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_prediction_batch_preflight(path: str | Path) -> PredictionBatchPreflightV1:
    """Load and replay every derived identity and invariant."""

    return PredictionBatchPreflightV1.from_record(
        _strict_json_object(Path(path), name="prediction batch preflight artifact")
    )


def _policy_from_arguments(arguments: argparse.Namespace) -> PredictionBatchPolicyV1:
    return PredictionBatchPolicyV1(
        require_nonempty=not arguments.allow_empty,
        require_common_frame_count=not arguments.allow_ragged_frames,
        require_common_spatial_shape=not arguments.allow_ragged_spatial,
        require_common_point_dtype=not arguments.allow_mixed_dtypes,
        require_common_optional_fields=not arguments.allow_mixed_optional_fields,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify a causal provider prediction-batch preflight artifact."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build", help="build a preflight artifact")
    build.add_argument("manifest")
    build.add_argument("output")
    build.add_argument("--causal-frame-stop", type=int)
    build.add_argument("--allow-empty", action="store_true")
    build.add_argument("--allow-ragged-frames", action="store_true")
    build.add_argument("--allow-ragged-spatial", action="store_true")
    build.add_argument("--allow-mixed-dtypes", action="store_true")
    build.add_argument("--allow-mixed-optional-fields", action="store_true")
    verify = subparsers.add_parser("verify", help="verify a persisted artifact")
    verify.add_argument("artifact")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    if arguments.action == "verify":
        artifact = load_prediction_batch_preflight(arguments.artifact)
        print(json.dumps(artifact.to_record(), indent=2, sort_keys=True))
        return 0
    artifact = preflight_prediction_batch(
        arguments.manifest,
        causal_frame_stop=arguments.causal_frame_stop,
        policy=_policy_from_arguments(arguments),
    )
    write_prediction_batch_preflight(arguments.output, artifact)
    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "status": artifact.status,
                "selected_payload_count": len(artifact.entries),
                "violation_count": len(artifact.violations),
                "future_prediction_payloads_opened": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if artifact.compatible else 2


__all__ = [
    "PREDICTION_BATCH_POLICY_SCHEMA",
    "PREDICTION_BATCH_POLICY_VERSION",
    "PREDICTION_BATCH_PREFLIGHT_SCHEMA",
    "PREDICTION_BATCH_PREFLIGHT_VERSION",
    "PredictionBatchEntryV1",
    "PredictionBatchIntegrityError",
    "PredictionBatchPolicyV1",
    "PredictionBatchPreflightV1",
    "PredictionBatchViolationV1",
    "load_prediction_batch_preflight",
    "main",
    "preflight_prediction_batch",
    "write_prediction_batch_preflight",
]


if __name__ == "__main__":
    raise SystemExit(main())
