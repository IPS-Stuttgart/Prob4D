"""Causal, content-addressed preflight for provider prediction batches.

Only manifest entries admitted by the exclusive causal cutoff are opened.  Batch
shape differences become structured infrastructure evidence before a scorer
attempts to stack arrays; malformed or changed selected bytes remain integrity
errors rather than scientific negatives.
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


class PredictionBatchIntegrityError(ValueError):
    """Selected provider bytes are missing, changed, or malformed."""


def _canonical_bytes(record: Mapping[str, object]) -> bytes:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact_id(record: Mapping[str, object]) -> str:
    identity_record = dict(record)
    identity_record.pop("artifact_id", None)
    return _sha256(_canonical_bytes(identity_record))


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_fields(
    record: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    actual = frozenset(record)
    if actual != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    result = tuple(_string(item, name=f"{name} item") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return tuple(_integer(item, name=f"{name} item") for item in value)


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
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
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must contain UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


@dataclass(frozen=True)
class PredictionBatchPolicyV1:
    """Frozen scorer compatibility requirements."""

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
        record = _mapping(value, name="prediction batch policy")
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
        _exact_fields(record, expected, name="prediction batch policy")
        if record["schema"] != PREDICTION_BATCH_POLICY_SCHEMA:
            raise ValueError("unsupported prediction batch policy schema")
        if record["schema_version"] != PREDICTION_BATCH_POLICY_VERSION:
            raise ValueError("unsupported prediction batch policy version")
        return cls(
            require_nonempty=_boolean(record["require_nonempty"], name="require_nonempty"),
            require_common_frame_count=_boolean(
                record["require_common_frame_count"],
                name="require_common_frame_count",
            ),
            require_common_spatial_shape=_boolean(
                record["require_common_spatial_shape"],
                name="require_common_spatial_shape",
            ),
            require_common_point_dtype=_boolean(
                record["require_common_point_dtype"],
                name="require_common_point_dtype",
            ),
            require_common_optional_fields=_boolean(
                record["require_common_optional_fields"],
                name="require_common_optional_fields",
            ),
        )


@dataclass(frozen=True)
class PredictionBatchEntryV1:
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
        if len(self.point_shape) < 2 or self.point_shape[-1] != 3:
            raise ValueError("point_shape must end in coordinate dimension 3")
        if len(self.output_frame_ids) != self.point_shape[0]:
            raise ValueError("output frame count differs from point tensor")
        if self.scene_flow_shape is not None and self.scene_flow_shape != self.point_shape:
            raise ValueError("scene-flow shape must equal point shape")
        if self.ray_shape is not None and self.ray_shape != self.point_shape:
            raise ValueError("ray shape must equal point shape")
        if len(self.payload_sha256) != 64:
            raise ValueError("payload_sha256 must contain 64 hexadecimal characters")
        try:
            int(self.payload_sha256, 16)
        except ValueError as error:
            raise ValueError("payload_sha256 must be hexadecimal") from error

    @property
    def frame_count(self) -> int:
        return self.point_shape[0]

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self.point_shape[1:-1]

    @property
    def optional_signature(self) -> tuple[bool, bool]:
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
        record = _mapping(value, name="prediction batch entry")
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
        _exact_fields(record, expected, name="prediction batch entry")
        scene = record["scene_flow_shape"]
        rays = record["ray_shape"]
        return cls(
            payload_id=_string(record["payload_id"], name="payload_id"),
            window_id=_string(record["window_id"], name="window_id"),
            relative_path=_string(record["relative_path"], name="relative_path"),
            output_frame_ids=_integer_tuple(
                record["output_frame_ids"], name="output_frame_ids"
            ),
            point_shape=_integer_tuple(record["point_shape"], name="point_shape"),
            point_dtype=_string(record["point_dtype"], name="point_dtype"),
            scene_flow_shape=(
                None if scene is None else _integer_tuple(scene, name="scene_flow_shape")
            ),
            ray_shape=None if rays is None else _integer_tuple(rays, name="ray_shape"),
            dense_storage_dtype=_string(
                record["dense_storage_dtype"], name="dense_storage_dtype"
            ),
            payload_sha256=_string(record["payload_sha256"], name="payload_sha256"),
            payload_byte_count=_integer(
                record["payload_byte_count"], name="payload_byte_count"
            ),
        )


@dataclass(frozen=True)
class PredictionBatchViolationV1:
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
        record = _mapping(value, name="prediction batch violation")
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
        _exact_fields(record, expected_fields, name="prediction batch violation")
        reference = record["reference_payload_id"]
        payload = record["payload_id"]
        return cls(
            code=_string(record["code"], name="code"),
            field_name=_string(record["field_name"], name="field_name"),
            reference_payload_id=(
                None if reference is None else _string(reference, name="reference_payload_id")
            ),
            payload_id=None if payload is None else _string(payload, name="payload_id"),
            expected=record["expected"],
            observed=record["observed"],
            message=_string(record["message"], name="message"),
        )


@dataclass(frozen=True)
class PredictionBatchPreflightV1:
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
        if self.status not in {"pass", "batch-incompatible"}:
            raise ValueError("unsupported prediction batch status")
        if self.future_prediction_payloads_opened != 0:
            raise ValueError("causal preflight must not open future payloads")
        if tuple(entry.payload_id for entry in self.entries) != self.selected_payload_ids:
            raise ValueError("entry order differs from selected payload order")
        if set(self.selected_payload_ids) & set(self.excluded_future_payload_ids):
            raise ValueError("selected and excluded payload rosters overlap")
        if self.status == "pass" and self.violations:
            raise ValueError("passing preflight cannot retain violations")
        if self.status == "batch-incompatible" and not self.violations:
            raise ValueError("incompatible preflight requires a violation")
        _canonical_bytes(dict(self.metadata))
        expected = _artifact_id(self.to_record(include_artifact_id=False))
        if self.artifact_id is None:
            object.__setattr__(self, "artifact_id", expected)
        elif self.artifact_id != expected:
            raise ValueError("prediction batch preflight artifact ID mismatch")

    @property
    def compatible(self) -> bool:
        return self.status == "pass"

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
        record = _mapping(value, name="prediction batch preflight")
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
        _exact_fields(record, expected, name="prediction batch preflight")
        if record["schema"] != PREDICTION_BATCH_PREFLIGHT_SCHEMA:
            raise ValueError("unsupported prediction batch preflight schema")
        if record["schema_version"] != PREDICTION_BATCH_PREFLIGHT_VERSION:
            raise ValueError("unsupported prediction batch preflight version")
        raw_entries = record["entries"]
        raw_violations = record["violations"]
        if not isinstance(raw_entries, list) or not isinstance(raw_violations, list):
            raise ValueError("entries and violations must be arrays")
        cutoff_value = record["causal_frame_stop"]
        cutoff = None if cutoff_value is None else _integer(cutoff_value, name="cutoff")
        return cls(
            manifest_artifact_id=_string(
                record["manifest_artifact_id"], name="manifest_artifact_id"
            ),
            causal_frame_stop=cutoff,
            policy=PredictionBatchPolicyV1.from_record(record["policy"]),
            selected_payload_ids=_string_tuple(
                record["selected_payload_ids"], name="selected_payload_ids"
            ),
            excluded_future_payload_ids=_string_tuple(
                record["excluded_future_payload_ids"],
                name="excluded_future_payload_ids",
            ),
            entries=tuple(PredictionBatchEntryV1.from_record(item) for item in raw_entries),
            violations=tuple(
                PredictionBatchViolationV1.from_record(item) for item in raw_violations
            ),
            status=_string(record["status"], name="status"),
            future_prediction_payloads_opened=_integer(
                record["future_prediction_payloads_opened"],
                name="future_prediction_payloads_opened",
            ),
            metadata=dict(_mapping(record["metadata"], name="metadata")),
            artifact_id=_string(record["artifact_id"], name="artifact_id"),
        )


def _payload_path(manifest_path: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PredictionBatchIntegrityError("payload path escapes the manifest root")
    root = manifest_path.parent.resolve()
    candidate = manifest_path.parent / relative
    if candidate.is_symlink():
        raise PredictionBatchIntegrityError("payload path is a symbolic link")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PredictionBatchIntegrityError("payload path escapes the manifest root") from error
    return resolved


def _load_selected_window(manifest_path: Path, descriptor: Any) -> PredictionBatchEntryV1:
    path = _payload_path(manifest_path, descriptor.path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PredictionBatchIntegrityError("cannot read selected prediction payload") from error
    if len(payload) != descriptor.byte_count:
        raise PredictionBatchIntegrityError("selected prediction payload byte count mismatch")
    if _sha256(payload) != descriptor.sha256:
        raise PredictionBatchIntegrityError("selected prediction payload SHA-256 mismatch")
    with tempfile.TemporaryDirectory(prefix="prob4d-batch-preflight-") as directory:
        snapshot = Path(directory) / "window.npz"
        snapshot.write_bytes(payload)
        try:
            window = PredictionWindow.from_npz(snapshot)
        except (OSError, KeyError, ValueError) as error:
            raise PredictionBatchIntegrityError(
                "selected payload is not a canonical PredictionWindow"
            ) from error
    if path.read_bytes() != payload:
        raise PredictionBatchIntegrityError("selected prediction payload changed during read")
    if window.window_id != descriptor.window_id:
        raise PredictionBatchIntegrityError("descriptor and payload window IDs differ")
    frame_ids = tuple(int(value) for value in window.frame_indices)
    if frame_ids != tuple(descriptor.output_frame_ids):
        raise PredictionBatchIntegrityError("descriptor and payload frame identities differ")
    if (window.scene_flow is not None) != bool(descriptor.has_scene_flow):
        raise PredictionBatchIntegrityError("scene-flow declaration differs from payload")
    if (window.ray_directions is not None) != bool(descriptor.has_ray_directions):
        raise PredictionBatchIntegrityError("ray declaration differs from payload")
    if window.dense_storage_dtype != descriptor.dense_storage_dtype:
        raise PredictionBatchIntegrityError("storage dtype declaration differs from payload")
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
        output_frame_ids=frame_ids,
        point_shape=point_shape,
        point_dtype=np.dtype(window.point_map.dtype).str,
        scene_flow_shape=scene_shape,
        ray_shape=ray_shape,
        dense_storage_dtype=window.dense_storage_dtype,
        payload_sha256=_sha256(payload),
        payload_byte_count=len(payload),
    )


def _mismatch(
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


def _violations(
    entries: tuple[PredictionBatchEntryV1, ...], policy: PredictionBatchPolicyV1
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
                message="no payload is admitted by the causal cutoff",
            ),
        )
    reference = entries[0]
    result: list[PredictionBatchViolationV1] = []
    for entry in entries[1:]:
        if policy.require_common_frame_count and entry.frame_count != reference.frame_count:
            result.append(
                _mismatch(
                    "frame-count-mismatch",
                    "frame_count",
                    reference,
                    entry,
                    reference.frame_count,
                    entry.frame_count,
                )
            )
        if (
            policy.require_common_spatial_shape
            and entry.spatial_shape != reference.spatial_shape
        ):
            result.append(
                _mismatch(
                    "spatial-shape-mismatch",
                    "spatial_shape",
                    reference,
                    entry,
                    list(reference.spatial_shape),
                    list(entry.spatial_shape),
                )
            )
        if policy.require_common_point_dtype and entry.point_dtype != reference.point_dtype:
            result.append(
                _mismatch(
                    "point-dtype-mismatch",
                    "point_dtype",
                    reference,
                    entry,
                    reference.point_dtype,
                    entry.point_dtype,
                )
            )
        if (
            policy.require_common_optional_fields
            and entry.optional_signature != reference.optional_signature
        ):
            result.append(
                _mismatch(
                    "optional-field-mismatch",
                    "optional_signature",
                    reference,
                    entry,
                    list(reference.optional_signature),
                    list(entry.optional_signature),
                )
            )
    return tuple(result)


def preflight_prediction_batch(
    manifest_path: str | Path,
    *,
    causal_frame_stop: int | None = None,
    policy: PredictionBatchPolicyV1 | None = None,
    metadata: Mapping[str, object] | None = None,
) -> PredictionBatchPreflightV1:
    """Inspect the causally admitted payloads and classify scorer compatibility."""

    path_input = Path(manifest_path)
    if path_input.is_symlink():
        raise PredictionBatchIntegrityError("provider manifest is a symbolic link")
    path = path_input.resolve()
    if causal_frame_stop is not None:
        _integer(causal_frame_stop, name="causal_frame_stop")
    manifest = load_prediction_provider_manifest(path)
    selected = tuple(
        descriptor
        for descriptor in manifest.payloads
        if causal_frame_stop is None
        or descriptor.is_causally_admitted(causal_frame_stop)
    )
    selected_ids = {descriptor.payload_id for descriptor in selected}
    excluded = tuple(
        descriptor for descriptor in manifest.payloads if descriptor.payload_id not in selected_ids
    )
    entries = tuple(_load_selected_window(path, descriptor) for descriptor in selected)
    active_policy = policy or PredictionBatchPolicyV1()
    violations = _violations(entries, active_policy)
    return PredictionBatchPreflightV1(
        manifest_artifact_id=manifest.artifact_id,
        causal_frame_stop=causal_frame_stop,
        policy=active_policy,
        selected_payload_ids=tuple(entry.payload_id for entry in entries),
        excluded_future_payload_ids=tuple(item.payload_id for item in excluded),
        entries=entries,
        violations=violations,
        status="pass" if not violations else "batch-incompatible",
        metadata={} if metadata is None else dict(metadata),
    )


def write_prediction_batch_preflight(
    path: str | Path, artifact: PredictionBatchPreflightV1
) -> Path:
    """Write atomically and refuse to replace a different artifact."""

    destination_input = Path(path)
    if destination_input.is_symlink():
        raise ValueError("preflight output is a symbolic link")
    destination = destination_input.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(artifact.to_record(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") == content:
            return destination
        raise FileExistsError("refusing to replace a different preflight artifact")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
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
    return PredictionBatchPreflightV1.from_record(
        _strict_json(Path(path), name="prediction batch preflight artifact")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or verify prediction-batch preflight")
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build")
    build.add_argument("manifest")
    build.add_argument("output")
    build.add_argument("--causal-frame-stop", type=int)
    build.add_argument("--allow-empty", action="store_true")
    build.add_argument("--allow-ragged-frames", action="store_true")
    build.add_argument("--allow-ragged-spatial", action="store_true")
    build.add_argument("--allow-mixed-dtypes", action="store_true")
    build.add_argument("--allow-mixed-optional-fields", action="store_true")
    verify = actions.add_parser("verify")
    verify.add_argument("artifact")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    if arguments.action == "verify":
        artifact = load_prediction_batch_preflight(arguments.artifact)
    else:
        policy = PredictionBatchPolicyV1(
            require_nonempty=not arguments.allow_empty,
            require_common_frame_count=not arguments.allow_ragged_frames,
            require_common_spatial_shape=not arguments.allow_ragged_spatial,
            require_common_point_dtype=not arguments.allow_mixed_dtypes,
            require_common_optional_fields=not arguments.allow_mixed_optional_fields,
        )
        artifact = preflight_prediction_batch(
            arguments.manifest,
            causal_frame_stop=arguments.causal_frame_stop,
            policy=policy,
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
    return 0 if arguments.action == "verify" or artifact.compatible else 2


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
