"""Append-only, content-addressed streams of unfused observation-factor bundles.

Each update references one schema-v4 :class:`ObservationFactorBundle` on disk,
adds observations from one non-overlapping causal frame interval, and binds the
previous update ID. Bundle paths are retrieval metadata; hashes, identities,
and frame boundaries determine the portable update and stream content addresses.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_finite_json_mapping,
    require_mapping,
    require_nonempty_string,
    require_sha256,
    require_string_sequence,
)
from .observation_factors import (
    OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION,
    ObservationFactorBundle,
    load_observation_factor_bundle,
)

OBSERVATION_FACTOR_STREAM_SCHEMA = "prob4d.observation-factor-stream"
OBSERVATION_FACTOR_STREAM_VERSION = 1

_STREAM_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
        "metadata",
        "updates",
        "artifact_id",
    }
)
_UPDATE_FIELDS = frozenset(
    {
        "update_index",
        "admitted_frame_start",
        "causal_frame_stop",
        "bundle_manifest_path",
        "bundle_manifest_sha256",
        "bundle_payload_sha256",
        "bundle_sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
        "factor_count",
        "observation_count",
        "persistent_identity_count",
        "observation_identity_sha256",
        "gauge_ids",
        "previous_update_id",
        "update_id",
    }
)
_PAYLOAD_FIELDS = frozenset({"path", "sha256", "allow_pickle"})
_GAUGE_COVARIANCE_FIELDS = frozenset(
    {
        "semantics",
        "joint_covariance_key",
        "ordered_gauge_ids",
        "cross_window_covariance_preserved",
        "diagonal_blocks_match_gauge_marginals",
    }
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_bytes(path, name="stream member"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bytes(path: Path, *, name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {name} {path.name!r}") from error


def _load_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON object key {key!r}")
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


@contextmanager
def _exclusive_stream_lock(path: Path) -> Iterator[None]:
    lock = path.with_name(f".{path.name}.lock")
    try:
        descriptor = os.open(
            lock,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"observation-factor stream is already being written: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    return require_exact_integer(value, name=name, minimum=0)


def _safe_relative_path(value: object, *, name: str) -> str:
    path = require_nonempty_string(value, name=name)
    if "\\" in path:
        raise ValueError(f"{name} must be a safe POSIX relative path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    return pure.as_posix()


def _resolved_member(root: Path, relative_path: str, *, name: str) -> Path:
    safe = _safe_relative_path(relative_path, name=name)
    root_resolved = root.resolve()
    candidate = (root_resolved / Path(*PurePosixPath(safe).parts)).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} escapes the stream directory") from error
    return candidate


def _relative_member(path: Path, *, root: Path, name: str) -> str:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        relative = path_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} must lie inside the stream directory") from error
    return _safe_relative_path(relative.as_posix(), name=name)


def _observation_identity_summary(
    bundle: ObservationFactorBundle,
) -> tuple[int, int, str]:
    persistent_identities: set[tuple[str, str, int]] = set()
    observation_identities: list[tuple[int, str, str, int]] = []
    for factor in bundle.factors:
        for point_id in np.asarray(factor.point_ids, dtype=np.int64):
            persistent = (factor.view_id, factor.window_id, int(point_id))
            observation = (factor.frame_index, *persistent)
            persistent_identities.add(persistent)
            observation_identities.append(observation)
    if len(set(observation_identities)) != len(observation_identities):
        raise ValueError("an observation-factor stream update contains duplicate frame identities")
    ordered = sorted(observation_identities)
    digest = hashlib.sha256(_canonical_json({"observations": ordered})).hexdigest()
    return len(persistent_identities), len(observation_identities), digest


@dataclass(frozen=True)
class ObservationFactorStreamUpdateV1:
    """One causally disjoint, hash-chained observation-factor update."""

    update_index: int
    admitted_frame_start: int
    causal_frame_stop: int
    bundle_manifest_path: str
    bundle_manifest_sha256: str
    bundle_payload_sha256: str
    bundle_sequence_id: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    factor_count: int
    observation_count: int
    persistent_identity_count: int
    observation_identity_sha256: str
    gauge_ids: tuple[str, ...]
    previous_update_id: str | None = None
    update_id: str | None = None

    def __post_init__(self) -> None:
        update_index = _require_nonnegative_integer(
            self.update_index,
            name="update_index",
        )
        frame_start = _require_nonnegative_integer(
            self.admitted_frame_start,
            name="admitted_frame_start",
        )
        frame_stop = _require_nonnegative_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
        )
        if frame_stop <= frame_start:
            raise ValueError("causal_frame_stop must exceed admitted_frame_start")
        factor_count = _require_nonnegative_integer(
            self.factor_count,
            name="factor_count",
        )
        observation_count = _require_nonnegative_integer(
            self.observation_count,
            name="observation_count",
        )
        identity_count = _require_nonnegative_integer(
            self.persistent_identity_count,
            name="persistent_identity_count",
        )
        if factor_count < 1 or observation_count < 1 or identity_count < 1:
            raise ValueError("stream updates must contain factors and observations")

        identifiers = {
            "bundle_sequence_id": require_nonempty_string(
                self.bundle_sequence_id,
                name="bundle_sequence_id",
            ),
            "case_id": require_nonempty_string(self.case_id, name="case_id"),
            "stream_id": require_nonempty_string(self.stream_id, name="stream_id"),
            "source_repository": require_nonempty_string(
                self.source_repository,
                name="source_repository",
            ),
            "source_revision": require_nonempty_string(
                self.source_revision,
                name="source_revision",
            ),
        }
        gauge_ids = require_string_sequence(self.gauge_ids, name="gauge_ids")
        if len(set(gauge_ids)) != len(gauge_ids):
            raise ValueError("gauge_ids must be unique")

        previous = self.previous_update_id
        if previous is not None:
            previous = require_sha256(previous, name="previous_update_id")
        manifest_path = _safe_relative_path(
            self.bundle_manifest_path,
            name="bundle_manifest_path",
        )
        manifest_sha = require_sha256(
            self.bundle_manifest_sha256,
            name="bundle_manifest_sha256",
        )
        payload_sha = require_sha256(
            self.bundle_payload_sha256,
            name="bundle_payload_sha256",
        )
        identity_sha = require_sha256(
            self.observation_identity_sha256,
            name="observation_identity_sha256",
        )

        object.__setattr__(self, "update_index", update_index)
        object.__setattr__(self, "admitted_frame_start", frame_start)
        object.__setattr__(self, "causal_frame_stop", frame_stop)
        object.__setattr__(self, "factor_count", factor_count)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "persistent_identity_count", identity_count)
        object.__setattr__(self, "bundle_manifest_path", manifest_path)
        object.__setattr__(self, "bundle_manifest_sha256", manifest_sha)
        object.__setattr__(self, "bundle_payload_sha256", payload_sha)
        object.__setattr__(self, "observation_identity_sha256", identity_sha)
        object.__setattr__(self, "gauge_ids", gauge_ids)
        object.__setattr__(self, "previous_update_id", previous)
        for name, value in identifiers.items():
            object.__setattr__(self, name, value)

        expected = _sha256_json(self.identity_record())
        supplied = self.update_id
        if supplied is not None and require_sha256(supplied, name="update_id") != expected:
            raise ValueError("observation-factor stream update ID mismatch")
        object.__setattr__(self, "update_id", expected)

    def identity_record(self) -> dict[str, object]:
        """Return the portable path-independent update identity payload."""

        return {
            "update_index": self.update_index,
            "admitted_frame_start": self.admitted_frame_start,
            "causal_frame_stop": self.causal_frame_stop,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "bundle_payload_sha256": self.bundle_payload_sha256,
            "bundle_sequence_id": self.bundle_sequence_id,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "factor_count": self.factor_count,
            "observation_count": self.observation_count,
            "persistent_identity_count": self.persistent_identity_count,
            "observation_identity_sha256": self.observation_identity_sha256,
            "gauge_ids": list(self.gauge_ids),
            "previous_update_id": self.previous_update_id,
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "bundle_manifest_path": self.bundle_manifest_path,
            "update_id": self.update_id,
        }

    @classmethod
    def from_record(cls, value: object) -> ObservationFactorStreamUpdateV1:
        mapping = require_mapping(value, name="observation-factor stream update")
        require_exact_fields(mapping, _UPDATE_FIELDS, name="stream update")
        gauge_ids = require_string_sequence(mapping["gauge_ids"], name="gauge_ids")
        return cls(
            update_index=mapping["update_index"],
            admitted_frame_start=mapping["admitted_frame_start"],
            causal_frame_stop=mapping["causal_frame_stop"],
            bundle_manifest_path=mapping["bundle_manifest_path"],
            bundle_manifest_sha256=mapping["bundle_manifest_sha256"],
            bundle_payload_sha256=mapping["bundle_payload_sha256"],
            bundle_sequence_id=mapping["bundle_sequence_id"],
            case_id=mapping["case_id"],
            stream_id=mapping["stream_id"],
            source_repository=mapping["source_repository"],
            source_revision=mapping["source_revision"],
            factor_count=mapping["factor_count"],
            observation_count=mapping["observation_count"],
            persistent_identity_count=mapping["persistent_identity_count"],
            observation_identity_sha256=mapping["observation_identity_sha256"],
            gauge_ids=gauge_ids,
            previous_update_id=mapping["previous_update_id"],
            update_id=mapping["update_id"],
        )


@dataclass(frozen=True)
class ObservationFactorStreamV1:
    """A portable append-only chain of causal observation-factor updates."""

    sequence_id: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    updates: tuple[ObservationFactorStreamUpdateV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        identifiers = {
            "sequence_id": require_nonempty_string(
                self.sequence_id,
                name="sequence_id",
            ),
            "case_id": require_nonempty_string(self.case_id, name="case_id"),
            "stream_id": require_nonempty_string(self.stream_id, name="stream_id"),
            "source_repository": require_nonempty_string(
                self.source_repository,
                name="source_repository",
            ),
            "source_revision": require_nonempty_string(
                self.source_revision,
                name="source_revision",
            ),
        }
        updates = tuple(self.updates)
        if not updates:
            raise ValueError("an observation-factor stream must contain updates")
        if any(not isinstance(update, ObservationFactorStreamUpdateV1) for update in updates):
            raise ValueError("updates must contain ObservationFactorStreamUpdateV1 values")
        previous: ObservationFactorStreamUpdateV1 | None = None
        for index, update in enumerate(updates):
            if update.update_index != index:
                raise ValueError("stream update indices must be contiguous from zero")
            if update.bundle_sequence_id != identifiers["sequence_id"]:
                raise ValueError("stream update sequence_id changed")
            for name in ("case_id", "stream_id", "source_repository", "source_revision"):
                if getattr(update, name) != identifiers[name]:
                    raise ValueError(f"stream update {name} changed")
            expected_previous = None if previous is None else previous.update_id
            if update.previous_update_id != expected_previous:
                raise ValueError("stream update hash chain is broken")
            if previous is not None and update.admitted_frame_start != previous.causal_frame_stop:
                raise ValueError("stream frame intervals must be contiguous")
            previous = update

        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(
                self.metadata,
                name="observation-factor stream metadata",
            ),
            name="observation-factor stream metadata",
        )
        object.__setattr__(self, "updates", updates)
        object.__setattr__(self, "metadata", metadata)
        for name, value in identifiers.items():
            object.__setattr__(self, name, value)

        expected = _sha256_json(self.identity_record())
        supplied = self.artifact_id
        if supplied is not None and require_sha256(supplied, name="artifact_id") != expected:
            raise ValueError("observation-factor stream artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected)

    @property
    def admitted_frame_start(self) -> int:
        return self.updates[0].admitted_frame_start

    @property
    def causal_frame_stop(self) -> int:
        return self.updates[-1].causal_frame_stop

    @property
    def factor_count(self) -> int:
        return sum(update.factor_count for update in self.updates)

    @property
    def observation_count(self) -> int:
        return sum(update.observation_count for update in self.updates)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": OBSERVATION_FACTOR_STREAM_SCHEMA,
            "schema_version": OBSERVATION_FACTOR_STREAM_VERSION,
            "sequence_id": self.sequence_id,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "metadata": plain_json(self.metadata),
            "updates": [
                update.identity_record() | {"update_id": update.update_id}
                for update in self.updates
            ],
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "artifact_id": self.artifact_id,
            "updates": [update.to_record() for update in self.updates],
        }


def _bundle_update(
    bundle_manifest_path: Path,
    *,
    stream_directory: Path,
    update_index: int,
    admitted_frame_start: int | None,
    previous_update_id: str | None,
) -> ObservationFactorStreamUpdateV1:
    manifest = bundle_manifest_path.resolve()
    relative_manifest = _relative_member(
        manifest,
        root=stream_directory,
        name="bundle_manifest_path",
    )
    manifest_bytes = _read_bytes(
        manifest,
        name="observation-factor bundle manifest",
    )
    manifest_sha = _sha256_bytes(manifest_bytes)
    record = _load_json_bytes(
        manifest_bytes,
        name="observation-factor bundle manifest",
    )
    if record.get("schema") != OBSERVATION_FACTOR_SCHEMA:
        raise ValueError("stream updates require an observation-factor bundle")
    schema_version = require_exact_integer(
        record.get("schema_version"),
        name="observation-factor schema_version",
        minimum=1,
    )
    if schema_version != OBSERVATION_FACTOR_SCHEMA_VERSION:
        raise ValueError("stream updates require observation-factor schema v4")
    covariance = require_mapping(
        record.get("gauge_covariance"),
        name="gauge_covariance",
    )
    require_exact_fields(
        covariance,
        _GAUGE_COVARIANCE_FIELDS,
        name="gauge_covariance",
    )
    if (
        covariance.get("semantics") != "joint-cross-window"
        or covariance.get("cross_window_covariance_preserved") is not True
        or covariance.get("diagonal_blocks_match_gauge_marginals") is not True
    ):
        raise ValueError("stream updates require joint cross-window gauge covariance")
    require_nonempty_string(
        covariance.get("joint_covariance_key"),
        name="joint_covariance_key",
    )
    require_string_sequence(
        covariance.get("ordered_gauge_ids"),
        name="ordered_gauge_ids",
    )

    payload_record = require_mapping(
        record.get("payload"),
        name="observation-factor bundle payload record",
    )
    require_exact_fields(
        payload_record,
        _PAYLOAD_FIELDS,
        name="payload record",
    )
    if payload_record.get("allow_pickle") is not False:
        raise ValueError("observation-factor payload must disable pickle")
    payload_relative = _safe_relative_path(
        payload_record.get("path"),
        name="observation-factor payload path",
    )
    payload_path = _resolved_member(
        manifest.parent,
        payload_relative,
        name="observation-factor payload path",
    )
    payload_sha = require_sha256(
        payload_record.get("sha256"),
        name="observation-factor payload SHA-256",
    )
    payload_bytes = _read_bytes(
        payload_path,
        name="observation-factor payload",
    )
    if _sha256_bytes(payload_bytes) != payload_sha:
        raise ValueError("observation-factor payload checksum mismatch")

    with tempfile.TemporaryDirectory(prefix="prob4d-stream-bundle-") as temporary:
        snapshot_manifest = Path(temporary) / manifest.name
        snapshot_manifest.write_bytes(manifest_bytes)
        snapshot_payload = snapshot_manifest.parent / Path(*PurePosixPath(payload_relative).parts)
        snapshot_payload.parent.mkdir(parents=True, exist_ok=True)
        snapshot_payload.write_bytes(payload_bytes)
        bundle = load_observation_factor_bundle(snapshot_manifest)

    frame_indices = [factor.frame_index for factor in bundle.factors]
    if not frame_indices:
        raise ValueError("stream updates require at least one factor")
    effective_start = (
        min(frame_indices)
        if admitted_frame_start is None
        else _require_nonnegative_integer(
            admitted_frame_start,
            name="admitted_frame_start",
        )
    )
    if min(frame_indices) < effective_start:
        raise ValueError("stream update reintroduces an already admitted frame")
    if max(frame_indices) >= bundle.causal_frame_stop:
        raise ValueError("stream update crosses its exclusive causal frame stop")
    persistent_count, observation_count, identity_sha = _observation_identity_summary(bundle)
    case_id = bundle.case_id
    stream_id = bundle.stream_id
    if case_id is None or stream_id is None:
        raise RuntimeError("validated observation-factor bundle lost case or stream ID")

    return ObservationFactorStreamUpdateV1(
        update_index=update_index,
        admitted_frame_start=effective_start,
        causal_frame_stop=bundle.causal_frame_stop,
        bundle_manifest_path=relative_manifest,
        bundle_manifest_sha256=manifest_sha,
        bundle_payload_sha256=payload_sha,
        bundle_sequence_id=bundle.sequence_id,
        case_id=case_id,
        stream_id=stream_id,
        source_repository=bundle.source_repository,
        source_revision=bundle.source_revision,
        factor_count=len(bundle.factors),
        observation_count=observation_count,
        persistent_identity_count=persistent_count,
        observation_identity_sha256=identity_sha,
        gauge_ids=tuple(gauge.window_id for gauge in bundle.gauges),
        previous_update_id=previous_update_id,
    )


def append_observation_factor_bundle(
    stream: ObservationFactorStreamV1 | None,
    bundle_manifest_path: str | Path,
    *,
    stream_manifest_path: str | Path,
    admitted_frame_start: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ObservationFactorStreamV1:
    """Append one schema-v4 delta bundle without reopening prior frame intervals."""

    stream_path = Path(stream_manifest_path)
    stream_directory = stream_path.resolve().parent
    manifest = Path(bundle_manifest_path)
    if stream is None:
        update_index = 0
        previous_update_id = None
        admitted_start = (
            None
            if admitted_frame_start is None
            else _require_nonnegative_integer(
                admitted_frame_start,
                name="admitted_frame_start",
            )
        )
    else:
        update_index = len(stream.updates)
        previous_update_id = stream.updates[-1].update_id
        expected_start = stream.causal_frame_stop
        admitted_start = (
            expected_start
            if admitted_frame_start is None
            else _require_nonnegative_integer(
                admitted_frame_start,
                name="admitted_frame_start",
            )
        )
        if admitted_start != expected_start:
            raise ValueError("the next stream interval must start at the prior causal stop")

    update = _bundle_update(
        manifest,
        stream_directory=stream_directory,
        update_index=update_index,
        admitted_frame_start=admitted_start,
        previous_update_id=previous_update_id,
    )
    if stream is None:
        return ObservationFactorStreamV1(
            sequence_id=update.bundle_sequence_id,
            case_id=update.case_id,
            stream_id=update.stream_id,
            source_repository=update.source_repository,
            source_revision=update.source_revision,
            updates=(update,),
            metadata={} if metadata is None else metadata,
        )
    if metadata is not None and plain_json(metadata) != plain_json(stream.metadata):
        raise ValueError("stream metadata cannot change while appending")
    return replace(
        stream,
        updates=(*stream.updates, update),
        artifact_id=None,
    )


def _require_append_only_rewrite(
    existing: ObservationFactorStreamV1,
    candidate: ObservationFactorStreamV1,
) -> None:
    for name in (
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
    ):
        if getattr(existing, name) != getattr(candidate, name):
            raise ValueError(f"persisted observation-factor stream {name} changed")
    if plain_json(existing.metadata) != plain_json(candidate.metadata):
        raise ValueError("persisted observation-factor stream metadata changed")
    if len(existing.updates) > len(candidate.updates):
        raise ValueError("observation-factor stream persistence cannot roll back updates")
    if candidate.updates[: len(existing.updates)] != existing.updates:
        raise ValueError("observation-factor stream persistence cannot fork its update chain")


def write_observation_factor_stream(
    stream: ObservationFactorStreamV1,
    path: str | Path,
) -> Path:
    """Persist an idempotent append-only stream under an exclusive lock."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_stream_lock(output):
        existing_bytes: bytes | None = None
        if output.exists():
            existing_bytes = _read_bytes(
                output,
                name="observation-factor stream manifest",
            )
            existing = load_observation_factor_stream(
                output,
                validate_bundles=False,
            )
            _require_append_only_rewrite(existing, stream)

        serialized = (
            json.dumps(
                stream.to_record(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            text=True,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            if existing_bytes is None:
                os.link(temporary, output)
            else:
                current_bytes = _read_bytes(
                    output,
                    name="observation-factor stream manifest",
                )
                if current_bytes != existing_bytes:
                    raise RuntimeError("observation-factor stream changed during publication")
                os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


def load_observation_factor_stream(
    path: str | Path,
    *,
    validate_bundles: bool = True,
) -> ObservationFactorStreamV1:
    """Load a stream and optionally revalidate every referenced bundle and payload."""

    manifest = Path(path)
    record = load_json_object(manifest, name="observation-factor stream manifest")
    require_exact_fields(record, _STREAM_FIELDS, name="stream manifest")
    if record.get("schema") != OBSERVATION_FACTOR_STREAM_SCHEMA:
        raise ValueError("manifest is not an observation-factor stream")
    version = require_exact_integer(
        record.get("schema_version"),
        name="observation-factor stream schema_version",
        minimum=1,
    )
    if version != OBSERVATION_FACTOR_STREAM_VERSION:
        raise ValueError("unsupported observation-factor stream version")
    raw_updates = record.get("updates")
    if type(raw_updates) is not list or not raw_updates:
        raise ValueError("observation-factor stream has no updates")
    updates = tuple(ObservationFactorStreamUpdateV1.from_record(value) for value in raw_updates)
    stream = ObservationFactorStreamV1(
        sequence_id=record.get("sequence_id"),
        case_id=record.get("case_id"),
        stream_id=record.get("stream_id"),
        source_repository=record.get("source_repository"),
        source_revision=record.get("source_revision"),
        updates=updates,
        metadata=require_finite_json_mapping(
            record.get("metadata"),
            name="observation-factor stream metadata",
        ),
        artifact_id=record.get("artifact_id"),
    )
    if validate_bundles:
        for update in stream.updates:
            bundle_path = _resolved_member(
                manifest.parent,
                update.bundle_manifest_path,
                name="bundle_manifest_path",
            )
            recomputed = _bundle_update(
                bundle_path,
                stream_directory=manifest.parent,
                update_index=update.update_index,
                admitted_frame_start=update.admitted_frame_start,
                previous_update_id=update.previous_update_id,
            )
            if recomputed != update:
                raise ValueError("observation-factor stream update no longer matches its bundle")
    return stream


__all__ = [
    "OBSERVATION_FACTOR_STREAM_SCHEMA",
    "OBSERVATION_FACTOR_STREAM_VERSION",
    "ObservationFactorStreamUpdateV1",
    "ObservationFactorStreamV1",
    "append_observation_factor_bundle",
    "load_observation_factor_stream",
    "write_observation_factor_stream",
]
