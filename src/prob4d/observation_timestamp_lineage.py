"""Content-addressed observation timestamps and clock lineage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ._immutable_array import immutable_array, immutable_integer_array
from ._immutable_json import frozen_finite_json_mapping, plain_json

OBSERVATION_TIMESTAMP_LINEAGE_SCHEMA = "prob4d.observation-timestamp-lineage"
OBSERVATION_TIMESTAMP_LINEAGE_VERSION = 1
TIMESTAMP_UNCERTAINTY_SEMANTICS = "conditional-jitter-excludes-shared-clock-offset"
_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "sequence_id",
        "case_id",
        "stream_id",
        "source_revision",
        "source_artifact_sha256",
        "causal_frame_stop",
        "clock_domain",
        "time_scale",
        "timestamp_source",
        "factor_ids",
        "frame_indices",
        "timestamps_ns",
        "conditional_timestamp_std_ns",
        "timestamp_uncertainty_semantics",
        "shared_clock_offset_prior_artifact_id",
        "metadata",
    }
)


class _FactorLike(Protocol):
    factor_id: str
    frame_index: int


class _BundleLike(Protocol):
    sequence_id: str
    case_id: str | None
    stream_id: str | None
    source_revision: str
    causal_frame_stop: int
    factors: Sequence[_FactorLike]


def _require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_nonempty_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_revision(value: object, *, name: str) -> str:
    revision = _require_nonempty_string(value, name=name)
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must be an exact lowercase Git commit")
    return revision


def _require_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _unique_strings(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(_require_nonempty_string(value, name=f"{name} entry") for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


@dataclass(frozen=True, slots=True)
class ObservationTimestampLineageV1:
    """Portable factor timestamps with explicit local and shared uncertainty."""

    sequence_id: str
    case_id: str
    stream_id: str
    source_revision: str
    source_artifact_sha256: str
    causal_frame_stop: int
    clock_domain: str
    time_scale: str
    timestamp_source: str
    factor_ids: tuple[str, ...]
    frame_indices: np.ndarray
    timestamps_ns: np.ndarray
    conditional_timestamp_std_ns: np.ndarray
    shared_clock_offset_prior_artifact_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp_uncertainty_semantics: str = TIMESTAMP_UNCERTAINTY_SEMANTICS
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        sequence_id = _require_nonempty_string(self.sequence_id, name="sequence_id")
        case_id = _require_nonempty_string(self.case_id, name="case_id")
        stream_id = _require_nonempty_string(self.stream_id, name="stream_id")
        revision = _require_revision(self.source_revision, name="source_revision")
        source_artifact = _require_sha256(
            self.source_artifact_sha256,
            name="source_artifact_sha256",
        )
        causal_frame_stop = _require_positive_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
        )
        clock_domain = _require_nonempty_string(
            self.clock_domain,
            name="clock_domain",
        )
        time_scale = _require_nonempty_string(self.time_scale, name="time_scale")
        timestamp_source = _require_nonempty_string(
            self.timestamp_source,
            name="timestamp_source",
        )
        factor_ids = _unique_strings(self.factor_ids, name="factor_ids")
        frame_indices = immutable_integer_array(
            self.frame_indices,
            name="frame_indices",
        )
        timestamps = immutable_integer_array(
            self.timestamps_ns,
            name="timestamps_ns",
        )
        conditional_std = immutable_array(
            self.conditional_timestamp_std_ns,
            dtype=np.float64,
        )
        expected_shape = (len(factor_ids),)
        for name, value in (
            ("frame_indices", frame_indices),
            ("timestamps_ns", timestamps),
            ("conditional_timestamp_std_ns", conditional_std),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
        if np.any(frame_indices < 0) or np.any(frame_indices >= causal_frame_stop):
            raise ValueError("frame_indices cross the declared causal frame stop")
        if np.any(timestamps < 0):
            raise ValueError("timestamps_ns must be nonnegative")
        if not np.all(np.isfinite(conditional_std)) or np.any(conditional_std < 0.0):
            raise ValueError("conditional_timestamp_std_ns must be finite and nonnegative")
        if self.timestamp_uncertainty_semantics != TIMESTAMP_UNCERTAINTY_SEMANTICS:
            raise ValueError("timestamp uncertainty semantics changed")
        shared_prior = self.shared_clock_offset_prior_artifact_id
        if shared_prior is not None:
            shared_prior = _require_sha256(
                shared_prior,
                name="shared_clock_offset_prior_artifact_id",
            )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="observation timestamp metadata",
        )

        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "source_artifact_sha256", source_artifact)
        object.__setattr__(self, "causal_frame_stop", causal_frame_stop)
        object.__setattr__(self, "clock_domain", clock_domain)
        object.__setattr__(self, "time_scale", time_scale)
        object.__setattr__(self, "timestamp_source", timestamp_source)
        object.__setattr__(self, "factor_ids", factor_ids)
        object.__setattr__(self, "frame_indices", frame_indices)
        object.__setattr__(self, "timestamps_ns", timestamps)
        object.__setattr__(self, "conditional_timestamp_std_ns", conditional_std)
        object.__setattr__(self, "shared_clock_offset_prior_artifact_id", shared_prior)
        object.__setattr__(self, "metadata", metadata)

        expected_id = _sha256_json(self.identity_record())
        supplied_id = self.artifact_id
        if (
            supplied_id is not None
            and _require_sha256(
                supplied_id,
                name="artifact_id",
            )
            != expected_id
        ):
            raise ValueError("observation timestamp lineage artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected_id)

    def identity_record(self) -> dict[str, Any]:
        """Return the content-addressed payload without its derived ID."""

        return {
            "schema": OBSERVATION_TIMESTAMP_LINEAGE_SCHEMA,
            "schema_version": OBSERVATION_TIMESTAMP_LINEAGE_VERSION,
            "sequence_id": self.sequence_id,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "source_revision": self.source_revision,
            "source_artifact_sha256": self.source_artifact_sha256,
            "causal_frame_stop": self.causal_frame_stop,
            "clock_domain": self.clock_domain,
            "time_scale": self.time_scale,
            "timestamp_source": self.timestamp_source,
            "factor_ids": list(self.factor_ids),
            "frame_indices": self.frame_indices.tolist(),
            "timestamps_ns": self.timestamps_ns.tolist(),
            "conditional_timestamp_std_ns": (self.conditional_timestamp_std_ns.tolist()),
            "timestamp_uncertainty_semantics": self.timestamp_uncertainty_semantics,
            "shared_clock_offset_prior_artifact_id": self.shared_clock_offset_prior_artifact_id,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, Any]:
        """Return the complete portable JSON record."""

        return {**self.identity_record(), "artifact_id": self.artifact_id}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> ObservationTimestampLineageV1:
        """Load and fully revalidate a closed-schema record."""

        if set(value) != _FIELDS:
            missing = sorted(_FIELDS - value.keys())
            extra = sorted(value.keys() - _FIELDS)
            raise ValueError(
                f"observation timestamp lineage fields changed; missing={missing}, extra={extra}"
            )
        if value.get("schema") != OBSERVATION_TIMESTAMP_LINEAGE_SCHEMA:
            raise ValueError("unexpected observation timestamp lineage schema")
        if value.get("schema_version") != OBSERVATION_TIMESTAMP_LINEAGE_VERSION:
            raise ValueError("unsupported observation timestamp lineage version")
        metadata = value["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("observation timestamp metadata must be a mapping")
        return cls(
            sequence_id=value["sequence_id"],
            case_id=value["case_id"],
            stream_id=value["stream_id"],
            source_revision=value["source_revision"],
            source_artifact_sha256=value["source_artifact_sha256"],
            causal_frame_stop=value["causal_frame_stop"],
            clock_domain=value["clock_domain"],
            time_scale=value["time_scale"],
            timestamp_source=value["timestamp_source"],
            factor_ids=value["factor_ids"],
            frame_indices=np.asarray(value["frame_indices"]),
            timestamps_ns=np.asarray(value["timestamps_ns"]),
            conditional_timestamp_std_ns=np.asarray(
                value["conditional_timestamp_std_ns"],
                dtype=np.float64,
            ),
            timestamp_uncertainty_semantics=value["timestamp_uncertainty_semantics"],
            shared_clock_offset_prior_artifact_id=value["shared_clock_offset_prior_artifact_id"],
            metadata=metadata,
            artifact_id=value["artifact_id"],
        )


def validate_timestamp_lineage_for_bundle(
    lineage: ObservationTimestampLineageV1, bundle: _BundleLike
) -> None:
    """Require an exact factor-order and causal-identity match."""

    case_id = bundle.sequence_id if bundle.case_id is None else str(bundle.case_id)
    stream_id = bundle.sequence_id if bundle.stream_id is None else str(bundle.stream_id)
    expected_scalars = {
        "sequence_id": bundle.sequence_id,
        "case_id": case_id,
        "stream_id": stream_id,
        "source_revision": bundle.source_revision,
        "causal_frame_stop": int(bundle.causal_frame_stop),
    }
    for name, expected in expected_scalars.items():
        if getattr(lineage, name) != expected:
            raise ValueError(f"timestamp lineage differs from bundle field {name}")
    factor_ids = tuple(factor.factor_id for factor in bundle.factors)
    frame_indices = np.asarray(
        [factor.frame_index for factor in bundle.factors],
        dtype=np.int64,
    )
    if lineage.factor_ids != factor_ids:
        raise ValueError("timestamp lineage factor order differs from bundle")
    if not np.array_equal(lineage.frame_indices, frame_indices):
        raise ValueError("timestamp lineage frame indices differ from bundle")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_observation_timestamp_lineage(
    path: str | Path,
) -> ObservationTimestampLineageV1:
    """Read one exact JSON snapshot and revalidate its content identity."""

    artifact_path = Path(path)
    try:
        payload = artifact_path.read_bytes()
    except OSError as error:
        raise ValueError("observation timestamp lineage is unreadable") from error
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("observation timestamp lineage is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("observation timestamp lineage must be a JSON object")
    return ObservationTimestampLineageV1.from_record(value)


def write_observation_timestamp_lineage(
    lineage: ObservationTimestampLineageV1,
    path: str | Path,
) -> None:
    """Publish one complete sidecar without replacing different content."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.is_symlink():
        raise ValueError("observation timestamp lineage target must not be a symlink")
    if artifact_path.exists():
        existing = load_observation_timestamp_lineage(artifact_path)
        if existing.artifact_id != lineage.artifact_id:
            raise ValueError("observation timestamp lineage path contains different content")
        return
    payload = (
        json.dumps(
            lineage.to_record(),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact_path.name}.",
        suffix=".tmp",
        dir=artifact_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, artifact_path)
        except FileExistsError:
            if artifact_path.is_symlink():
                raise ValueError(
                    "observation timestamp lineage target must not be a symlink"
                ) from None
            existing = load_observation_timestamp_lineage(artifact_path)
            if existing.artifact_id != lineage.artifact_id:
                raise ValueError(
                    "observation timestamp lineage publication raced with different content"
                ) from None
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "OBSERVATION_TIMESTAMP_LINEAGE_SCHEMA",
    "OBSERVATION_TIMESTAMP_LINEAGE_VERSION",
    "TIMESTAMP_UNCERTAINTY_SEMANTICS",
    "ObservationTimestampLineageV1",
    "load_observation_timestamp_lineage",
    "validate_timestamp_lineage_for_bundle",
    "write_observation_timestamp_lineage",
]
