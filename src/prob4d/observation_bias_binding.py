"""Exact bindings between recursive observation factors and visual-bias streams.

The two source streams remain independently reusable.  This additive contract
proves that every visual-bias update belongs to the same ordered observation
update, frame interval, row count, and observation identity before a downstream
estimator consumes the pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_sha256,
)
from .observation_factor_stream import (
    ObservationFactorStreamV1,
    load_observation_factor_stream,
)
from .visual_bias_stream import (
    VisualBiasNuisanceStreamV1,
    load_visual_bias_nuisance_stream,
)

OBSERVATION_BIAS_BINDING_SCHEMA: Final = "prob4d.observation-bias-stream-binding"
OBSERVATION_BIAS_BINDING_VERSION: Final = 1
OBSERVATION_BIAS_BINDING_UPDATE_SCHEMA: Final = "prob4d.observation-bias-stream-binding-update.v1"
OBSERVATION_BIAS_BINDING_CLAIM_BOUNDARY: Final = (
    "This artifact proves exact structural agreement between one observation-factor "
    "stream and one persistent visual-bias stream. It does not establish provider "
    "competence, bias-model completeness, target calibration, physical-state "
    "identifiability, guarded-query benefit, Causal4D intervention benefit, "
    "deployment safety, or state of the art."
)

_BINDING_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
        "observation_factor_stream_artifact_id",
        "visual_bias_stream_artifact_id",
        "visual_bias_stream_key",
        "visual_bias_model_id",
        "updates",
        "metadata",
        "claim_boundary",
    }
)
_UPDATE_FIELDS: Final = frozenset(
    {
        "schema",
        "update_index",
        "observation_factor_update_id",
        "visual_bias_update_id",
        "visual_bias_artifact_id",
        "observation_artifact_id",
        "observation_identity_sha256",
        "bundle_manifest_sha256",
        "bundle_payload_sha256",
        "frame_start",
        "frame_stop_exclusive",
        "observation_count",
        "row_start",
        "row_stop_exclusive",
        "previous_update_id",
        "update_id",
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


@contextmanager
def _exclusive_binding_lock(path: Path) -> Iterator[None]:
    lock = path.with_name(f".{path.name}.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            f"observation-bias binding is already being written: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


def _atomic_write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ObservationBiasBindingUpdateV1:
    """One exact factor-update to bias-update correspondence."""

    update_index: int
    observation_factor_update_id: str
    visual_bias_update_id: str
    visual_bias_artifact_id: str
    observation_artifact_id: str
    observation_identity_sha256: str
    bundle_manifest_sha256: str
    bundle_payload_sha256: str
    frame_start: int
    frame_stop_exclusive: int
    observation_count: int
    row_start: int
    row_stop_exclusive: int
    previous_update_id: str | None = None
    update_id: str | None = None

    def __post_init__(self) -> None:
        update_index = require_exact_integer(
            self.update_index,
            name="update_index",
            minimum=0,
        )
        frame_start = require_exact_integer(
            self.frame_start,
            name="frame_start",
            minimum=0,
        )
        frame_stop = require_exact_integer(
            self.frame_stop_exclusive,
            name="frame_stop_exclusive",
            minimum=1,
        )
        observation_count = require_exact_integer(
            self.observation_count,
            name="observation_count",
            minimum=1,
        )
        row_start = require_exact_integer(
            self.row_start,
            name="row_start",
            minimum=0,
        )
        row_stop = require_exact_integer(
            self.row_stop_exclusive,
            name="row_stop_exclusive",
            minimum=1,
        )
        if frame_stop <= frame_start:
            raise ValueError("binding frame interval must be nonempty")
        if row_stop - row_start != observation_count:
            raise ValueError("binding row interval differs from observation_count")

        for name in (
            "observation_factor_update_id",
            "visual_bias_update_id",
            "visual_bias_artifact_id",
            "observation_artifact_id",
            "observation_identity_sha256",
            "bundle_manifest_sha256",
            "bundle_payload_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=name),
            )
        previous = self.previous_update_id
        if previous is not None:
            previous = require_sha256(previous, name="previous_update_id")

        object.__setattr__(self, "update_index", update_index)
        object.__setattr__(self, "frame_start", frame_start)
        object.__setattr__(self, "frame_stop_exclusive", frame_stop)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "row_start", row_start)
        object.__setattr__(self, "row_stop_exclusive", row_stop)
        object.__setattr__(self, "previous_update_id", previous)

        expected = _sha256_json(self.identity_record())
        supplied = self.update_id
        if (
            supplied is not None
            and require_sha256(
                supplied,
                name="update_id",
            )
            != expected
        ):
            raise ValueError("observation-bias binding update ID mismatch")
        object.__setattr__(self, "update_id", expected)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": OBSERVATION_BIAS_BINDING_UPDATE_SCHEMA,
            "update_index": self.update_index,
            "observation_factor_update_id": self.observation_factor_update_id,
            "visual_bias_update_id": self.visual_bias_update_id,
            "visual_bias_artifact_id": self.visual_bias_artifact_id,
            "observation_artifact_id": self.observation_artifact_id,
            "observation_identity_sha256": self.observation_identity_sha256,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "bundle_payload_sha256": self.bundle_payload_sha256,
            "frame_start": self.frame_start,
            "frame_stop_exclusive": self.frame_stop_exclusive,
            "observation_count": self.observation_count,
            "row_start": self.row_start,
            "row_stop_exclusive": self.row_stop_exclusive,
            "previous_update_id": self.previous_update_id,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "update_id": self.update_id}

    @classmethod
    def from_record(cls, value: object) -> ObservationBiasBindingUpdateV1:
        mapping = require_mapping(value, name="observation-bias binding update")
        require_exact_fields(
            mapping,
            _UPDATE_FIELDS,
            name="observation-bias binding update",
        )
        if mapping["schema"] != OBSERVATION_BIAS_BINDING_UPDATE_SCHEMA:
            raise ValueError("unsupported observation-bias binding update schema")
        previous_value = mapping["previous_update_id"]
        previous = (
            None
            if previous_value is None
            else require_sha256(previous_value, name="previous_update_id")
        )
        return cls(
            update_index=require_exact_integer(
                mapping["update_index"],
                name="update_index",
                minimum=0,
            ),
            observation_factor_update_id=require_sha256(
                mapping["observation_factor_update_id"],
                name="observation_factor_update_id",
            ),
            visual_bias_update_id=require_sha256(
                mapping["visual_bias_update_id"],
                name="visual_bias_update_id",
            ),
            visual_bias_artifact_id=require_sha256(
                mapping["visual_bias_artifact_id"],
                name="visual_bias_artifact_id",
            ),
            observation_artifact_id=require_sha256(
                mapping["observation_artifact_id"],
                name="observation_artifact_id",
            ),
            observation_identity_sha256=require_sha256(
                mapping["observation_identity_sha256"],
                name="observation_identity_sha256",
            ),
            bundle_manifest_sha256=require_sha256(
                mapping["bundle_manifest_sha256"],
                name="bundle_manifest_sha256",
            ),
            bundle_payload_sha256=require_sha256(
                mapping["bundle_payload_sha256"],
                name="bundle_payload_sha256",
            ),
            frame_start=require_exact_integer(
                mapping["frame_start"],
                name="frame_start",
                minimum=0,
            ),
            frame_stop_exclusive=require_exact_integer(
                mapping["frame_stop_exclusive"],
                name="frame_stop_exclusive",
                minimum=1,
            ),
            observation_count=require_exact_integer(
                mapping["observation_count"],
                name="observation_count",
                minimum=1,
            ),
            row_start=require_exact_integer(
                mapping["row_start"],
                name="row_start",
                minimum=0,
            ),
            row_stop_exclusive=require_exact_integer(
                mapping["row_stop_exclusive"],
                name="row_stop_exclusive",
                minimum=1,
            ),
            previous_update_id=previous,
            update_id=require_sha256(mapping["update_id"], name="update_id"),
        )


@dataclass(frozen=True)
class ObservationBiasStreamBindingV1:
    """Content-addressed proof that two recursive streams agree update by update."""

    sequence_id: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    observation_factor_stream_artifact_id: str
    visual_bias_stream_artifact_id: str
    visual_bias_stream_key: str
    visual_bias_model_id: str
    updates: tuple[ObservationBiasBindingUpdateV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "sequence_id",
            "case_id",
            "stream_id",
            "source_repository",
            "source_revision",
            "visual_bias_stream_key",
        ):
            object.__setattr__(
                self,
                name,
                require_exact_string(getattr(self, name), name=name),
            )
        for name in (
            "observation_factor_stream_artifact_id",
            "visual_bias_stream_artifact_id",
            "visual_bias_model_id",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=name),
            )

        if type(self.updates) is not tuple or not self.updates:
            raise ValueError("binding updates must be a nonempty canonical tuple")
        if any(not isinstance(update, ObservationBiasBindingUpdateV1) for update in self.updates):
            raise ValueError("binding updates must contain ObservationBiasBindingUpdateV1 values")
        previous: ObservationBiasBindingUpdateV1 | None = None
        for index, update in enumerate(self.updates):
            if update.update_index != index:
                raise ValueError("binding update indices must be contiguous from zero")
            expected_previous = None if previous is None else previous.update_id
            if update.previous_update_id != expected_previous:
                raise ValueError("observation-bias binding update chain is broken")
            if previous is None:
                if update.row_start != 0:
                    raise ValueError("the first binding update must start at row zero")
            else:
                if update.frame_start != previous.frame_stop_exclusive:
                    raise ValueError("binding frame intervals must be contiguous")
                if update.row_start != previous.row_stop_exclusive:
                    raise ValueError("binding row intervals must be contiguous")
            previous = update

        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(
                self.metadata,
                name="observation-bias binding metadata",
            ),
            name="observation-bias binding metadata",
        )
        object.__setattr__(self, "metadata", metadata)

        expected = _sha256_json(self.identity_record())
        supplied = self.artifact_id
        if (
            supplied is not None
            and require_sha256(
                supplied,
                name="artifact_id",
            )
            != expected
        ):
            raise ValueError("observation-bias binding artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected)

    @property
    def observation_count(self) -> int:
        return self.updates[-1].row_stop_exclusive

    @property
    def admitted_frame_start(self) -> int:
        return self.updates[0].frame_start

    @property
    def causal_frame_stop(self) -> int:
        return self.updates[-1].frame_stop_exclusive

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": OBSERVATION_BIAS_BINDING_SCHEMA,
            "schema_version": OBSERVATION_BIAS_BINDING_VERSION,
            "sequence_id": self.sequence_id,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "observation_factor_stream_artifact_id": (self.observation_factor_stream_artifact_id),
            "visual_bias_stream_artifact_id": self.visual_bias_stream_artifact_id,
            "visual_bias_stream_key": self.visual_bias_stream_key,
            "visual_bias_model_id": self.visual_bias_model_id,
            "updates": [update.to_record() for update in self.updates],
            "metadata": plain_json(self.metadata),
            "claim_boundary": OBSERVATION_BIAS_BINDING_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "artifact_id": self.artifact_id}

    def summary(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "sequence_id": self.sequence_id,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "observation_factor_stream_artifact_id": (self.observation_factor_stream_artifact_id),
            "visual_bias_stream_artifact_id": self.visual_bias_stream_artifact_id,
            "visual_bias_model_id": self.visual_bias_model_id,
            "update_count": len(self.updates),
            "observation_count": self.observation_count,
            "admitted_frame_start": self.admitted_frame_start,
            "causal_frame_stop": self.causal_frame_stop,
            "claim_boundary": OBSERVATION_BIAS_BINDING_CLAIM_BOUNDARY,
        }


def build_observation_bias_binding(
    observation_stream: ObservationFactorStreamV1,
    visual_bias_stream: VisualBiasNuisanceStreamV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ObservationBiasStreamBindingV1:
    """Prove exact ordered agreement between one factor stream and one bias stream."""

    if not isinstance(observation_stream, ObservationFactorStreamV1):
        raise TypeError("observation_stream must be an ObservationFactorStreamV1")
    if not isinstance(visual_bias_stream, VisualBiasNuisanceStreamV1):
        raise TypeError("visual_bias_stream must be a VisualBiasNuisanceStreamV1")
    if len(observation_stream.updates) != len(visual_bias_stream.updates):
        raise ValueError("observation and visual-bias streams have different update counts")
    if observation_stream.observation_count != visual_bias_stream.observation_count:
        raise ValueError("observation and visual-bias streams have different row counts")

    observation_artifact_id = observation_stream.artifact_id
    visual_artifact_id = visual_bias_stream.artifact_id
    bias_model_id = visual_bias_stream.bias_model_id
    if observation_artifact_id is None or visual_artifact_id is None or bias_model_id is None:
        raise ValueError("source streams must have complete content identities")

    updates: list[ObservationBiasBindingUpdateV1] = []
    previous_update_id: str | None = None
    for index, (observation_update, visual_update) in enumerate(
        zip(observation_stream.updates, visual_bias_stream.updates, strict=True)
    ):
        factor_update_id = observation_update.update_id
        visual_update_id = visual_update.update_id
        if factor_update_id is None or visual_update_id is None:
            raise ValueError("source stream update lacks a content identity")
        if visual_update.observation_stream_update_id != factor_update_id:
            raise ValueError(f"visual-bias update {index} references another observation update")
        if (
            visual_update.frame_start != observation_update.admitted_frame_start
            or visual_update.frame_stop_exclusive != observation_update.causal_frame_stop
        ):
            raise ValueError(f"visual-bias update {index} frame interval differs")
        row_count = visual_update.row_stop_exclusive - visual_update.row_start
        if row_count != observation_update.observation_count:
            raise ValueError(f"visual-bias update {index} row count differs")
        if (
            visual_update.observation_identity_sha256
            != observation_update.observation_identity_sha256
        ):
            raise ValueError(f"visual-bias update {index} observation identity differs")

        binding_update = ObservationBiasBindingUpdateV1(
            update_index=index,
            observation_factor_update_id=factor_update_id,
            visual_bias_update_id=visual_update_id,
            visual_bias_artifact_id=visual_update.visual_bias_artifact_id,
            observation_artifact_id=visual_update.observation_artifact_id,
            observation_identity_sha256=visual_update.observation_identity_sha256,
            bundle_manifest_sha256=observation_update.bundle_manifest_sha256,
            bundle_payload_sha256=observation_update.bundle_payload_sha256,
            frame_start=visual_update.frame_start,
            frame_stop_exclusive=visual_update.frame_stop_exclusive,
            observation_count=row_count,
            row_start=visual_update.row_start,
            row_stop_exclusive=visual_update.row_stop_exclusive,
            previous_update_id=previous_update_id,
        )
        updates.append(binding_update)
        previous_update_id = binding_update.update_id

    return ObservationBiasStreamBindingV1(
        sequence_id=observation_stream.sequence_id,
        case_id=observation_stream.case_id,
        stream_id=observation_stream.stream_id,
        source_repository=observation_stream.source_repository,
        source_revision=observation_stream.source_revision,
        observation_factor_stream_artifact_id=observation_artifact_id,
        visual_bias_stream_artifact_id=visual_artifact_id,
        visual_bias_stream_key=visual_bias_stream.stream_key,
        visual_bias_model_id=bias_model_id,
        updates=tuple(updates),
        metadata={} if metadata is None else metadata,
    )


def build_observation_bias_binding_from_paths(
    observation_stream_path: str | Path,
    visual_bias_stream_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    validate_bundles: bool = True,
) -> ObservationBiasStreamBindingV1:
    """Load exact source artifacts and build their cross-stream binding."""

    if type(validate_bundles) is not bool:
        raise TypeError("validate_bundles must be Boolean")
    observation_stream = load_observation_factor_stream(
        observation_stream_path,
        validate_bundles=validate_bundles,
    )
    visual_bias_stream = load_visual_bias_nuisance_stream(visual_bias_stream_path)
    return build_observation_bias_binding(
        observation_stream,
        visual_bias_stream,
        metadata=metadata,
    )


def verify_observation_bias_binding(
    binding: ObservationBiasStreamBindingV1,
    observation_stream: ObservationFactorStreamV1,
    visual_bias_stream: VisualBiasNuisanceStreamV1,
) -> ObservationBiasStreamBindingV1:
    """Rebuild a retained binding from current source objects and compare exactly."""

    if not isinstance(binding, ObservationBiasStreamBindingV1):
        raise TypeError("binding must be an ObservationBiasStreamBindingV1")
    rebuilt = build_observation_bias_binding(
        observation_stream,
        visual_bias_stream,
        metadata=binding.metadata,
    )
    if rebuilt != binding:
        raise ValueError("observation-bias binding no longer matches its source streams")
    return binding


def verify_observation_bias_binding_from_paths(
    binding_path: str | Path,
    observation_stream_path: str | Path,
    visual_bias_stream_path: str | Path,
    *,
    validate_bundles: bool = True,
) -> ObservationBiasStreamBindingV1:
    """Revalidate a retained binding against exact current source artifacts."""

    binding = load_observation_bias_binding(binding_path)
    observation_stream = load_observation_factor_stream(
        observation_stream_path,
        validate_bundles=validate_bundles,
    )
    visual_bias_stream = load_visual_bias_nuisance_stream(visual_bias_stream_path)
    return verify_observation_bias_binding(
        binding,
        observation_stream,
        visual_bias_stream,
    )


def _require_append_only_rewrite(
    existing: ObservationBiasStreamBindingV1,
    candidate: ObservationBiasStreamBindingV1,
) -> None:
    for name in (
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
        "visual_bias_stream_key",
        "visual_bias_model_id",
    ):
        if getattr(existing, name) != getattr(candidate, name):
            raise ValueError(f"persisted observation-bias binding {name} changed")
    if plain_json(existing.metadata) != plain_json(candidate.metadata):
        raise ValueError("persisted observation-bias binding metadata changed")
    if len(existing.updates) > len(candidate.updates):
        raise ValueError("observation-bias binding persistence cannot roll back updates")
    if candidate.updates[: len(existing.updates)] != existing.updates:
        raise ValueError("observation-bias binding persistence cannot fork its update chain")
    if len(existing.updates) == len(candidate.updates):
        raise ValueError(
            "observation-bias binding can change source stream IDs only when appending"
        )


def write_observation_bias_binding(
    binding: ObservationBiasStreamBindingV1,
    path: str | Path,
) -> Path:
    """Persist one idempotent append-only cross-stream binding."""

    if not isinstance(binding, ObservationBiasStreamBindingV1):
        raise TypeError("binding must be an ObservationBiasStreamBindingV1")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_binding_lock(output):
        if output.exists():
            existing = load_observation_bias_binding(output)
            if existing.artifact_id == binding.artifact_id:
                return output
            _require_append_only_rewrite(existing, binding)
        _atomic_write_json(output, binding.to_record())
    return output


def load_observation_bias_binding(path: str | Path) -> ObservationBiasStreamBindingV1:
    """Load one strict content-addressed observation/bias stream binding."""

    record = load_json_object(path, name="observation-bias binding manifest")
    require_exact_fields(record, _BINDING_FIELDS, name="observation-bias binding manifest")
    if record["schema"] != OBSERVATION_BIAS_BINDING_SCHEMA:
        raise ValueError("manifest is not an observation-bias stream binding")
    version = require_exact_integer(
        record["schema_version"],
        name="observation-bias binding schema_version",
        minimum=1,
    )
    if version != OBSERVATION_BIAS_BINDING_VERSION:
        raise ValueError("unsupported observation-bias binding version")
    if record["claim_boundary"] != OBSERVATION_BIAS_BINDING_CLAIM_BOUNDARY:
        raise ValueError("observation-bias binding claim boundary changed")
    raw_updates = record["updates"]
    if type(raw_updates) is not list or not raw_updates:
        raise ValueError("observation-bias binding has no updates")
    return ObservationBiasStreamBindingV1(
        sequence_id=require_exact_string(record["sequence_id"], name="sequence_id"),
        case_id=require_exact_string(record["case_id"], name="case_id"),
        stream_id=require_exact_string(record["stream_id"], name="stream_id"),
        source_repository=require_exact_string(
            record["source_repository"],
            name="source_repository",
        ),
        source_revision=require_exact_string(
            record["source_revision"],
            name="source_revision",
        ),
        observation_factor_stream_artifact_id=require_sha256(
            record["observation_factor_stream_artifact_id"],
            name="observation_factor_stream_artifact_id",
        ),
        visual_bias_stream_artifact_id=require_sha256(
            record["visual_bias_stream_artifact_id"],
            name="visual_bias_stream_artifact_id",
        ),
        visual_bias_stream_key=require_exact_string(
            record["visual_bias_stream_key"],
            name="visual_bias_stream_key",
        ),
        visual_bias_model_id=require_sha256(
            record["visual_bias_model_id"],
            name="visual_bias_model_id",
        ),
        updates=tuple(ObservationBiasBindingUpdateV1.from_record(value) for value in raw_updates),
        metadata=require_finite_json_mapping(
            record["metadata"],
            name="observation-bias binding metadata",
        ),
        artifact_id=require_sha256(record["artifact_id"], name="artifact_id"),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d observation bias-binding",
        description=(
            "Build, validate, or replay an exact binding between recursive "
            "observation factors and a persistent visual-bias stream."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build and persist an exact binding")
    build.add_argument("observation_stream")
    build.add_argument("visual_bias_stream")
    build.add_argument("output")

    validate = subparsers.add_parser("validate", help="validate a retained binding")
    validate.add_argument("binding")

    verify = subparsers.add_parser(
        "verify",
        help="replay a retained binding against both current source streams",
    )
    verify.add_argument("binding")
    verify.add_argument("observation_stream")
    verify.add_argument("visual_bias_stream")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    if arguments.command == "build":
        binding = build_observation_bias_binding_from_paths(
            arguments.observation_stream,
            arguments.visual_bias_stream,
        )
        write_observation_bias_binding(binding, arguments.output)
    elif arguments.command == "validate":
        binding = load_observation_bias_binding(arguments.binding)
    elif arguments.command == "verify":
        binding = verify_observation_bias_binding_from_paths(
            arguments.binding,
            arguments.observation_stream,
            arguments.visual_bias_stream,
        )
    else:
        parser.error("unsupported observation-bias binding command")
        return 2
    print(json.dumps(binding.summary(), indent=2, sort_keys=True))
    return 0


__all__ = [
    "OBSERVATION_BIAS_BINDING_CLAIM_BOUNDARY",
    "OBSERVATION_BIAS_BINDING_SCHEMA",
    "OBSERVATION_BIAS_BINDING_UPDATE_SCHEMA",
    "OBSERVATION_BIAS_BINDING_VERSION",
    "ObservationBiasBindingUpdateV1",
    "ObservationBiasStreamBindingV1",
    "build_observation_bias_binding",
    "build_observation_bias_binding_from_paths",
    "load_observation_bias_binding",
    "main",
    "verify_observation_bias_binding",
    "verify_observation_bias_binding_from_paths",
    "write_observation_bias_binding",
]


if __name__ == "__main__":
    raise SystemExit(main())
