"""Strict provider-v2 envelopes for claim-bearing observation-factor bundles.

The neutral schema-v4 :class:`ObservationFactorBundle` remains reusable by
exploratory and frozen consumers.  This module adds a path-independent,
content-addressed envelope for new claim-bearing experiments.  The envelope
binds the bundle manifest and payload bytes, causal source lineage, covariance
calibration identities, the complete provider-v2 attestation, and exact runtime
revision evidence without changing the neutral bundle schema.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ._causal_observation_source import CausalOverlapSelection
from ._immutable_json import frozen_finite_json_mapping, plain_json
from .calibration_compatibility import (
    assert_calibration_pair_compatible,
    load_prediction_calibration_target,
)
from .observation_factors import (
    OBSERVATION_FACTOR_SCHEMA_VERSION,
    ObservationFactorBundle,
    load_observation_factor_bundle,
    write_observation_factor_bundle,
)
from .provider_attestation import (
    PROVIDER_SOURCE_REPOSITORY,
    build_provider_attestation,
    validate_provider_attestation,
)
from .provider_v1 import (
    GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1,
)
from .runtime_revision import assert_runtime_revision

CLAIM_BEARING_FACTOR_BUNDLE_SCHEMA = (
    "prob4d.claim-bearing-observation-factor-bundle"
)
CLAIM_BEARING_FACTOR_BUNDLE_VERSION = 1
_REQUIRED_PROVIDER_CAPABILITIES = frozenset(
    {
        "joint_cross_window_sim3_gauge_covariance_in_factor_bundle",
        "provider_attested_observation_artifacts",
        "strict_claim_bearing_observation_loading",
    }
)
_ENVELOPE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "bundle_manifest_path",
        "bundle_manifest_sha256",
        "bundle_payload_sha256",
        "bundle_schema_version",
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
        "causal_frame_stop",
        "factor_count",
        "observation_count",
        "gauge_ids",
        "gauge_covariance_semantics",
        "cross_window_gauge_covariance_preserved",
        "causal_source_lineage",
        "provider_manifest_id",
        "calibration_artifact_ids",
        "runtime_revision_source",
        "runtime_revision_independently_verified",
        "provider_attestation",
        "metadata",
    }
)
_CALIBRATION_FIELDS = frozenset(
    {"gauge_artifact_id", "point_artifact_id"}
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
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read claim-bearing member {path.name!r}") from error
    return digest.hexdigest()


def _require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_nonempty_string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
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
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _safe_relative_path(value: object, *, name: str) -> str:
    path = _require_nonempty_string(value, name=name)
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
        raise ValueError(f"{name} escapes the envelope directory") from error
    return candidate


def _relative_member(path: Path, *, root: Path, name: str) -> str:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        relative = path_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} must lie inside the envelope directory") from error
    return _safe_relative_path(relative.as_posix(), name=name)


def _validated_calibration_ids(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("calibration_artifact_ids must be a mapping")
    if set(value) != _CALIBRATION_FIELDS:
        raise ValueError(
            "calibration_artifact_ids must contain exactly gauge_artifact_id "
            "and point_artifact_id"
        )
    result: dict[str, str] = {}
    for name in sorted(_CALIBRATION_FIELDS):
        result[name] = _require_sha256(
            value[name],
            name=f"calibration {name}",
        )
    return frozen_finite_json_mapping(
        result,
        name="claim-bearing calibration artifact IDs",
    )


def _validated_gauge_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("gauge_ids must be a nonempty sequence")
    result = tuple(
        _require_nonempty_string(item, name="gauge_id") for item in value
    )
    if len(set(result)) != len(result):
        raise ValueError("gauge_ids must be unique")
    return result


def _validated_lineage(
    value: object,
    *,
    causal_frame_stop: int,
    gauge_ids: tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("causal_source_lineage must be a mapping")
    lineage = frozen_finite_json_mapping(
        value,
        name="claim-bearing causal source lineage",
    )
    if lineage.get("schema_version") != 1:
        raise ValueError("claim-bearing causal source lineage changed schema version")
    if lineage.get("producer") != "Prob4D":
        raise ValueError("claim-bearing causal source lineage changed producer")
    if lineage.get("source_product") != "independently_decoded_overlap_windows":
        raise ValueError("claim-bearing factors require independent overlap windows")
    if lineage.get("causal_frame_stop_exclusive") != causal_frame_stop:
        raise ValueError("causal source lineage differs from the bundle cutoff")
    if lineage.get("future_prediction_payloads_opened") != 0:
        raise ValueError("claim-bearing factor export opened future prediction payloads")
    _require_sha256(
        lineage.get("source_artifact_sha256"),
        name="causal source artifact SHA-256",
    )

    selected = lineage.get("selected_windows")
    if not isinstance(selected, list) or not selected:
        raise ValueError("claim-bearing factors require selected source-window lineage")
    selected_ids: set[str] = set()
    for raw_window in selected:
        if not isinstance(raw_window, Mapping):
            raise ValueError("selected source-window lineage must contain mappings")
        window_id = _require_nonempty_string(
            raw_window.get("window_id"),
            name="selected source window_id",
        )
        if window_id in selected_ids:
            raise ValueError("selected source window IDs must be unique")
        selected_ids.add(window_id)
        start = raw_window.get("source_frame_start")
        stop = raw_window.get("source_frame_stop_exclusive")
        maximum = raw_window.get("source_frame_max")
        for name, frame in (
            ("source_frame_start", start),
            ("source_frame_stop_exclusive", stop),
            ("source_frame_max", maximum),
        ):
            if isinstance(frame, bool) or not isinstance(frame, int):
                raise ValueError(f"selected source-window {name} must be an integer")
        start = int(start)
        stop = int(stop)
        maximum = int(maximum)
        if start < 0 or stop <= start or maximum < start or maximum >= stop:
            raise ValueError("selected source-window frame bounds are inconsistent")
        if maximum >= causal_frame_stop:
            raise ValueError("selected source window crosses the causal frame boundary")
        _require_sha256(
            raw_window.get("payload_sha256"),
            name="selected source-window payload SHA-256",
        )
        _require_sha256(
            raw_window.get("frame_indices_sha256"),
            name="selected source-window frame-index SHA-256",
        )
    if not set(gauge_ids).issubset(selected_ids):
        raise ValueError("bundle gauges are not all present in causal source lineage")
    return lineage


def _provider_fields(
    attestation: Mapping[str, Any],
    *,
    source_revision: str,
) -> tuple[Mapping[str, Any], str, Mapping[str, str], str, bool]:
    validated = validate_provider_attestation(
        attestation,
        source_revision=source_revision,
        require_claim_bearing=True,
    )
    manifest = validated["provider_manifest"]
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, list) or not _REQUIRED_PROVIDER_CAPABILITIES.issubset(
        capabilities
    ):
        raise ValueError(
            "provider manifest lacks claim-bearing observation-factor capabilities"
        )
    provider_manifest_id = _require_sha256(
        validated.get("provider_manifest_id"),
        name="provider_manifest_id",
    )
    calibration_ids = _validated_calibration_ids(
        validated.get("calibration_artifact_ids")
    )
    runtime = validated.get("runtime_revision")
    if not isinstance(runtime, Mapping):
        raise ValueError("validated provider runtime revision must be a mapping")
    runtime_source = _require_nonempty_string(
        runtime.get("source"),
        name="runtime_revision_source",
    )
    runtime_verified = runtime.get("independently_verified")
    if runtime_verified is not True:
        raise ValueError("claim-bearing factor runtime was not independently verified")
    return (
        frozen_finite_json_mapping(
            validated,
            name="claim-bearing provider attestation",
        ),
        provider_manifest_id,
        calibration_ids,
        runtime_source,
        True,
    )


@dataclass(frozen=True, slots=True)
class ClaimBearingObservationFactorBundleEnvelopeV1:
    """Portable identity and provenance for one neutral schema-v4 factor bundle."""

    bundle_manifest_path: str
    bundle_manifest_sha256: str
    bundle_payload_sha256: str
    sequence_id: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    causal_frame_stop: int
    factor_count: int
    observation_count: int
    gauge_ids: tuple[str, ...]
    causal_source_lineage: Mapping[str, Any]
    provider_manifest_id: str
    calibration_artifact_ids: Mapping[str, str]
    runtime_revision_source: str
    runtime_revision_independently_verified: bool
    provider_attestation: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    bundle_schema_version: int = OBSERVATION_FACTOR_SCHEMA_VERSION
    gauge_covariance_semantics: str = "joint-cross-window"
    cross_window_gauge_covariance_preserved: bool = True
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        manifest_path = _safe_relative_path(
            self.bundle_manifest_path,
            name="bundle_manifest_path",
        )
        manifest_sha = _require_sha256(
            self.bundle_manifest_sha256,
            name="bundle_manifest_sha256",
        )
        payload_sha = _require_sha256(
            self.bundle_payload_sha256,
            name="bundle_payload_sha256",
        )
        sequence_id = _require_nonempty_string(self.sequence_id, name="sequence_id")
        case_id = _require_nonempty_string(self.case_id, name="case_id")
        stream_id = _require_nonempty_string(self.stream_id, name="stream_id")
        repository = _require_nonempty_string(
            self.source_repository,
            name="source_repository",
        )
        if repository != PROVIDER_SOURCE_REPOSITORY:
            raise ValueError("claim-bearing factor bundle must be produced by Prob4D")
        revision = _require_revision(self.source_revision, name="source_revision")
        causal_frame_stop = _require_positive_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
        )
        factor_count = _require_positive_integer(
            self.factor_count,
            name="factor_count",
        )
        observation_count = _require_positive_integer(
            self.observation_count,
            name="observation_count",
        )
        gauge_ids = _validated_gauge_ids(self.gauge_ids)
        if (
            isinstance(self.bundle_schema_version, bool)
            or not isinstance(self.bundle_schema_version, int)
            or self.bundle_schema_version != OBSERVATION_FACTOR_SCHEMA_VERSION
        ):
            raise ValueError("claim-bearing factor envelope requires schema-v4 bundles")
        covariance_semantics = _require_nonempty_string(
            self.gauge_covariance_semantics,
            name="gauge_covariance_semantics",
        )
        if covariance_semantics != "joint-cross-window":
            raise ValueError("claim-bearing factor envelope requires joint gauge covariance")
        if self.cross_window_gauge_covariance_preserved is not True:
            raise ValueError("claim-bearing factor envelope lost cross-window covariance")
        lineage = _validated_lineage(
            self.causal_source_lineage,
            causal_frame_stop=causal_frame_stop,
            gauge_ids=gauge_ids,
        )
        (
            attestation,
            attested_manifest_id,
            attested_calibration_ids,
            attested_runtime_source,
            attested_runtime_verified,
        ) = _provider_fields(self.provider_attestation, source_revision=revision)
        manifest_id = _require_sha256(
            self.provider_manifest_id,
            name="provider_manifest_id",
        )
        calibration_ids = _validated_calibration_ids(
            self.calibration_artifact_ids
        )
        runtime_source = _require_nonempty_string(
            self.runtime_revision_source,
            name="runtime_revision_source",
        )
        if self.runtime_revision_independently_verified is not True:
            raise ValueError(
                "runtime_revision_independently_verified must be literally True"
            )
        if manifest_id != attested_manifest_id:
            raise ValueError("provider_manifest_id differs from provider attestation")
        if dict(calibration_ids) != dict(attested_calibration_ids):
            raise ValueError("calibration_artifact_ids differ from provider attestation")
        if runtime_source != attested_runtime_source:
            raise ValueError("runtime_revision_source differs from provider attestation")
        if self.runtime_revision_independently_verified is not attested_runtime_verified:
            raise ValueError("runtime verification differs from provider attestation")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="claim-bearing factor envelope metadata",
        )

        object.__setattr__(self, "bundle_manifest_path", manifest_path)
        object.__setattr__(self, "bundle_manifest_sha256", manifest_sha)
        object.__setattr__(self, "bundle_payload_sha256", payload_sha)
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "source_repository", repository)
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "causal_frame_stop", causal_frame_stop)
        object.__setattr__(self, "factor_count", factor_count)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "gauge_ids", gauge_ids)
        object.__setattr__(self, "causal_source_lineage", lineage)
        object.__setattr__(self, "provider_manifest_id", manifest_id)
        object.__setattr__(self, "calibration_artifact_ids", calibration_ids)
        object.__setattr__(self, "runtime_revision_source", runtime_source)
        object.__setattr__(
            self,
            "runtime_revision_independently_verified",
            True,
        )
        object.__setattr__(self, "provider_attestation", attestation)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "gauge_covariance_semantics", covariance_semantics)

        expected = _sha256_json(self.identity_record())
        supplied = self.artifact_id
        if supplied is not None and _require_sha256(
            supplied,
            name="artifact_id",
        ) != expected:
            raise ValueError("claim-bearing factor envelope artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected)

    def identity_record(self) -> dict[str, Any]:
        """Return the path-independent identity payload."""

        return {
            "schema": CLAIM_BEARING_FACTOR_BUNDLE_SCHEMA,
            "schema_version": CLAIM_BEARING_FACTOR_BUNDLE_VERSION,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "bundle_payload_sha256": self.bundle_payload_sha256,
            "bundle_schema_version": self.bundle_schema_version,
            "sequence_id": self.sequence_id,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "causal_frame_stop": self.causal_frame_stop,
            "factor_count": self.factor_count,
            "observation_count": self.observation_count,
            "gauge_ids": list(self.gauge_ids),
            "gauge_covariance_semantics": self.gauge_covariance_semantics,
            "cross_window_gauge_covariance_preserved": (
                self.cross_window_gauge_covariance_preserved
            ),
            "causal_source_lineage": plain_json(self.causal_source_lineage),
            "provider_manifest_id": self.provider_manifest_id,
            "calibration_artifact_ids": plain_json(
                self.calibration_artifact_ids
            ),
            "runtime_revision_source": self.runtime_revision_source,
            "runtime_revision_independently_verified": (
                self.runtime_revision_independently_verified
            ),
            "provider_attestation": plain_json(self.provider_attestation),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.identity_record(),
            "bundle_manifest_path": self.bundle_manifest_path,
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_record(
        cls,
        value: Mapping[str, Any],
    ) -> ClaimBearingObservationFactorBundleEnvelopeV1:
        if set(value) != _ENVELOPE_FIELDS:
            missing = sorted(_ENVELOPE_FIELDS - value.keys())
            extra = sorted(value.keys() - _ENVELOPE_FIELDS)
            raise ValueError(
                "claim-bearing factor envelope fields changed; "
                f"missing={missing}, extra={extra}"
            )
        if value.get("schema") != CLAIM_BEARING_FACTOR_BUNDLE_SCHEMA:
            raise ValueError("unexpected claim-bearing factor envelope schema")
        if value.get("schema_version") != CLAIM_BEARING_FACTOR_BUNDLE_VERSION:
            raise ValueError("unsupported claim-bearing factor envelope version")
        return cls(
            bundle_manifest_path=value["bundle_manifest_path"],
            bundle_manifest_sha256=value["bundle_manifest_sha256"],
            bundle_payload_sha256=value["bundle_payload_sha256"],
            bundle_schema_version=value["bundle_schema_version"],
            sequence_id=value["sequence_id"],
            case_id=value["case_id"],
            stream_id=value["stream_id"],
            source_repository=value["source_repository"],
            source_revision=value["source_revision"],
            causal_frame_stop=value["causal_frame_stop"],
            factor_count=value["factor_count"],
            observation_count=value["observation_count"],
            gauge_ids=value["gauge_ids"],
            gauge_covariance_semantics=value["gauge_covariance_semantics"],
            cross_window_gauge_covariance_preserved=value[
                "cross_window_gauge_covariance_preserved"
            ],
            causal_source_lineage=value["causal_source_lineage"],
            provider_manifest_id=value["provider_manifest_id"],
            calibration_artifact_ids=value["calibration_artifact_ids"],
            runtime_revision_source=value["runtime_revision_source"],
            runtime_revision_independently_verified=value[
                "runtime_revision_independently_verified"
            ],
            provider_attestation=value["provider_attestation"],
            metadata=value["metadata"],
            artifact_id=value["artifact_id"],
        )


@dataclass(frozen=True, slots=True)
class ValidatedClaimBearingObservationFactorBundle:
    """A loaded neutral factor bundle bound to a validated provider-v2 envelope."""

    bundle: ObservationFactorBundle
    envelope: ClaimBearingObservationFactorBundleEnvelopeV1

    def __post_init__(self) -> None:
        _validate_bundle_against_envelope(self.bundle, self.envelope)

    @property
    def artifact_id(self) -> str:
        artifact_id = self.envelope.artifact_id
        if artifact_id is None:
            raise RuntimeError("validated claim-bearing envelope lost its artifact ID")
        return artifact_id

    @property
    def provider_manifest_id(self) -> str:
        return self.envelope.provider_manifest_id

    @property
    def gauge_calibration_id(self) -> str:
        return self.envelope.calibration_artifact_ids["gauge_artifact_id"]

    @property
    def point_calibration_id(self) -> str:
        return self.envelope.calibration_artifact_ids["point_artifact_id"]


def _factor_observation_count(bundle: ObservationFactorBundle) -> int:
    return sum(len(factor.point_ids) for factor in bundle.factors)


def _lineage_window_bounds(
    lineage: Mapping[str, Any],
) -> dict[str, tuple[int, int]]:
    selected = lineage.get("selected_windows")
    if not isinstance(selected, list):
        raise ValueError("causal source lineage selected_windows changed type")
    result: dict[str, tuple[int, int]] = {}
    for raw_window in selected:
        if not isinstance(raw_window, Mapping):
            raise ValueError("causal source lineage contains a non-mapping window")
        window_id = str(raw_window["window_id"])
        result[window_id] = (
            int(raw_window["source_frame_start"]),
            int(raw_window["source_frame_stop_exclusive"]),
        )
    return result


def _validate_bundle_claim_semantics(
    bundle: ObservationFactorBundle,
    lineage: Mapping[str, Any],
) -> None:
    if bundle.schema_version != OBSERVATION_FACTOR_SCHEMA_VERSION:
        raise ValueError("claim-bearing factor bundle must use schema version 4")
    if bundle.gauge_covariance_semantics != "joint-cross-window":
        raise ValueError("claim-bearing factor bundle requires joint gauge covariance")
    if not bundle.cross_window_gauge_covariance_preserved:
        raise ValueError("claim-bearing factor bundle lost cross-window covariance")
    bounds = _lineage_window_bounds(lineage)
    for factor in bundle.factors:
        if factor.window_id != factor.gauge_id:
            raise ValueError("claim-bearing factors require window_id == gauge_id")
        if factor.window_id not in bounds:
            raise ValueError("factor window is absent from causal source lineage")
        start, stop = bounds[factor.window_id]
        if not start <= factor.frame_index < stop:
            raise ValueError("factor frame lies outside its causal source window")


def _validate_bundle_against_envelope(
    bundle: ObservationFactorBundle,
    envelope: ClaimBearingObservationFactorBundleEnvelopeV1,
) -> None:
    _validate_bundle_claim_semantics(bundle, envelope.causal_source_lineage)
    expected = {
        "sequence_id": bundle.sequence_id,
        "case_id": bundle.case_id,
        "stream_id": bundle.stream_id,
        "source_repository": bundle.source_repository,
        "source_revision": bundle.source_revision,
        "causal_frame_stop": bundle.causal_frame_stop,
        "factor_count": len(bundle.factors),
        "observation_count": _factor_observation_count(bundle),
        "gauge_ids": tuple(gauge.window_id for gauge in bundle.gauges),
    }
    for name, value in expected.items():
        if getattr(envelope, name) != value:
            raise ValueError(f"factor bundle differs from envelope field {name}")


def validate_claim_bearing_observation_factor_bundle(
    bundle: ObservationFactorBundle,
    envelope: ClaimBearingObservationFactorBundleEnvelopeV1,
) -> ValidatedClaimBearingObservationFactorBundle:
    """Validate one in-memory bundle/envelope pair without reading files."""

    return ValidatedClaimBearingObservationFactorBundle(
        bundle=bundle,
        envelope=envelope,
    )


def _default_bundle_paths(envelope_path: Path) -> tuple[Path, Path]:
    name = envelope_path.name
    base = name[:-5] if name.endswith(".json") else name
    manifest = envelope_path.with_name(f"{base}.bundle.json")
    return manifest, manifest.with_suffix(".npz")


def _bundle_payload_path(manifest_path: Path) -> tuple[Path, str]:
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("observation-factor bundle manifest is unreadable") from error
    if not isinstance(record, Mapping):
        raise ValueError("observation-factor bundle manifest must be a JSON object")
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("observation-factor bundle payload record is missing")
    payload_path = _resolved_member(
        manifest_path.parent,
        _safe_relative_path(
            payload.get("path"),
            name="observation-factor payload path",
        ),
        name="observation-factor payload path",
    )
    payload_sha = _require_sha256(
        payload.get("sha256"),
        name="observation-factor payload SHA-256",
    )
    if _sha256_file(payload_path) != payload_sha:
        raise ValueError("observation-factor payload checksum mismatch")
    return payload_path, payload_sha


def seal_claim_bearing_observation_factor_bundle(
    bundle: ObservationFactorBundle,
    envelope_path: str | Path,
    *,
    causal_source_lineage: Mapping[str, Any],
    provider_attestation: Mapping[str, Any],
    bundle_manifest_path: str | Path | None = None,
    payload_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ValidatedClaimBearingObservationFactorBundle:
    """Write a neutral bundle and bind it into one validated provider-v2 envelope.

    This lower-level operation accepts an already constructed claim-bearing
    provider attestation. New production code should normally call
    :func:`write_claim_bearing_observation_factor_bundle`, which independently
    verifies calibration compatibility and the executing Prob4D revision first.
    """

    gauge_ids = tuple(gauge.window_id for gauge in bundle.gauges)
    validated_lineage = _validated_lineage(
        causal_source_lineage,
        causal_frame_stop=bundle.causal_frame_stop,
        gauge_ids=gauge_ids,
    )
    (
        validated_attestation,
        manifest_id,
        calibration_ids,
        runtime_source,
        runtime_verified,
    ) = _provider_fields(provider_attestation, source_revision=bundle.source_revision)
    _validate_bundle_claim_semantics(bundle, validated_lineage)

    envelope_file = Path(envelope_path)
    envelope_file.parent.mkdir(parents=True, exist_ok=True)
    default_manifest, default_payload = _default_bundle_paths(envelope_file)
    manifest_file = (
        default_manifest
        if bundle_manifest_path is None
        else Path(bundle_manifest_path)
    )
    payload_file = default_payload if payload_path is None else Path(payload_path)
    if manifest_file.resolve() == envelope_file.resolve():
        raise ValueError("bundle manifest and claim-bearing envelope must differ")
    if payload_file.resolve() in {envelope_file.resolve(), manifest_file.resolve()}:
        raise ValueError("bundle payload path must differ from manifest and envelope")
    _relative_member(
        payload_file,
        root=manifest_file.parent,
        name="bundle payload path",
    )
    relative_manifest = _relative_member(
        manifest_file,
        root=envelope_file.parent,
        name="bundle_manifest_path",
    )

    write_observation_factor_bundle(
        bundle,
        manifest_file,
        payload_path=payload_file,
    )
    _, payload_sha = _bundle_payload_path(manifest_file)
    manifest_sha = _sha256_file(manifest_file)
    envelope = ClaimBearingObservationFactorBundleEnvelopeV1(
        bundle_manifest_path=relative_manifest,
        bundle_manifest_sha256=manifest_sha,
        bundle_payload_sha256=payload_sha,
        sequence_id=bundle.sequence_id,
        case_id=bundle.case_id,
        stream_id=bundle.stream_id,
        source_repository=bundle.source_repository,
        source_revision=bundle.source_revision,
        causal_frame_stop=bundle.causal_frame_stop,
        factor_count=len(bundle.factors),
        observation_count=_factor_observation_count(bundle),
        gauge_ids=gauge_ids,
        causal_source_lineage=validated_lineage,
        provider_manifest_id=manifest_id,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source=runtime_source,
        runtime_revision_independently_verified=runtime_verified,
        provider_attestation=validated_attestation,
        metadata={} if metadata is None else metadata,
    )
    _validate_bundle_against_envelope(bundle, envelope)
    encoded = (
        json.dumps(
            envelope.to_record(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = envelope_file.with_name(f".{envelope_file.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, envelope_file)
    return load_claim_bearing_observation_factor_bundle(envelope_file)


def write_claim_bearing_observation_factor_bundle(
    bundle: ObservationFactorBundle,
    envelope_path: str | Path,
    *,
    causal_selection: CausalOverlapSelection,
    gauge_covariance_calibration: GaugeCovarianceCalibrationV1,
    point_uncertainty_calibration: PointUncertaintyCalibrationV1,
    bundle_manifest_path: str | Path | None = None,
    payload_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ValidatedClaimBearingObservationFactorBundle:
    """Validate producer evidence and write a claim-bearing factor envelope.

    Calibration compatibility is checked from prediction-manifest metadata before
    bundle files are written. Runtime provenance must independently identify the
    exact clean Prob4D revision declared by the bundle.
    """

    target = load_prediction_calibration_target(causal_selection.manifest_path)
    assert_calibration_pair_compatible(
        gauge_covariance_calibration,
        point_uncertainty_calibration,
        target,
    )
    runtime = assert_runtime_revision(bundle.source_revision)
    from .provider_v2 import prob4d_provider_manifest

    attestation = build_provider_attestation(
        provider_manifest=prob4d_provider_manifest(
            provider_revision=bundle.source_revision
        ),
        provider_revision=bundle.source_revision,
        export_mode="calibrated",
        calibration_compatibility_validated=True,
        calibration_artifact_ids={
            "gauge_artifact_id": gauge_covariance_calibration.artifact_id,
            "point_artifact_id": point_uncertainty_calibration.artifact_id,
        },
        covariance_root_mode="canonical_eigenspaces",
        composition_jacobian_mode="analytic",
        runtime_revision=runtime.as_metadata(),
    )
    lineage = causal_selection.artifact_lineage_metadata(
        causal_frame_stop=bundle.causal_frame_stop
    )
    return seal_claim_bearing_observation_factor_bundle(
        bundle,
        envelope_path,
        causal_source_lineage=lineage,
        provider_attestation=attestation,
        bundle_manifest_path=bundle_manifest_path,
        payload_path=payload_path,
        metadata=metadata,
    )


def load_claim_bearing_observation_factor_bundle(
    envelope_path: str | Path,
) -> ValidatedClaimBearingObservationFactorBundle:
    """Load and validate an envelope, neutral manifest, and NPZ payload."""

    envelope_file = Path(envelope_path)
    try:
        record = json.loads(envelope_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("claim-bearing factor envelope is unreadable") from error
    if not isinstance(record, Mapping):
        raise ValueError("claim-bearing factor envelope must be a JSON object")
    envelope = ClaimBearingObservationFactorBundleEnvelopeV1.from_record(record)
    manifest_file = _resolved_member(
        envelope_file.parent,
        envelope.bundle_manifest_path,
        name="bundle_manifest_path",
    )
    if _sha256_file(manifest_file) != envelope.bundle_manifest_sha256:
        raise ValueError("claim-bearing envelope no longer matches its bundle manifest")
    _, payload_sha = _bundle_payload_path(manifest_file)
    if payload_sha != envelope.bundle_payload_sha256:
        raise ValueError("claim-bearing envelope no longer matches its bundle payload")
    bundle = load_observation_factor_bundle(manifest_file)
    return validate_claim_bearing_observation_factor_bundle(bundle, envelope)


__all__ = [
    "CLAIM_BEARING_FACTOR_BUNDLE_SCHEMA",
    "CLAIM_BEARING_FACTOR_BUNDLE_VERSION",
    "ClaimBearingObservationFactorBundleEnvelopeV1",
    "ValidatedClaimBearingObservationFactorBundle",
    "load_claim_bearing_observation_factor_bundle",
    "seal_claim_bearing_observation_factor_bundle",
    "validate_claim_bearing_observation_factor_bundle",
    "write_claim_bearing_observation_factor_bundle",
]
