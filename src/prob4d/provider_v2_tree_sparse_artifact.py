"""Claim-bearing provider-v2 envelopes for tree-sparse observation artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ._causal_observation_source import CausalOverlapSelection
from ._gauge_tree_artifact_common import canonical_json_bytes
from ._gauge_tree_artifact_io import (
    _read_stable_bytes,
    _reject_duplicate_keys,
    _reject_json_constant,
    _write_create_if_absent,
)
from ._gauge_tree_common import GAUGE_TREE_PRIOR_SEMANTICS
from ._immutable_json import frozen_finite_json_mapping, plain_json
from .calibration_compatibility import (
    assert_calibration_pair_compatible,
    load_prediction_calibration_target,
)
from .provider_attestation import (
    PROVIDER_SOURCE_REPOSITORY,
    build_provider_attestation,
)
from .provider_v1 import (
    GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1,
)
from .provider_v2_factor_bundle import (
    _lineage_window_bounds,
    _provider_fields,
    _relative_member,
    _require_nonempty_string,
    _require_positive_integer,
    _require_revision,
    _require_sha256,
    _resolved_member,
    _safe_relative_path,
    _sha256_file,
    _sha256_json,
    _validated_calibration_ids,
    _validated_gauge_ids,
    _validated_lineage,
)
from .provider_v2_tree_sparse_manifest import (
    CLAIM_BEARING_TREE_SPARSE_OBSERVATION_ENVELOPE_VERSION,
    TREE_SPARSE_PROVIDER_CAPABILITIES,
    prob4d_tree_sparse_provider_manifest,
)
from .runtime_revision import assert_runtime_revision
from .tree_sparse_observation_artifact import (
    TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION,
    LoadedTreeSparseObservationArtifactV1,
    load_tree_sparse_observation_artifact,
    write_tree_sparse_observation_artifact,
)
from .tree_sparse_observation_factors import TreeSparseStackedObservationFactors

CLAIM_BEARING_TREE_SPARSE_OBSERVATION_SCHEMA = (
    "prob4d.claim-bearing-tree-sparse-observation-artifact"
)
CLAIM_BEARING_TREE_SPARSE_OBSERVATION_VERSION = (
    CLAIM_BEARING_TREE_SPARSE_OBSERVATION_ENVELOPE_VERSION
)
CLAIM_BEARING_TREE_SPARSE_MAX_ENVELOPE_BYTES = 4_194_304
_REQUIRED_TREE_SPARSE_CAPABILITIES = frozenset(TREE_SPARSE_PROVIDER_CAPABILITIES)
_ENVELOPE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "observation_manifest_path",
        "observation_manifest_sha256",
        "observation_artifact_id",
        "observation_artifact_schema_version",
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
        "causal_frame_stop",
        "observation_count",
        "gauge_ids",
        "gauge_tree_prior_artifact_id",
        "gauge_tree_prior_id",
        "causal_source_lineage",
        "provider_manifest_id",
        "calibration_artifact_ids",
        "runtime_revision_source",
        "runtime_revision_independently_verified",
        "provider_attestation",
        "metadata",
    }
)


def _tree_provider_fields(
    attestation: Mapping[str, Any],
    *,
    source_revision: str,
) -> tuple[Mapping[str, Any], str, Mapping[str, str], str, bool]:
    result = _provider_fields(attestation, source_revision=source_revision)
    provider_attestation = result[0]
    manifest = provider_attestation.get("provider_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("provider attestation lacks a provider manifest")
    capabilities = manifest.get("capabilities")
    if isinstance(capabilities, (str, bytes)) or not isinstance(
        capabilities,
        Sequence,
    ):
        raise ValueError("provider manifest capabilities changed type")
    if not _REQUIRED_TREE_SPARSE_CAPABILITIES.issubset(capabilities):
        raise ValueError("provider manifest lacks tree-sparse claim capabilities")
    artifact_versions = manifest.get("artifact_schema_versions")
    if not isinstance(artifact_versions, Mapping):
        raise ValueError("provider manifest artifact schema versions changed type")
    observation_version = artifact_versions.get("TreeSparseObservationArtifactV1")
    if (
        isinstance(observation_version, bool)
        or not isinstance(observation_version, int)
        or observation_version != TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION
    ):
        raise ValueError("provider manifest lacks the tree-sparse artifact version")
    envelope_version = artifact_versions.get("ClaimBearingTreeSparseObservationEnvelopeV1")
    if (
        isinstance(envelope_version, bool)
        or not isinstance(envelope_version, int)
        or envelope_version != CLAIM_BEARING_TREE_SPARSE_OBSERVATION_VERSION
    ):
        raise ValueError("provider manifest lacks the tree-sparse envelope version")
    return result


def _validate_tree_claim_semantics(
    factors: TreeSparseStackedObservationFactors,
    lineage: Mapping[str, Any],
) -> None:
    if factors.gauge_tree_prior.representation_semantics != GAUGE_TREE_PRIOR_SEMANTICS:
        raise ValueError("claim-bearing tree-sparse factors changed prior semantics")
    bounds = _lineage_window_bounds(lineage)
    for row_index in range(factors.observation_count):
        gauge_index = int(factors.gauge_indices[row_index])
        gauge_id = factors.gauge_ids[gauge_index]
        if gauge_id not in bounds:
            raise ValueError("tree-sparse row gauge is absent from causal source lineage")
        start, stop = bounds[gauge_id]
        frame_index = int(factors.frame_indices[row_index])
        if not start <= frame_index < stop:
            raise ValueError("tree-sparse row lies outside its causal source window")


@dataclass(frozen=True, slots=True)
class ClaimBearingTreeSparseObservationEnvelopeV1:
    """Portable claim identity for one tree-sparse observation artifact."""

    observation_manifest_path: str
    observation_manifest_sha256: str
    observation_artifact_id: str
    sequence_id: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    causal_frame_stop: int
    observation_count: int
    gauge_ids: tuple[str, ...]
    gauge_tree_prior_artifact_id: str
    gauge_tree_prior_id: str
    causal_source_lineage: Mapping[str, Any]
    provider_manifest_id: str
    calibration_artifact_ids: Mapping[str, str]
    runtime_revision_source: str
    runtime_revision_independently_verified: bool
    provider_attestation: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    observation_artifact_schema_version: int = TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        manifest_path = _safe_relative_path(
            self.observation_manifest_path,
            name="observation_manifest_path",
        )
        manifest_sha = _require_sha256(
            self.observation_manifest_sha256,
            name="observation_manifest_sha256",
        )
        observation_artifact_id = _require_sha256(
            self.observation_artifact_id,
            name="observation_artifact_id",
        )
        if (
            isinstance(self.observation_artifact_schema_version, bool)
            or not isinstance(self.observation_artifact_schema_version, int)
            or self.observation_artifact_schema_version != TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION
        ):
            raise ValueError("claim-bearing envelope requires tree-sparse artifact v1")
        sequence_id = _require_nonempty_string(self.sequence_id, name="sequence_id")
        case_id = _require_nonempty_string(self.case_id, name="case_id")
        stream_id = _require_nonempty_string(self.stream_id, name="stream_id")
        repository = _require_nonempty_string(
            self.source_repository,
            name="source_repository",
        )
        if repository != PROVIDER_SOURCE_REPOSITORY:
            raise ValueError("claim-bearing tree-sparse artifact must be produced by Prob4D")
        revision = _require_revision(self.source_revision, name="source_revision")
        causal_frame_stop = _require_positive_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
        )
        observation_count = _require_positive_integer(
            self.observation_count,
            name="observation_count",
        )
        gauge_ids = _validated_gauge_ids(self.gauge_ids)
        prior_artifact_id = _require_sha256(
            self.gauge_tree_prior_artifact_id,
            name="gauge_tree_prior_artifact_id",
        )
        prior_id = _require_sha256(
            self.gauge_tree_prior_id,
            name="gauge_tree_prior_id",
        )
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
        ) = _tree_provider_fields(self.provider_attestation, source_revision=revision)
        manifest_id = _require_sha256(
            self.provider_manifest_id,
            name="provider_manifest_id",
        )
        calibration_ids = _validated_calibration_ids(self.calibration_artifact_ids)
        runtime_source = _require_nonempty_string(
            self.runtime_revision_source,
            name="runtime_revision_source",
        )
        if self.runtime_revision_independently_verified is not True:
            raise ValueError("runtime_revision_independently_verified must be literally True")
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
            name="claim-bearing tree-sparse metadata",
        )

        object.__setattr__(self, "observation_manifest_path", manifest_path)
        object.__setattr__(self, "observation_manifest_sha256", manifest_sha)
        object.__setattr__(self, "observation_artifact_id", observation_artifact_id)
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "source_repository", repository)
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "causal_frame_stop", causal_frame_stop)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "gauge_ids", gauge_ids)
        object.__setattr__(self, "gauge_tree_prior_artifact_id", prior_artifact_id)
        object.__setattr__(self, "gauge_tree_prior_id", prior_id)
        object.__setattr__(self, "causal_source_lineage", lineage)
        object.__setattr__(self, "provider_manifest_id", manifest_id)
        object.__setattr__(self, "calibration_artifact_ids", calibration_ids)
        object.__setattr__(self, "runtime_revision_source", runtime_source)
        object.__setattr__(self, "provider_attestation", attestation)
        object.__setattr__(self, "metadata", metadata)

        expected_id = _sha256_json(self.identity_record())
        if self.artifact_id is not None:
            supplied_id = _require_sha256(self.artifact_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("claim-bearing tree-sparse artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected_id)

    def identity_record(self) -> dict[str, Any]:
        return {
            "schema": CLAIM_BEARING_TREE_SPARSE_OBSERVATION_SCHEMA,
            "schema_version": CLAIM_BEARING_TREE_SPARSE_OBSERVATION_VERSION,
            "observation_manifest_sha256": self.observation_manifest_sha256,
            "observation_artifact_id": self.observation_artifact_id,
            "observation_artifact_schema_version": (self.observation_artifact_schema_version),
            "sequence_id": self.sequence_id,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "causal_frame_stop": self.causal_frame_stop,
            "observation_count": self.observation_count,
            "gauge_ids": list(self.gauge_ids),
            "gauge_tree_prior_artifact_id": self.gauge_tree_prior_artifact_id,
            "gauge_tree_prior_id": self.gauge_tree_prior_id,
            "causal_source_lineage": plain_json(self.causal_source_lineage),
            "provider_manifest_id": self.provider_manifest_id,
            "calibration_artifact_ids": plain_json(self.calibration_artifact_ids),
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
            "observation_manifest_path": self.observation_manifest_path,
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_record(
        cls,
        value: Mapping[str, Any],
    ) -> ClaimBearingTreeSparseObservationEnvelopeV1:
        if set(value) != _ENVELOPE_FIELDS:
            missing = sorted(_ENVELOPE_FIELDS - value.keys())
            extra = sorted(value.keys() - _ENVELOPE_FIELDS)
            raise ValueError(
                f"claim-bearing tree-sparse fields changed; missing={missing}, extra={extra}"
            )
        if value.get("schema") != CLAIM_BEARING_TREE_SPARSE_OBSERVATION_SCHEMA:
            raise ValueError("unexpected claim-bearing tree-sparse schema")
        schema_version = _require_positive_integer(
            value.get("schema_version"),
            name="schema_version",
        )
        if schema_version != CLAIM_BEARING_TREE_SPARSE_OBSERVATION_VERSION:
            raise ValueError("unsupported claim-bearing tree-sparse version")
        return cls(
            observation_manifest_path=value["observation_manifest_path"],
            observation_manifest_sha256=value["observation_manifest_sha256"],
            observation_artifact_id=value["observation_artifact_id"],
            observation_artifact_schema_version=value["observation_artifact_schema_version"],
            sequence_id=value["sequence_id"],
            case_id=value["case_id"],
            stream_id=value["stream_id"],
            source_repository=value["source_repository"],
            source_revision=value["source_revision"],
            causal_frame_stop=value["causal_frame_stop"],
            observation_count=value["observation_count"],
            gauge_ids=value["gauge_ids"],
            gauge_tree_prior_artifact_id=value["gauge_tree_prior_artifact_id"],
            gauge_tree_prior_id=value["gauge_tree_prior_id"],
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
class ValidatedClaimBearingTreeSparseObservation:
    """A strict tree-sparse artifact bound to provider-v2 evidence."""

    observation: LoadedTreeSparseObservationArtifactV1
    envelope: ClaimBearingTreeSparseObservationEnvelopeV1

    def __post_init__(self) -> None:
        _validate_observation_against_envelope(self.observation, self.envelope)

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


def _validate_observation_against_envelope(
    observation: LoadedTreeSparseObservationArtifactV1,
    envelope: ClaimBearingTreeSparseObservationEnvelopeV1,
) -> None:
    manifest = observation.manifest
    factors = observation.factors
    _validate_tree_claim_semantics(factors, envelope.causal_source_lineage)
    expected = {
        "observation_artifact_id": manifest.artifact_id,
        "sequence_id": manifest.sequence_id,
        "case_id": manifest.case_id,
        "stream_id": manifest.stream_id,
        "source_repository": manifest.source_repository,
        "source_revision": manifest.source_revision,
        "causal_frame_stop": manifest.causal_frame_stop,
        "observation_count": manifest.observation_count,
        "gauge_ids": manifest.gauge_ids,
        "gauge_tree_prior_artifact_id": manifest.gauge_tree_prior_artifact_id,
        "gauge_tree_prior_id": manifest.gauge_tree_prior_id,
    }
    for name, value in expected.items():
        if getattr(envelope, name) != value:
            raise ValueError(f"tree-sparse observation differs from envelope field {name}")


def validate_claim_bearing_tree_sparse_observation(
    observation: LoadedTreeSparseObservationArtifactV1,
    envelope: ClaimBearingTreeSparseObservationEnvelopeV1,
) -> ValidatedClaimBearingTreeSparseObservation:
    """Validate one already loaded tree-sparse artifact/envelope pair."""

    return ValidatedClaimBearingTreeSparseObservation(
        observation=observation,
        envelope=envelope,
    )


def _default_observation_path(envelope_path: Path) -> Path:
    name = envelope_path.name
    base = name[:-5] if name.endswith(".json") else name
    return envelope_path.with_name(f"{base}.tree-sparse.json")


def seal_claim_bearing_tree_sparse_observation(
    factors: TreeSparseStackedObservationFactors,
    envelope_path: str | Path,
    *,
    sequence_id: str,
    case_id: str,
    stream_id: str,
    source_revision: str,
    causal_source_lineage: Mapping[str, Any],
    provider_attestation: Mapping[str, Any],
    observation_manifest_path: str | Path | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ValidatedClaimBearingTreeSparseObservation:
    """Write and seal a tree-sparse artifact using preconstructed attestation."""

    if not isinstance(factors, TreeSparseStackedObservationFactors):
        raise TypeError("factors must be a TreeSparseStackedObservationFactors")
    revision = _require_revision(source_revision, name="source_revision")
    sequence = _require_nonempty_string(sequence_id, name="sequence_id")
    case = _require_nonempty_string(case_id, name="case_id")
    stream = _require_nonempty_string(stream_id, name="stream_id")
    gauge_ids = factors.gauge_ids
    lineage = _validated_lineage(
        causal_source_lineage,
        causal_frame_stop=factors.causal_frame_stop,
        gauge_ids=gauge_ids,
    )
    (
        attestation,
        provider_manifest_id,
        calibration_ids,
        runtime_source,
        runtime_verified,
    ) = _tree_provider_fields(provider_attestation, source_revision=revision)
    _validate_tree_claim_semantics(factors, lineage)
    artifact_meta = frozen_finite_json_mapping(
        {} if artifact_metadata is None else artifact_metadata,
        name="claim-bearing tree-sparse artifact metadata",
    )
    envelope_meta = frozen_finite_json_mapping(
        {} if metadata is None else metadata,
        name="claim-bearing tree-sparse envelope metadata",
    )

    envelope_file = Path(envelope_path)
    envelope_file.parent.mkdir(parents=True, exist_ok=True)
    observation_file = (
        _default_observation_path(envelope_file)
        if observation_manifest_path is None
        else Path(observation_manifest_path)
    )
    if observation_file.resolve() == envelope_file.resolve():
        raise ValueError("tree-sparse observation manifest and envelope must differ")
    relative_observation = _relative_member(
        observation_file,
        root=envelope_file.parent,
        name="observation_manifest_path",
    )

    loaded_observation = write_tree_sparse_observation_artifact(
        factors,
        observation_file,
        sequence_id=sequence,
        case_id=case,
        stream_id=stream,
        source_repository=PROVIDER_SOURCE_REPOSITORY,
        source_revision=revision,
        metadata=artifact_meta,
    )
    observation_sha = _sha256_file(observation_file)
    manifest = loaded_observation.manifest
    envelope = ClaimBearingTreeSparseObservationEnvelopeV1(
        observation_manifest_path=relative_observation,
        observation_manifest_sha256=observation_sha,
        observation_artifact_id=cast(str, manifest.artifact_id),
        sequence_id=manifest.sequence_id,
        case_id=manifest.case_id,
        stream_id=manifest.stream_id,
        source_repository=manifest.source_repository,
        source_revision=manifest.source_revision,
        causal_frame_stop=manifest.causal_frame_stop,
        observation_count=manifest.observation_count,
        gauge_ids=manifest.gauge_ids,
        gauge_tree_prior_artifact_id=manifest.gauge_tree_prior_artifact_id,
        gauge_tree_prior_id=manifest.gauge_tree_prior_id,
        causal_source_lineage=lineage,
        provider_manifest_id=provider_manifest_id,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source=runtime_source,
        runtime_revision_independently_verified=runtime_verified,
        provider_attestation=attestation,
        metadata=envelope_meta,
    )
    _validate_observation_against_envelope(loaded_observation, envelope)
    _write_create_if_absent(
        envelope_file,
        canonical_json_bytes(envelope.to_record()),
        name="claim-bearing tree-sparse observation envelope",
    )
    return load_claim_bearing_tree_sparse_observation(envelope_file)


def write_claim_bearing_tree_sparse_observation(
    factors: TreeSparseStackedObservationFactors,
    envelope_path: str | Path,
    *,
    sequence_id: str,
    case_id: str,
    stream_id: str,
    source_revision: str,
    causal_selection: CausalOverlapSelection,
    gauge_covariance_calibration: GaugeCovarianceCalibrationV1,
    point_uncertainty_calibration: PointUncertaintyCalibrationV1,
    observation_manifest_path: str | Path | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ValidatedClaimBearingTreeSparseObservation:
    """Validate calibration/runtime evidence and write one claim-bearing artifact."""

    target = load_prediction_calibration_target(causal_selection.manifest_path)
    assert_calibration_pair_compatible(
        gauge_covariance_calibration,
        point_uncertainty_calibration,
        target,
    )
    revision = _require_revision(source_revision, name="source_revision")
    runtime = assert_runtime_revision(revision)
    attestation = build_provider_attestation(
        provider_manifest=prob4d_tree_sparse_provider_manifest(provider_revision=revision),
        provider_revision=revision,
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
        causal_frame_stop=factors.causal_frame_stop
    )
    return seal_claim_bearing_tree_sparse_observation(
        factors,
        envelope_path,
        sequence_id=sequence_id,
        case_id=case_id,
        stream_id=stream_id,
        source_revision=revision,
        causal_source_lineage=lineage,
        provider_attestation=attestation,
        observation_manifest_path=observation_manifest_path,
        artifact_metadata=artifact_metadata,
        metadata=metadata,
    )


def load_claim_bearing_tree_sparse_observation(
    envelope_path: str | Path,
) -> ValidatedClaimBearingTreeSparseObservation:
    """Load and validate a tree-sparse artifact and its provider-v2 envelope."""

    envelope_file = Path(envelope_path)
    payload = _read_stable_bytes(
        envelope_file,
        name="claim-bearing tree-sparse observation envelope",
        maximum_bytes=CLAIM_BEARING_TREE_SPARSE_MAX_ENVELOPE_BYTES,
    )
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("claim-bearing tree-sparse envelope is invalid JSON") from error
    if not isinstance(raw, Mapping):
        raise ValueError("claim-bearing tree-sparse envelope must be a JSON object")
    envelope = ClaimBearingTreeSparseObservationEnvelopeV1.from_record(raw)
    observation_file = _resolved_member(
        envelope_file.parent,
        envelope.observation_manifest_path,
        name="observation_manifest_path",
    )
    if _sha256_file(observation_file) != envelope.observation_manifest_sha256:
        raise ValueError("claim-bearing envelope no longer matches its observation manifest")
    observation = load_tree_sparse_observation_artifact(observation_file)
    return validate_claim_bearing_tree_sparse_observation(observation, envelope)


__all__ = [
    "CLAIM_BEARING_TREE_SPARSE_OBSERVATION_SCHEMA",
    "CLAIM_BEARING_TREE_SPARSE_OBSERVATION_VERSION",
    "ClaimBearingTreeSparseObservationEnvelopeV1",
    "ValidatedClaimBearingTreeSparseObservation",
    "load_claim_bearing_tree_sparse_observation",
    "seal_claim_bearing_tree_sparse_observation",
    "validate_claim_bearing_tree_sparse_observation",
    "write_claim_bearing_tree_sparse_observation",
]
