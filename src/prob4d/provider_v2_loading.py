"""Strict loading boundary for claim-bearing Prob4D provider-v2 observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .causal_stream_contract import (
    PROB4D_CAUSAL_STREAM_CONTRACT_VERSION,
    PROB4D_CAUSAL_STREAM_ID,
    PROB4D_SOURCE_REPOSITORY,
)
from .observation_contract import ObservationBeliefExportV1
from .observation_validation import load_observation_belief_export
from .provider_attestation import validate_provider_attestation


def _required_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _require_sha256(value: object, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_revision(value: object, *, name: str) -> str:
    revision = str(value)
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must be an exact lowercase Git commit")
    return revision


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ValidatedClaimBearingObservation:
    """A strict provider-v2 observation plus its validated producer identities."""

    observation: ObservationBeliefExportV1
    provider_manifest_id: str
    gauge_calibration_id: str
    point_calibration_id: str
    runtime_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_manifest_id",
            _require_sha256(
                self.provider_manifest_id,
                name="provider_manifest_id",
            ),
        )
        object.__setattr__(
            self,
            "gauge_calibration_id",
            _require_sha256(
                self.gauge_calibration_id,
                name="gauge_calibration_id",
            ),
        )
        object.__setattr__(
            self,
            "point_calibration_id",
            _require_sha256(
                self.point_calibration_id,
                name="point_calibration_id",
            ),
        )
        runtime_revision = _require_revision(
            self.runtime_revision,
            name="runtime_revision",
        )
        if runtime_revision != self.observation.source_revision:
            raise ValueError(
                "validated runtime revision differs from the observation source revision"
            )
        object.__setattr__(self, "runtime_revision", runtime_revision)

    @property
    def artifact_id(self) -> str:
        """Return the immutable observation content address."""

        return self.observation.artifact_id


def validate_claim_bearing_observation_belief(
    artifact: ObservationBeliefExportV1,
) -> ValidatedClaimBearingObservation:
    """Require the complete causal, calibrated, and attested provider-v2 boundary."""

    if artifact.source_repository != PROB4D_SOURCE_REPOSITORY:
        raise ValueError("claim-bearing observation must be produced by Prob4D")
    if artifact.stream_id != PROB4D_CAUSAL_STREAM_ID:
        raise ValueError("claim-bearing observation must use the strict Prob4D stream")

    metadata = artifact.metadata
    if (
        metadata.get("prob4d_causal_stream_contract_version")
        != PROB4D_CAUSAL_STREAM_CONTRACT_VERSION
    ):
        raise ValueError("claim-bearing observation requires causal stream contract v2")
    stream_contract = _required_mapping(
        metadata.get("prob4d_causal_stream_contract"),
        name="Prob4D causal stream contract",
    )
    if stream_contract.get("version") != PROB4D_CAUSAL_STREAM_CONTRACT_VERSION:
        raise ValueError("Prob4D causal stream contract metadata changed version")
    if metadata.get("gauge_mode") != "sequential":
        raise ValueError("claim-bearing observation requires sequential gauge mode")
    if metadata.get("joint_cross_window_gauge_covariance_represented") is not True:
        raise ValueError(
            "claim-bearing observation requires joint cross-window gauge covariance"
        )
    if metadata.get("metric_anchor_covariance_in_joint_factor") is not True:
        raise ValueError(
            "claim-bearing observation requires metric-anchor covariance in the joint factor"
        )
    expected_factor_names = tuple(
        f"joint_gauge_latent_{index:04d}" for index in range(len(artifact.factor_names))
    )
    if not expected_factor_names or artifact.factor_names != expected_factor_names:
        raise ValueError("claim-bearing observation requires canonical joint gauge factors")
    if {int(value) for value in artifact.factor_group_ids} != {0}:
        raise ValueError("claim-bearing observation requires one shared joint factor group")

    lineage = _required_mapping(
        metadata.get("causal_source_lineage"),
        name="Prob4D causal source lineage",
    )
    if lineage.get("causal_frame_stop_exclusive") != artifact.causal_frame_stop:
        raise ValueError("causal source lineage differs from the observation cutoff")
    if lineage.get("future_prediction_payloads_opened") != 0:
        raise ValueError("claim-bearing observation opened future prediction payloads")
    selected_windows = lineage.get("selected_windows")
    if not isinstance(selected_windows, list) or not selected_windows:
        raise ValueError("claim-bearing observation requires selected source-window lineage")
    for selected_window in selected_windows:
        window = _required_mapping(selected_window, name="selected source window")
        source_frame_max = window.get("source_frame_max")
        if isinstance(source_frame_max, bool) or not isinstance(source_frame_max, int):
            raise ValueError("selected source-window frame maximum must be an integer")
        if source_frame_max >= artifact.causal_frame_stop:
            raise ValueError("selected source window crosses the causal frame boundary")

    posterior = _required_mapping(
        metadata.get("gauge_posterior"),
        name="Prob4D gauge posterior",
    )
    if posterior.get("model") != "sequential_joint_spanning_tree_v1":
        raise ValueError("claim-bearing observation requires the sequential joint gauge tree")
    if posterior.get("cross_window_covariance_preserved") is not True:
        raise ValueError("claim-bearing gauge posterior lost cross-window covariance")
    if posterior.get("fixed_lag_boundary_covariance_is_approximate") is not False:
        raise ValueError("claim-bearing observation cannot use approximate fixed-lag covariance")
    if posterior.get("exported_factor_rank") != len(artifact.factor_names):
        raise ValueError("claim-bearing gauge rank differs from the exported factor rank")

    calibration = _required_mapping(
        metadata.get("covariance_calibration"),
        name="Prob4D covariance calibration metadata",
    )
    if calibration.get("status") != "calibrated":
        raise ValueError("claim-bearing observation requires both covariance calibrations")
    if calibration.get("uncalibrated_exploratory_covariance_allowed") is not False:
        raise ValueError(
            "claim-bearing observation cannot allow uncalibrated covariance"
        )
    if calibration.get("pointwise_covariance_fallback_allowed") is not False:
        raise ValueError(
            "claim-bearing observation cannot allow pointwise covariance fallback"
        )
    alignment_count = _require_nonnegative_integer(
        calibration.get("alignment_count"),
        name="claim-bearing alignment_count",
    )
    calibrated_alignment_count = _require_nonnegative_integer(
        calibration.get("gauge_calibrated_alignment_count"),
        name="claim-bearing gauge_calibrated_alignment_count",
    )
    if calibrated_alignment_count != alignment_count:
        raise ValueError("claim-bearing observation has uncalibrated gauge alignments")
    fallback_counts = _required_mapping(
        calibration.get("covariance_fallback_counts"),
        name="claim-bearing covariance fallback counts",
    )
    if fallback_counts:
        raise ValueError("claim-bearing observation reports covariance fallback use")

    raw_attestation = _required_mapping(
        metadata.get("prob4d_provider_attestation"),
        name="Prob4D provider attestation",
    )
    validated_attestation = validate_provider_attestation(
        raw_attestation,
        source_revision=artifact.source_revision,
        require_claim_bearing=True,
    )
    attested_calibration = _required_mapping(
        validated_attestation["calibration_artifact_ids"],
        name="attested calibration identities",
    )
    gauge_calibration_id = _require_sha256(
        attested_calibration.get("gauge_artifact_id"),
        name="attested gauge calibration ID",
    )
    point_calibration_id = _require_sha256(
        attested_calibration.get("point_artifact_id"),
        name="attested point calibration ID",
    )
    if calibration.get("gauge_artifact_id") != gauge_calibration_id:
        raise ValueError("gauge calibration metadata differs from provider attestation")
    if calibration.get("point_artifact_id") != point_calibration_id:
        raise ValueError("point calibration metadata differs from provider attestation")

    runtime = _required_mapping(
        validated_attestation["runtime_revision"],
        name="validated runtime revision",
    )
    runtime_revision = _require_revision(
        runtime.get("observed_revision"),
        name="validated runtime revision",
    )
    provider_manifest_id = _require_sha256(
        validated_attestation["provider_manifest_id"],
        name="provider manifest ID",
    )
    return ValidatedClaimBearingObservation(
        observation=artifact,
        provider_manifest_id=provider_manifest_id,
        gauge_calibration_id=gauge_calibration_id,
        point_calibration_id=point_calibration_id,
        runtime_revision=runtime_revision,
    )


def load_claim_bearing_observation_belief(
    path: str | Path,
) -> ValidatedClaimBearingObservation:
    """Load one artifact and require all claim-bearing provider-v2 invariants."""

    return validate_claim_bearing_observation_belief(
        load_observation_belief_export(path)
    )


__all__ = [
    "ValidatedClaimBearingObservation",
    "load_claim_bearing_observation_belief",
    "validate_claim_bearing_observation_belief",
]
