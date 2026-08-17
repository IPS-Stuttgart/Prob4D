"""Provider-neutral case evaluation with legacy MotionCrafter replay.

Aggregation and metric computation remain in ``_provider_evaluation_compute``.
This module replaces only the artifact-identity boundary so claim-bearing
provider evaluation can consume any validated ``PredictionProviderManifestV1``
route without fabricating MotionCrafter fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._provider_evaluation_compute import (
    _method_support_summary,
    _numeric_leaves,
    _paired_common_support,
    aggregate_provider_records,
)
from ._provider_evaluation_manifest import ProviderEvaluationCase
from .evaluation_modes import EvaluationModes, evaluate_sequence_modes
from .io import FusedPredictionArtifact, load_fused_prediction_artifact, load_truth
from .metrics import DEFAULT_EVALUATION_CHUNK_SIZE
from .provider_evaluation_identity import (
    PROVIDER_EVALUATION_IDENTITY_METADATA_KEY,
    ProviderEvaluationIdentity,
    provider_evaluation_method_signature,
    provider_identity_report_record,
    validate_provider_evaluation_metadata,
)


def evaluate_provider_cases(
    cases: list[ProviderEvaluationCase],
    *,
    allow_legacy_artifacts: bool,
    evaluation_chunk_size: int = DEFAULT_EVALUATION_CHUNK_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Evaluate paired cases and enforce one provider contract per method."""

    if evaluation_chunk_size < 1:
        raise ValueError("evaluation_chunk_size must be positive")
    records: list[dict[str, Any]] = []
    signatures: dict[str, tuple[object, ...]] = {}
    method_metadata: dict[str, dict[str, Any]] = {}
    for case in cases:
        truth = load_truth(case.truth_path)
        artifacts: dict[str, FusedPredictionArtifact] = {}
        identities: dict[str, ProviderEvaluationIdentity] = {}
        for method_id, prediction_path in sorted(case.predictions.items()):
            artifact = load_fused_prediction_artifact(prediction_path)
            artifacts[method_id] = artifact
            metadata = artifact.metadata
            if metadata.legacy_unspecified and not allow_legacy_artifacts:
                raise ValueError(
                    f"{prediction_path} has legacy unspecified covariance semantics; "
                    "regenerate it or pass --allow-legacy-artifacts for an explicitly "
                    "non-claim-bearing diagnostic"
                )
            if not metadata.legacy_unspecified and metadata.method_id != method_id:
                raise ValueError(
                    f"prediction method label {method_id!r} disagrees with artifact "
                    f"method_id {metadata.method_id!r}"
                )
            identity = None
            if not metadata.legacy_unspecified:
                identity = validate_provider_evaluation_metadata(
                    metadata,
                    path=Path(prediction_path),
                )
                identities[method_id] = identity
            signature = (
                (
                    metadata.fusion_method,
                    metadata.covariance_semantics,
                    metadata.correlation_assumption,
                )
                if identity is None
                else provider_evaluation_method_signature(metadata, identity)
            )
            previous = signatures.setdefault(method_id, signature)
            if signature != previous:
                raise ValueError(
                    f"method {method_id!r} mixes covariance, model, estimator, "
                    "revision, or provider semantics"
                )
            reported_metadata = metadata.to_dict()
            if (
                identity is not None
                and identity.identity_format == "prediction-provider-manifest-v1"
            ):
                artifact_details = reported_metadata.get("metadata")
                if not isinstance(artifact_details, dict):
                    raise AssertionError("validated fused metadata is not a mapping")
                invariant_details = dict(artifact_details)
                invariant_details.pop(PROVIDER_EVALUATION_IDENTITY_METADATA_KEY)
                reported_metadata["metadata"] = invariant_details
                reported_metadata["provider_contract"] = identity.contract_record()
            method_metadata.setdefault(method_id, reported_metadata)

        common_point_support, common_flow_support, paired_support = (
            _paired_common_support(
                truth,
                {
                    method_id: artifact.sequence
                    for method_id, artifact in artifacts.items()
                },
            )
        )
        for method_id, prediction_path in sorted(case.predictions.items()):
            artifact = artifacts[method_id]
            native_modes: EvaluationModes = evaluate_sequence_modes(
                artifact.sequence,
                truth,
                boundary_frames=list(case.boundary_frames),
                prefix_frame_stop_exclusive=case.prefix_frame_stop_exclusive,
                evaluation_chunk_size=evaluation_chunk_size,
            )
            common_modes: EvaluationModes = evaluate_sequence_modes(
                artifact.sequence,
                truth,
                boundary_frames=list(case.boundary_frames),
                prefix_frame_stop_exclusive=case.prefix_frame_stop_exclusive,
                truth_support_mask=common_point_support,
                truth_flow_support_mask=common_flow_support,
                evaluation_chunk_size=evaluation_chunk_size,
            )
            evaluation = common_modes.to_dict()
            native_evaluation = native_modes.to_dict()
            support = _method_support_summary(
                common=common_modes,
                native=native_modes,
                paired=paired_support,
            )
            numeric = _numeric_leaves(evaluation)
            numeric.update(
                _numeric_leaves(
                    native_evaluation,
                    prefix="native_support",
                )
            )
            numeric.update(_numeric_leaves(support, prefix="support"))
            record: dict[str, Any] = {
                "case_id": case.case_id,
                "group_id": case.group_id,
                "method_id": method_id,
                "prediction_path": str(prediction_path),
                "truth_path": str(case.truth_path),
                "artifact": artifact.metadata.to_dict(),
                "evaluation": evaluation,
                "native_support_evaluation": native_evaluation,
                "support": support,
                "_numeric": numeric,
            }
            identity = identities.get(method_id)
            if (
                identity is not None
                and identity.identity_format == "prediction-provider-manifest-v1"
            ):
                record["provider_identity"] = provider_identity_report_record(identity)
            records.append(record)
    return records, method_metadata


__all__ = ["aggregate_provider_records", "evaluate_provider_cases"]
