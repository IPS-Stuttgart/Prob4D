"""Strict shared definitions for audited CUT3R source competence."""

from __future__ import annotations

from typing import Final, Literal

from ._cut3r_source_competence_v2_common import PROPER_SCORE_SEMANTICS

LOCK_SCHEMA: Final = "prob4d.cut3r-source-competence-support-audit-lock"
MANIFEST_SCHEMA: Final = "prob4d.cut3r-metric-support-manifest"
REPORT_SCHEMA: Final = "prob4d.cut3r-source-competence-support-audit-report"
VERSION: Final = 1
PROPER_SCORE_REFERENCE_FIT_SCOPE: Final = "development-and-calibration-only"
CLAIM_BOUNDARY: Final = (
    "This source-only audit independently reconstructs exact metric-support "
    "identities from retained canonical row manifests and binds the exact "
    "arm-neutral proper-score reference artifact. It cannot change a CUT3R arm, "
    "metric value, support row, source roster, policy, target-access boundary, "
    "BayesianPhysTwin decision, Causal4D decision, or state-of-the-art claim."
)
Status = Literal["pass", "fail", "technical-failure", "not-evaluated"]

_LOCK_SPEC_FIELDS: Final = frozenset(
    {
        "common_support_definition_sha256",
        "proper_score_reference_artifact_id",
        "proper_score_reference_fit_scope",
        "proper_score_semantics",
        "require_complete_manifest_roster",
    }
)
_LOCK_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "comparison_lock_id",
        "source_competence_lock_id",
        "common_support_lock_id",
        "record_definition_sha256",
        "common_support_definition_sha256",
        "source_evaluation_group_ids",
        "random_seeds",
        "contrast",
        "proper_score_semantics",
        "proper_score_reference_artifact_id",
        "proper_score_reference_sha256",
        "proper_score_reference_fit_scope",
        "require_complete_manifest_roster",
        "source_truth_required",
        "target_access",
        "claim_boundary",
        "support_audit_lock_id",
    }
)
_MANIFEST_INPUT_FIELDS: Final = frozenset(
    {
        "source_truth_used",
        "target_payloads_opened",
        "target_outcomes_opened",
        "entries",
    }
)
_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "comparison_lock_id",
        "source_competence_lock_id",
        "common_support_lock_id",
        "support_audit_lock_id",
        "common_support_definition_sha256",
        "proper_score_reference_artifact_id",
        "proper_score_reference_sha256",
        "source_truth_used",
        "target_payloads_opened",
        "target_outcomes_opened",
        "entries",
        "metric_support_manifest_id",
    }
)
_ENTRY_FIELDS: Final = frozenset(
    {
        "group_id",
        "case_id",
        "frame_index",
        "random_seed",
        "point_rows",
        "endpoint_rows",
        "proper_score_rows",
        "seam_rows",
    }
)
_REPORT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "comparison_lock_id",
        "source_competence_lock_id",
        "common_support_lock_id",
        "support_audit_lock_id",
        "records_id",
        "metric_support_manifest_id",
        "source_competence_report_v2_id",
        "proper_score_semantics",
        "proper_score_reference_artifact_id",
        "proper_score_reference_sha256",
        "proper_score_reference_fit_scope",
        "verified_pair_count",
        "support_manifest_status",
        "proper_score_reference_binding_status",
        "mean_quality_status",
        "identity_reliability_status",
        "mean_quality_reasons",
        "identity_reliability_reasons",
        "audited_source_competence_pass",
        "source_truth_used",
        "target_payloads_opened",
        "target_outcomes_opened",
        "claim_boundary",
        "source_competence_support_audit_report_id",
    }
)

__all__ = [
    "CLAIM_BOUNDARY",
    "LOCK_SCHEMA",
    "MANIFEST_SCHEMA",
    "PROPER_SCORE_REFERENCE_FIT_SCOPE",
    "PROPER_SCORE_SEMANTICS",
    "REPORT_SCHEMA",
    "Status",
    "VERSION",
]
