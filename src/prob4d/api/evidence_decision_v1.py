"""Stable Prob4D consumer API for evidence-decision wire version 1.

This module is intentionally separate from the producer-oriented :mod:`v1` and
:mod:`v2` façades.  It validates decisions emitted by BayesianPhysTwin without
making Prob4D depend on that package at runtime.
"""

from __future__ import annotations

from ..evidence_decision_v1 import (
    EVIDENCE_DECISION_JSON_SCHEMA_SHA256,
    EVIDENCE_DECISION_SCHEMA,
    EVIDENCE_DECISION_SCHEMA_VERSION,
    EVIDENCE_DECISION_SOURCE_REPOSITORY,
    EVIDENCE_DECISION_SOURCE_REVISION,
    DecisionMetricV1,
    DecisionRepositoryStateV1,
    DecisionStatus,
    RepositoryRole,
    RunClassification,
    ValidatedEvidenceDecisionV1,
    evidence_decision_contract_identity,
    load_evidence_decision_v1,
    require_authorized_evidence_decision_v1,
    require_prob4d_evidence_binding_v1,
    require_repository_binding_v1,
    validate_evidence_decision_v1,
)

API_VERSION = 1

__all__ = [
    "API_VERSION",
    "DecisionMetricV1",
    "DecisionRepositoryStateV1",
    "DecisionStatus",
    "EVIDENCE_DECISION_JSON_SCHEMA_SHA256",
    "EVIDENCE_DECISION_SCHEMA",
    "EVIDENCE_DECISION_SCHEMA_VERSION",
    "EVIDENCE_DECISION_SOURCE_REPOSITORY",
    "EVIDENCE_DECISION_SOURCE_REVISION",
    "RepositoryRole",
    "RunClassification",
    "ValidatedEvidenceDecisionV1",
    "evidence_decision_contract_identity",
    "load_evidence_decision_v1",
    "require_authorized_evidence_decision_v1",
    "require_prob4d_evidence_binding_v1",
    "require_repository_binding_v1",
    "validate_evidence_decision_v1",
]
