from __future__ import annotations

import prob4d.api.evidence_decision_v1 as api
import prob4d.evidence_decision_v1 as implementation


EXPECTED_PUBLIC_API = {
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
}


def test_evidence_decision_api_v1_is_small_and_versioned() -> None:
    assert api.API_VERSION == 1
    assert set(api.__all__) == EXPECTED_PUBLIC_API
    assert len(api.__all__) == len(set(api.__all__))
    assert api.validate_evidence_decision_v1 is (
        implementation.validate_evidence_decision_v1
    )
    assert api.require_prob4d_evidence_binding_v1 is (
        implementation.require_prob4d_evidence_binding_v1
    )
