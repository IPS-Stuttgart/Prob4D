"""Bayesian beliefs that preserve unresolved compact-group symmetry.

A physical observation can identify a quotient state while remaining invariant
along a compact group orbit. This module represents the posterior as a quotient
belief times an explicit conditional group law, preserves that law under
orbit-invariant evidence, and exposes shared-group query, decision, and
continuous-cover certificates without selecting an unsupported group
representative.

The code does not infer a symmetry group, certify a quadrature cover, validate a
physical likelihood or state--action coupling, or authorize deployment. Those
are caller-owned source and provenance obligations.
"""

from __future__ import annotations

from ._symmetry_complete_action import (
    GaugeCoupledActionCertificateV1,
    GaugeCouplingReceiptV1,
    certify_gauge_coupled_action_orbit,
)
from ._symmetry_complete_base import (
    SYMMETRY_COMPLETE_BELIEF_CLAIM_BOUNDARY,
    SYMMETRY_COMPLETE_BELIEF_VERSION,
    CompactGroupQuadratureV1,
    SymmetryCompleteBeliefV1,
)
from ._symmetry_complete_decision import (
    CompactGroupDecisionCertificateV1,
    certify_compact_group_decision,
)
from ._symmetry_complete_point import (
    PointCompletionAuditV1,
    audit_point_completion,
    pushforward_shared_group_query,
)
from ._symmetry_complete_query import (
    OrbitInvarianceCertificateV1,
    certify_compact_group_query,
)
from ._symmetry_complete_update import (
    SymmetryInformationV1,
    SymmetryUpdateV1,
    update_symmetry_complete_belief,
)

__all__ = [
    "CompactGroupDecisionCertificateV1",
    "CompactGroupQuadratureV1",
    "GaugeCoupledActionCertificateV1",
    "GaugeCouplingReceiptV1",
    "OrbitInvarianceCertificateV1",
    "PointCompletionAuditV1",
    "SYMMETRY_COMPLETE_BELIEF_CLAIM_BOUNDARY",
    "SYMMETRY_COMPLETE_BELIEF_VERSION",
    "SymmetryCompleteBeliefV1",
    "SymmetryInformationV1",
    "SymmetryUpdateV1",
    "audit_point_completion",
    "certify_compact_group_decision",
    "certify_compact_group_query",
    "certify_gauge_coupled_action_orbit",
    "pushforward_shared_group_query",
    "update_symmetry_complete_belief",
]
