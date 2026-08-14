"""Exact bounded posterior over globally consistent local material identities.

Each input :class:`MaterialIdentityMixtureV1` selects one null or linked
predecessor for one target-local track. The posterior conditions several such
source-calibrated mixtures on a window-unique forest constraint while preserving
all local ``(window_id, track_id)`` identities and mandatory null fallbacks.
"""

from ._joint_material_identity_common import (
    CLAIM_BOUNDARY,
    JOINT_ASSIGNMENT_SEMANTICS,
    JOINT_LIKELIHOOD_SEMANTICS,
    JOINT_MATERIAL_IDENTITY_POSTERIOR_SCHEMA,
    JOINT_MATERIAL_IDENTITY_POSTERIOR_VERSION,
)
from ._joint_material_identity_compute import (
    assignment_components,
    build_joint_material_identity_posterior,
    joint_candidate_marginals,
    marginalize_joint_assignment_log_likelihoods,
)
from ._joint_material_identity_io import (
    load_joint_material_identity_posterior,
    write_joint_material_identity_posterior,
)
from ._joint_material_identity_likelihood import MarginalizedJointIdentityLikelihood
from ._joint_material_identity_model import JointMaterialIdentityPosteriorV1
from ._joint_material_identity_records import (
    JointIdentityAssignmentV1,
    JointIdentityMarginalV1,
)

__all__ = [
    "CLAIM_BOUNDARY",
    "JOINT_ASSIGNMENT_SEMANTICS",
    "JOINT_LIKELIHOOD_SEMANTICS",
    "JOINT_MATERIAL_IDENTITY_POSTERIOR_SCHEMA",
    "JOINT_MATERIAL_IDENTITY_POSTERIOR_VERSION",
    "JointIdentityAssignmentV1",
    "JointIdentityMarginalV1",
    "JointMaterialIdentityPosteriorV1",
    "MarginalizedJointIdentityLikelihood",
    "assignment_components",
    "build_joint_material_identity_posterior",
    "joint_candidate_marginals",
    "load_joint_material_identity_posterior",
    "marginalize_joint_assignment_log_likelihoods",
    "write_joint_material_identity_posterior",
]
