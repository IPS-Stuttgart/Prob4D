"""Extended provider-v2 manifest for claim-bearing tree-sparse artifacts."""

from __future__ import annotations

from typing import cast

from .provider_attestation import compute_provider_manifest_id
from .provider_v2 import prob4d_provider_manifest
from .tree_sparse_observation_artifact import (
    TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION,
)

TREE_SPARSE_PROVIDER_CAPABILITIES = (
    "content_addressed_tree_sparse_observation_artifacts",
    "strict_claim_bearing_tree_sparse_observation_loading",
)
CLAIM_BEARING_TREE_SPARSE_OBSERVATION_ENVELOPE_VERSION = 1


def prob4d_tree_sparse_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return provider-v2 plus the prospective tree-sparse artifact contracts."""

    descriptor = dict(prob4d_provider_manifest(provider_revision=provider_revision))
    descriptor.pop("manifest_id", None)
    capabilities = list(cast(list[str], descriptor["capabilities"]))
    for capability in TREE_SPARSE_PROVIDER_CAPABILITIES:
        if capability not in capabilities:
            capabilities.append(capability)
    schema_versions = dict(cast(dict[str, int], descriptor["artifact_schema_versions"]))
    schema_versions["TreeSparseObservationArtifactV1"] = TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION
    schema_versions["ClaimBearingTreeSparseObservationEnvelopeV1"] = (
        CLAIM_BEARING_TREE_SPARSE_OBSERVATION_ENVELOPE_VERSION
    )
    metadata = dict(cast(dict[str, object], descriptor["metadata"]))
    metadata["tree_sparse_observation_artifact_semantics"] = (
        "selected explicit-gauge rows are stored as content-addressed non-pickled "
        "NPY members and bound to one portable causal gauge-tree prior; neither "
        "marginal point covariance nor a dense joint gauge covariance is serialized"
    )
    metadata["claim_bearing_tree_sparse_loading_semantics"] = (
        "the strict envelope binds artifact bytes, causal source-window lineage, "
        "calibration identities, the complete extended provider manifest, and an "
        "independently verified runtime revision before downstream admission"
    )
    descriptor["capabilities"] = capabilities
    descriptor["artifact_schema_versions"] = schema_versions
    descriptor["metadata"] = metadata
    manifest_id = compute_provider_manifest_id(descriptor)
    return {"manifest_id": manifest_id, **descriptor}


__all__ = [
    "CLAIM_BEARING_TREE_SPARSE_OBSERVATION_ENVELOPE_VERSION",
    "TREE_SPARSE_PROVIDER_CAPABILITIES",
    "prob4d_tree_sparse_provider_manifest",
]
