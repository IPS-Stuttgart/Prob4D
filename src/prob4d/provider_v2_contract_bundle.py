"""Normative conformance corpus for Prob4D provider-v2 factor contracts.

The corpus is data-only and fixes one valid joint-gauge/tree-sparse vector plus
ten adversarial mutations. Independent downstream implementations can carry the
same bytes while retaining their own validators.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ._provider_v2_contract_common import (
    PROVIDER_V2_CONTRACT_BUNDLE,
    PROVIDER_V2_CONTRACT_BUNDLE_SHA256,
    PROVIDER_V2_CONTRACT_BUNDLE_VERSION,
    InvalidProviderV2ContractVector,
    ProviderV2ContractVector,
    invalid_provider_v2_contract_vectors,
    provider_v2_contract_bundle_manifest,
    provider_v2_contract_schema,
    provider_v2_contract_vector,
)
from ._provider_v2_contract_materialization import (
    PROVIDER_V2_CONTRACT_MINIMAL_STACK_SEMANTIC_SHA256,
    PROVIDER_V2_CONTRACT_NUMERICAL_ATOL,
    PROVIDER_V2_CONTRACT_NUMERICAL_RTOL,
    PROVIDER_V2_CONTRACT_STACK_SEMANTIC_SCHEMA,
    PROVIDER_V2_CONTRACT_STACK_SEMANTIC_VERSION,
    ProviderV2ContractMaterialization,
    materialize_provider_v2_contract_vector,
    provider_v2_contract_array_sha256,
    provider_v2_contract_runtime_stack_sha256,
    provider_v2_contract_stack_semantic_sha256,
    provider_v2_contract_stack_sha256,
    validate_provider_v2_contract_materialization,
)
from .api import v2 as provider_api


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    manifest = provider_v2_contract_bundle_manifest()
    vector = provider_v2_contract_vector()
    materialization = materialize_provider_v2_contract_vector(vector)
    validate_provider_v2_contract_materialization(vector, materialization)
    summary = {
        "bundle_sha256": manifest["bundle_sha256"],
        "valid_vectors": 1,
        "invalid_vectors": len(invalid_provider_v2_contract_vectors()),
        "provider_api_version": provider_api.PROVIDER_API_VERSION,
        "provider_factor_api_version": provider_api.PROVIDER_FACTOR_API_VERSION,
        "observation_factor_schema_version": (
            provider_api.OBSERVATION_FACTOR_SCHEMA_VERSION
        ),
        "tree_sparse_observation_schema_version": (
            provider_api.TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION
        ),
        "minimal_prior_id": materialization.gauge_tree_prior.prior_id,
        "minimal_stack_semantic_sha256": (
            provider_v2_contract_stack_semantic_sha256(
                materialization.tree_sparse_stack
            )
        ),
        "minimal_reference_runtime_stack_sha256": (
            vector.payload["expected"]["stack_sha256"]
        ),
        "minimal_observed_runtime_stack_sha256": (
            provider_v2_contract_runtime_stack_sha256(
                materialization.tree_sparse_stack
            )
        ),
        "numerical_atol": PROVIDER_V2_CONTRACT_NUMERICAL_ATOL,
        "numerical_rtol": PROVIDER_V2_CONTRACT_NUMERICAL_RTOL,
    }
    print(
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":") if args.compact else None,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "InvalidProviderV2ContractVector",
    "PROVIDER_V2_CONTRACT_BUNDLE",
    "PROVIDER_V2_CONTRACT_BUNDLE_SHA256",
    "PROVIDER_V2_CONTRACT_BUNDLE_VERSION",
    "PROVIDER_V2_CONTRACT_MINIMAL_STACK_SEMANTIC_SHA256",
    "PROVIDER_V2_CONTRACT_NUMERICAL_ATOL",
    "PROVIDER_V2_CONTRACT_NUMERICAL_RTOL",
    "PROVIDER_V2_CONTRACT_STACK_SEMANTIC_SCHEMA",
    "PROVIDER_V2_CONTRACT_STACK_SEMANTIC_VERSION",
    "ProviderV2ContractMaterialization",
    "ProviderV2ContractVector",
    "invalid_provider_v2_contract_vectors",
    "main",
    "materialize_provider_v2_contract_vector",
    "provider_v2_contract_array_sha256",
    "provider_v2_contract_bundle_manifest",
    "provider_v2_contract_runtime_stack_sha256",
    "provider_v2_contract_schema",
    "provider_v2_contract_stack_semantic_sha256",
    "provider_v2_contract_stack_sha256",
    "provider_v2_contract_vector",
    "validate_provider_v2_contract_materialization",
]
