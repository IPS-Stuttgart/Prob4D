from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from prob4d.api import v2 as api_v2
from prob4d.provider_v2_contract_bundle import (
    PROVIDER_V2_CONTRACT_BUNDLE,
    PROVIDER_V2_CONTRACT_BUNDLE_SHA256,
    invalid_provider_v2_contract_vectors,
    materialize_provider_v2_contract_vector,
    provider_v2_contract_bundle_manifest,
    provider_v2_contract_schema,
    provider_v2_contract_stack_sha256,
    provider_v2_contract_vector,
    validate_provider_v2_contract_materialization,
)


def test_provider_v2_contract_bundle_is_content_locked() -> None:
    manifest = provider_v2_contract_bundle_manifest()

    assert manifest["bundle_name"] == PROVIDER_V2_CONTRACT_BUNDLE
    assert manifest["bundle_sha256"] == PROVIDER_V2_CONTRACT_BUNDLE_SHA256
    assert manifest["canonical_repository"] == "IPS-Stuttgart/Prob4D"
    assert set(manifest["files"]) == {
        "invalid_cases.json",
        "schema.json",
        "vectors/minimal.json",
    }


def test_provider_v2_contract_schema_fixes_advanced_versions_and_semantics() -> None:
    schema = provider_v2_contract_schema()

    assert schema["provider_api_version"] == api_v2.PROVIDER_API_VERSION == 2
    assert (
        schema["provider_factor_api_version"]
        == api_v2.PROVIDER_FACTOR_API_VERSION
        == 2
    )
    assert (
        schema["observation_factor_schema_version"]
        == api_v2.OBSERVATION_FACTOR_SCHEMA_VERSION
        == 4
    )
    assert schema["required_semantics"]["gauge_covariance"] == (
        "joint-cross-window"
    )
    assert schema["required_semantics"]["causal_frame_stop"] == "exclusive"


def test_minimal_vector_materializes_dense_sparse_and_tree_sparse_contracts() -> None:
    vector = provider_v2_contract_vector()
    materialization = materialize_provider_v2_contract_vector(vector)
    validate_provider_v2_contract_materialization(vector, materialization)

    assert isinstance(materialization.bundle, api_v2.ObservationFactorBundle)
    assert isinstance(
        materialization.sparse_stack,
        api_v2.SparseStackedObservationFactors,
    )
    assert isinstance(
        materialization.tree_sparse_stack,
        api_v2.TreeSparseStackedObservationFactors,
    )
    assert isinstance(
        materialization.gauge_tree_prior,
        api_v2.GaugeTreeSquareRootPriorV1,
    )
    assert materialization.bundle.gauge_covariance_semantics == (
        "joint-cross-window"
    )
    assert materialization.bundle.cross_window_gauge_covariance_preserved
    assert materialization.tree_sparse_stack.observation_count == 4
    assert materialization.tree_sparse_stack.gauge_ids == (
        "window-0",
        "window-1",
    )
    assert provider_v2_contract_stack_sha256(
        materialization.tree_sparse_stack
    ) == vector.payload["expected"]["stack_sha256"]

    for value in (
        materialization.tree_sparse_stack.world_mean_m,
        materialization.tree_sparse_stack.conditional_world_covariance_m2,
        materialization.tree_sparse_stack.local_gauge_jacobian,
        materialization.gauge_tree_prior.parent_indices,
        materialization.gauge_tree_prior.transition_matrices,
    ):
        assert not np.asarray(value).flags.writeable


def test_all_invalid_vectors_fail_closed_at_the_producer_boundary() -> None:
    invalid = invalid_provider_v2_contract_vectors()

    assert len(invalid) == 10
    assert len({case.case_id for case in invalid}) == len(invalid)
    for case in invalid:
        with pytest.raises((TypeError, ValueError), match=case.expected_error):
            materialize_provider_v2_contract_vector(case.payload)


def test_contract_vector_is_defensively_reloaded() -> None:
    first = provider_v2_contract_vector()
    first.payload["bundle"]["sequence_id"] = "mutated"

    second = provider_v2_contract_vector()
    assert second.payload["bundle"]["sequence_id"] == "sequence-a"


def test_contract_bundle_cli_reports_verified_identities() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prob4d.provider_v2_contract_bundle",
            "--compact",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)

    assert summary["bundle_sha256"] == PROVIDER_V2_CONTRACT_BUNDLE_SHA256
    assert summary["valid_vectors"] == 1
    assert summary["invalid_vectors"] == 10
    assert summary["provider_api_version"] == 2
    assert summary["provider_factor_api_version"] == 2
    assert summary["observation_factor_schema_version"] == 4
    assert summary["tree_sparse_observation_schema_version"] == 1
