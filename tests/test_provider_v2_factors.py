from __future__ import annotations

import subprocess
import sys

import prob4d.observation_factors as neutral
import prob4d.provider_v2_factor_bundle as strict
import prob4d.sparse_observation_factors as sparse
import prob4d.tree_sparse_observation_factors as tree_sparse
from prob4d import provider_v2_factors as provider


def test_provider_v2_factor_facade_reexports_one_contract_surface() -> None:
    assert provider.PROVIDER_FACTOR_API_VERSION == 2
    assert provider.PROB4D_PROVIDER_FACTOR_API_VERSION == 2
    assert provider.OBSERVATION_FACTOR_SCHEMA_VERSION == 4
    assert provider.CLAIM_BEARING_FACTOR_BUNDLE_VERSION == 1

    assert provider.ObservationFactor is neutral.ObservationFactor
    assert provider.ObservationFactorBundle is neutral.ObservationFactorBundle
    assert provider.StackedObservationFactors is neutral.StackedObservationFactors
    assert provider.load_observation_factor_bundle is neutral.load_observation_factor_bundle
    assert provider.write_observation_factor_bundle is neutral.write_observation_factor_bundle

    assert (
        provider.ClaimBearingObservationFactorBundleEnvelopeV1
        is strict.ClaimBearingObservationFactorBundleEnvelopeV1
    )
    assert (
        provider.ValidatedClaimBearingObservationFactorBundle
        is strict.ValidatedClaimBearingObservationFactorBundle
    )
    assert (
        provider.load_claim_bearing_observation_factor_bundle
        is strict.load_claim_bearing_observation_factor_bundle
    )
    assert (
        provider.write_claim_bearing_observation_factor_bundle
        is strict.write_claim_bearing_observation_factor_bundle
    )

    assert (
        provider.SparseStackedObservationFactors
        is sparse.SparseStackedObservationFactors
    )
    assert (
        provider.stack_sparse_observation_factors
        is sparse.stack_sparse_observation_factors
    )
    assert (
        provider.TreeSparseStackedObservationFactors
        is tree_sparse.TreeSparseStackedObservationFactors
    )
    assert provider.bind_gauge_tree_prior is tree_sparse.bind_gauge_tree_prior
    assert (
        provider.stack_tree_sparse_observation_factors
        is tree_sparse.stack_tree_sparse_observation_factors
    )


def test_provider_v2_factor_facade_all_is_explicit_and_resolvable() -> None:
    assert provider.__all__ == sorted(provider.__all__)
    assert len(provider.__all__) == len(set(provider.__all__))
    assert all(not name.startswith("_") for name in provider.__all__)
    assert all(hasattr(provider, name) for name in provider.__all__)


def test_provider_v2_factor_facade_import_remains_numpy_only() -> None:
    code = """
import sys
import prob4d.provider_v2_factors
forbidden = {"torch", "diffusers", "decord"}
loaded = sorted(forbidden & set(sys.modules))
if loaded:
    raise SystemExit(f"optional GPU dependencies loaded: {loaded}")
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
