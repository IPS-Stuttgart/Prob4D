from __future__ import annotations

import importlib.util
from importlib.metadata import version

import prob4d.api.v2 as api_v2
from prob4d import provider_v2, provider_v2_factors
from prob4d.gauge import GaugeEstimate
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1
from prob4d.project_identity import PROB4D_PROJECT_ID
from prob4d.sim3 import Sim3


def test_api_v2_is_the_only_current_versioned_facade() -> None:
    assert importlib.util.find_spec("prob4d.api.v1") is None
    assert api_v2.API_VERSION == 2
    assert api_v2.PROVIDER_API_VERSION == provider_v2.PROVIDER_API_VERSION == 2
    assert (
        api_v2.PROVIDER_FACTOR_API_VERSION
        == provider_v2_factors.PROVIDER_FACTOR_API_VERSION
        == 2
    )
    assert api_v2.__version__ == version("prob4d")


def test_api_v2_reexports_claim_bearing_provider_contracts() -> None:
    assert (
        api_v2.load_claim_bearing_observation_belief
        is provider_v2.load_claim_bearing_observation_belief
    )
    assert (
        api_v2.load_claim_bearing_observation_factor_bundle
        is provider_v2_factors.load_claim_bearing_observation_factor_bundle
    )
    assert (
        api_v2.load_claim_bearing_tree_sparse_observation
        is provider_v2_factors.load_claim_bearing_tree_sparse_observation
    )
    assert api_v2.ObservationFactorBundle is provider_v2_factors.ObservationFactorBundle
    assert (
        api_v2.TreeSparseStackedObservationFactors
        is provider_v2_factors.TreeSparseStackedObservationFactors
    )
    assert api_v2.GaugeTreeSquareRootPriorV1 is GaugeTreeSquareRootPriorV1
    assert api_v2.GaugeEstimate is GaugeEstimate
    assert api_v2.Sim3 is Sim3


def test_api_v2_exposes_transfer_safe_project_identity() -> None:
    assert api_v2.PROB4D_PROJECT_ID == PROB4D_PROJECT_ID
    descriptor = api_v2.prob4d_project_identity()
    assert descriptor["project_id"] == PROB4D_PROJECT_ID
    assert api_v2.validate_prob4d_project_identity(descriptor) == descriptor
    assert api_v2.is_prob4d_repository("IPS-Stuttgart/Prob4D")
    assert api_v2.is_prob4d_repository("FlorianPfaff/Prob4D")


def test_api_v2_declares_only_supported_public_names() -> None:
    required = {
        "API_VERSION",
        "PROVIDER_API_VERSION",
        "PROVIDER_FACTOR_API_VERSION",
        "PROB4D_PROJECT_ID",
        "GaugeEstimate",
        "GaugeTreeSquareRootPriorV1",
        "ObservationFactorBundle",
        "Sim3",
        "TreeSparseStackedObservationFactors",
        "load_claim_bearing_observation_belief",
        "load_claim_bearing_observation_factor_bundle",
        "load_claim_bearing_tree_sparse_observation",
        "prob4d_provider_manifest",
        "prob4d_tree_sparse_provider_manifest",
    }
    assert required <= set(api_v2.__all__)
    assert len(api_v2.__all__) == len(set(api_v2.__all__))
    assert all(
        not name.startswith("_") or name == "__version__"
        for name in api_v2.__all__
    )
    assert "main" not in api_v2.__all__
