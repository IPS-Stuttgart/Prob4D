from __future__ import annotations

from importlib.metadata import version

import prob4d.api.v1 as api_v1
from prob4d import provider_v1


def test_api_v1_is_versioned_and_matches_the_installed_distribution() -> None:
    assert api_v1.API_VERSION == 1
    assert api_v1.PROVIDER_API_VERSION == provider_v1.PROVIDER_API_VERSION == 1
    assert api_v1.__version__ == version("prob4d")


def test_api_v1_reexports_the_frozen_provider_contract() -> None:
    assert api_v1.ObservationBeliefExportV1 is provider_v1.ObservationBeliefExportV1
    assert api_v1.ObservationFactorBundle is provider_v1.ObservationFactorBundle
    assert (
        api_v1.export_calibrated_observation_belief
        is provider_v1.export_calibrated_observation_belief
    )
    assert (
        api_v1.load_observation_factor_bundle
        is provider_v1.load_observation_factor_bundle
    )
    assert api_v1.prob4d_provider_manifest is provider_v1.prob4d_provider_manifest


def test_api_v1_declares_only_supported_public_names() -> None:
    required = {
        "API_VERSION",
        "PROVIDER_API_VERSION",
        "MetricGaugeAnchor",
        "ObservationBeliefExportV1",
        "ObservationFactorBundle",
        "export_calibrated_observation_belief",
        "load_observation_belief_export",
        "prob4d_provider_manifest",
    }
    assert required <= set(api_v1.__all__)
    assert all(
        not name.startswith("_") or name == "__version__"
        for name in api_v1.__all__
    )
