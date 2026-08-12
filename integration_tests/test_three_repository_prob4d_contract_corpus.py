from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import pytest

EXPECTED_BUNDLE_SHA256 = "a62c693a14c227daa1f4c8db850e691a1d0081df0c853cf0174c33d0b8504ce9"
MODULE_NAMES = (
    "prob4d.observation_contract_bundle",
    "bayesian_phystwin.observation_contract_bundle",
    "causal4d.observation_contract_bundle",
)


def _modules() -> tuple[ModuleType, ...]:
    return tuple(import_module(name) for name in MODULE_NAMES)


def _assert_equal_arrays(
    expected: Mapping[str, np.ndarray],
    observed: Mapping[str, np.ndarray],
) -> None:
    assert tuple(sorted(observed)) == tuple(sorted(expected))
    for name in sorted(expected):
        expected_array = expected[name]
        observed_array = observed[name]
        assert observed_array.dtype == expected_array.dtype
        assert observed_array.shape == expected_array.shape
        assert observed_array.flags.writeable is False
        np.testing.assert_array_equal(observed_array, expected_array)


def test_all_installed_packages_validate_the_same_normative_bundle() -> None:
    manifests = tuple(module.observation_contract_bundle_manifest() for module in _modules())

    assert manifests[0]["bundle_sha256"] == EXPECTED_BUNDLE_SHA256
    assert manifests[1:] == (manifests[0], manifests[0])
    assert manifests[0]["bundle_name"] == "phys4d.observation_belief.v1"
    assert manifests[0]["bundle_version"] == 1


@pytest.mark.parametrize("vector_name", ("minimal", "zero_rank"))
def test_valid_vector_payloads_and_hashes_match_across_installed_packages(
    vector_name: str,
) -> None:
    modules = _modules()
    vectors = tuple(module.observation_contract_vector(vector_name) for module in modules)
    reference = vectors[0]

    for module, vector in zip(modules, vectors, strict=True):
        assert vector.name == reference.name
        assert vector.descriptor == reference.descriptor
        assert vector.expected_artifact_id == reference.expected_artifact_id
        _assert_equal_arrays(reference.arrays, vector.arrays)
        assert (
            module.observation_contract_artifact_id(vector.descriptor, vector.arrays)
            == reference.expected_artifact_id
        )
        for name, values in vector.arrays.items():
            assert module.observation_contract_array_sha256(values) == modules[
                0
            ].observation_contract_array_sha256(reference.arrays[name])


def _invalid_case_ids(module: ModuleType) -> tuple[str, ...]:
    cases: tuple[Mapping[str, Any], ...] = module.observation_contract_invalid_cases()
    return tuple(str(case["id"]) for case in cases)


def test_invalid_case_rosters_match_across_installed_packages() -> None:
    modules = _modules()
    reference_ids = _invalid_case_ids(modules[0])

    assert reference_ids
    assert _invalid_case_ids(modules[1]) == reference_ids
    assert _invalid_case_ids(modules[2]) == reference_ids


@pytest.mark.parametrize(
    "case_id",
    _invalid_case_ids(import_module("prob4d.observation_contract_bundle")),
)
def test_invalid_vectors_materialize_identically_across_installed_packages(
    case_id: str,
) -> None:
    vectors = tuple(
        module.invalid_observation_contract_vector(case_id) for module in _modules()
    )
    reference = vectors[0]

    for vector in vectors:
        assert vector.case_id == reference.case_id
        assert vector.mode == reference.mode
        assert vector.descriptor == reference.descriptor
        assert vector.original_artifact_id == reference.original_artifact_id
        _assert_equal_arrays(reference.arrays, vector.arrays)
