from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from importlib import import_module
from typing import Any

import numpy as np
import pytest
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.observation_belief_gauge_adapter import (
    build_gauge_aware_batch_from_observation_belief,
)

MODULE_NAMES = (
    "prob4d.observation_contract_bundle",
    "bayesian_phystwin.observation_contract_bundle",
    "causal4d.observation_contract_bundle",
)
ROW_ARRAY_NAMES = (
    "mean_xyz_m",
    "frame_ids",
    "entity_ids",
    "view_indices",
    "window_indices",
    "correlation_group_ids",
    "factor_group_ids",
    "prior_reliability",
    "association_probability",
    "local_covariance_m2",
    "low_rank_factor_m",
)


def _contract_vector() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    modules = tuple(import_module(name) for name in MODULE_NAMES)
    vectors = tuple(module.observation_contract_vector("minimal") for module in modules)
    reference = vectors[0]

    assert vectors[1].descriptor == reference.descriptor
    assert vectors[2].descriptor == reference.descriptor
    assert vectors[1].expected_artifact_id == reference.expected_artifact_id
    assert vectors[2].expected_artifact_id == reference.expected_artifact_id

    arrays = {name: np.asarray(value).copy() for name, value in reference.arrays.items()}
    for vector in vectors[1:]:
        assert tuple(sorted(vector.arrays)) == tuple(sorted(arrays))
        for name, expected in arrays.items():
            np.testing.assert_array_equal(vector.arrays[name], expected)
    return dict(reference.descriptor), arrays


def _belief(
    *,
    arrays_override: dict[str, np.ndarray] | None = None,
    stream_id: str = "cross-stack-metamorphic",
) -> ObservationBeliefV1:
    descriptor, arrays = _contract_vector()
    if arrays_override is not None:
        arrays.update(
            {name: np.asarray(value).copy() for name, value in arrays_override.items()}
        )
    return ObservationBeliefV1(
        case_id=str(descriptor["case_id"]),
        stream_id=stream_id,
        causal_frame_stop=int(descriptor["causal_frame_stop"]),
        view_names=tuple(descriptor["view_names"]),
        window_names=tuple(descriptor["window_names"]),
        factor_names=tuple(descriptor["factor_names"]),
        source_repository="cross-stack/metamorphic-provider",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=arrays["declared_frame_ids"],
        mean_xyz_m=arrays["mean_xyz_m"],
        frame_ids=arrays["frame_ids"],
        entity_ids=arrays["entity_ids"],
        view_indices=arrays["view_indices"],
        window_indices=arrays["window_indices"],
        correlation_group_ids=arrays["correlation_group_ids"],
        factor_group_ids=arrays["factor_group_ids"],
        prior_reliability=arrays["prior_reliability"],
        association_probability=arrays["association_probability"],
        local_covariance_m2=arrays["local_covariance_m2"],
        low_rank_factor_m=arrays["low_rank_factor_m"],
        group_ids=arrays["group_ids"],
        group_prior_nominal_probability=arrays["group_prior_nominal_probability"],
        group_composite_weight=arrays["group_composite_weight"],
        metadata={"purpose": "installed-wheel-cross-stack-metamorphic-v1"},
    )


def _base_physical_prediction(belief: ObservationBeliefV1) -> np.ndarray:
    residual = np.column_stack(
        (
            1.0e-3 * (belief.frame_ids + 1),
            2.0e-3 * (belief.entity_ids + 1),
            -1.5e-3 * (belief.window_indices + 1),
        )
    )
    return belief.mean_xyz_m - residual


def _base_state_jacobian(belief: ObservationBeliefV1) -> np.ndarray:
    state = np.zeros((belief.observation_count, 3, 2), dtype=np.float64)
    state[:, 0, 0] = 1.0
    state[:, 1, 1] = 1.0
    state[:, 2, 0] = 0.25
    state[:, 2, 1] = -0.5
    return state


def _base_query_jacobian() -> np.ndarray:
    query = np.zeros((2, 3, 2), dtype=np.float64)
    query[0, 0, 0] = 1.0
    query[0, 2, 1] = 0.25
    query[1, 1, 1] = 1.0
    query[1, 2, 0] = -0.5
    return query


def _adapter(
    belief: ObservationBeliefV1,
    *,
    physical_prediction: np.ndarray | None = None,
    linear_transform: np.ndarray | None = None,
    response_scale_m: float = 0.1,
):
    state = _base_state_jacobian(belief)
    query = _base_query_jacobian()
    if linear_transform is not None:
        transform = np.asarray(linear_transform, dtype=np.float64)
        state = np.einsum("ij,njk->nik", transform, state)
        query = np.einsum("ij,qjk->qik", transform, query)
    return build_gauge_aware_batch_from_observation_belief(
        belief,
        physical_prediction_xyz_m=(
            _base_physical_prediction(belief)
            if physical_prediction is None
            else physical_prediction
        ),
        state_jacobian=state,
        query_state_jacobian=query,
        physical_response_scale_m=response_scale_m,
    )


def _marginal_observation_covariance(batch: Any) -> np.ndarray:
    count = len(batch.innovation_m)
    covariance = np.zeros((3 * count, 3 * count), dtype=np.float64)
    for index, block in enumerate(batch.observation_covariance_m2):
        start = 3 * index
        covariance[start : start + 3, start : start + 3] = block
    gauge = batch.gauge_jacobian.reshape(3 * count, -1)
    return covariance + gauge @ batch.gauge_prior_covariance @ gauge.T


def test_shared_root_basis_rotation_is_invariant_across_installed_contracts() -> None:
    belief = _belief()
    reference = _adapter(belief).batch
    angle = 0.731
    orthogonal = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    rotated_root = np.einsum(
        "nij,jk->nik",
        belief.low_rank_factor_m,
        orthogonal,
    )
    rotated = _adapter(
        _belief(
            arrays_override={"low_rank_factor_m": rotated_root},
            stream_id="cross-stack-metamorphic-root-rotation",
        )
    ).batch

    np.testing.assert_allclose(rotated.innovation_m, reference.innovation_m, atol=0.0)
    np.testing.assert_allclose(
        _marginal_observation_covariance(rotated),
        _marginal_observation_covariance(reference),
        atol=1.0e-14,
        rtol=1.0e-12,
    )


def test_global_frame_and_unit_change_is_equivariant() -> None:
    belief = _belief()
    prediction = _base_physical_prediction(belief)
    reference = _adapter(belief, physical_prediction=prediction).batch

    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = 1000.0
    translation = np.array([250.0, -75.0, 10.0], dtype=np.float64)
    linear = scale * rotation
    transformed_belief = belief.transformed(
        rotation=rotation,
        translation_m=translation,
        scale=scale,
        stream_id="cross-stack-metamorphic-millimetres",
    )
    transformed_prediction = np.einsum(
        "ij,nj->ni",
        linear,
        prediction,
    ) + translation
    transformed = _adapter(
        transformed_belief,
        physical_prediction=transformed_prediction,
        linear_transform=linear,
        response_scale_m=scale * 0.1,
    ).batch

    np.testing.assert_allclose(
        transformed.innovation_m,
        np.einsum("ij,nj->ni", linear, reference.innovation_m),
        atol=1.0e-10,
        rtol=1.0e-12,
    )
    row_transform = np.kron(
        np.eye(belief.observation_count, dtype=np.float64),
        linear,
    )
    expected_covariance = (
        row_transform
        @ _marginal_observation_covariance(reference)
        @ row_transform.T
    )
    np.testing.assert_allclose(
        _marginal_observation_covariance(transformed),
        expected_covariance,
        atol=1.0e-8,
        rtol=1.0e-12,
    )


def test_observation_row_permutation_only_permutes_the_batch() -> None:
    belief = _belief()
    reference = _adapter(belief).batch
    permutation = np.array([2, 0, 3, 1], dtype=np.int64)
    overrides = {
        name: np.asarray(getattr(belief, name))[permutation]
        for name in ROW_ARRAY_NAMES
    }
    permuted = _adapter(
        _belief(
            arrays_override=overrides,
            stream_id="cross-stack-metamorphic-row-permutation",
        )
    ).batch

    np.testing.assert_allclose(
        permuted.innovation_m,
        reference.innovation_m[permutation],
        atol=0.0,
    )
    flat_indices = np.concatenate(
        [np.arange(3 * index, 3 * index + 3) for index in permutation]
    )
    expected_covariance = _marginal_observation_covariance(reference)[
        np.ix_(flat_indices, flat_indices)
    ]
    np.testing.assert_allclose(
        _marginal_observation_covariance(permuted),
        expected_covariance,
        atol=1.0e-14,
        rtol=1.0e-12,
    )


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _guarded_handoff_type():
    try:
        module = import_module("causal4d.guarded_bpt_belief_handoff_v2")
    except ModuleNotFoundError:
        if os.environ.get("PROB4D_REQUIRE_CROSS_STACK_METAMORPHIC") == "1":
            raise
        pytest.skip("installed Causal4D revision predates guarded handoff v2")
    return module.BayesianPhysTwinGuardedBeliefHandoffReceiptV2


def _fallback_receipt():
    receipt_type = _guarded_handoff_type()
    baseline_bpt = _sha256("baseline-bpt")
    baseline_causal = _sha256("baseline-causal4d")
    return receipt_type(
        protocol_id="cross-stack-metamorphic-v1",
        case_id="fallback-case",
        causal_frame_stop=12,
        update_id=_sha256("update"),
        admission_id=_sha256("admission"),
        tree_block_result_id=_sha256("tree-block-result"),
        observation_artifact_id=_sha256("observation"),
        linearization_artifact_id=_sha256("linearization"),
        provider_manifest_id=_sha256("provider-manifest"),
        runtime_identity_id=_sha256("runtime"),
        prob4d_source_repository="IPS-Stuttgart/Prob4D",
        prob4d_runtime_revision="c" * 40,
        runtime_revision_evidence_source="installed-wheel-build-identity",
        candidate_construction_receipt_id=_sha256("candidate-construction"),
        guarded_selection_receipt_id=_sha256("guarded-selection"),
        guard_certificate_id=_sha256("guard-certificate"),
        guard_decision_id=_sha256("guard-decision"),
        selection_id=_sha256("selection"),
        baseline_bpt_belief_id=baseline_bpt,
        candidate_bpt_belief_id=_sha256("candidate-bpt"),
        selected_bpt_belief_id=baseline_bpt,
        baseline_causal4d_belief_id=baseline_causal,
        delivered_causal4d_belief_id=baseline_causal,
        update_inference_admissible=False,
        selected_candidate=False,
        exact_fallback=True,
        evidence_consumed_count=0,
        covariance_consumed_count=0,
        covariance_result_id=None,
        evidence_ledger_id=_sha256("evidence-ledger"),
        raw_prob4d_reinterpreted=False,
        metadata={"metamorphic_test": True},
    )


def test_exact_fallback_consumes_no_prob4d_evidence_or_query_covariance() -> None:
    receipt = _fallback_receipt()

    assert receipt.exact_fallback is True
    assert receipt.selected_candidate is False
    assert receipt.selected_bpt_belief_id == receipt.baseline_bpt_belief_id
    assert (
        receipt.delivered_causal4d_belief_id
        == receipt.baseline_causal4d_belief_id
    )
    assert receipt.evidence_consumed_count == 0
    assert receipt.covariance_consumed_count == 0
    assert receipt.covariance_result_id is None

    with pytest.raises(ValueError, match="zero observation evidence"):
        replace(receipt, evidence_consumed_count=1)
    with pytest.raises(ValueError, match="zero query covariance"):
        replace(receipt, covariance_consumed_count=1)
    with pytest.raises(ValueError, match="must not bind query covariance"):
        replace(receipt, covariance_result_id=_sha256("forbidden-covariance"))
    with pytest.raises(ValueError, match="changed the baseline Causal4D belief"):
        replace(
            receipt,
            delivered_causal4d_belief_id=_sha256("changed-delivered-belief"),
        )
