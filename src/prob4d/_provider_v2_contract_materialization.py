"""Producer-owned materialization of provider-v2 conformance vectors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .api import v2 as provider_api
from ._provider_v2_contract_common import (
    ProviderV2ContractVector,
    _validate_sha256,
)


@dataclass(frozen=True, slots=True)
class ProviderV2ContractMaterialization:
    """Producer-owned objects reconstructed from one neutral vector."""

    bundle: provider_api.ObservationFactorBundle
    gauge_tree_prior: provider_api.GaugeTreeSquareRootPriorV1
    sparse_stack: provider_api.SparseStackedObservationFactors
    tree_sparse_stack: provider_api.TreeSparseStackedObservationFactors


def provider_v2_contract_array_sha256(values: object) -> str:
    """Hash one array using the corpus' dtype/shape/byte convention."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def provider_v2_contract_stack_sha256(
    stack: provider_api.TreeSparseStackedObservationFactors,
) -> str:
    """Hash the execution-relevant tree-sparse row contract."""

    if not isinstance(stack, provider_api.TreeSparseStackedObservationFactors):
        raise TypeError("stack must be TreeSparseStackedObservationFactors")
    digest = hashlib.sha256()
    array_names = (
        "world_mean_m",
        "conditional_world_covariance_m2",
        "marginal_world_covariance_m2",
        "local_gauge_jacobian",
        "gauge_indices",
        "association_probability",
        "prior_reliability",
        "prior_nominal_probability",
        "composite_weight",
        "point_ids",
        "frame_indices",
    )
    for name in array_names:
        digest.update(name.encode("utf-8"))
        digest.update(
            provider_v2_contract_array_sha256(getattr(stack, name)).encode("ascii")
        )
    tuple_names = (
        "gauge_ids",
        "view_ids",
        "factor_ids",
        "correlation_group_ids",
    )
    for name in tuple_names:
        digest.update(name.encode("utf-8"))
        digest.update(
            json.dumps(list(getattr(stack, name)), separators=(",", ":")).encode(
                "utf-8"
            )
        )
    digest.update(str(stack.causal_frame_stop).encode("ascii"))
    digest.update(stack.gauge_tree_prior.prior_id.encode("ascii"))
    return digest.hexdigest()


def _array_record(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, Mapping) or set(value) != {"dtype", "shape", "values"}:
        raise ValueError(f"{name} must be a closed array record")
    dtype = np.dtype(value["dtype"])
    shape_value = value["shape"]
    if isinstance(shape_value, (str, bytes)) or not isinstance(
        shape_value,
        Sequence,
    ):
        raise ValueError(f"{name} shape must be a sequence")
    shape = tuple(int(item) for item in shape_value)
    array = np.asarray(value["values"], dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    return array


def _factor(value: object) -> provider_api.ObservationFactor:
    if not isinstance(value, Mapping):
        raise ValueError("factor record must be a mapping")
    return provider_api.ObservationFactor(
        factor_id=value["factor_id"],
        frame_index=value["frame_index"],
        view_id=value["view_id"],
        window_id=value["window_id"],
        gauge_id=value["gauge_id"],
        point_ids=_array_record(value["point_ids"], name="point_ids"),
        points_local_m=_array_record(
            value["points_local_m"],
            name="points_local_m",
        ),
        valid_mask=_array_record(value["valid_mask"], name="valid_mask"),
        local_covariance_m2=_array_record(
            value["local_covariance_m2"],
            name="local_covariance_m2",
        ),
        association_probability=_array_record(
            value["association_probability"],
            name="association_probability",
        ),
        prior_reliability=_array_record(
            value["prior_reliability"],
            name="prior_reliability",
        ),
        prior_nominal_probability=value["prior_nominal_probability"],
        composite_weight=value["composite_weight"],
        correlation_group_id=value["correlation_group_id"],
        causal_frame_stop=value["causal_frame_stop"],
    )


def materialize_provider_v2_contract_vector(
    vector: ProviderV2ContractVector | Mapping[str, Any],
) -> ProviderV2ContractMaterialization:
    """Construct and cross-check the provider-owned objects for one vector."""

    payload = vector.payload if isinstance(vector, ProviderV2ContractVector) else vector
    if not isinstance(payload, Mapping):
        raise TypeError("provider-v2 contract vector must be a mapping")
    bundle_record = payload["bundle"]
    tree_record = payload["tree_prior"]
    if not isinstance(bundle_record, Mapping) or not isinstance(tree_record, Mapping):
        raise ValueError("provider-v2 contract vector sections must be mappings")
    gauges = tuple(
        provider_api.GaugeEstimate(
            gauge["window_id"],
            provider_api.Sim3.from_vector(
                _array_record(gauge["sim3_vector"], name="sim3_vector")
            ),
            _array_record(gauge["covariance"], name="gauge covariance"),
        )
        for gauge in bundle_record["gauges"]
    )
    factors = tuple(_factor(factor) for factor in bundle_record["factors"])
    bundle = provider_api.ObservationFactorBundle(
        sequence_id=bundle_record["sequence_id"],
        case_id=bundle_record["case_id"],
        stream_id=bundle_record["stream_id"],
        factors=factors,
        gauges=gauges,
        source_repository=bundle_record["source_repository"],
        source_revision=bundle_record["source_revision"],
        causal_frame_stop=bundle_record["causal_frame_stop"],
        joint_gauge_covariance=_array_record(
            bundle_record["joint_gauge_covariance"],
            name="joint_gauge_covariance",
        ),
        gauge_covariance_semantics=bundle_record["gauge_covariance_semantics"],
        metadata=bundle_record["metadata"],
    )
    prior = provider_api.GaugeTreeSquareRootPriorV1.from_transition_covariances(
        gauge_ids=tuple(tree_record["gauge_ids"]),
        parent_indices=_array_record(
            tree_record["parent_indices"],
            name="parent_indices",
        ),
        transition_matrices=_array_record(
            tree_record["transition_matrices"],
            name="transition_matrices",
        ),
        innovation_covariances=_array_record(
            tree_record["innovation_covariances"],
            name="innovation_covariances",
        ),
    )
    sparse = provider_api.stack_sparse_observation_factors(bundle)
    tree_sparse = provider_api.stack_tree_sparse_observation_factors(bundle, prior)
    return ProviderV2ContractMaterialization(bundle, prior, sparse, tree_sparse)


def validate_provider_v2_contract_materialization(
    vector: ProviderV2ContractVector,
    materialization: ProviderV2ContractMaterialization,
) -> None:
    """Validate the expected proper construction and execution identities."""

    expected = vector.payload["expected"]
    if not isinstance(expected, Mapping):
        raise ValueError("provider-v2 contract expected section must be a mapping")
    prior_id = _validate_sha256(expected["prior_id"], name="expected prior_id")
    if materialization.gauge_tree_prior.prior_id != prior_id:
        raise ValueError("provider-v2 contract prior identity changed")
    count = expected["observation_count"]
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError("expected observation_count must be an integer")
    if materialization.tree_sparse_stack.observation_count != count:
        raise ValueError("provider-v2 contract observation count changed")
    stack_digest = _validate_sha256(
        expected["stack_sha256"],
        name="expected stack_sha256",
    )
    observed_digest = provider_v2_contract_stack_sha256(
        materialization.tree_sparse_stack
    )
    if observed_digest != stack_digest:
        raise ValueError("provider-v2 contract tree-sparse stack identity changed")
    expected_mean = _array_record(
        expected["world_mean_m"],
        name="expected world_mean_m",
    )
    if not np.allclose(
        materialization.tree_sparse_stack.world_mean_m,
        expected_mean,
        atol=1e-14,
        rtol=0.0,
    ):
        raise ValueError("provider-v2 contract world means changed")


__all__ = [
    "ProviderV2ContractMaterialization",
    "materialize_provider_v2_contract_vector",
    "provider_v2_contract_array_sha256",
    "provider_v2_contract_stack_sha256",
    "validate_provider_v2_contract_materialization",
]
