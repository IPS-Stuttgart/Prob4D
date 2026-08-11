"""Producer-owned materialization of provider-v2 conformance vectors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._provider_v2_contract_common import (
    ProviderV2ContractVector,
    _validate_sha256,
)
from .api import v2 as provider_api

PROVIDER_V2_CONTRACT_NUMERICAL_ATOL = 1e-12
PROVIDER_V2_CONTRACT_NUMERICAL_RTOL = 1e-10
PROVIDER_V2_CONTRACT_STACK_SEMANTIC_SCHEMA = (
    "prob4d.provider-v2-tree-sparse-stack-semantic"
)
PROVIDER_V2_CONTRACT_STACK_SEMANTIC_VERSION = 1
PROVIDER_V2_CONTRACT_MINIMAL_STACK_SEMANTIC_SHA256 = (
    "58621710b5b22a64163c47b4756f200cea13e56491d85a3852af96ec1cb0f4fb"
)

_ROW_ARRAY_NAMES = (
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
_COMMON_FLOAT_ROW_ARRAY_NAMES = (
    "world_mean_m",
    "conditional_world_covariance_m2",
    "marginal_world_covariance_m2",
    "association_probability",
    "prior_reliability",
    "prior_nominal_probability",
    "composite_weight",
)
_INTEGER_ROW_ARRAY_NAMES = (
    "gauge_indices",
    "point_ids",
    "frame_indices",
)
_TUPLE_NAMES = (
    "gauge_ids",
    "view_ids",
    "factor_ids",
    "correlation_group_ids",
)


@dataclass(frozen=True, slots=True)
class ProviderV2ContractMaterialization:
    """Producer-owned objects reconstructed from one neutral vector."""

    bundle: provider_api.ObservationFactorBundle
    dense_stack: provider_api.StackedObservationFactors
    gauge_tree_prior: provider_api.GaugeTreeSquareRootPriorV1
    sparse_stack: provider_api.SparseStackedObservationFactors
    tree_sparse_stack: provider_api.TreeSparseStackedObservationFactors


def provider_v2_contract_array_sha256(values: object) -> str:
    """Hash one array using its runtime dtype, shape, and exact bytes."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def provider_v2_contract_runtime_stack_sha256(
    stack: provider_api.TreeSparseStackedObservationFactors,
) -> str:
    """Hash exact derived bytes for same-runtime replay diagnostics.

    This digest deliberately is not a cross-runtime conformance criterion. Last-bit
    floating-point differences between supported NumPy or BLAS implementations do not
    alter the provider-v2 contract when all explicit numerical tolerances pass.
    """

    if not isinstance(stack, provider_api.TreeSparseStackedObservationFactors):
        raise TypeError("stack must be TreeSparseStackedObservationFactors")
    digest = hashlib.sha256()
    for name in _ROW_ARRAY_NAMES:
        digest.update(name.encode("utf-8"))
        digest.update(
            provider_v2_contract_array_sha256(getattr(stack, name)).encode("ascii")
        )
    for name in _TUPLE_NAMES:
        digest.update(name.encode("utf-8"))
        digest.update(
            json.dumps(list(getattr(stack, name)), separators=(",", ":")).encode(
                "utf-8"
            )
        )
    digest.update(str(stack.causal_frame_stop).encode("ascii"))
    digest.update(stack.gauge_tree_prior.prior_id.encode("ascii"))
    return digest.hexdigest()


def provider_v2_contract_stack_sha256(
    stack: provider_api.TreeSparseStackedObservationFactors,
) -> str:
    """Compatibility alias for the runtime-specific stack digest."""

    return provider_v2_contract_runtime_stack_sha256(stack)


def _semantic_dtype(value: object, *, name: str) -> str:
    dtype = np.asarray(value).dtype
    if dtype == np.dtype(np.float64):
        return "float64"
    if dtype == np.dtype(np.int64):
        return "int64"
    raise TypeError(f"provider-v2 semantic stack {name} has unsupported dtype {dtype}")


def _float_hex_vector(value: object) -> list[str]:
    array = np.asarray(value, dtype=np.float64)
    return [float(item).hex() for item in array.reshape(-1)]


def provider_v2_contract_stack_semantic_sha256(
    stack: provider_api.TreeSparseStackedObservationFactors,
) -> str:
    """Hash exact structural semantics without derived floating-point bytes."""

    if not isinstance(stack, provider_api.TreeSparseStackedObservationFactors):
        raise TypeError("stack must be TreeSparseStackedObservationFactors")
    payload = {
        "schema": PROVIDER_V2_CONTRACT_STACK_SEMANTIC_SCHEMA,
        "version": PROVIDER_V2_CONTRACT_STACK_SEMANTIC_VERSION,
        "observation_count": stack.observation_count,
        "array_contracts": {
            name: {
                "dtype": _semantic_dtype(getattr(stack, name), name=name),
                "shape": list(np.asarray(getattr(stack, name)).shape),
            }
            for name in _ROW_ARRAY_NAMES
        },
        "gauge_ids": list(stack.gauge_ids),
        "view_ids": list(stack.view_ids),
        "factor_ids": list(stack.factor_ids),
        "correlation_group_ids": list(stack.correlation_group_ids),
        "gauge_indices": [int(value) for value in stack.gauge_indices],
        "point_ids": [int(value) for value in stack.point_ids],
        "frame_indices": [int(value) for value in stack.frame_indices],
        "association_probability_hex": _float_hex_vector(
            stack.association_probability
        ),
        "prior_reliability_hex": _float_hex_vector(stack.prior_reliability),
        "prior_nominal_probability_hex": _float_hex_vector(
            stack.prior_nominal_probability
        ),
        "composite_weight_hex": _float_hex_vector(stack.composite_weight),
        "causal_frame_stop": stack.causal_frame_stop,
        "gauge_tree_prior_id": stack.gauge_tree_prior.prior_id,
    }
    content = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


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
    dense = provider_api.stack_observation_factors(bundle)
    sparse = provider_api.stack_sparse_observation_factors(bundle)
    tree_sparse = provider_api.stack_tree_sparse_observation_factors(bundle, prior)
    return ProviderV2ContractMaterialization(
        bundle,
        dense,
        prior,
        sparse,
        tree_sparse,
    )


def _require_float_parity(
    left: object,
    right: object,
    *,
    name: str,
) -> None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != np.dtype(np.float64) or right_array.dtype != np.dtype(
        np.float64
    ):
        raise ValueError(f"provider-v2 contract {name} dtype changed")
    if left_array.shape != right_array.shape or not np.allclose(
        left_array,
        right_array,
        atol=PROVIDER_V2_CONTRACT_NUMERICAL_ATOL,
        rtol=PROVIDER_V2_CONTRACT_NUMERICAL_RTOL,
        equal_nan=True,
    ):
        raise ValueError(f"provider-v2 contract {name} numerical parity changed")


def _require_integer_parity(
    left: object,
    right: object,
    *,
    name: str,
) -> None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != np.dtype(np.int64) or right_array.dtype != np.dtype(
        np.int64
    ):
        raise ValueError(f"provider-v2 contract {name} dtype changed")
    if not np.array_equal(left_array, right_array):
        raise ValueError(f"provider-v2 contract {name} identity changed")


def _validate_stack_parity(
    materialization: ProviderV2ContractMaterialization,
) -> None:
    dense = materialization.dense_stack
    sparse = materialization.sparse_stack
    tree = materialization.tree_sparse_stack

    for name in _COMMON_FLOAT_ROW_ARRAY_NAMES:
        _require_float_parity(
            getattr(dense, name),
            getattr(sparse, name),
            name=f"dense/sparse {name}",
        )
        _require_float_parity(
            getattr(sparse, name),
            getattr(tree, name),
            name=f"sparse/tree-sparse {name}",
        )
    _require_float_parity(
        sparse.local_gauge_jacobian,
        tree.local_gauge_jacobian,
        name="sparse/tree-sparse local_gauge_jacobian",
    )
    for name in ("point_ids", "frame_indices"):
        _require_integer_parity(
            getattr(dense, name),
            getattr(sparse, name),
            name=f"dense/sparse {name}",
        )
    for name in _INTEGER_ROW_ARRAY_NAMES:
        if name != "gauge_indices":
            _require_integer_parity(
                getattr(sparse, name),
                getattr(tree, name),
                name=f"sparse/tree-sparse {name}",
            )
    _require_integer_parity(
        sparse.gauge_indices,
        tree.gauge_indices,
        name="sparse/tree-sparse gauge_indices",
    )
    _require_float_parity(
        dense.gauge_jacobian,
        sparse.dense_gauge_jacobian(),
        name="dense/sparse gauge_jacobian",
    )
    _require_float_parity(
        dense.gauge_prior_covariance,
        sparse.gauge_prior_covariance,
        name="dense/sparse gauge_prior_covariance",
    )
    for name in _TUPLE_NAMES:
        dense_value = getattr(dense, name)
        sparse_value = getattr(sparse, name)
        tree_value = getattr(tree, name)
        if dense_value != sparse_value or sparse_value != tree_value:
            raise ValueError(f"provider-v2 contract {name} identity changed")
    if not (
        dense.causal_frame_stop
        == sparse.causal_frame_stop
        == tree.causal_frame_stop
    ):
        raise ValueError("provider-v2 contract causal frame stop changed")


def validate_provider_v2_contract_materialization(
    vector: ProviderV2ContractVector,
    materialization: ProviderV2ContractMaterialization,
) -> None:
    """Validate construction, exact structure, and tolerance-bound numerics."""

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

    reference_runtime_digest = _validate_sha256(
        expected["stack_sha256"],
        name="expected reference runtime stack_sha256",
    )
    if not reference_runtime_digest:
        raise AssertionError("validated SHA-256 digest unexpectedly empty")
    semantic_digest = provider_v2_contract_stack_semantic_sha256(
        materialization.tree_sparse_stack
    )
    if semantic_digest != PROVIDER_V2_CONTRACT_MINIMAL_STACK_SEMANTIC_SHA256:
        raise ValueError("provider-v2 contract tree-sparse semantic identity changed")

    expected_mean = _array_record(
        expected["world_mean_m"],
        name="expected world_mean_m",
    )
    if not np.allclose(
        materialization.tree_sparse_stack.world_mean_m,
        expected_mean,
        atol=PROVIDER_V2_CONTRACT_NUMERICAL_ATOL,
        rtol=PROVIDER_V2_CONTRACT_NUMERICAL_RTOL,
    ):
        raise ValueError("provider-v2 contract world means changed")

    _validate_stack_parity(materialization)
    _require_float_parity(
        materialization.bundle.joint_gauge_covariance,
        materialization.sparse_stack.gauge_prior_covariance,
        name="bundle/sparse joint gauge covariance",
    )
    materialization.gauge_tree_prior.verify_dense_covariance(
        materialization.bundle.joint_gauge_covariance,
        atol=PROVIDER_V2_CONTRACT_NUMERICAL_ATOL,
        rtol=PROVIDER_V2_CONTRACT_NUMERICAL_RTOL,
    )


__all__ = [
    "PROVIDER_V2_CONTRACT_MINIMAL_STACK_SEMANTIC_SHA256",
    "PROVIDER_V2_CONTRACT_NUMERICAL_ATOL",
    "PROVIDER_V2_CONTRACT_NUMERICAL_RTOL",
    "PROVIDER_V2_CONTRACT_STACK_SEMANTIC_SCHEMA",
    "PROVIDER_V2_CONTRACT_STACK_SEMANTIC_VERSION",
    "ProviderV2ContractMaterialization",
    "materialize_provider_v2_contract_vector",
    "provider_v2_contract_array_sha256",
    "provider_v2_contract_runtime_stack_sha256",
    "provider_v2_contract_stack_semantic_sha256",
    "provider_v2_contract_stack_sha256",
    "validate_provider_v2_contract_materialization",
]
