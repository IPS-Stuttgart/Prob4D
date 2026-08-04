from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prob4d.gauge import GaugeEstimate
from prob4d.observation_factors import ObservationFactor, ObservationFactorBundle
from prob4d.sim3 import Sim3
from prob4d.sparse_observation_factors import (
    SparseStackedObservationFactors,
    stack_sparse_observation_factors,
)


def _bundle(*, with_invalid_row: bool = False) -> ObservationFactorBundle:
    covariance_0 = np.eye(7, dtype=np.float64) * 2.0e-4
    covariance_1 = np.eye(7, dtype=np.float64) * 3.0e-4
    cross = np.eye(7, dtype=np.float64) * 5.0e-5
    joint = np.block(
        [
            [covariance_0, cross],
            [cross, covariance_1],
        ]
    )
    gauges = (
        GaugeEstimate("window-0", Sim3.identity(), covariance_0),
        GaugeEstimate("window-1", Sim3.identity(), covariance_1),
    )
    common = {
        "valid_mask": np.asarray([True, not with_invalid_row]),
        "local_covariance_m2": np.repeat(
            np.eye(3, dtype=np.float64)[None] * 1.0e-3,
            2,
            axis=0,
        ),
        "association_probability": np.asarray([0.9, 0.8]),
        "prior_reliability": np.asarray([0.85, 0.75]),
        "prior_nominal_probability": 0.95,
        "composite_weight": 0.5,
        "causal_frame_stop": 6,
    }
    factors = (
        ObservationFactor(
            factor_id="factor-0",
            frame_index=2,
            view_id="camera-0",
            window_id="window-0",
            gauge_id="window-0",
            point_ids=np.asarray([10, 11], dtype=np.int64),
            points_local_m=np.asarray(
                [[0.0, 0.0, 1.0], [0.2, 0.0, 1.1]],
                dtype=np.float64,
            ),
            correlation_group_id="camera-0:frame-2",
            **common,
        ),
        ObservationFactor(
            factor_id="factor-1",
            frame_index=4,
            view_id="camera-0",
            window_id="window-1",
            gauge_id="window-1",
            point_ids=np.asarray([20, 21], dtype=np.int64),
            points_local_m=np.asarray(
                [[0.1, 0.2, 1.2], [0.3, 0.1, 1.3]],
                dtype=np.float64,
            ),
            correlation_group_id="camera-0:frame-4",
            **common,
        ),
    )
    return ObservationFactorBundle(
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:explicit-gauge:camera-0",
        factors=factors,
        gauges=gauges,
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        causal_frame_stop=6,
        joint_gauge_covariance=joint,
        gauge_covariance_semantics="joint-cross-window",
    )


def test_sparse_stack_is_exactly_equivalent_to_dense_stack() -> None:
    bundle = _bundle()
    dense = bundle.stack()
    sparse = stack_sparse_observation_factors(bundle)

    np.testing.assert_allclose(sparse.world_mean_m, dense.world_mean_m)
    np.testing.assert_allclose(
        sparse.conditional_world_covariance_m2,
        dense.conditional_world_covariance_m2,
    )
    np.testing.assert_allclose(
        sparse.marginal_world_covariance_m2,
        dense.marginal_world_covariance_m2,
    )
    np.testing.assert_allclose(
        sparse.dense_gauge_jacobian(),
        dense.gauge_jacobian,
    )
    np.testing.assert_allclose(
        sparse.gauge_prior_covariance,
        dense.gauge_prior_covariance,
    )
    np.testing.assert_array_equal(
        sparse.association_probability,
        dense.association_probability,
    )
    np.testing.assert_array_equal(
        sparse.prior_reliability,
        dense.prior_reliability,
    )
    np.testing.assert_array_equal(
        sparse.prior_nominal_probability,
        dense.prior_nominal_probability,
    )
    np.testing.assert_array_equal(
        sparse.composite_weight,
        dense.composite_weight,
    )
    np.testing.assert_array_equal(sparse.point_ids, dense.point_ids)
    np.testing.assert_array_equal(sparse.frame_indices, dense.frame_indices)
    assert sparse.view_ids == dense.view_ids
    assert sparse.factor_ids == dense.factor_ids
    assert sparse.correlation_group_ids == dense.correlation_group_ids
    assert sparse.gauge_ids == dense.gauge_ids


def test_sparse_gauge_operations_match_dense_linear_algebra() -> None:
    bundle = _bundle()
    dense = bundle.stack()
    sparse = stack_sparse_observation_factors(bundle)
    delta = np.linspace(-0.02, 0.03, 14, dtype=np.float64)

    dense_response = np.einsum(
        "nij,j->ni",
        dense.gauge_jacobian,
        delta,
        optimize=True,
    )
    np.testing.assert_allclose(sparse.apply_gauge_delta(delta), dense_response)
    np.testing.assert_allclose(
        sparse.apply_gauge_delta(delta.reshape(2, 7)),
        dense_response,
    )
    np.testing.assert_allclose(
        sparse.gauge_marginal_covariance_m2(),
        sparse.marginal_world_covariance_m2
        - sparse.conditional_world_covariance_m2,
        atol=1e-15,
        rtol=1e-12,
    )
    assert np.any(sparse.gauge_prior_covariance[:7, 7:] != 0.0)


def test_sparse_stack_preserves_include_invalid_semantics() -> None:
    bundle = _bundle(with_invalid_row=True)
    dense_default = bundle.stack()
    sparse_default = stack_sparse_observation_factors(bundle)
    dense_all = bundle.stack(include_invalid=True)
    sparse_all = stack_sparse_observation_factors(bundle, include_invalid=True)

    assert sparse_default.observation_count == len(dense_default.world_mean_m) == 2
    assert sparse_all.observation_count == len(dense_all.world_mean_m) == 4
    np.testing.assert_allclose(
        sparse_all.dense_gauge_jacobian(),
        dense_all.gauge_jacobian,
    )


def test_sparse_stack_owns_readonly_arrays_and_rejects_lossy_indices() -> None:
    sparse = stack_sparse_observation_factors(_bundle())

    with pytest.raises(ValueError):
        sparse.local_gauge_jacobian[0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        sparse.gauge_indices[0] = 1
    with pytest.raises(TypeError, match="genuine integers"):
        replace(
            sparse,
            gauge_indices=sparse.gauge_indices.astype(np.float64),
        )


def test_sparse_stack_reduces_expanded_gauge_design_storage() -> None:
    sparse = stack_sparse_observation_factors(_bundle())

    assert sparse.gauge_count == 2
    assert sparse.dense_gauge_dimension == 14
    assert sparse.sparse_gauge_design_nbytes == (
        sparse.local_gauge_jacobian.nbytes + sparse.gauge_indices.nbytes
    )
    assert sparse.sparse_gauge_design_nbytes < sparse.dense_gauge_design_nbytes


def test_sparse_contract_rejects_wrong_gauge_delta_shape() -> None:
    sparse = stack_sparse_observation_factors(_bundle())

    with pytest.raises(ValueError, match="gauge_delta must have shape"):
        sparse.apply_gauge_delta(np.zeros(7, dtype=np.float64))
    with pytest.raises(ValueError, match="must be finite"):
        delta = np.zeros(14, dtype=np.float64)
        delta[0] = np.nan
        sparse.apply_gauge_delta(delta)


def test_sparse_contract_requires_observation_factor_bundle() -> None:
    with pytest.raises(TypeError, match="ObservationFactorBundle"):
        stack_sparse_observation_factors(object())  # type: ignore[arg-type]


def test_sparse_result_type_is_public_and_slotted() -> None:
    sparse = stack_sparse_observation_factors(_bundle())

    assert isinstance(sparse, SparseStackedObservationFactors)
    with pytest.raises((AttributeError, TypeError)):
        sparse.new_attribute = 1  # type: ignore[misc]
