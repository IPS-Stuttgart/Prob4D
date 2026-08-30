"""Exercise the new kernel through the existing production operator surfaces."""

import numpy as np

from prob4d.api.v2 import (
    StackedObservationFactors,
    build_observation_gaussian_operator,
)
from prob4d.posterior_preserving_compression import (
    compress_shared_factor_for_posterior,
)
from prob4d.query_posterior import (
    augment_observation_gaussian_operator,
    condition_gaussian_query,
)


def test_existing_structured_operator_and_query_conditioner_parity():
    conditional = np.array([np.diag([0.3, 0.4, 0.5]), np.diag([0.6, 0.7, 0.8])])
    factors = StackedObservationFactors(
        world_mean_m=np.zeros((2, 3)),
        conditional_world_covariance_m2=conditional,
        marginal_world_covariance_m2=conditional,
        gauge_jacobian=np.zeros((2, 3, 7)),
        gauge_prior_covariance=np.zeros((7, 7)),
        association_probability=np.ones(2),
        prior_reliability=np.ones(2),
        prior_nominal_probability=np.ones(2),
        composite_weight=np.ones(2),
        point_ids=np.array([10, 20], dtype=np.int64),
        frame_indices=np.array([0, 1], dtype=np.int64),
        view_ids=("camera", "camera"),
        factor_ids=("factor-0", "factor-1"),
        correlation_group_ids=("group-0", "group-1"),
        gauge_ids=("gauge-0",),
        causal_frame_stop=2,
    )
    base = build_observation_gaussian_operator(factors)
    rng = np.random.default_rng(20260830)
    shared = rng.normal(size=(2, 3, 7)) / 3
    physical = rng.normal(size=(2, 3, 4)) / 2
    query = rng.normal(size=(2, 4))
    prior = query @ query.T
    cross = query @ physical.reshape(6, 4).T
    full = augment_observation_gaussian_operator(
        base, np.concatenate([shared, physical], axis=2)
    )
    result = compress_shared_factor_for_posterior(
        shared, prior_query_covariance=prior,
        query_observation_cross_covariance=cross,
        innovation_operator=full, maximum_rank=2,
    )
    assert not result.exact_fallback
    assert result.retained_rank == 2
    reduced = augment_observation_gaussian_operator(
        base, np.concatenate([result.compressed_factor_m, physical], axis=2)
    )
    for _ in range(8):
        arguments = {
            "prior_mean": np.zeros(2), "prior_covariance": prior,
            "innovation": rng.normal(size=(2, 3)),
            "query_observation_cross_covariance": cross,
        }
        reference = condition_gaussian_query(**arguments, innovation_operator=full)
        actual = condition_gaussian_query(**arguments, innovation_operator=reduced)
        np.testing.assert_allclose(actual.posterior_mean, reference.posterior_mean, atol=1e-11)
        np.testing.assert_allclose(
            actual.posterior_covariance, reference.posterior_covariance, atol=1e-11
        )
    # The observation evidence is deliberately NOT preserved by the theorem.
    assert not np.isclose(full.log_determinant, reduced.log_determinant)
