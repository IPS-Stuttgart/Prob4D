from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prob4d.gauge_tree_prior import (
    GAUGE_TREE_PRIOR_SCHEMA,
    GAUGE_TREE_PRIOR_VERSION,
    GaugeTreeSquareRootPriorV1,
)


def _prior(*, gauge_count: int = 5) -> GaugeTreeSquareRootPriorV1:
    parents = np.asarray([-1] + [(index - 1) // 2 for index in range(1, gauge_count)])
    transitions = np.zeros((gauge_count, 7, 7), dtype=np.float64)
    scales = np.empty_like(transitions)
    scales[0] = np.diag(np.linspace(0.05, 0.11, 7))
    for index in range(1, gauge_count):
        transitions[index] = np.eye(7) * (0.7 + 0.02 * index)
        scales[index] = np.diag(np.linspace(0.02, 0.04, 7) * (1.0 + 0.05 * index))
    return GaugeTreeSquareRootPriorV1(
        gauge_ids=tuple(f"window-{index}" for index in range(gauge_count)),
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
    )


def test_identity_is_stable_and_changes_with_factor_content() -> None:
    prior = _prior()
    manifest = prior.to_dict()
    assert manifest["schema"] == GAUGE_TREE_PRIOR_SCHEMA
    assert manifest["version"] == GAUGE_TREE_PRIOR_VERSION
    assert manifest["prior_id"] == prior.prior_id
    modified = prior.transition_matrices.copy()
    modified[1, 0, 0] += 1e-6
    assert replace(prior, transition_matrices=modified).prior_id != prior.prior_id


def test_factor_arrays_are_irreversibly_readonly() -> None:
    prior = _prior()
    with pytest.raises(ValueError):
        prior.parent_indices[1] = 0
    with pytest.raises(ValueError):
        prior.transition_matrices[1, 0, 0] = 0.0
    with pytest.raises(ValueError):
        prior.innovation_scale_tril[0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        prior.transition_matrices.setflags(write=True)


def test_factor_storage_is_linear_and_below_dense_storage() -> None:
    small = _prior(gauge_count=4)
    large = _prior(gauge_count=8)
    assert large.factor_storage_nbytes == 2 * small.factor_storage_nbytes
    assert large.dense_covariance_nbytes == 4 * small.dense_covariance_nbytes
    assert large.factor_storage_nbytes < large.dense_covariance_nbytes
    assert large.storage_ratio_to_dense < small.storage_ratio_to_dense


def test_dense_materialization_has_an_explicit_size_guard() -> None:
    prior = _prior()
    with pytest.raises(ValueError, match="limited to 4 gauges"):
        prior.materialize_dense_covariance(maximum_gauges=4)
    with pytest.raises(TypeError, match="genuine integer"):
        prior.materialize_dense_covariance(maximum_gauges=True)  # type: ignore[arg-type]


def test_contract_rejects_invalid_tree_and_square_root_inputs() -> None:
    prior = _prior()
    with pytest.raises(ValueError, match="first gauge may be a root"):
        replace(prior, parent_indices=np.asarray([-1, -1, 0, 1, 1]))
    with pytest.raises(ValueError, match="precede its child"):
        replace(prior, parent_indices=np.asarray([-1, 0, 2, 1, 1]))
    bad_transition = prior.transition_matrices.copy()
    bad_transition[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="root transition"):
        replace(prior, transition_matrices=bad_transition)
    bad_scale = prior.innovation_scale_tril.copy()
    bad_scale[2, 0, 1] = 0.2
    with pytest.raises(ValueError, match="lower triangular"):
        replace(prior, innovation_scale_tril=bad_scale)
    bad_scale = prior.innovation_scale_tril.copy()
    bad_scale[2, 0, 0] = 0.0
    with pytest.raises(ValueError, match="positive diagonal"):
        replace(prior, innovation_scale_tril=bad_scale)


def test_contract_rejects_bad_shapes_unknown_ids_and_lossy_indices() -> None:
    prior = _prior()
    with pytest.raises(ValueError, match="value must have shape"):
        prior.covariance_action(np.zeros(7))
    with pytest.raises(ValueError, match="unknown gauges"):
        prior.selected_covariance(("missing",))
    with pytest.raises(ValueError, match="duplicate"):
        prior.selected_covariance(("window-0", "window-0"))
    jacobian = np.zeros((2, 3, 7))
    with pytest.raises(ValueError, match="genuine integers"):
        prior.row_marginal_covariance(jacobian, np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="unknown gauge"):
        prior.row_marginal_covariance(jacobian, np.asarray([0, 9]))
    with pytest.raises(ValueError, match=r"shape \(M, 3\)"):
        prior.observation_covariance_action(jacobian, np.asarray([0, 1]), np.zeros((2, 4)))


def test_information_quadratic_rejects_multiple_states() -> None:
    prior = _prior()
    with pytest.raises(ValueError, match="exactly one"):
        prior.information_quadratic(np.zeros((prior.dimension, 2)))
