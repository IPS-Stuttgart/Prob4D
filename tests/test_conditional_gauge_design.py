"""Numerical, causal-input, and exact-replay controls for conditional windows."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from prob4d.conditional_gauge_design import (
    ConditionalGaugeSession,
    CorrelatedGaugeDesign,
    GaussianGaugeBelief,
    QueryWindowUtility,
    UnsupportedDeterministicConstraint,
    select_query_window,
)


def _model(design, noise, sizes, ids=None):
    return CorrelatedGaugeDesign(
        "normalized-test-chart", "known-synthetic-noise", tuple(ids or [f"w{i}" for i in range(len(sizes))]),
        tuple(sizes), design, noise,
    )


def _prior(mean=None, covariance=None):
    return GaussianGaugeBelief(
        "normalized-test-chart", np.zeros(7) if mean is None else mean,
        np.eye(7) if covariance is None else covariance,
    )


@pytest.mark.parametrize("seed", range(5))
def test_sequential_correlated_updates_equal_dense_joint_posterior(seed):
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(9, 7))
    root = rng.normal(size=(9, 9))
    noise = root @ root.T + 0.3 * np.eye(9)
    p_root = rng.normal(size=(7, 7))
    prior = _prior(rng.normal(size=7), p_root @ p_root.T + np.eye(7))
    values = rng.normal(size=9)
    model = _model(design, noise, [2, 3, 4])
    prior_information = np.linalg.solve(prior.covariance, np.eye(7))
    expected_cov = np.linalg.inv(prior_information + design.T @ np.linalg.solve(noise, design))
    expected_mean = expected_cov @ (prior_information @ prior.mean + design.T @ np.linalg.solve(noise, values))
    for order in itertools.permutations(model.window_ids):
        session = ConditionalGaugeSession(model, prior)
        for window in order:
            session.assimilate(window, values[list(model.rows((window,)))])
        np.testing.assert_allclose(session.belief.covariance, expected_cov, rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(session.belief.mean, expected_mean, rtol=2e-10, atol=2e-11)


def test_exact_replay_has_zero_utility_and_preserves_complete_belief_identity():
    design = np.eye(7)[[0, 2, 3, 4, 5, 6]]
    covariance = 0.1 * np.eye(6)
    model = _model(np.vstack((design, design)), np.tile(covariance, (2, 2)), [6, 6])
    session = ConditionalGaugeSession(model, _prior())
    first = session.assimilate("w0", np.arange(6.0))
    utility = session.preview_query("w1", np.eye(7))
    assert utility.conditional_information_rank == 0
    assert utility.variance_reduction == 0
    assert select_query_window((utility,)) is None
    replay = session.assimilate("w1", np.arange(6.0))
    assert replay is first
    assert session.history_ids == ("w0", "w1")


def test_inconsistent_duplicate_fails_before_mutating_session():
    h = np.eye(7)[[0]]
    session = ConditionalGaugeSession(_model(np.vstack((h, h)), np.ones((2, 2)), [1, 1]), _prior())
    before = session.assimilate("w0", np.array([1.0]))
    with pytest.raises(ValueError, match="noise-support identity"):
        session.assimilate("w1", np.array([2.0]))
    assert session.belief is before
    assert session.history_ids == ("w0",)


def test_rank_deficiency_is_preserved_without_fabricated_precision():
    h = np.eye(7)[[0, 2, 3, 4, 5, 6]]
    session = ConditionalGaugeSession(_model(h, 0.1 * np.eye(6), [6]), _prior())
    posterior = session.assimilate("w0", np.ones(6))
    assert posterior.mean[1] == 0
    assert posterior.covariance[1, 1] == 1
    assert session.model.conditional_factor((), "w0").information_rank == 6


def test_prior_mediated_gain_is_not_falsely_forbidden():
    p = np.eye(7)
    p[0, 1] = p[1, 0] = 0.7
    session = ConditionalGaugeSession(_model(np.eye(7)[[0]], np.eye(1), [1]), _prior(covariance=p))
    assert session.preview_query("w0", np.eye(7)[[1]]).variance_reduction > 0
    assert session.assimilate("w0", np.array([1.0])).covariance[1, 1] < 1


def test_zero_standalone_state_information_can_have_conditional_query_value():
    h = np.vstack((np.eye(7)[0], np.zeros(7)))
    model = _model(h, np.array([[1.0, 1.0], [1.0, 1.1]]), [1, 1])
    untouched = ConditionalGaugeSession(model, _prior())
    assert untouched.preview_query("w1", np.eye(7)[[0]]).variance_reduction == 0
    untouched.assimilate("w0", np.array([1.0]))
    conditional = untouched.preview_query("w1", np.eye(7)[[0]])
    assert conditional.conditional_information_rank == 1
    assert conditional.variance_reduction > 0.4
    posterior = untouched.assimilate("w1", np.array([0.5]))
    assert posterior.covariance[0, 0] < 0.1


def test_candidate_selection_does_not_depend_on_observed_history_values():
    h = np.eye(7)[[0, 0, 1]]
    noise = np.array([[1.0, 0.9, 0], [0.9, 1.0, 0], [0, 0, 2.0]])
    model = _model(h, noise, [1, 1, 1])
    utilities = []
    for value in [-1e6, 0.0, 1e6]:
        session = ConditionalGaugeSession(model, _prior())
        session.assimilate("w0", np.array([value]))
        utilities.append(tuple(session.preview_query(w, np.eye(7)) for w in ["w1", "w2"]))
    assert utilities[0] == utilities[1] == utilities[2]


def test_global_chart_change_and_query_units_preserve_utility():
    rng = np.random.default_rng(80)
    h = rng.normal(size=(5, 7))
    n = rng.normal(size=(5, 5))
    r = n @ n.T + np.eye(5)
    j = rng.normal(size=(3, 7))
    transform = rng.normal(size=(7, 7)) + 4 * np.eye(7)
    inverse = np.linalg.inv(transform)
    original = ConditionalGaugeSession(_model(h, r, [2, 3]), _prior())
    changed = ConditionalGaugeSession(_model(h @ inverse, r, [2, 3]), _prior(covariance=transform @ transform.T))
    original.assimilate("w0", np.ones(2))
    changed.assimilate("w0", np.ones(2))
    a = original.preview_query("w1", j)
    b = changed.preview_query("w1", 1000 * j @ inverse, query_metric=1e-6 * np.eye(3))
    np.testing.assert_allclose(a.variance_reduction, b.variance_reduction, rtol=1e-10)


@pytest.mark.parametrize("scale", [1e-7, 1.0, 1e7])
def test_measurement_unit_scaling_preserves_utility(scale):
    h = np.eye(7)[[0, 1]]
    r = np.array([[1.0, 0.5], [0.5, 1.0]])
    session = ConditionalGaugeSession(_model(scale * h, scale**2 * r, [1, 1]), _prior())
    session.assimilate("w0", np.array([scale]))
    reference = ConditionalGaugeSession(_model(h, r, [1, 1]), _prior())
    reference.assimilate("w0", np.ones(1))
    np.testing.assert_allclose(session.preview_query("w1", np.eye(7)).variance_reduction,
                               reference.preview_query("w1", np.eye(7)).variance_reduction, rtol=1e-10)


def test_informative_zero_noise_is_rejected_not_silently_discarded():
    with pytest.raises(UnsupportedDeterministicConstraint):
        _model(np.eye(7)[[0]], np.zeros((1, 1)), [1])
    model = _model(np.zeros((1, 7)), np.zeros((1, 1)), [1])
    session = ConditionalGaugeSession(model, _prior())
    prior = session.belief
    assert session.assimilate("w0", np.zeros(1)) is prior


def test_arrays_are_defensive_copies_and_immutable():
    h = np.eye(7)
    r = np.eye(7)
    model = _model(h, r, [7])
    h[:] = 3
    r[:] = 4
    np.testing.assert_array_equal(model.design_matrix, np.eye(7))
    with pytest.raises(ValueError):
        model.noise_covariance[0, 0] = 7
    prior = _prior()
    with pytest.raises(ValueError):
        prior.covariance[0, 0] = 7


@pytest.mark.parametrize("bad_noise", [np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([[1.0, 0.2], [0.0, 1.0]]), np.full((2, 2), np.nan)])
def test_invalid_joint_covariance_is_rejected(bad_noise):
    with pytest.raises(ValueError):
        _model(np.eye(7)[[0, 1]], bad_noise, [1, 1])


def test_validation_and_no_reconsumption():
    model = _model(np.eye(7)[[0]], np.eye(1), [1])
    with pytest.raises(ValueError, match="same declared"):
        ConditionalGaugeSession(model, GaussianGaugeBelief("different", np.zeros(7), np.eye(7)))
    session = ConditionalGaugeSession(model, _prior())
    for bad in [np.zeros((1, 1)), np.zeros(2), np.array([np.nan])]:
        with pytest.raises(ValueError):
            session.assimilate("w0", bad)
    assert session.history_ids == ()
    with pytest.raises(ValueError):
        session.preview_query("w0", np.ones((1, 6)))
    with pytest.raises(ValueError):
        session.preview_query("w0", np.ones((2, 7)), query_metric=np.diag([1.0, -1.0]))
    assert session.preview_query("w0", np.zeros((1, 7))).variance_reduction == 0
    session.assimilate("w0", np.zeros(1))
    with pytest.raises(ValueError, match="already been assimilated"):
        session.assimilate("w0", np.zeros(1))


def test_selection_threshold_ties_cost_and_empty_cases():
    a = QueryWindowUtility("a", 1, 10.0, 8.0, 2.0)
    b = QueryWindowUtility("b", 1, 10.0, 9.0, 1.0)
    assert select_query_window((b, a)) == "a"
    assert select_query_window((a, b), minimum_gain_per_cost=1) is None
    assert select_query_window(()) is None
    with pytest.raises(ValueError):
        select_query_window((a, a))
    with pytest.raises(ValueError):
        select_query_window((a,), minimum_gain_per_cost=-1)


@pytest.mark.parametrize("cost", [0, -1, np.inf, np.nan, True])
def test_invalid_utility_cannot_enter_selector(cost):
    with pytest.raises(ValueError):
        QueryWindowUtility("bad", 1, 1.0, 0.5, cost)


def test_study_reproducibility_and_predeclared_choice():
    from prob4d.conditional_gauge_study import run_study

    first = run_study(episodes=200, bootstrap_replicates=20)
    second = run_study(episodes=200, bootstrap_replicates=20)
    assert first == second
    arms = first["arms"]
    assert arms["conditional_query_selection_correct_update"]["selected_window"] == "complement"
    assert arms["marginal_query_selection_independent_update"]["selected_window"] == "near_repeat"
    assert arms["global_variance_selection_correct_update"]["selected_window"] == "global_only"
    assert first["maximum_kernel_vs_independent_dense_reference_error"] < 1e-10
    assert first["correlation_sweep"][-1]["query_variance_reduction_mm2"]["near_repeat"] == 0
    assert first["protocol"]["targets_opened"] is False


def test_singular_repeated_history_still_conditions_future_noise_correctly():
    h = np.eye(7)[[0, 0, 1]]
    r = np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [0.5, 0.5, 1.0]])
    model = _model(h, r, [1, 1, 1])
    a = ConditionalGaugeSession(model, _prior())
    b = ConditionalGaugeSession(model, _prior())
    a.assimilate("w0", np.array([1.0]))
    a.assimilate("w1", np.array([1.0]))
    a.assimilate("w2", np.array([2.0]))
    b.assimilate("w0", np.array([1.0]))
    before_replay = b.assimilate("w2", np.array([2.0]))
    assert b.assimilate("w1", np.array([1.0])) is before_replay
    np.testing.assert_allclose(a.belief.mean, b.belief.mean, atol=1e-12)
    np.testing.assert_allclose(a.belief.covariance, b.belief.covariance, atol=1e-12)
