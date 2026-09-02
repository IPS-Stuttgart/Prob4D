from __future__ import annotations

import json
import sys
from importlib import util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_dot_r11_r20_query_message_real_v1.py"
PROTOCOL = ROOT / "protocols/dot-r11-r20-query-message-real-v1.json"


def _module():
    spec = util.spec_from_file_location("dot_query_message_real", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_is_content_addressed_and_source_only() -> None:
    module = _module()
    protocol = module._load_protocol(PROTOCOL)
    assert protocol["stage"] == "development-only-source-cross-validated"
    assert protocol["dataset"]["source_sequences"] == [
        f"R{index:02d}" for index in range(11, 21)
    ]
    assert protocol["dataset"]["confirmation_sequences"] == [
        f"R{index:02d}" for index in range(21, 31)
    ]
    assert protocol["dataset"]["reserved_sequences"] == "R31-R70"
    boundary = protocol["information_boundary"]
    assert boundary["sealed_provider_reuse_only"] is True
    assert boundary["provider_rerun_authorized"] is False
    assert boundary["confirmation_access_authorized"] is False
    assert boundary["r21_r30_payloads_opened"] is False
    assert boundary["r31_r70_payloads_opened"] is False
    assert boundary["target_side_retuning_allowed"] is False
    assert boundary["bayesian_phystwin_executed"] is False
    assert boundary["causal4d_executed"] is False


def test_equal_sequence_moments_do_not_pool_marker_counts() -> None:
    module = _module()
    first = np.asarray([[0.0, 0.0, 0.0]])
    second = np.repeat(np.asarray([[10.0, 0.0, 0.0]]), 100, axis=0)
    mean = module._equal_sequence_mean((first, second))
    np.testing.assert_allclose(mean, np.asarray([5.0, 0.0, 0.0]))
    second_moment = module._equal_sequence_second_moment(
        (first, second), np.zeros(3)
    )
    assert second_moment[0, 0] == pytest.approx(50.0)


def test_regularized_covariance_is_spd_for_rank_deficient_samples() -> None:
    module = _module()
    raw = np.asarray(
        [
            [2.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    result = module._regularize_covariance(
        raw,
        diagonal_shrinkage=0.2,
        absolute_floor=1.0e-4,
    )
    np.linalg.cholesky(result)
    assert float(np.linalg.eigvalsh(result)[0]) >= 1.0e-4 - 1.0e-12


def test_bounded_joint_covariance_preserves_marginals() -> None:
    module = _module()
    covariance_a = np.diag([1.0, 2.0, 3.0])
    covariance_b = np.diag([1.5, 2.5, 3.5])
    cross = np.asarray(
        [
            [3.0, 0.2, 0.0],
            [0.0, 4.0, 0.1],
            [0.2, 0.0, 5.0],
        ]
    )
    joint, bounded, canonical = module._joint_covariance_with_bounded_correlation(
        covariance_a,
        covariance_b,
        cross,
        cross_shrinkage=0.0,
        maximum_canonical_correlation=0.9,
    )
    np.linalg.cholesky(joint)
    np.testing.assert_allclose(joint[:3, :3], covariance_a)
    np.testing.assert_allclose(joint[3:, 3:], covariance_b)
    np.testing.assert_allclose(joint[:3, 3:], bounded)
    assert float(np.max(canonical)) <= 0.9 + 1.0e-12


def test_query_message_and_duplicate_controls_are_exact() -> None:
    module = _module()
    prior = np.diag([2.0, 3.0, 4.0])
    noise = np.diag([0.4, 0.8, 1.2])
    observation = np.asarray([0.2, -0.1, 0.4])
    posterior_mean, posterior_covariance = module._posterior_identity_measurement(
        prior, observation, noise
    )
    message = module._message(
        prior_covariance=prior,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        sequence="R11",
        marker_index=4,
        run="window_a",
    )
    belief = module.apply_gaussian_query_message(message)
    np.testing.assert_allclose(belief.mean, posterior_mean, atol=1.0e-12)
    np.testing.assert_allclose(
        belief.covariance, posterior_covariance, atol=1.0e-12
    )
    duplicate = module.fuse_gaussian_query_messages_covariance_intersection(
        (message, message),
        weights=(0.5, 0.5),
    )
    duplicate_belief = module.apply_gaussian_query_message(duplicate)
    np.testing.assert_allclose(duplicate_belief.mean, belief.mean, atol=1.0e-12)
    np.testing.assert_allclose(
        duplicate_belief.covariance, belief.covariance, atol=1.0e-12
    )


def test_update_probability_matches_scalar_geometry() -> None:
    module = _module()
    means = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    covariance = np.eye(3)
    probability = module._probability_update_beats_fallback(means, covariance)
    assert probability[0] == pytest.approx(0.6914624612740131)
    assert probability[1] == pytest.approx(0.5)


def test_source_decision_distinguishes_empirical_and_algebraic_failures() -> None:
    module = _module()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    row = {
        "mean_gaussian_nll_per_dimension": 1.0,
        "coverage_90": 0.9,
        "harmful_execute_fraction_all": 0.0,
        "deployed_decision_loss_per_coordinate": 1.0,
    }
    methods = {
        name: dict(row)
        for name in (
            "two_window_query_ci_equal",
            "naive_independent_message_sum",
            "window_a_only",
            "window_b_only",
        )
    }
    aggregate = {"sequence_count": 10, "methods": methods}
    parity = {
        "single_mean": 0.0,
        "single_covariance": 0.0,
        "duplicate_mean": 0.0,
        "duplicate_covariance": 0.0,
    }
    decision, checks = module._source_decision(aggregate, parity, protocol)
    assert decision == "source-real-overlap-positive"
    assert all(checks.values())

    methods["two_window_query_ci_equal"]["mean_gaussian_nll_per_dimension"] = 2.0
    decision, checks = module._source_decision(aggregate, parity, protocol)
    assert decision == "source-real-overlap-mixed"
    assert checks["single_message_parity"] is True

    parity["single_mean"] = 1.0
    decision, _ = module._source_decision(aggregate, parity, protocol)
    assert decision == "source-real-overlap-negative"
