from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_tracking_cloth_query_portfolio_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("query_portfolio", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_query_marker_selection_is_unique_deterministic_and_nested_at_endpoints():
    module = _module()
    np.testing.assert_array_equal(module.select_query_indices(12, 1), [0])
    np.testing.assert_array_equal(
        module.select_query_indices(12, 12), np.arange(12)
    )
    selected = module.select_query_indices(20, 8)
    assert len(selected) == len(np.unique(selected)) == 8
    assert selected[0] == 0 and selected[-1] == 19


def test_block_low_rank_solver_matches_dense_covariance():
    module = _module()
    rng = np.random.default_rng(4)
    blocks = np.stack(
        [np.diag(rng.uniform(0.2, 1.0, 3)) for _ in range(5)]
    )
    conditional = np.zeros((15, 15))
    for index, block in enumerate(blocks):
        conditional[
            3 * index : 3 * index + 3,
            3 * index : 3 * index + 3,
        ] = block
    factor = rng.normal(size=(15, 6)) / 3
    rhs = rng.normal(size=(15, 7))
    actual = module.BlockLowRankSolver(conditional, factor).solve(rhs)
    expected = np.linalg.solve(conditional + factor @ factor.T, rhs)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_spectral_joint_baseline_is_positive_definite():
    module = _module()
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(21, 8))
    joint = latent @ latent.T + 0.1 * np.eye(21)
    prior = joint[:6, :6]
    cross = joint[:6, 6:]
    observation = joint[6:, 6:]
    gain, posterior, details = module._spectral_joint_baseline(
        prior, cross, observation, rank=3
    )
    assert gain.shape == (6, 15)
    assert np.linalg.eigvalsh(posterior).min() > 0.0
    assert details["low_rank_dimension"] == 3
    assert details["representation_bytes"] > 0


def test_shared_decomposition_reconstructs_covariance():
    module = _module()
    rng = np.random.default_rng(11)
    blocks = np.stack(
        [np.diag(rng.uniform(0.4, 1.0, 3)) for _ in range(4)]
    )
    diagonal = np.zeros((12, 12))
    for index, block in enumerate(blocks):
        diagonal[
            3 * index : 3 * index + 3,
            3 * index : 3 * index + 3,
        ] = block
    shared = rng.normal(size=(12, 5)) / 5
    covariance = diagonal + shared @ shared.T
    conditional, factor, beta = module.decompose_shared_covariance(
        covariance, 0.2, 1e-12
    )
    assert 0.0 < beta <= 0.2
    np.testing.assert_allclose(
        conditional + factor @ factor.T,
        covariance,
        rtol=1e-9,
        atol=1e-12,
    )
