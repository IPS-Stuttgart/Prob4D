from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_dot_rope_cut3r_measured_querybank_v1.py"


def _module():
    name = "dot_cut3r_measured_querybank"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_diagonal_low_rank_solver_matches_dense_covariance() -> None:
    module = _module()
    rng = np.random.default_rng(7)
    diagonal = rng.uniform(0.2, 1.0, size=24)
    factor = rng.normal(size=(24, 7)) / 4.0
    rhs = rng.normal(size=(24, 13))
    solver = module.DiagonalLowRankSolver(diagonal, factor)
    expected = np.linalg.solve(np.diag(diagonal) + factor @ factor.T, rhs)
    np.testing.assert_allclose(solver.solve(rhs), expected, rtol=1e-11, atol=1e-12)


def test_tempered_joint_model_preserves_marginal_variances_and_is_spd() -> None:
    module = _module()
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(24, 7)) / 5.0
    query_rows = module.coordinate_indices(module.select_query_indices(8, 4))
    prior, cross, diagonal, factor = module.tempered_joint_model(
        raw,
        query_rows,
        dependence_strength=0.85,
        noise_standard_deviation=0.03,
    )
    observation = np.diag(diagonal) + factor @ factor.T
    np.testing.assert_allclose(
        np.diag(observation),
        0.03**2 + np.sum(raw * raw, axis=1),
        rtol=1e-12,
        atol=1e-12,
    )
    query_raw = raw[query_rows]
    np.testing.assert_allclose(
        np.diag(prior),
        0.03**2 + np.sum(query_raw * query_raw, axis=1),
        rtol=1e-12,
        atol=1e-12,
    )
    joint = np.block([[prior, cross], [cross.T, observation]])
    assert np.linalg.eigvalsh(joint).min() > 0.0


def test_query_compression_matches_full_measured_factor() -> None:
    module = _module()
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(24, 7)) / 6.0
    query_rows = module.coordinate_indices(module.select_query_indices(8, 2))
    prior, cross, diagonal, factor = module.tempered_joint_model(
        raw,
        query_rows,
        dependence_strength=0.85,
        noise_standard_deviation=0.04,
    )
    full_solver = module.DiagonalLowRankSolver(diagonal, factor)
    full_gain, full_posterior = module.posterior(prior, cross, full_solver)
    compression = module.compress_shared_factor_for_posterior(
        factor.reshape(8, 3, 7),
        prior_query_covariance=prior,
        query_observation_cross_covariance=cross,
        innovation_operator=full_solver,
        maximum_rank=6,
        rank_relative_tolerance=1e-12,
        parity_relative_tolerance=1e-9,
    )
    reduced_factor = compression.compressed_factor_m.reshape(24, -1)
    reduced_solver = module.DiagonalLowRankSolver(diagonal, reduced_factor)
    reduced_gain, reduced_posterior = module.posterior(prior, cross, reduced_solver)
    assert compression.retained_rank <= min(6, factor.shape[1])
    np.testing.assert_allclose(reduced_gain, full_gain, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        reduced_posterior,
        full_posterior,
        rtol=1e-9,
        atol=1e-11,
    )


def test_spectral_baseline_is_psd_and_respects_byte_budget() -> None:
    module = _module()
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(24, 7)) / 5.0
    query_rows = module.coordinate_indices(module.select_query_indices(8, 4))
    prior, cross, diagonal, factor = module.tempered_joint_model(
        raw,
        query_rows,
        dependence_strength=0.85,
        noise_standard_deviation=0.05,
    )
    observation = np.diag(diagonal) + factor @ factor.T
    gain, posterior, details = module.spectral_joint_baseline(
        prior,
        cross,
        observation,
        byte_budget=4096,
    )
    assert gain.shape == cross.shape
    assert np.linalg.eigvalsh(posterior).min() > 0.0
    assert details["raw_representation_bytes"] <= details["byte_budget"]
