from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from prob4d.visual_bias import (
    VisualBiasNuisanceV1,
    load_visual_bias_nuisance,
    orthogonalize_visual_bias_basis,
    write_visual_bias_nuisance,
)


def _nuisance() -> VisualBiasNuisanceV1:
    return VisualBiasNuisanceV1(
        observation_artifact_id="a" * 64,
        observation_identity_sha256="b" * 64,
        bias_ids=("camera-0", "camera-1"),
        basis_names=("ray-depth",),
        row_bias_indices=np.asarray([0, 1], dtype=np.int64),
        bias_jacobian=np.asarray(
            [
                [[1.0], [0.0], [0.0]],
                [[0.0], [1.0], [0.0]],
            ],
            dtype=np.float64,
        ),
        joint_bias_covariance=np.asarray(
            [[4.0, 1.0], [1.0, 9.0]],
            dtype=np.float64,
        ),
        orthogonalization_semantics="not-orthogonalized",
        maximum_gauge_projection=0.5,
        gauge_projection_tolerance=1.0,
        metadata={"uses_truth": False},
    )


def test_low_rank_factor_preserves_complete_cross_scope_covariance() -> None:
    nuisance = _nuisance()
    design = nuisance.global_design()
    expected = design @ nuisance.joint_bias_covariance @ design.T
    factor = nuisance.low_rank_factor()
    actual = factor.reshape(6, -1) @ factor.reshape(6, -1).T
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        nuisance.marginal_covariance(),
        np.stack((expected[:3, :3], expected[3:, 3:])),
        atol=1e-12,
    )


def test_roundtrip_binds_payload_and_arrays(tmp_path: Path) -> None:
    nuisance = _nuisance()
    manifest = tmp_path / "bias.json"
    write_visual_bias_nuisance(nuisance, manifest)
    loaded = load_visual_bias_nuisance(manifest)
    assert loaded.artifact_id == nuisance.artifact_id
    np.testing.assert_array_equal(loaded.row_bias_indices, nuisance.row_bias_indices)
    np.testing.assert_allclose(loaded.bias_jacobian, nuisance.bias_jacobian)
    assert loaded.summary()["latent_dimension"] == 2


def test_payload_tamper_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "bias.json"
    _, payload = write_visual_bias_nuisance(_nuisance(), manifest)
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="byte count mismatch"):
        load_visual_bias_nuisance(manifest)


def test_global_gauge_projection_removes_shared_translation() -> None:
    rows = 4
    bias = np.zeros((rows, 3, 2), dtype=np.float64)
    bias[:, 0, 0] = 1.0
    bias[:, 1, 1] = np.asarray([-1.0, 1.0, -1.0, 1.0])
    gauge = np.zeros((rows, 3, 1), dtype=np.float64)
    gauge[:, 0, 0] = 1.0
    covariance = np.repeat(np.eye(3)[None, :, :], rows, axis=0)

    result = orthogonalize_visual_bias_basis(bias, gauge, covariance)
    assert result.gauge_rank == 1
    assert result.maximum_projection_before > 0.99
    assert result.maximum_projection_after < 1e-12
    np.testing.assert_allclose(result.bias_jacobian[:, 0, 0], 0.0, atol=1e-12)
    assert np.linalg.norm(result.bias_jacobian[:, :, 1]) > 0.0


def test_orthogonalized_contract_fails_when_projection_exceeds_tolerance() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        replace(
            _nuisance(),
            orthogonalization_semantics=(
                "conditional-whitened-global-gauge-projection-v1"
            ),
            maximum_gauge_projection=0.1,
            gauge_projection_tolerance=0.01,
            artifact_id=None,
        )


def test_arrays_are_immutable() -> None:
    nuisance = _nuisance()
    with pytest.raises(ValueError):
        nuisance.bias_jacobian[0, 0, 0] = 2.0
