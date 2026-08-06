from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prob4d.visual_bias_calibration import (
    VisualBiasCalibrationGroup,
    build_visual_bias_nuisance_from_calibration,
    fit_visual_bias_calibration,
    load_visual_bias_calibration,
    write_visual_bias_calibration,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _design(row_count: int) -> np.ndarray:
    design = np.zeros((row_count, 3, 2), dtype=np.float64)
    design[:, 0, 0] = 1.0
    design[:, 1, 1] = np.linspace(-1.0, 1.0, row_count)
    return design


def _group(
    group_id: str,
    coefficient: tuple[float, float],
    *,
    row_count: int = 24,
    noise_scale: float = 0.02,
    gauge_design: np.ndarray | None = None,
) -> VisualBiasCalibrationGroup:
    design = _design(row_count)
    residual = np.einsum("nir,r->ni", design, np.asarray(coefficient))
    covariance = np.broadcast_to(
        noise_scale**2 * np.eye(3, dtype=np.float64),
        (row_count, 3, 3),
    ).copy()
    return VisualBiasCalibrationGroup(
        group_id=group_id,
        residual=residual.astype(np.float64),
        bias_jacobian=design,
        conditional_covariance=covariance,
        gauge_design=gauge_design,
        metadata={"source_only": True},
    )


def _fit(groups: tuple[VisualBiasCalibrationGroup, ...], **kwargs):
    return fit_visual_bias_calibration(
        groups,
        basis_names=("ray-depth", "depth-bowl"),
        provider_manifest_id=_SHA_A,
        calibration_source_id=_SHA_B,
        group_definition="complete-physical-object-v1",
        residual_definition="source-metric-minus-provider-point-v1",
        uses_truth=True,
        **kwargs,
    )


def test_selects_only_supported_coherent_mode() -> None:
    groups = tuple(
        _group(f"object-{index:02d}", (value, 0.0))
        for index, value in enumerate((-0.8, -0.5, -0.2, 0.2, 0.5, 0.8))
    )
    calibration = _fit(groups, minimum_nll_improvement=1e-3)

    assert calibration.promoted
    assert calibration.selected_rank == 1
    assert calibration.selected_basis_names == ("ray-depth",)
    assert calibration.rank_mean_nll[1] < calibration.rank_mean_nll[0]
    assert calibration.selected_covariance.shape == (1, 1)
    assert calibration.selected_covariance[0, 0] > 0.0
    with pytest.raises(ValueError):
        calibration.selected_covariance.setflags(write=True)


def test_rank_zero_is_valid_when_no_shared_bias_is_supported() -> None:
    groups = tuple(_group(f"object-{index:02d}", (0.0, 0.0)) for index in range(5))
    calibration = _fit(groups, minimum_nll_improvement=1e-4)

    assert not calibration.promoted
    assert calibration.selected_rank == 0
    assert calibration.selected_covariance.shape == (0, 0)
    with pytest.raises(ValueError, match="rank-zero"):
        build_visual_bias_nuisance_from_calibration(
            calibration,
            observation_artifact_id=_SHA_C,
            observation_identity_sha256=_SHA_D,
            bias_id="camera-0",
            bias_jacobian=_design(4),
        )


def test_equal_group_weighting_is_invariant_to_exact_row_duplication() -> None:
    original = tuple(
        _group(f"object-{index:02d}", (value, 0.0), row_count=12)
        for index, value in enumerate((-0.7, -0.3, 0.3, 0.7))
    )
    duplicated_first = VisualBiasCalibrationGroup(
        group_id=original[0].group_id,
        residual=np.repeat(original[0].residual, 2, axis=0),
        bias_jacobian=np.repeat(original[0].bias_jacobian, 2, axis=0),
        conditional_covariance=np.repeat(
            original[0].conditional_covariance,
            2,
            axis=0,
        ),
        metadata=original[0].metadata,
    )
    duplicated = (duplicated_first, *original[1:])

    first = _fit(original, minimum_nll_improvement=1e-3)
    second = _fit(duplicated, minimum_nll_improvement=1e-3)

    assert first.selected_rank == second.selected_rank
    np.testing.assert_allclose(first.rank_mean_nll, second.rank_mean_nll, atol=1e-12)
    np.testing.assert_allclose(
        first.selected_covariance,
        second.selected_covariance,
        atol=1e-12,
    )


def test_gauge_collinear_mode_is_not_promoted() -> None:
    groups = []
    for index, value in enumerate((-0.7, -0.2, 0.2, 0.7)):
        base = _group(f"object-{index:02d}", (value, 0.0), row_count=16)
        groups.append(
            VisualBiasCalibrationGroup(
                group_id=base.group_id,
                residual=base.residual,
                bias_jacobian=base.bias_jacobian,
                conditional_covariance=base.conditional_covariance,
                gauge_design=base.bias_jacobian[:, :, :1],
            )
        )
    calibration = _fit(tuple(groups), minimum_nll_improvement=1e-4)

    assert calibration.orthogonalization_semantics.startswith("conditional-whitened")
    assert np.max(calibration.group_maximum_gauge_projection) <= (
        calibration.gauge_projection_tolerance
    )
    assert calibration.selected_rank != 1


def test_round_trip_tamper_rejection_and_sidecar_binding(tmp_path: Path) -> None:
    groups = tuple(
        _group(f"object-{index:02d}", (value, 0.0))
        for index, value in enumerate((-0.8, -0.4, 0.4, 0.8))
    )
    calibration = _fit(groups, minimum_nll_improvement=1e-3)
    manifest = tmp_path / "calibration.json"
    _, payload = write_visual_bias_calibration(calibration, manifest)
    loaded = load_visual_bias_calibration(manifest)

    assert loaded.artifact_id == calibration.artifact_id
    assert loaded.summary() == calibration.summary()
    sidecar = build_visual_bias_nuisance_from_calibration(
        loaded,
        observation_artifact_id=_SHA_C,
        observation_identity_sha256=_SHA_D,
        bias_id="camera-0",
        bias_jacobian=_design(8),
        metadata={"case_id": "case-a"},
    )
    assert sidecar.basis_names == loaded.selected_basis_names
    assert sidecar.metadata["visual_bias_calibration_artifact_id"] == loaded.artifact_id
    assert sidecar.metadata["uses_target_outcomes"] is False

    with np.load(payload, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["rank_mean_nll"] = arrays["rank_mean_nll"] + 1.0
    np.savez_compressed(payload, **arrays)
    with pytest.raises(ValueError, match="SHA-256"):
        load_visual_bias_calibration(manifest)


def test_duplicate_json_and_existing_writer_lock_fail_closed(tmp_path: Path) -> None:
    groups = tuple(
        _group(f"object-{index:02d}", (value, 0.0))
        for index, value in enumerate((-0.6, -0.2, 0.2, 0.6))
    )
    calibration = _fit(groups, minimum_nll_improvement=1e-3)

    locked_manifest = tmp_path / "locked.json"
    lock = locked_manifest.with_name(f".{locked_manifest.name}.lock")
    lock.write_text("retained-lock\n", encoding="utf-8")
    with pytest.raises(ValueError, match="writer lock"):
        write_visual_bias_calibration(calibration, locked_manifest)

    manifest = tmp_path / "calibration.json"
    write_visual_bias_calibration(calibration, manifest)
    original = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        '{"schema":"duplicate",' + original.lstrip()[1:],
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_visual_bias_calibration(manifest)
