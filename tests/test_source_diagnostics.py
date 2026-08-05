from __future__ import annotations

import numpy as np
import pytest

from prob4d.data import PredictionWindow
from prob4d.source_diagnostics import (
    CommonModeFailureAudit,
    SourceOnlyDiagnosticGrid,
    audit_common_mode_failures,
    augment_source_reliability_features,
    build_common_gauge_seed_dispersion_diagnostic,
    build_flow_point_consistency_diagnostic,
)
from prob4d.source_reliability import SourceReliabilityFeatures


class _StringLike:
    def __str__(self) -> str:
        return "coerced-identifier"


class _StringSubclass(str):
    pass


def _window(
    window_id: str,
    *,
    offset: float = 0.0,
    flow_scale: float = 1.0,
    frame_indices: np.ndarray | None = None,
) -> PredictionWindow:
    points = np.zeros((3, 1, 2, 3), dtype=np.float64)
    for local_index in range(3):
        points[local_index, ..., 0] = 10.0 + offset + local_index
    flow = np.zeros_like(points)
    flow[:-1, ..., 0] = flow_scale
    valid = np.ones(points.shape[:-1], dtype=bool)
    deform = np.zeros_like(valid)
    deform[:-1] = True
    return PredictionWindow(
        window_id=window_id,
        frame_indices=(np.arange(3, dtype=np.int64) if frame_indices is None else frame_indices),
        point_map=points,
        valid_mask=valid,
        scene_flow=flow,
        deform_mask=deform,
    )


def _diagnostic_metadata() -> dict[str, bool]:
    return {
        "uses_truth": False,
        "uses_downstream_physical_innovation": False,
        "uses_association_probability": False,
    }


def _diagnostic_with_names(feature_names: object) -> SourceOnlyDiagnosticGrid:
    return SourceOnlyDiagnosticGrid(
        feature_names=feature_names,  # type: ignore[arg-type]
        values=np.zeros((1, 1, 1), dtype=np.float64),
        available_mask=np.ones((1, 1), dtype=bool),
        metadata=_diagnostic_metadata(),
    )


def test_flow_point_consistency_is_zero_for_matching_one_step_flow() -> None:
    diagnostic = build_flow_point_consistency_diagnostic(_window("seed-1"))

    assert diagnostic.feature_names == (
        "has_flow_point_consistency",
        "log1p_relative_flow_point_residual",
        "flow_point_direction_disagreement",
    )
    assert np.all(diagnostic.available_mask[:-1])
    assert not np.any(diagnostic.available_mask[-1])
    assert np.allclose(diagnostic.values[:-1, ..., 1:], 0.0)
    assert diagnostic.metadata["uses_truth"] is False


def test_flow_point_consistency_detects_magnitude_and_direction_mismatch() -> None:
    diagnostic = build_flow_point_consistency_diagnostic(_window("seed-1", flow_scale=-1.0))

    assert np.all(diagnostic.values[:-1, ..., 1] > 0.0)
    assert np.allclose(diagnostic.values[:-1, ..., 2], 1.0)


def test_diagnostic_feature_names_reject_coercion_and_normalization() -> None:
    with pytest.raises(TypeError, match="canonical tuple"):
        _diagnostic_with_names(["feature"])
    with pytest.raises(TypeError, match="canonical tuple"):
        _diagnostic_with_names("feature")

    invalid_tuples = (
        (1,),
        (" feature",),
        ("feature ",),
        ("",),
        (_StringLike(),),
        (_StringSubclass("feature"),),
    )
    for feature_names in invalid_tuples:
        with pytest.raises(ValueError):
            _diagnostic_with_names(feature_names)


def test_seed_dispersion_requires_common_grid_and_reports_sample_spread() -> None:
    first = _window("seed-1")
    identical = _window("seed-2")
    shifted = _window("seed-3", offset=2.0)

    zero = build_common_gauge_seed_dispersion_diagnostic(
        [first, identical],
        common_gauge_id="metric-anchor-a",
        model_set_id="model-set-a",
    )
    spread = build_common_gauge_seed_dispersion_diagnostic(
        [first, shifted],
        common_gauge_id="metric-anchor-a",
        model_set_id="model-set-a",
    )

    assert np.all(zero.available_mask)
    assert np.allclose(zero.values[..., 2], 0.0)
    assert np.all(spread.values[..., 2] > 0.0)
    assert np.allclose(spread.values[..., 1], 1.0)
    assert spread.metadata["alignment_performed_by_diagnostic"] is False


def test_seed_dispersion_rejects_noncanonical_identifiers() -> None:
    windows = [_window("seed-1"), _window("seed-2")]
    invalid_values = (
        " identifier",
        "identifier ",
        "",
        _StringLike(),
        _StringSubclass("identifier"),
        1,
    )
    for value in invalid_values:
        with pytest.raises(ValueError):
            build_common_gauge_seed_dispersion_diagnostic(
                windows,
                common_gauge_id=value,  # type: ignore[arg-type]
                model_set_id="model-set-a",
            )
        with pytest.raises(ValueError):
            build_common_gauge_seed_dispersion_diagnostic(
                windows,
                common_gauge_id="metric-anchor-a",
                model_set_id=value,  # type: ignore[arg-type]
            )


def test_seed_dispersion_rejects_different_frame_identity() -> None:
    with pytest.raises(ValueError, match="frame indices differ"):
        build_common_gauge_seed_dispersion_diagnostic(
            [
                _window("seed-1"),
                _window(
                    "seed-2",
                    frame_indices=np.asarray([1, 2, 3], dtype=np.int64),
                ),
            ],
            common_gauge_id="metric-anchor-a",
            model_set_id="model-set-a",
        )


def test_diagnostics_augment_source_reliability_without_using_target_data() -> None:
    window = _window("seed-1")
    base = SourceReliabilityFeatures(
        feature_names=("base",),
        values=np.zeros(window.shape + (1,), dtype=np.float64),
        valid_mask=window.valid_mask,
        metadata={
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
            "uses_association_probability": False,
        },
    )
    diagnostic = build_flow_point_consistency_diagnostic(window)

    augmented = augment_source_reliability_features(base, [diagnostic])

    assert augmented.feature_names == ("base", *diagnostic.feature_names)
    assert np.array_equal(augmented.valid_mask, base.valid_mask)
    assert augmented.values.shape[-1] == 4
    assert augmented.metadata["uses_truth"] is False
    assert len(augmented.metadata["source_only_diagnostics"]) == 1


def test_common_mode_audit_reports_all_four_quadrants() -> None:
    audit = audit_common_mode_failures(
        np.asarray([0.1, 0.9, 0.1, 0.9]),
        np.asarray([0.1, 0.1, 0.9, 0.9]),
        disagreement_threshold=0.5,
        error_threshold=0.5,
    )

    assert audit.low_disagreement_low_error_count == 1
    assert audit.high_disagreement_low_error_count == 1
    assert audit.low_disagreement_high_error_count == 1
    assert audit.high_disagreement_high_error_count == 1
    assert audit.low_disagreement_high_error_rate == pytest.approx(0.25)
    assert audit.low_disagreement_high_error_mean == pytest.approx(0.9)
    assert audit.to_dict()["low_disagreement_high_error_rate"] == pytest.approx(0.25)


def test_common_mode_audit_rejects_coercible_scalars() -> None:
    with pytest.raises(TypeError, match="genuine real scalar"):
        audit_common_mode_failures(
            np.asarray([0.1]),
            np.asarray([0.1]),
            disagreement_threshold="0.5",  # type: ignore[arg-type]
            error_threshold=0.5,
        )
    with pytest.raises(TypeError, match="genuine real scalar"):
        audit_common_mode_failures(
            np.asarray([0.1]),
            np.asarray([0.1]),
            disagreement_threshold=False,
            error_threshold=0.5,
        )
    with pytest.raises(TypeError, match="genuine integer"):
        CommonModeFailureAudit(
            disagreement_threshold=0.5,
            error_threshold=0.5,
            valid_count=1.0,  # type: ignore[arg-type]
            low_disagreement_low_error_count=1,
            high_disagreement_low_error_count=0,
            low_disagreement_high_error_count=0,
            high_disagreement_high_error_count=0,
            low_disagreement_high_error_mean=0.0,
            low_disagreement_high_error_max=0.0,
        )


def test_common_mode_audit_rejects_inconsistent_severity() -> None:
    with pytest.raises(ValueError, match="require zero severity"):
        CommonModeFailureAudit(
            disagreement_threshold=0.5,
            error_threshold=0.5,
            valid_count=1,
            low_disagreement_low_error_count=1,
            high_disagreement_low_error_count=0,
            low_disagreement_high_error_count=0,
            high_disagreement_high_error_count=0,
            low_disagreement_high_error_mean=0.1,
            low_disagreement_high_error_max=0.1,
        )
    with pytest.raises(ValueError, match="mean cannot exceed"):
        CommonModeFailureAudit(
            disagreement_threshold=0.5,
            error_threshold=0.5,
            valid_count=1,
            low_disagreement_low_error_count=0,
            high_disagreement_low_error_count=0,
            low_disagreement_high_error_count=1,
            high_disagreement_high_error_count=0,
            low_disagreement_high_error_mean=2.0,
            low_disagreement_high_error_max=1.0,
        )
