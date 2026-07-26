from __future__ import annotations

import numpy as np
import pytest

from prob4d.alignment import (
    POINTWISE_COVARIANCE_FALLBACK,
    align_windows,
    alignment_covariance_context,
)
from prob4d.data import PredictionWindow
from prob4d.sim3 import Sim3


class DoubleCalibration:
    artifact_id = "d" * 64

    def apply(self, covariance: np.ndarray) -> np.ndarray:
        return 2.0 * covariance


def _windows() -> tuple[PredictionWindow, PredictionWindow]:
    generator = np.random.default_rng(10)
    global_points = generator.normal(size=(10, 4, 6, 3))
    moving_to_reference = Sim3.from_vector(
        np.array([-0.08, 0.03, 0.02, -0.04, 0.3, -0.1, 0.2])
    )
    reference = PredictionWindow(
        "reference",
        np.arange(3, 8),
        global_points[3:8],
        np.ones((5, 4, 6), dtype=bool),
    )
    moving = PredictionWindow(
        "moving",
        np.arange(5, 10),
        moving_to_reference.inverse().transform_points(global_points[5:10]),
        np.ones((5, 4, 6), dtype=bool),
    )
    return reference, moving


def test_pointwise_covariance_fallback_is_explicit() -> None:
    reference, moving = _windows()
    alignment = align_windows(reference, moving)

    assert alignment.result.covariance_fallback == POINTWISE_COVARIANCE_FALLBACK
    assert alignment.result.num_covariance_clusters == 72


def test_provider_context_can_fail_closed_on_cluster_fallback() -> None:
    reference, moving = _windows()
    with alignment_covariance_context(fallback_policy="error"):
        with pytest.raises(ValueError, match="fewer than eight"):
            align_windows(reference, moving)

    # Context reset preserves the low-level compatibility behavior.
    assert align_windows(reference, moving).result.covariance_fallback is not None


def test_provider_context_applies_calibration_and_records_artifact() -> None:
    reference, moving = _windows()
    baseline = align_windows(reference, moving).result
    with alignment_covariance_context(
        calibration=DoubleCalibration(),
        fallback_policy="pointwise",
    ) as diagnostics:
        calibrated = align_windows(reference, moving).result

    np.testing.assert_allclose(calibrated.covariance, 2.0 * baseline.covariance)
    assert calibrated.covariance_calibration_id == DoubleCalibration.artifact_id
    assert diagnostics.alignment_count == 1
    assert diagnostics.calibrated_alignment_count == 1
    assert diagnostics.fallback_counts == {POINTWISE_COVARIANCE_FALLBACK: 1}
