from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.cycle_guard_group_robust import (
    GROUP_ROBUST_THRESHOLD_SEMANTICS,
    finite_sample_conformal_threshold,
    fit_group_robust_cycle_calibration,
    fit_observable_uncertainty_strata,
)


def test_observable_strata_are_development_frozen_tertiles() -> None:
    strata = fit_observable_uncertainty_strata(np.arange(1.0, 13.0))

    assert strata.assign(1.0) == "low"
    assert strata.assign(6.0) == "medium"
    assert strata.assign(12.0) == "high"
    assert strata.to_dict()["stratum_ids"] == ["low", "medium", "high"]


def test_conformal_threshold_uses_declared_finite_sample_order() -> None:
    fitted = finite_sample_conformal_threshold(
        [0.1, 0.4, 0.2, 0.3, 0.5],
        alpha=0.20,
    )

    assert fitted.order_index == math.ceil(6 * 0.8)
    assert fitted.threshold == 0.5
    assert fitted.finite_sample_coverage_level == fitted.order_index / 6


def test_conformal_threshold_fails_when_sample_cannot_support_alpha() -> None:
    with pytest.raises(ValueError, match="too small"):
        finite_sample_conformal_threshold([0.2], alpha=0.05)


def test_group_robust_threshold_is_maximum_observable_stratum_threshold() -> None:
    development = np.linspace(1.0, 30.0, 30)
    calibration_features = np.linspace(1.0, 30.0, 30)
    calibration_scores = np.concatenate(
        (
            np.linspace(0.1, 0.2, 10),
            np.linspace(0.2, 0.4, 10),
            np.linspace(0.3, 0.9, 10),
        )
    )

    fitted = fit_group_robust_cycle_calibration(
        development,
        calibration_features,
        calibration_scores,
        alpha=0.20,
        minimum_calibration_per_stratum=5,
    )

    assert fitted.threshold_semantics == GROUP_ROBUST_THRESHOLD_SEMANTICS
    assert fitted.worst_group_threshold == max(
        fitted.threshold_by_stratum.values()
    )
    assert fitted.threshold_for_feature(29.0) == fitted.threshold_by_stratum["high"]


def test_group_robust_calibration_rejects_unsupported_strata() -> None:
    with pytest.raises(ValueError, match="only"):
        fit_group_robust_cycle_calibration(
            np.linspace(1.0, 30.0, 30),
            [1.0, 2.0, 29.0],
            [0.1, 0.2, 0.3],
            alpha=0.5,
            minimum_calibration_per_stratum=2,
        )
