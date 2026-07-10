from pathlib import Path

import numpy as np

from prob4d.fusion import FusedSequence
from prob4d.metrics import TruthSequence
from prob4d.sintel_uncertainty import (
    SequenceInputs,
    _bootstrap_method_metric,
    _resize_bilinear,
    evaluate_prediction_uncertainty,
    held_out_split,
)


def test_held_out_split_is_disjoint_and_exhaustive() -> None:
    names = [
        "alley_1",
        "alley_2",
        "ambush_2",
        "ambush_4",
        "ambush_5",
        "ambush_6",
        "ambush_7",
        "bamboo_1",
        "bamboo_2",
        "bandage_1",
        "bandage_2",
        "cave_2",
        "cave_4",
        "market_2",
        "market_5",
        "market_6",
        "mountain_1",
        "shaman_2",
        "shaman_3",
        "sleeping_1",
        "sleeping_2",
        "temple_2",
        "temple_3",
    ]
    inputs = [SequenceInputs(name, Path(name), Path(name)) for name in names]

    calibration, test = held_out_split(inputs)

    assert len(calibration) == 11
    assert len(test) == 12
    assert {item.sequence for item in calibration} | {item.sequence for item in test} == set(names)
    calibration_families = {item.sequence.rsplit("_", 1)[0] for item in calibration}
    test_families = {item.sequence.rsplit("_", 1)[0] for item in test}
    assert calibration_families.isdisjoint(test_families)


def test_resize_bilinear_preserves_corners() -> None:
    values = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)

    resized = _resize_bilinear(values, (3, 3))

    np.testing.assert_allclose(resized[0, 0, 0], values[0, 0, 0])
    np.testing.assert_allclose(resized[0, -1, -1], values[0, -1, -1])
    np.testing.assert_allclose(resized[0, 1, 1], np.mean(values[0], axis=(0, 1)))


def test_prediction_uncertainty_reports_overlap_scope() -> None:
    truth_points = np.ones((3, 2, 2, 3))
    prediction_points = truth_points.copy()
    prediction_points[..., 0] += 0.1
    mask = np.ones((3, 2, 2), dtype=bool)
    covariance = np.broadcast_to(np.eye(3) * 0.01, truth_points.shape + (3,)).copy()
    contributors = np.ones_like(mask, dtype=np.uint16)
    contributors[1] = 2
    prediction = FusedSequence(np.arange(3), prediction_points, mask, covariance, contributors)
    truth = TruthSequence(np.arange(3), truth_points, mask)

    result = evaluate_prediction_uncertainty(
        prediction,
        truth,
        maximum_points=100,
        seed=1,
    )

    assert result["scopes"]["all"]["count"] == 12
    assert result["scopes"]["overlap"]["count"] == 4


def test_method_bootstrap_uses_sequence_level_values() -> None:
    results = {
        "first": {"ci": {"scopes": {"all": {"selective_gain_80": 0.1}}}},
        "second": {"ci": {"scopes": {"all": {"selective_gain_80": 0.3}}}},
    }

    summary = _bootstrap_method_metric(
        results,
        "ci",
        "all",
        "selective_gain_80",
        seed=4,
    )

    assert summary["mean"] == 0.2
    assert summary["bootstrap_95_interval"] == [0.1, 0.3]
    assert summary["sequences"] == 2
