from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "science"
    / "run_tracking_cloth_continuous_calibrated_so2_v1.py"
)


def module():
    spec = importlib.util.spec_from_file_location("continuous_cloth_study", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_minimal_axis_rotation_maps_source_to_target() -> None:
    study = module()
    rng = np.random.default_rng(3)
    for _ in range(100):
        source = rng.normal(size=3)
        target = rng.normal(size=3)
        source /= np.linalg.norm(source)
        target /= np.linalg.norm(target)
        rotation = study._rotation_align(source, target)
        np.testing.assert_allclose(rotation @ source, target, atol=1e-11)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-11)
        assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_constant_angular_velocity_predictor_is_exact_on_constructed_motion() -> None:
    study = module()
    axis = np.array([0.0, 0.0, 1.0])
    anchor_a = np.zeros(3)
    anchor_b = axis
    delta = 0.2

    def point(angle: float) -> np.ndarray:
        return np.array([math.cos(angle), math.sin(angle), 0.4])

    earlier = np.stack((anchor_a, anchor_b, point(0.0)))
    previous = np.stack((anchor_a, anchor_b, point(delta)))
    current = np.stack((anchor_a, anchor_b, point(2.0 * delta)))
    representative, center, predicted_axis = study._predict_probe(
        earlier,
        previous,
        current,
        "constructed",
    )
    np.testing.assert_allclose(representative, current[2], atol=1e-12)
    np.testing.assert_allclose(center, [0.0, 0.0, 0.4], atol=1e-12)
    np.testing.assert_allclose(predicted_axis, axis, atol=1e-12)


def test_triplet_selection_uses_only_prefix_and_finds_off_axis_probe() -> None:
    study = module()
    frames = 20
    positions = np.zeros((frames, 4, 3))
    positions[:, 0] = [0.0, 0.0, 0.0]
    positions[:, 1] = [0.0, 0.0, 100.0]
    positions[:, 2] = [40.0, 0.0, 50.0]
    positions[:, 3] = [2.0, 0.0, 50.0]
    indices, details = study._select_triplet(
        positions,
        ("a", "b", "probe", "near"),
        prefix_frames=12,
        minimum_anchor_distance_mm=20.0,
        minimum_probe_radius_mm=5.0,
    )
    assert indices == (0, 1, 2)
    assert details["probe_radius_q10_mm"] == pytest.approx(40.0)


def test_calibrated_policy_accepts_small_orbit_and_rejects_full_circle() -> None:
    study = module()
    case = study.Case(
        angle_radians=0.05,
        orbit_residual_mm=0.0,
        radial_scale_mm=10.0,
        normalized_score=0.05 / (math.pi / 3.0),
        representative_mm=np.array([10.0, 0.0, 0.0]),
        truth_mm=np.array(
            [10.0 * math.cos(0.05), 10.0 * math.sin(0.05), 0.0]
        ),
        axis_center_mm=np.zeros(3),
        origin_mm=np.zeros(3),
        axis=np.array([0.0, 0.0, 1.0]),
        gauge_id="case",
    )
    result = study._evaluate_cases(
        [case],
        threshold=0.1,
        angle_normalizer_rad=math.pi / 3.0,
        grid_sizes=[8],
    )
    assert result["calibrated_acceptance"] == 1.0
    assert result["full_circle_acceptance"] == 0.0
    assert result["support_coverage"] == 1.0
    assert result["continuous_query_interval_coverage"] == 1.0
    assert result["exact_fallback_fraction"] == 1.0


def test_hash_fold_is_deterministic_and_size_bound() -> None:
    study = module()
    first = study._fold("path/file.csv", "A2", 5, "salt")
    assert first == study._fold("path/file.csv", "A2", 5, "salt")
    assert 0 <= first < 5
    assert first != study._fold("path/file.csv", "A3", 5, "salt")
