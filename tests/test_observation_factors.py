from pathlib import Path

import numpy as np
import pytest

from prob4d.gauge import GaugeEstimate
from prob4d.observation_factors import (
    ObservationFactor,
    ObservationFactorBundle,
    load_observation_factor_bundle,
    sim3_point_jacobian,
    write_observation_factor_bundle,
)
from prob4d.sim3 import Sim3


def _factor(
    factor_id: str,
    *,
    gauge_id: str,
    view_id: str = "camera-0",
    group_id: str = "backbone-window-0",
    frame_index: int = 4,
    causal_frame_limit: int = 8,
) -> ObservationFactor:
    return ObservationFactor(
        factor_id=factor_id,
        frame_index=frame_index,
        view_id=view_id,
        window_id=gauge_id,
        gauge_id=gauge_id,
        point_ids=np.asarray([11, 12]),
        points_local_m=np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        valid_mask=np.asarray([True, True]),
        local_covariance_m2=np.tile(np.eye(3) * 0.01, (2, 1, 1)),
        association_probability=np.asarray([0.9, 0.6]),
        correlation_group_id=group_id,
        causal_frame_limit=causal_frame_limit,
        ray_directions_local=np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
    )


def _bundle() -> ObservationFactorBundle:
    gauges = (
        GaugeEstimate("window-0", Sim3.identity(), np.diag([0.04] + [0.0] * 6)),
        GaugeEstimate(
            "window-1",
            Sim3.from_vector(np.asarray([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])),
            np.eye(7) * 1e-4,
        ),
    )
    return ObservationFactorBundle(
        sequence_id="sequence-a",
        factors=(
            _factor("factor-0", gauge_id="window-0"),
            _factor(
                "factor-1",
                gauge_id="window-1",
                view_id="camera-1",
                group_id="backbone-window-1",
            ),
        ),
        gauges=gauges,
        source_revision="0123456789abcdef",
        causal_frame_limit=8,
        metadata={"producer": "unit-test", "metric": True},
    )


def test_bundle_roundtrip_preserves_unfused_provenance(tmp_path: Path) -> None:
    bundle = _bundle()
    manifest, payload = write_observation_factor_bundle(
        bundle, tmp_path / "factors.json"
    )

    loaded = load_observation_factor_bundle(manifest)

    assert payload.is_file()
    assert loaded.sequence_id == bundle.sequence_id
    assert loaded.source_revision == bundle.source_revision
    assert [factor.factor_id for factor in loaded.factors] == ["factor-0", "factor-1"]
    assert [factor.view_id for factor in loaded.factors] == ["camera-0", "camera-1"]
    assert loaded.correlation_group_counts == {
        "backbone-window-0": 1,
        "backbone-window-1": 1,
    }
    np.testing.assert_array_equal(loaded.factors[0].point_ids, [11, 12])
    np.testing.assert_allclose(loaded.gauges[1].global_from_local.translation, [0.5, 0, 0])


def test_linearized_covariance_includes_sim3_gauge_uncertainty() -> None:
    bundle = _bundle()

    factor = bundle.linearize("factor-0")

    np.testing.assert_allclose(factor.world_mean_m[0], [1.0, 0.0, 0.0])
    assert factor.gauge_jacobian[0, 0, 0] == pytest.approx(1.0)
    assert factor.gauge_jacobian[0, 1, 0] == pytest.approx(0.0)
    assert factor.conditional_world_covariance_m2[0, 0, 0] == pytest.approx(0.01)
    assert factor.marginal_world_covariance_m2[0, 0, 0] == pytest.approx(0.05)
    assert factor.marginal_world_covariance_m2[0, 1, 1] == pytest.approx(0.01)
    np.testing.assert_allclose(factor.ray_directions_world[1], [0.0, 1.0, 0.0])


def test_sim3_rotation_jacobian_matches_finite_difference() -> None:
    transform = Sim3.from_vector(
        np.asarray([0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.4])
    )
    point = np.asarray([[0.4, -0.2, 0.7]])
    analytic = sim3_point_jacobian(transform, point)[0]
    vector = transform.as_vector()
    baseline = transform.transform_points(point)[0]
    numerical = np.empty((3, 7))
    for parameter in range(7):
        step = 1e-7
        perturbed = vector.copy()
        perturbed[parameter] += step
        numerical[:, parameter] = (
            Sim3.from_vector(perturbed).transform_points(point)[0] - baseline
        ) / step

    np.testing.assert_allclose(analytic, numerical, atol=2e-6, rtol=2e-5)


def test_stacked_factors_retain_separate_gauge_blocks() -> None:
    stacked = _bundle().stack()

    assert stacked.world_mean_m.shape == (4, 3)
    assert stacked.gauge_jacobian.shape == (4, 3, 14)
    np.testing.assert_allclose(
        stacked.conditional_world_covariance_m2[0], np.eye(3) * 0.01
    )
    assert stacked.marginal_world_covariance_m2[0, 0, 0] == pytest.approx(0.05)
    assert stacked.gauge_prior_covariance.shape == (14, 14)
    assert np.any(stacked.gauge_jacobian[:2, :, :7])
    np.testing.assert_array_equal(stacked.gauge_jacobian[:2, :, 7:], 0.0)
    np.testing.assert_array_equal(stacked.gauge_jacobian[2:, :, :7], 0.0)
    assert np.any(stacked.gauge_jacobian[2:, :, 7:])
    assert stacked.correlation_group_ids == (
        "backbone-window-0",
        "backbone-window-0",
        "backbone-window-1",
        "backbone-window-1",
    )
    np.testing.assert_allclose(
        stacked.gauge_prior_covariance[:7, :7],
        np.diag([0.04] + [0.0] * 6),
    )


def test_bundle_rejects_factor_with_different_causal_limit() -> None:
    with pytest.raises(ValueError, match="causal frame limits differ"):
        ObservationFactorBundle(
            sequence_id="sequence-a",
            factors=(
                _factor(
                    "factor-0",
                    gauge_id="window-0",
                    causal_frame_limit=9,
                ),
            ),
            gauges=(
                GaugeEstimate("window-0", Sim3.identity(), np.eye(7) * 1e-4),
            ),
            source_revision="revision",
            causal_frame_limit=8,
        )


def test_bundle_loader_rejects_changed_payload(tmp_path: Path) -> None:
    manifest, payload = write_observation_factor_bundle(
        _bundle(), tmp_path / "factors.json"
    )
    payload.write_bytes(payload.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_observation_factor_bundle(manifest)
