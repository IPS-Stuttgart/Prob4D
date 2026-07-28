import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import prob4d.provider_v1 as provider_v1
from prob4d.gauge import GaugeEstimate
from prob4d.observation_factors import (
    OBSERVATION_FACTOR_SCHEMA_VERSION,
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
    causal_frame_stop: int = 9,
    reliability: np.ndarray | None = None,
    association: np.ndarray | None = None,
    nominal_probability: float = 0.8,
    composite_weight: float = 0.5,
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
        association_probability=(
            np.asarray([0.9, 0.6]) if association is None else association
        ),
        prior_reliability=(
            np.asarray([0.7, 0.4]) if reliability is None else reliability
        ),
        prior_nominal_probability=nominal_probability,
        composite_weight=composite_weight,
        correlation_group_id=group_id,
        causal_frame_stop=causal_frame_stop,
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
    joint_gauge_covariance = np.zeros((14, 14), dtype=np.float64)
    joint_gauge_covariance[:7, :7] = gauges[0].covariance
    joint_gauge_covariance[7:, 7:] = gauges[1].covariance
    joint_gauge_covariance[0, 7] = 1e-3
    joint_gauge_covariance[7, 0] = 1e-3
    return ObservationFactorBundle(
        sequence_id="sequence-a",
        factors=(
            _factor("factor-0", gauge_id="window-0"),
            _factor(
                "factor-1",
                gauge_id="window-1",
                view_id="camera-1",
                group_id="backbone-window-1",
                nominal_probability=0.6,
                composite_weight=0.25,
            ),
        ),
        gauges=gauges,
        source_revision="0123456789abcdef",
        causal_frame_stop=9,
        case_id="double-stretch-sloth",
        stream_id="prob4d:camera-points",
        metadata={"producer": "unit-test", "metric": True},
        joint_gauge_covariance=joint_gauge_covariance,
        gauge_covariance_semantics="joint-cross-window",
    )


def test_bundle_roundtrip_preserves_unfused_provenance(tmp_path: Path) -> None:
    bundle = _bundle()
    manifest, payload = write_observation_factor_bundle(
        bundle, tmp_path / "factors.json"
    )

    loaded = load_observation_factor_bundle(manifest)
    record = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload.is_file()
    assert record["schema_version"] == 4
    assert record["gauge_covariance"] == {
        "cross_window_covariance_preserved": True,
        "diagonal_blocks_match_gauge_marginals": True,
        "joint_covariance_key": "joint_gauge_covariance",
        "ordered_gauge_ids": ["window-0", "window-1"],
        "semantics": "joint-cross-window",
    }
    assert record["causal_frame_stop"] == 9
    assert record["causal_frame_stop_convention"] == "exclusive"
    assert loaded.sequence_id == bundle.sequence_id
    assert loaded.source_revision == bundle.source_revision
    assert loaded.case_id == "double-stretch-sloth"
    assert loaded.stream_id == "prob4d:camera-points"
    assert loaded.source_repository == "FlorianPfaff/Prob4D"
    assert loaded.causal_frame_stop == 9
    assert loaded.causal_frame_limit == 8
    assert loaded.gauge_covariance_semantics == "joint-cross-window"
    assert loaded.cross_window_gauge_covariance_preserved
    np.testing.assert_allclose(
        loaded.joint_gauge_covariance,
        bundle.joint_gauge_covariance,
    )
    assert [factor.factor_id for factor in loaded.factors] == ["factor-0", "factor-1"]
    np.testing.assert_allclose(loaded.factors[0].prior_reliability, [0.7, 0.4])
    assert loaded.factors[0].prior_nominal_probability == pytest.approx(0.8)
    assert loaded.factors[1].composite_weight == pytest.approx(0.25)
    assert loaded.correlation_group_parameters == {
        "backbone-window-0": {
            "prior_nominal_probability": 0.8,
            "composite_weight": 0.5,
        },
        "backbone-window-1": {
            "prior_nominal_probability": 0.6,
            "composite_weight": 0.25,
        },
    }


def test_linearized_covariance_includes_sim3_gauge_uncertainty() -> None:
    factor = _bundle().linearize("factor-0")

    np.testing.assert_allclose(factor.world_mean_m[0], [1.0, 0.0, 0.0])
    assert factor.gauge_jacobian[0, 0, 0] == pytest.approx(1.0)
    assert factor.conditional_world_covariance_m2[0, 0, 0] == pytest.approx(0.01)
    assert factor.marginal_world_covariance_m2[0, 0, 0] == pytest.approx(0.05)
    np.testing.assert_allclose(factor.prior_reliability, [0.7, 0.4])
    assert factor.prior_nominal_probability == pytest.approx(0.8)
    assert factor.composite_weight == pytest.approx(0.5)
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


def test_stacked_factors_keep_association_reliability_and_group_weights_separate() -> None:
    stacked = _bundle().stack()

    assert stacked.world_mean_m.shape == (4, 3)
    assert stacked.gauge_jacobian.shape == (4, 3, 14)
    np.testing.assert_allclose(stacked.association_probability, [0.9, 0.6, 0.9, 0.6])
    np.testing.assert_allclose(stacked.prior_reliability, [0.7, 0.4, 0.7, 0.4])
    np.testing.assert_allclose(stacked.prior_nominal_probability, [0.8, 0.8, 0.6, 0.6])
    np.testing.assert_allclose(stacked.composite_weight, [0.5, 0.5, 0.25, 0.25])
    assert stacked.causal_frame_stop == 9
    assert stacked.causal_frame_limit == 8
    assert stacked.gauge_prior_covariance[0, 7] == pytest.approx(1e-3)
    assert stacked.correlation_group_ids == (
        "backbone-window-0",
        "backbone-window-0",
        "backbone-window-1",
        "backbone-window-1",
    )


def test_stacked_joint_gauge_prior_preserves_cross_factor_covariance() -> None:
    stacked = _bundle().stack()

    cross_covariance = (
        stacked.gauge_jacobian[0]
        @ stacked.gauge_prior_covariance
        @ stacked.gauge_jacobian[2].T
    )

    assert cross_covariance[0, 0] == pytest.approx(1e-3)
    assert np.linalg.norm(cross_covariance) == pytest.approx(1e-3)


def test_zero_reliability_row_is_not_stacked_even_with_high_association() -> None:
    factor = _factor(
        "factor-0",
        gauge_id="window-0",
        association=np.ones(2),
        reliability=np.asarray([1.0, 0.0]),
    )
    bundle = ObservationFactorBundle(
        sequence_id="sequence",
        factors=(factor,),
        gauges=(GaugeEstimate("window-0", Sim3.identity(), np.eye(7) * 1e-4),),
        source_revision="revision",
        causal_frame_stop=9,
    )

    stacked = bundle.stack()

    assert len(stacked.world_mean_m) == 1
    np.testing.assert_allclose(stacked.association_probability, [1.0])
    np.testing.assert_allclose(stacked.prior_reliability, [1.0])


def test_factor_rejects_frame_at_exclusive_causal_stop() -> None:
    with pytest.raises(ValueError, match="exclusive causal frame stop"):
        _factor(
            "factor-0",
            gauge_id="window-0",
            frame_index=9,
            causal_frame_stop=9,
        )


def test_bundle_rejects_factor_with_different_causal_stop() -> None:
    with pytest.raises(ValueError, match="causal frame stops differ"):
        ObservationFactorBundle(
            sequence_id="sequence-a",
            factors=(
                _factor(
                    "factor-0",
                    gauge_id="window-0",
                    causal_frame_stop=10,
                ),
            ),
            gauges=(
                GaugeEstimate("window-0", Sim3.identity(), np.eye(7) * 1e-4),
            ),
            source_revision="revision",
            causal_frame_stop=9,
        )


def test_bundle_rejects_inconsistent_group_parameters() -> None:
    with pytest.raises(ValueError, match="one correlation group"):
        ObservationFactorBundle(
            sequence_id="sequence-a",
            factors=(
                _factor("factor-0", gauge_id="window-0", group_id="shared"),
                _factor(
                    "factor-1",
                    gauge_id="window-0",
                    group_id="shared",
                    composite_weight=0.25,
                ),
            ),
            gauges=(
                GaugeEstimate("window-0", Sim3.identity(), np.eye(7) * 1e-4),
            ),
            source_revision="revision",
            causal_frame_stop=9,
        )


def test_bundle_rejects_joint_covariance_with_mismatched_marginal_block() -> None:
    bundle = _bundle()
    invalid = bundle.joint_gauge_covariance.copy()
    invalid[0, 0] += 0.01

    with pytest.raises(ValueError, match="diagonal blocks must match"):
        ObservationFactorBundle(
            sequence_id=bundle.sequence_id,
            factors=bundle.factors,
            gauges=bundle.gauges,
            source_revision=bundle.source_revision,
            causal_frame_stop=bundle.causal_frame_stop,
            joint_gauge_covariance=invalid,
            gauge_covariance_semantics="joint-cross-window",
        )


def test_bundle_rejects_cross_window_covariance_under_marginal_semantics() -> None:
    bundle = _bundle()

    with pytest.raises(ValueError, match="zero cross-window"):
        ObservationFactorBundle(
            sequence_id=bundle.sequence_id,
            factors=bundle.factors,
            gauges=bundle.gauges,
            source_revision=bundle.source_revision,
            causal_frame_stop=bundle.causal_frame_stop,
            joint_gauge_covariance=bundle.joint_gauge_covariance,
            gauge_covariance_semantics="marginal-blocks-only",
        )


def test_schema_v3_loader_upgrades_to_explicit_marginal_only_semantics(
    tmp_path: Path,
) -> None:
    manifest, _ = write_observation_factor_bundle(
        _bundle(), tmp_path / "factors.json"
    )
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["schema_version"] = 3
    record.pop("gauge_covariance")
    manifest.write_text(json.dumps(record), encoding="utf-8")

    loaded = load_observation_factor_bundle(manifest)

    assert loaded.schema_version == OBSERVATION_FACTOR_SCHEMA_VERSION
    assert loaded.metadata["loaded_from_schema_version"] == 3
    assert loaded.gauge_covariance_semantics == "marginal-blocks-only"
    assert not loaded.cross_window_gauge_covariance_preserved
    assert loaded.joint_gauge_covariance[0, 7] == 0.0


def test_schema_v4_loader_rejects_changed_joint_gauge_order(tmp_path: Path) -> None:
    manifest, _ = write_observation_factor_bundle(
        _bundle(), tmp_path / "factors.json"
    )
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["gauge_covariance"]["ordered_gauge_ids"] = ["window-1", "window-0"]
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="order differs"):
        load_observation_factor_bundle(manifest)


def test_provider_v1_writer_remains_frozen_at_schema_v3(tmp_path: Path) -> None:
    legacy_bundle = replace(
        _bundle(),
        joint_gauge_covariance=None,
        gauge_covariance_semantics="marginal-blocks-only",
    )

    manifest, _ = provider_v1.write_observation_factor_bundle(
        legacy_bundle,
        tmp_path / "legacy-factors.json",
    )
    record = json.loads(manifest.read_text(encoding="utf-8"))

    assert provider_v1.OBSERVATION_FACTOR_SCHEMA_VERSION == 3
    assert record["schema_version"] == 3
    assert "gauge_covariance" not in record
    loaded = provider_v1.load_observation_factor_bundle(manifest)
    assert loaded.schema_version == 3
    assert loaded.gauge_covariance_semantics == "marginal-blocks-only"


def test_provider_v1_loader_rejects_schema_v4_joint_covariance(
    tmp_path: Path,
) -> None:
    manifest, _ = write_observation_factor_bundle(
        _bundle(), tmp_path / "factors.json"
    )

    with pytest.raises(ValueError, match="use prob4d.provider_v2"):
        provider_v1.load_observation_factor_bundle(manifest)


def test_schema_v2_loader_upgrades_inclusive_limit_and_missing_reliability(
    tmp_path: Path,
) -> None:
    manifest, _ = write_observation_factor_bundle(
        _bundle(), tmp_path / "factors.json"
    )
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["schema_version"] = 2
    record.pop("gauge_covariance")
    record["causal_frame_limit"] = record.pop("causal_frame_stop") - 1
    record.pop("causal_frame_stop_convention")
    for factor in record["factors"]:
        factor["causal_frame_limit"] = factor.pop("causal_frame_stop") - 1
        factor.pop("prior_nominal_probability")
        factor.pop("composite_weight")
        factor["arrays"].pop("prior_reliability")
    manifest.write_text(json.dumps(record), encoding="utf-8")

    loaded = load_observation_factor_bundle(manifest)

    assert loaded.schema_version == OBSERVATION_FACTOR_SCHEMA_VERSION
    assert loaded.causal_frame_stop == 9
    assert loaded.metadata["loaded_from_schema_version"] == 2
    np.testing.assert_array_equal(loaded.factors[0].prior_reliability, np.ones(2))
    assert loaded.factors[0].prior_nominal_probability == 1.0
    assert loaded.factors[0].composite_weight == 1.0


def test_bundle_loader_rejects_changed_payload(tmp_path: Path) -> None:
    manifest, payload = write_observation_factor_bundle(
        _bundle(), tmp_path / "factors.json"
    )
    payload.write_bytes(payload.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_observation_factor_bundle(manifest)
