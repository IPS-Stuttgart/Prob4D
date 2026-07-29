import json

import numpy as np
import pytest

from prob4d.data import PredictionWindow
from prob4d.source_reliability import (
    SourceReliabilityModelV1,
    build_source_reliability_features,
    fit_group_balanced_source_reliability,
    load_source_reliability_model,
    save_source_reliability_model,
)
from prob4d.uncertainty import DisagreementEvidence, StructuredCovariance


def feature_window() -> PredictionWindow:
    point_map = np.zeros((3, 2, 2, 3), dtype=np.float64)
    point_map[..., 2] = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.5, 2.5], [3.5, 4.5]],
            [[2.0, 3.0], [4.0, 5.0]],
        ]
    )
    valid = np.ones((3, 2, 2), dtype=bool)
    valid[1, 1, 1] = False
    flow = np.zeros_like(point_map)
    flow[..., 0] = 0.1
    deform = valid.copy()
    return PredictionWindow(
        window_id="features",
        frame_indices=np.array([10, 11, 12]),
        point_map=point_map,
        valid_mask=valid,
        scene_flow=flow,
        deform_mask=deform,
    )


def test_source_feature_builder_uses_only_source_side_inputs() -> None:
    window = feature_window()
    rays = window.rays()
    covariance = StructuredCovariance(
        ray_directions=rays,
        parallel_variance=np.full(window.shape, 0.04),
        lateral_variance=np.full(window.shape, 0.01),
    )
    parallel = np.zeros(window.shape)
    lateral = np.zeros(window.shape)
    count = np.zeros(window.shape)
    parallel[0, 0, 0] = 0.08
    lateral[0, 0, 0] = 0.01
    count[0, 0, 0] = 1.0
    evidence = DisagreementEvidence(parallel, lateral, count)

    features = build_source_reliability_features(window, covariance, evidence)

    assert features.values.shape == window.shape + (7,)
    assert features.feature_names[0] == "has_overlap"
    assert features.values[0, 0, 0, 0] == 1.0
    assert features.values[0, 0, 1, 0] == 0.0
    assert features.values[0, 0, 0, 1] > 0.0
    assert features.values[0, 0, 0, 2] == 1.0
    assert features.values[1, 0, 0, 2] == 0.0
    assert features.values[2, 0, 0, 2] == 1.0
    assert features.metadata["uses_truth"] is False
    assert features.metadata["uses_downstream_physical_innovation"] is False
    assert features.valid_mask[1, 1, 1] == 0


def test_group_balance_prevents_dense_sequence_domination() -> None:
    features = np.zeros((404, 1), dtype=np.float64)
    labels = np.r_[np.ones(400), np.zeros(4)]
    groups = np.asarray(["large"] * 400 + ["small"] * 4)

    model = fit_group_balanced_source_reliability(
        features,
        labels,
        groups,
        feature_names=("constant",),
        ridge=1e-3,
        label_definition="source point error below frozen threshold",
        group_definition="sequence",
    )

    probability = float(model.predict(np.zeros((1, 1)))[0])
    np.testing.assert_allclose(probability, 0.5, atol=1e-12)
    np.testing.assert_allclose(
        model.report.group_balanced_nominal_fraction,
        0.5,
        atol=1e-12,
    )
    assert model.report.group_count == 2
    assert model.report.converged is True


def test_logistic_model_learns_source_feature_ranking() -> None:
    generator = np.random.default_rng(42)
    feature = np.linspace(-3.0, 3.0, 300)
    latent = feature + generator.normal(scale=0.7, size=len(feature))
    labels = (latent > 0.0).astype(np.float64)
    groups = np.asarray([f"sequence-{index // 30}" for index in range(len(feature))])

    model = fit_group_balanced_source_reliability(
        feature[:, None],
        labels,
        groups,
        feature_names=("source_score",),
        ridge=1e-2,
        label_definition="source-only nominal correspondence label",
        group_definition="sequence",
    )
    predicted = model.predict(np.array([[-2.0], [0.0], [2.0]]))

    assert predicted[0] < predicted[1] < predicted[2]
    assert predicted[0] < 0.2
    assert predicted[2] > 0.8
    assert model.report.weighted_brier_score < 0.15


def test_fit_and_artifact_are_invariant_to_row_order() -> None:
    generator = np.random.default_rng(7)
    features = generator.normal(size=(120, 3))
    labels = (features[:, 0] - 0.5 * features[:, 1] > 0.0).astype(float)
    groups = np.asarray([f"group-{index // 20}" for index in range(len(features))])
    arguments = {
        "feature_names": ("first", "second", "third"),
        "ridge": 0.05,
        "label_definition": "source-only binary nominality",
        "group_definition": "object session",
        "metadata": {"split": "calibration-only"},
    }
    first = fit_group_balanced_source_reliability(
        features,
        labels,
        groups,
        **arguments,
    )
    permutation = generator.permutation(len(features))
    second = fit_group_balanced_source_reliability(
        features[permutation],
        labels[permutation],
        groups[permutation],
        **arguments,
    )

    assert first.artifact_id == second.artifact_id
    assert first.to_dict() == second.to_dict()
    np.testing.assert_array_equal(first.coefficients, second.coefficients)


def test_source_reliability_artifact_round_trip_and_tamper_rejection(
    tmp_path,
) -> None:
    features = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    groups = np.array(["a", "a", "b", "b"])
    model = fit_group_balanced_source_reliability(
        features,
        labels,
        groups,
        feature_names=("score",),
        label_definition="source-only nominality",
        group_definition="sequence",
    )
    path = tmp_path / "source-reliability.json"
    save_source_reliability_model(model, path)

    loaded = load_source_reliability_model(path)
    assert loaded.artifact_id == model.artifact_id
    np.testing.assert_array_equal(loaded.coefficients, model.coefficients)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["coefficients"][0] += 0.1
    with pytest.raises(ValueError, match="artifact_id"):
        SourceReliabilityModelV1.from_dict(payload)


def test_feature_contract_masks_invalid_rows_during_prediction() -> None:
    window = feature_window()
    covariance = StructuredCovariance(
        ray_directions=window.rays(),
        parallel_variance=np.full(window.shape, 0.04),
        lateral_variance=np.full(window.shape, 0.01),
    )
    features = build_source_reliability_features(window, covariance)
    active_values = features.flattened()
    labels = (active_values[:, 2] > 0.5).astype(float)
    groups = np.asarray(
        [f"frame-{index // 4}" for index in range(len(active_values))]
    )
    model = fit_group_balanced_source_reliability(
        active_values,
        labels,
        groups,
        feature_names=features.feature_names,
        label_definition="source temporal edge proximity above frozen threshold",
        group_definition="frame",
    )

    probability = model.predict(features)
    assert probability.shape == window.shape
    assert probability[1, 1, 1] == 0.0
    assert np.all(probability[features.valid_mask] > 0.0)


def test_source_reliability_fit_requires_both_label_classes() -> None:
    with pytest.raises(ValueError, match="both label classes"):
        fit_group_balanced_source_reliability(
            np.zeros((4, 1)),
            np.ones(4),
            np.asarray(["a", "a", "b", "b"]),
            feature_names=("constant",),
            label_definition="nominality",
            group_definition="sequence",
        )
