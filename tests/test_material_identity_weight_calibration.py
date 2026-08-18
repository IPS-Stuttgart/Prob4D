from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import prob4d.material_identity_weight_calibration as calibration_module
from prob4d.material_identity_mixture import (
    LocalTrackEndpoint,
    MaterialIdentityCandidateV1,
)
from prob4d.material_identity_weight_calibration import (
    MaterialIdentityCalibrationExampleV1,
    calibrated_mixture_from_config,
    calibration_from_config,
    fit_material_identity_weight_calibration,
    load_material_identity_weight_calibration,
    write_material_identity_weight_calibration,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_ids(group: int, example: int) -> tuple[str, str]:
    return _sha(f"g{group}-e{example}-null"), _sha(f"g{group}-e{example}-linked")


def _examples(*, reverse: bool = False) -> tuple[MaterialIdentityCalibrationExampleV1, ...]:
    result: list[MaterialIdentityCalibrationExampleV1] = []
    for group in range(6):
        for example, score in enumerate((0.1, 0.3, 0.7, 0.9)):
            null_id, linked_id = _candidate_ids(group, example)
            candidate_ids = (linked_id, null_id) if reverse else (null_id, linked_id)
            candidate_kinds = ("linked", "null") if reverse else ("null", "linked")
            features = (
                np.array([[score, 1.0 - score], [0.0, 0.0]])
                if reverse
                else np.array([[0.0, 0.0], [score, 1.0 - score]])
            )
            result.append(
                MaterialIdentityCalibrationExampleV1(
                    example_id=f"g{group}-e{example}",
                    group_id=f"group-{group}",
                    candidate_ids=candidate_ids,
                    candidate_kinds=candidate_kinds,
                    features=features,
                    true_candidate_id=linked_id if score > 0.5 else null_id,
                    metadata={"source_only": True},
                )
            )
    if reverse:
        result.reverse()
    return tuple(result)


def _fit(examples=None):
    return fit_material_identity_weight_calibration(
        _examples() if examples is None else examples,
        feature_names=("source_score", "inverse_score"),
        feature_schema_id="a" * 64,
        association_rule_id="b" * 64,
        tracklet_producer_revision="c" * 40,
        association_revision="d" * 40,
        label_definition="source material-label identity",
        group_definition="complete physical object or acquisition session",
        cross_fit_fold_count=3,
        ridge=0.05,
        metadata={"uses_target_outcomes": False},
    )


def test_group_cross_fitted_model_improves_over_uniform() -> None:
    model = _fit()

    assert model.report.final_fit_converged
    assert model.report.cross_fitted_top1_accuracy == pytest.approx(1.0)
    assert model.report.log_loss_advantage_vs_uniform > 0.3
    assert model.report.cross_fitted_mean_true_probability > 0.7
    assert model.report.group_count == 6
    assert model.calibration_group_ids == tuple(f"group-{index}" for index in range(6))


def test_training_order_and_candidate_order_do_not_change_identity() -> None:
    first = _fit(_examples())
    second = _fit(_examples(reverse=True))

    assert first.artifact_id == second.artifact_id
    np.testing.assert_array_equal(first.feature_center, second.feature_center)
    np.testing.assert_array_equal(first.feature_scale, second.feature_scale)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    assert first.null_bias == second.null_bias


def test_group_balanced_report_does_not_count_examples_as_groups() -> None:
    examples = list(_examples())
    retained = [item for item in examples if item.group_id == "group-0"]
    retained.extend(
        item
        for item in examples
        if item.group_id == "group-1" and item.true_kind == "linked"
    )
    retained.extend(item for item in examples if item.group_id in {"group-2", "group-3"})
    model = _fit(tuple(retained))

    expected = np.mean(
        [
            np.mean([item.true_kind == "null" for item in retained if item.group_id == group_id])
            for group_id in model.calibration_group_ids
        ]
    )
    assert model.report.observed_null_fraction == pytest.approx(expected)
    assert model.report.group_count == len(model.calibration_group_ids)


def test_round_trip_is_strict_and_no_clobber(tmp_path: Path) -> None:
    model = _fit()
    path = tmp_path / "identity-calibration.json"
    write_material_identity_weight_calibration(path, model)

    restored = load_material_identity_weight_calibration(path)
    assert restored.artifact_id == model.artifact_id
    np.testing.assert_array_equal(restored.coefficients, model.coefficients)
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError):
        write_material_identity_weight_calibration(path, model)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["null_bias"] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_id mismatch"):
        load_material_identity_weight_calibration(path)


def test_loader_rejects_duplicate_keys_and_coercive_vectors(tmp_path: Path) -> None:
    model = _fit()
    path = tmp_path / "identity-calibration.json"
    write_material_identity_weight_calibration(path, model)

    duplicate = path.read_text(encoding="utf-8").replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_material_identity_weight_calibration(path)

    write_material_identity_weight_calibration(path, model, overwrite=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["coefficients"][0] = "1.0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="real number"):
        load_material_identity_weight_calibration(path)


def _mixture_config(*, reverse: bool = False) -> dict[str, object]:
    null_candidate = MaterialIdentityCandidateV1(
        source_endpoint=None,
        association_result_id=None,
        source_score=None,
        calibrated_log_weight=0.0,
    )
    linked_candidate = MaterialIdentityCandidateV1(
        source_endpoint=LocalTrackEndpoint("window-1", 7),
        association_result_id="e" * 64,
        source_score=0.9,
        calibrated_log_weight=0.0,
    )
    candidates: list[dict[str, object]] = [
        {
            "source_endpoint": None,
            "association_result_id": None,
            "source_score": None,
            "features": [0.0, 0.0],
            "metadata": {"candidate_id": null_candidate.candidate_id(
                target_endpoint=LocalTrackEndpoint("window-2", 3)
            )},
        },
        {
            "source_endpoint": {"window_id": "window-1", "track_id": 7},
            "association_result_id": "e" * 64,
            "source_score": 0.9,
            "features": [0.9, 0.1],
            "metadata": {"candidate_id": linked_candidate.candidate_id(
                target_endpoint=LocalTrackEndpoint("window-2", 3)
            )},
        },
    ]
    if reverse:
        candidates.reverse()
    return {
        "target_endpoint": {"window_id": "window-2", "track_id": 3},
        "window_order": ["window-0", "window-1", "window-2"],
        "causal_frame_stop": 75,
        "association_rule_id": "b" * 64,
        "feature_schema_id": "a" * 64,
        "tracklet_producer_revision": "c" * 40,
        "association_revision": "d" * 40,
        "feature_names": ["source_score", "inverse_score"],
        "candidates": candidates,
        "metadata": {"claim_bearing": False},
    }


def test_calibrated_mixture_uses_model_identity_and_is_order_invariant() -> None:
    model = _fit()
    first = calibrated_mixture_from_config(model, _mixture_config())
    second = calibrated_mixture_from_config(model, _mixture_config(reverse=True))

    assert first.mixture_id == second.mixture_id
    assert first.calibration_id == model.artifact_id
    assert first.metadata["material_identity_weight_calibration_id"] == model.artifact_id
    assert first.probabilities[1] > first.probabilities[0]
    assert first.candidates[0].kind == "null"


def test_calibrated_mixture_rejects_compatibility_mismatch() -> None:
    model = _fit()
    config = _mixture_config()
    config["feature_schema_id"] = "f" * 64
    with pytest.raises(ValueError, match="feature_schema_id"):
        calibrated_mixture_from_config(model, config)

    config = _mixture_config()
    config["association_revision"] = "f" * 40
    with pytest.raises(ValueError, match="association revision"):
        calibrated_mixture_from_config(model, config)


def test_raw_configuration_rejects_target_outcomes_and_bad_truth() -> None:
    example = _examples()[0].to_dict()
    config = {
        "feature_names": ["source_score", "inverse_score"],
        "feature_schema_id": "a" * 64,
        "association_rule_id": "b" * 64,
        "tracklet_producer_revision": "c" * 40,
        "association_revision": "d" * 40,
        "label_definition": "source material-label identity",
        "group_definition": "complete object/session",
        "cross_fit_fold_count": 2,
        "ridge": 0.05,
        "maximum_iterations": 100,
        "convergence_tolerance": 1e-10,
        "examples": [example, _examples()[4].to_dict()],
        "metadata": {},
        "uses_target_outcomes": True,
    }
    with pytest.raises(ValueError, match="may not use target outcomes"):
        calibration_from_config(config)

    with pytest.raises(ValueError, match="must occur"):
        MaterialIdentityCalibrationExampleV1(
            example_id="bad",
            group_id="group",
            candidate_ids=("1" * 64, "2" * 64),
            candidate_kinds=("null", "linked"),
            features=np.zeros((2, 1)),
            true_candidate_id="3" * 64,
        )


def test_conditional_logit_derivatives_match_finite_differences() -> None:
    examples = _examples()[:8]
    weights = calibration_module._example_weights(examples)
    center, scale = calibration_module._feature_location_scale(examples, weights)
    parameters = np.array([0.2, -0.3, 0.4])
    objective, gradient, hessian = calibration_module._objective_gradient_hessian(
        examples,
        weights,
        center,
        scale,
        parameters,
        0.05,
    )
    assert np.isfinite(objective)

    epsilon = 1e-6
    numerical_gradient = np.empty_like(gradient)
    numerical_hessian = np.empty_like(hessian)
    for index in range(len(parameters)):
        offset = np.zeros_like(parameters)
        offset[index] = epsilon
        plus = calibration_module._objective_gradient_hessian(
            examples, weights, center, scale, parameters + offset, 0.05
        )
        minus = calibration_module._objective_gradient_hessian(
            examples, weights, center, scale, parameters - offset, 0.05
        )
        numerical_gradient[index] = (plus[0] - minus[0]) / (2.0 * epsilon)
        numerical_hessian[:, index] = (plus[1] - minus[1]) / (2.0 * epsilon)

    np.testing.assert_allclose(gradient, numerical_gradient, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(hessian, numerical_hessian, atol=2e-7, rtol=2e-6)


def test_null_only_application_preserves_exact_reference() -> None:
    model = _fit()
    config = _mixture_config()
    config["candidates"] = [config["candidates"][0]]
    mixture = calibrated_mixture_from_config(model, config)
    assert mixture.null_probability == 1.0
    np.testing.assert_array_equal(mixture.probabilities, [1.0])


def test_calibration_data_id_binds_complete_canonical_examples() -> None:
    first = _fit(_examples())
    second = _fit(_examples(reverse=True))
    assert first.calibration_data_id == second.calibration_data_id

    changed = list(_examples())
    original = changed[0]
    modified = original.features.copy()
    modified[0, 0] += 0.01
    changed[0] = MaterialIdentityCalibrationExampleV1(
        example_id=original.example_id,
        group_id=original.group_id,
        candidate_ids=original.candidate_ids,
        candidate_kinds=original.candidate_kinds,
        features=modified,
        true_candidate_id=original.true_candidate_id,
        metadata=original.metadata,
    )
    third = _fit(tuple(changed))
    assert third.calibration_data_id != first.calibration_data_id


def test_cross_fit_requires_both_outcomes_in_every_training_partition() -> None:
    examples: list[MaterialIdentityCalibrationExampleV1] = []
    for group, true_kind in (("group-null", "null"), ("group-linked", "linked")):
        null_id = _sha(f"{group}-null")
        linked_id = _sha(f"{group}-linked")
        examples.append(
            MaterialIdentityCalibrationExampleV1(
                example_id=f"{group}-example",
                group_id=group,
                candidate_ids=(null_id, linked_id),
                candidate_kinds=("null", "linked"),
                features=np.array([[0.0, 0.0], [1.0, 0.0]]),
                true_candidate_id=null_id if true_kind == "null" else linked_id,
            )
        )
    with pytest.raises(ValueError, match="training fold must contain"):
        fit_material_identity_weight_calibration(
            tuple(examples),
            feature_names=("source_score", "inverse_score"),
            feature_schema_id="a" * 64,
            association_rule_id="b" * 64,
            tracklet_producer_revision="c" * 40,
            association_revision="d" * 40,
            label_definition="source truth",
            group_definition="complete object/session",
            cross_fit_fold_count=2,
            ridge=0.05,
        )


def test_report_rejects_inconsistent_derived_metrics() -> None:
    model = _fit()
    payload = model.to_dict()
    payload["report"]["log_loss_advantage_vs_uniform"] += 0.1
    payload["artifact_id"] = model.artifact_id
    with pytest.raises(ValueError, match="inconsistent"):
        type(model).from_dict(payload)


def test_exact_64_character_revisions_are_supported() -> None:
    model = fit_material_identity_weight_calibration(
        _examples(),
        feature_names=("source_score", "inverse_score"),
        feature_schema_id="a" * 64,
        association_rule_id="b" * 64,
        tracklet_producer_revision="c" * 64,
        association_revision="d" * 64,
        label_definition="source truth",
        group_definition="complete object/session",
        cross_fit_fold_count=3,
        ridge=0.05,
    )
    assert len(model.tracklet_producer_revision) == 64
    assert len(model.association_revision) == 64


def test_numerical_nonconvergence_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="optimization did not converge"):
        fit_material_identity_weight_calibration(
            _examples(),
            feature_names=("source_score", "inverse_score"),
            feature_schema_id="a" * 64,
            association_rule_id="b" * 64,
            tracklet_producer_revision="c" * 40,
            association_revision="d" * 40,
            label_definition="source truth",
            group_definition="complete object/session",
            cross_fit_fold_count=3,
            ridge=0.05,
            maximum_iterations=1,
            convergence_tolerance=1e-16,
        )
