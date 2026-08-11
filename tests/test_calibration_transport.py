from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from prob4d.calibration_transport import (
    CALIBRATION_TRANSPORT_CLAIM_BOUNDARY,
    CALIBRATION_TRANSPORT_EVIDENCE_SCHEMA,
    CALIBRATION_TRANSPORT_MODEL_SCHEMA,
    CALIBRATION_TRANSPORT_VERSION,
    CalibrationTransportEvidenceV1,
    CalibrationTransportModelV1,
    CalibrationTransportPolicyV1,
    CalibrationTransportUnitV1,
    calibration_transport_feature_contract_id,
    evaluate_calibration_transport,
    fit_calibration_transport_model,
    load_calibration_transport_evidence,
    load_calibration_transport_model,
    save_calibration_transport_evidence,
    save_calibration_transport_model,
)

FEATURE_NAMES = ("overlap_disagreement", "relative_variance")
FEATURE_CONTRACT = calibration_transport_feature_contract_id(
    FEATURE_NAMES,
    semantics="source-only-reliability-features-v1",
    configuration={"window_size": 25, "uses_target_truth": False},
)


def policy(
    *,
    maximum_unsupported_group_fraction: float = 0.0,
    maximum_unsupported_row_fraction: float = 0.0,
) -> CalibrationTransportPolicyV1:
    return CalibrationTransportPolicyV1(
        quantile_levels=(0.1, 0.5, 0.9),
        miscoverage_rate=0.2,
        minimum_source_units=6,
        neighbor_count=1,
        maximum_unsupported_group_fraction=maximum_unsupported_group_fraction,
        maximum_unsupported_row_fraction=maximum_unsupported_row_fraction,
        absolute_scale_floor=1e-6,
        relative_scale_floor=1e-6,
    )


def unit(
    unit_id: str,
    center: tuple[float, float],
    *,
    seed: int,
    row_count: int = 101,
    scale: float = 0.08,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    feature_contract_id: str = FEATURE_CONTRACT,
) -> CalibrationTransportUnitV1:
    generator = np.random.default_rng(seed)
    values = generator.normal(center, scale, size=(row_count, len(feature_names)))
    return CalibrationTransportUnitV1(
        unit_id=unit_id,
        feature_contract_id=feature_contract_id,
        feature_names=feature_names,
        feature_values=values,
        metadata={"seed": seed, "source_only": True},
    )


def source_units() -> tuple[CalibrationTransportUnitV1, ...]:
    return tuple(
        unit(
            f"source-{index:02d}",
            (-0.15 + 0.05 * index, 0.1 - 0.03 * index),
            seed=100 + index,
        )
        for index in range(8)
    )


def fitted_model(
    *,
    selected_policy: CalibrationTransportPolicyV1 | None = None,
) -> CalibrationTransportModelV1:
    return fit_calibration_transport_model(
        source_units(),
        policy=policy() if selected_policy is None else selected_policy,
        metadata={"cohort": "source-objects-v1"},
    )


def test_feature_contract_is_deterministic_and_configuration_bound() -> None:
    repeated = calibration_transport_feature_contract_id(
        FEATURE_NAMES,
        semantics="source-only-reliability-features-v1",
        configuration={"uses_target_truth": False, "window_size": 25},
    )
    changed = calibration_transport_feature_contract_id(
        FEATURE_NAMES,
        semantics="source-only-reliability-features-v1",
        configuration={"uses_target_truth": False, "window_size": 26},
    )

    assert repeated == FEATURE_CONTRACT
    assert changed != FEATURE_CONTRACT
    assert len(FEATURE_CONTRACT) == 64


def test_in_support_target_is_accepted_and_shift_is_rejected() -> None:
    model = fitted_model()
    supported = unit("target-supported", (0.02, -0.01), seed=501)
    shifted = unit("target-shifted", (4.0, -0.01), seed=502)

    supported_evidence = evaluate_calibration_transport(model, [supported])
    shifted_evidence = evaluate_calibration_transport(model, [shifted])

    assert supported_evidence.accepted
    assert supported_evidence.group_results[0].supported
    assert supported_evidence.group_results[0].support_margin >= 0.0
    assert not shifted_evidence.accepted
    assert not shifted_evidence.group_results[0].supported
    assert shifted_evidence.group_results[0].support_margin < 0.0
    assert shifted_evidence.worst_feature_name == "overlap_disagreement"
    assert shifted_evidence.decision_reasons == (
        "unsupported-group-fraction",
        "unsupported-row-fraction",
    )


def test_constant_source_dimension_fails_closed_on_target_shift() -> None:
    sources = []
    for index in range(8):
        values = np.column_stack(
            (
                np.ones(80),
                np.linspace(-0.1, 0.1, 80) + 0.01 * index,
            )
        )
        sources.append(
            CalibrationTransportUnitV1(
                unit_id=f"constant-source-{index}",
                feature_contract_id=FEATURE_CONTRACT,
                feature_names=FEATURE_NAMES,
                feature_values=values,
            )
        )
    model = fit_calibration_transport_model(sources, policy=policy())
    target = CalibrationTransportUnitV1(
        unit_id="constant-target",
        feature_contract_id=FEATURE_CONTRACT,
        feature_names=FEATURE_NAMES,
        feature_values=np.column_stack((np.full(80, 1.01), np.linspace(-0.1, 0.1, 80))),
    )

    evidence = evaluate_calibration_transport(model, [target])

    assert not evidence.accepted
    assert evidence.group_results[0].feature_outside_range_max[
        "overlap_disagreement"
    ] > 100.0


def test_complete_unit_and_row_fraction_gates_are_conjunctive() -> None:
    permissive_model = fitted_model(
        selected_policy=policy(
            maximum_unsupported_group_fraction=0.5,
            maximum_unsupported_row_fraction=0.02,
        )
    )
    strict_model = fitted_model(
        selected_policy=policy(
            maximum_unsupported_group_fraction=0.49,
            maximum_unsupported_row_fraction=0.02,
        )
    )
    supported = unit("target-large-supported", (0.02, -0.01), seed=601, row_count=100)
    shifted = unit("target-small-shifted", (4.0, 0.0), seed=602, row_count=1)

    permissive = evaluate_calibration_transport(permissive_model, [shifted, supported])
    strict = evaluate_calibration_transport(strict_model, [supported, shifted])

    assert permissive.unsupported_group_fraction == pytest.approx(0.5)
    assert permissive.unsupported_row_fraction == pytest.approx(1.0 / 101.0)
    assert permissive.accepted
    assert not strict.accepted
    assert strict.decision_reasons == ("unsupported-group-fraction",)


def test_source_and_row_order_do_not_change_model_identity() -> None:
    original_units = source_units()
    reordered_units = []
    for item in reversed(original_units):
        reordered_units.append(
            CalibrationTransportUnitV1(
                unit_id=item.unit_id,
                feature_contract_id=item.feature_contract_id,
                feature_names=item.feature_names,
                feature_values=item.feature_values[::-1],
                metadata=item.metadata,
            )
        )

    original = fit_calibration_transport_model(original_units, policy=policy())
    reordered = fit_calibration_transport_model(reordered_units, policy=policy())

    assert original.model_id == reordered.model_id
    assert original.to_dict() == reordered.to_dict()


def test_target_order_does_not_change_evidence_identity() -> None:
    model = fitted_model()
    first = unit("target-a", (0.0, 0.0), seed=701)
    second = unit("target-b", (0.05, -0.02), seed=702)

    left = evaluate_calibration_transport(model, [first, second])
    right = evaluate_calibration_transport(model, [second, first])

    assert left.evidence_id == right.evidence_id
    assert left.to_dict() == right.to_dict()


def test_model_arrays_and_metadata_are_defensively_immutable() -> None:
    sources = list(source_units())
    model = fit_calibration_transport_model(sources, policy=policy())
    original = model.source_embeddings.copy()
    sources[0].feature_values.setflags(write=True)
    sources[0].feature_values[...] = 99.0

    np.testing.assert_array_equal(model.source_embeddings, original)
    for array in (
        model.source_embeddings,
        model.embedding_center,
        model.embedding_scale,
        model.source_nonconformity_scores,
    ):
        assert not array.flags.writeable
    with pytest.raises(TypeError):
        model.metadata["changed"] = True  # type: ignore[index]


def test_model_and_evidence_round_trip_with_no_clobber(tmp_path) -> None:
    model = fitted_model()
    evidence = evaluate_calibration_transport(
        model,
        [unit("target-roundtrip", (0.0, 0.0), seed=801)],
        metadata={"target_prefix_only": True},
    )
    model_path = tmp_path / "model.json"
    evidence_path = tmp_path / "evidence.json"

    save_calibration_transport_model(model, model_path)
    save_calibration_transport_evidence(evidence, evidence_path)
    loaded_model = load_calibration_transport_model(model_path)
    loaded_evidence = load_calibration_transport_evidence(
        evidence_path,
        model=loaded_model,
    )

    assert loaded_model.to_dict() == model.to_dict()
    assert loaded_evidence.to_dict() == evidence.to_dict()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_calibration_transport_model(model, model_path)
    save_calibration_transport_model(model, model_path, overwrite=True)


def test_tampered_derived_model_and_evidence_fields_are_rejected() -> None:
    model = fitted_model()
    evidence = evaluate_calibration_transport(
        model,
        [unit("target-tamper", (0.0, 0.0), seed=901)],
    )
    model_record = model.to_dict()
    model_record["support_threshold"] = float(model.support_threshold) + 1.0
    evidence_record = evidence.to_dict()
    evidence_record["accepted"] = not evidence.accepted

    with pytest.raises(ValueError, match="derived fields changed"):
        CalibrationTransportModelV1.from_dict(model_record)
    with pytest.raises(ValueError, match="derived fields changed"):
        CalibrationTransportEvidenceV1.from_dict(evidence_record, model=model)


def test_duplicate_json_keys_fail_closed(tmp_path) -> None:
    model = fitted_model()
    path = tmp_path / "duplicate.json"
    record = json.dumps(model.to_dict(), sort_keys=True)
    path.write_text(record[:-1] + ',"schema":"forged"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key 'schema'"):
        load_calibration_transport_model(path)


def test_feature_contract_names_and_unit_identity_boundaries_fail_closed() -> None:
    model = fitted_model()
    different_contract = "f" * 64
    target_contract = unit(
        "target-contract",
        (0.0, 0.0),
        seed=1001,
        feature_contract_id=different_contract,
    )
    target_names = unit(
        "target-names",
        (0.0, 0.0),
        seed=1002,
        feature_names=("relative_variance", "overlap_disagreement"),
    )
    overlapping = unit("source-00", (0.0, 0.0), seed=1003)

    with pytest.raises(ValueError, match="feature contract"):
        evaluate_calibration_transport(model, [target_contract])
    with pytest.raises(ValueError, match="feature names"):
        evaluate_calibration_transport(model, [target_names])
    with pytest.raises(ValueError, match="overlap source unit IDs"):
        evaluate_calibration_transport(model, [overlapping])


def test_insufficient_or_duplicate_source_units_fail_closed() -> None:
    sources = source_units()
    with pytest.raises(ValueError, match="below policy.minimum_source_units"):
        fit_calibration_transport_model(sources[:5], policy=policy())
    with pytest.raises(ValueError, match="source unit IDs must be unique"):
        fit_calibration_transport_model([*sources, sources[0]], policy=policy())


def test_policy_rejects_coercive_or_unidentified_configuration() -> None:
    with pytest.raises(ValueError, match="minimum_source_units"):
        replace(policy(), minimum_source_units=True)
    with pytest.raises(ValueError, match="at least one scale floor"):
        replace(policy(), absolute_scale_floor=0.0, relative_scale_floor=0.0)
    with pytest.raises(ValueError, match="neighbor_count"):
        replace(policy(), neighbor_count=6)
    with pytest.raises(TypeError, match="quantile_levels"):
        replace(policy(), quantile_levels=[0.1, 0.5, 0.9])  # type: ignore[arg-type]


def test_unit_rejects_coercion_nonfinite_values_and_mutable_name_form() -> None:
    values = np.ones((3, 2))
    with pytest.raises(TypeError, match="feature_names"):
        CalibrationTransportUnitV1(
            unit_id="bad-names",
            feature_contract_id=FEATURE_CONTRACT,
            feature_names=list(FEATURE_NAMES),  # type: ignore[arg-type]
            feature_values=values,
        )
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="feature_values must be finite"):
        CalibrationTransportUnitV1(
            unit_id="bad-values",
            feature_contract_id=FEATURE_CONTRACT,
            feature_names=FEATURE_NAMES,
            feature_values=values,
        )


def test_nonalphabetical_feature_names_are_supported() -> None:
    names = ("z_feature", "a_feature")
    contract = calibration_transport_feature_contract_id(
        names,
        semantics="source-only-test-v1",
        configuration={},
    )
    sources = tuple(
        CalibrationTransportUnitV1(
            unit_id=f"ordered-source-{index}",
            feature_contract_id=contract,
            feature_names=names,
            feature_values=np.column_stack(
                (
                    np.linspace(0.0, 1.0, 40) + 0.01 * index,
                    np.linspace(-1.0, 0.0, 40) - 0.01 * index,
                )
            ),
        )
        for index in range(6)
    )
    model = fit_calibration_transport_model(sources, policy=policy())
    target = CalibrationTransportUnitV1(
        unit_id="ordered-target",
        feature_contract_id=contract,
        feature_names=names,
        feature_values=np.column_stack(
            (np.linspace(0.0, 1.0, 40), np.linspace(-1.0, 0.0, 40))
        ),
    )

    evidence = evaluate_calibration_transport(model, [target])

    assert tuple(evidence.group_results[0].feature_distance_rms) == (
        "a_feature",
        "z_feature",
    )
    assert evidence.accepted


def test_records_state_schema_version_and_scientific_boundary() -> None:
    model = fitted_model()
    evidence = evaluate_calibration_transport(
        model,
        [unit("target-boundary", (0.0, 0.0), seed=1101)],
    )

    model_record = model.to_dict()
    evidence_record = evidence.to_dict()
    assert model_record["schema"] == CALIBRATION_TRANSPORT_MODEL_SCHEMA
    assert evidence_record["schema"] == CALIBRATION_TRANSPORT_EVIDENCE_SCHEMA
    assert model_record["schema_version"] == CALIBRATION_TRANSPORT_VERSION
    assert evidence_record["schema_version"] == CALIBRATION_TRANSPORT_VERSION
    assert model_record["claim_boundary"] == CALIBRATION_TRANSPORT_CLAIM_BOUNDARY
    assert evidence_record["claim_boundary"] == CALIBRATION_TRANSPORT_CLAIM_BOUNDARY
    assert "target truth" in CALIBRATION_TRANSPORT_CLAIM_BOUNDARY
    assert "BayesianPhysTwin" in CALIBRATION_TRANSPORT_CLAIM_BOUNDARY
