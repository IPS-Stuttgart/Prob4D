from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "verify_dot_rope_cut3r_heldout_result.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("dot_heldout_result_verifier", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scientific_fixture(module):
    rows = []
    selected_values = {}
    for index, sequence in enumerate(module.CONFIRMATION_SEQUENCES):
        selected = 0.5 + 0.02 * index
        selected_values[sequence] = selected
        for method, offset in (
            (module.SELECTED_METHOD, 0.0),
            ("pointwise_quadratic", 0.4),
            ("shared_quadratic_curvature", 0.8),
            ("local_first_order", 1.2),
        ):
            rows.append(
                {
                    "sequence": sequence,
                    "method": method,
                    "normalized_nll_per_dimension": selected + offset,
                }
            )
    comparisons = {}
    for comparator, offset in (
        ("pointwise_quadratic", 0.4),
        ("shared_quadratic_curvature", 0.8),
        ("local_first_order", 1.2),
    ):
        comparisons[comparator] = {
            "comparator": comparator,
            "mean_selected_minus_comparator": -offset,
            "lower_95": -offset - 0.05,
            "upper_95": -offset + 0.05,
            "sequence_wins": 7,
            "sequence_count": 7,
            "per_sequence": {
                sequence: -offset for sequence in module.CONFIRMATION_SEQUENCES
            },
        }
    aggregate = []
    for method in [module.SELECTED_METHOD, *module.COMPARATORS]:
        values = [
            float(row["normalized_nll_per_dimension"])
            for row in rows
            if row["method"] == method
        ]
        aggregate.append(
            {
                "method": method,
                "sequence_count": 7,
                "mean_normalized_nll_per_dimension": sum(values) / len(values),
            }
        )
    result = {
        "schema": module.RESULT_SCHEMA,
        "schema_version": 1,
        "decision": "heldout-strong-positive",
        "selected_dependence_alpha": 0.85,
        "selected_dependence_method": module.SELECTED_METHOD,
        "source_calibration": {"calibration_id": module.SOURCE_CALIBRATION_ID},
        "information_boundary": {
            "opened_sequences": module.CONFIRMATION_SEQUENCES,
            "reserved_sequences": "R11-R70",
            "source_alpha_refit": False,
            "provider_means_changed": False,
            "markers_opened_only_after_provider_seal": True,
        },
        "method_rows": rows,
        "aggregate_methods": aggregate,
        "heldout_statistics": {
            "independent_unit": "complete_sequence",
            "primary_metric": "normalized_joint_gaussian_nll_per_dimension",
            "comparisons": comparisons,
            "classification": "heldout-strong-positive",
        },
        "marker_support_id": "a" * 64,
    }
    result["evaluation_id"] = module._content_id(result)
    return result


def test_verifier_accepts_frozen_strong_positive_fixture() -> None:
    module = _load_script()
    result = _scientific_fixture(module)

    verified = module.verify(result, None)

    assert verified["decision"] == "heldout-strong-positive"
    assert verified["selected_alpha"] == 0.85
    assert verified["confirmation_sequence_count"] == 7


def test_verifier_rejects_posthoc_alpha_change() -> None:
    module = _load_script()
    result = _scientific_fixture(module)
    result["selected_dependence_alpha"] = 0.8
    unsigned = dict(result)
    unsigned.pop("evaluation_id")
    result["evaluation_id"] = module._content_id(unsigned)

    with pytest.raises(ValueError, match="selected alpha changed"):
        module.verify(result, None)


def test_verifier_recomputes_per_sequence_difference() -> None:
    module = _load_script()
    result = _scientific_fixture(module)
    result["heldout_statistics"]["comparisons"]["pointwise_quadratic"]["per_sequence"][
        "R04"
    ] = -0.3
    unsigned = dict(result)
    unsigned.pop("evaluation_id")
    result["evaluation_id"] = module._content_id(unsigned)

    with pytest.raises(ValueError, match="per-sequence difference changed"):
        module.verify(result, None)


def test_verifier_recomputes_terminal_classification() -> None:
    module = _load_script()
    result = _scientific_fixture(module)
    result["decision"] = "heldout-directional-positive"
    unsigned = dict(result)
    unsigned.pop("evaluation_id")
    result["evaluation_id"] = module._content_id(unsigned)

    with pytest.raises(ValueError, match="terminal decision changed"):
        module.verify(result, None)


def test_verifier_accepts_content_addressed_support_negative() -> None:
    module = _load_script()
    failure = {
        "schema": module.FAILURE_SCHEMA,
        "schema_version": 1,
        "decision": "heldout-support-negative",
        "selected_dependence_alpha": 0.85,
        "source_calibration_id": module.SOURCE_CALIBRATION_ID,
        "marker_support_id": "b" * 64,
    }
    failure["result_id"] = module._content_id(failure)

    verified = module.verify(failure, None)

    assert verified["decision"] == "heldout-support-negative"


def test_verifier_rejects_wrong_confirmation_roster() -> None:
    module = _load_script()
    result = _scientific_fixture(module)
    result["information_boundary"]["opened_sequences"] = module.CONFIRMATION_SEQUENCES[:-1]
    unsigned = dict(result)
    unsigned.pop("evaluation_id")
    result["evaluation_id"] = module._content_id(unsigned)

    with pytest.raises(ValueError, match="opened sequence roster changed"):
        module.verify(result, None)
