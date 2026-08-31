#!/usr/bin/env python3
"""Independently verify the frozen DOT R04--R10 held-out result contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RESULT_SCHEMA = "prob4d.dot-rope-cut3r-heldout-confirmation"
FAILURE_SCHEMA = "prob4d.dot-rope-cut3r-heldout-failure"
SUPPORT_SCHEMA = "prob4d.dot-rope-cut3r-heldout-support"
SCHEMA_VERSION = 1
CONFIRMATION_SEQUENCES = [f"R{index:02d}" for index in range(4, 11)]
RESERVED_SEQUENCES = "R11-R70"
SELECTED_ALPHA = 0.85
SELECTED_METHOD = "dependence_alpha_0850"
SOURCE_CALIBRATION_ID = "943339ac864fda04cc59081bc81a605576b3c90bf0aa996aea00b00335cfc0c7"
COMPARATORS = ["pointwise_quadratic", "shared_quadratic_curvature", "local_first_order"]
SCIENTIFIC_DECISIONS = {
    "heldout-strong-positive",
    "heldout-directional-positive",
    "heldout-mixed-or-negative",
    "heldout-support-negative",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--marker-support", type=Path)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_hex(value: object, *, name: str, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must have {length} lowercase hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must have {length} lowercase hexadecimal characters")
    return value


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _verify_content_id(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    identifier = unsigned.pop(field, None)
    identifier = _require_hex(identifier, name=field)
    if _content_id(unsigned) != identifier:
        raise ValueError(f"{field} does not match the canonical payload")
    return identifier


def _method_by_sequence(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, float]:
    selected = [row for row in rows if row.get("method") == method]
    by_sequence = {
        str(row["sequence"]): _finite(
            row["normalized_nll_per_dimension"],
            name=f"{method} normalized NLL",
        )
        for row in selected
    }
    if set(by_sequence) != set(CONFIRMATION_SEQUENCES) or len(selected) != len(
        CONFIRMATION_SEQUENCES
    ):
        raise ValueError(f"{method} does not have one row per confirmation sequence")
    return by_sequence


def _expected_classification(comparisons: Mapping[str, Mapping[str, Any]]) -> str:
    if set(comparisons) != set(COMPARATORS):
        raise ValueError("held-out comparison roster changed")
    values = [comparisons[name] for name in COMPARATORS]
    means_negative = all(
        _finite(value["mean_selected_minus_comparator"], name="paired mean") < 0.0
        for value in values
    )
    pointwise = comparisons["pointwise_quadratic"]
    shared = comparisons["shared_quadratic_curvature"]
    if (
        means_negative
        and _finite(pointwise["upper_95"], name="pointwise upper bound") < 0.0
        and _finite(shared["upper_95"], name="shared upper bound") < 0.0
    ):
        return "heldout-strong-positive"
    if means_negative:
        return "heldout-directional-positive"
    return "heldout-mixed-or-negative"


def _verify_comparisons(
    rows: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
) -> str:
    if statistics.get("independent_unit") != "complete_sequence":
        raise ValueError("held-out independent unit changed")
    if statistics.get("primary_metric") != "normalized_joint_gaussian_nll_per_dimension":
        raise ValueError("held-out primary metric changed")
    comparisons = statistics.get("comparisons")
    if not isinstance(comparisons, dict):
        raise ValueError("held-out comparisons are unavailable")
    selected = _method_by_sequence(rows, SELECTED_METHOD)
    for comparator in COMPARATORS:
        comparison = comparisons.get(comparator)
        if not isinstance(comparison, dict):
            raise ValueError(f"held-out comparison is missing for {comparator}")
        reference = _method_by_sequence(rows, comparator)
        differences = {
            sequence: selected[sequence] - reference[sequence]
            for sequence in CONFIRMATION_SEQUENCES
        }
        recorded = comparison.get("per_sequence")
        if not isinstance(recorded, dict) or set(recorded) != set(CONFIRMATION_SEQUENCES):
            raise ValueError(f"per-sequence comparison roster changed for {comparator}")
        for sequence, difference in differences.items():
            if not math.isclose(
                _finite(recorded[sequence], name="recorded per-sequence difference"),
                difference,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"per-sequence difference changed for {comparator}/{sequence}")
        mean = sum(differences.values()) / len(differences)
        if not math.isclose(
            _finite(comparison["mean_selected_minus_comparator"], name="recorded paired mean"),
            mean,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"paired mean changed for {comparator}")
        wins = sum(value < 0.0 for value in differences.values())
        if comparison.get("sequence_wins") != wins:
            raise ValueError(f"sequence-win count changed for {comparator}")
        if comparison.get("sequence_count") != len(CONFIRMATION_SEQUENCES):
            raise ValueError(f"sequence count changed for {comparator}")
        lower = _finite(comparison["lower_95"], name="lower bootstrap bound")
        upper = _finite(comparison["upper_95"], name="upper bootstrap bound")
        if lower > upper:
            raise ValueError(f"bootstrap interval is reversed for {comparator}")
    expected = _expected_classification(comparisons)
    if statistics.get("classification") != expected:
        raise ValueError("held-out classification does not follow the frozen decision rule")
    return expected


def _verify_aggregate(rows: Sequence[Mapping[str, Any]], aggregate: object) -> None:
    if not isinstance(aggregate, list):
        raise ValueError("aggregate method rows are unavailable")
    aggregate_by_method = {
        str(row["method"]): row for row in aggregate if isinstance(row, dict) and "method" in row
    }
    for method in [SELECTED_METHOD, *COMPARATORS]:
        by_sequence = _method_by_sequence(rows, method)
        expected = sum(by_sequence.values()) / len(by_sequence)
        row = aggregate_by_method.get(method)
        if row is None:
            raise ValueError(f"aggregate row is missing for {method}")
        measured = _finite(row["mean_normalized_nll_per_dimension"], name="aggregate NLL")
        if not math.isclose(measured, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"aggregate NLL changed for {method}")
        if row.get("sequence_count") != len(CONFIRMATION_SEQUENCES):
            raise ValueError(f"aggregate sequence count changed for {method}")


def _verify_support(value: Mapping[str, Any]) -> str:
    if value.get("schema") != SUPPORT_SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported held-out marker-support schema")
    support_id = _verify_content_id(value, "support_id")
    boundary = value.get("information_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("marker-support information boundary is missing")
    if boundary.get("opened_sequences") != CONFIRMATION_SEQUENCES:
        raise ValueError("marker-support opened sequence roster changed")
    if boundary.get("reserved_sequences") != RESERVED_SEQUENCES:
        raise ValueError("marker-support reserved sequence boundary changed")
    if boundary.get("sealed_confirmation_provider_predictions_opened") is not True:
        raise ValueError("marker support did not consume a sealed provider bundle")
    if boundary.get("confirmation_2d_markers_opened_after_provider_seal") is not True:
        raise ValueError("2-D marker custody changed")
    if boundary.get("confirmation_3d_markers_opened_after_provider_seal") is not True:
        raise ValueError("3-D marker custody changed")
    return support_id


def verify(value: Mapping[str, Any], marker_support: Mapping[str, Any] | None) -> dict[str, Any]:
    schema = value.get("schema")
    if schema == RESULT_SCHEMA:
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported held-out result schema version")
        evaluation_id = _verify_content_id(value, "evaluation_id")
        decision = value.get("decision")
        if decision not in SCIENTIFIC_DECISIONS - {"heldout-support-negative"}:
            raise ValueError("completed held-out result has an unsupported decision")
        if _finite(value.get("selected_dependence_alpha"), name="selected alpha") != SELECTED_ALPHA:
            raise ValueError("held-out selected alpha changed")
        if value.get("selected_dependence_method") != SELECTED_METHOD:
            raise ValueError("held-out selected method changed")
        source = value.get("source_calibration")
        if not isinstance(source, dict) or source.get("calibration_id") != SOURCE_CALIBRATION_ID:
            raise ValueError("held-out source calibration binding changed")
        boundary = value.get("information_boundary")
        if not isinstance(boundary, dict):
            raise ValueError("held-out information boundary is missing")
        if boundary.get("opened_sequences") != CONFIRMATION_SEQUENCES:
            raise ValueError("held-out opened sequence roster changed")
        if boundary.get("reserved_sequences") != RESERVED_SEQUENCES:
            raise ValueError("held-out reserved sequence boundary changed")
        if boundary.get("source_alpha_refit") is not False:
            raise ValueError("confirmation-side alpha refit was performed")
        if boundary.get("provider_means_changed") is not False:
            raise ValueError("confirmation-side provider means changed")
        if boundary.get("markers_opened_only_after_provider_seal") is not True:
            raise ValueError("held-out marker custody changed")
        rows = value.get("method_rows")
        if not isinstance(rows, list):
            raise ValueError("held-out method rows are unavailable")
        expected = _verify_comparisons(rows, value.get("heldout_statistics") or {})
        if decision != expected:
            raise ValueError("held-out terminal decision changed")
        _verify_aggregate(rows, value.get("aggregate_methods"))
        support_id = _require_hex(value.get("marker_support_id"), name="marker_support_id")
        if marker_support is not None and _verify_support(marker_support) != support_id:
            raise ValueError("held-out result references another marker-support artifact")
        return {
            "decision": decision,
            "result_id": evaluation_id,
            "selected_alpha": SELECTED_ALPHA,
            "confirmation_sequence_count": len(CONFIRMATION_SEQUENCES),
        }
    if schema == FAILURE_SCHEMA:
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported held-out failure schema version")
        result_id = _verify_content_id(value, "result_id")
        decision = value.get("decision")
        if decision not in {"heldout-support-negative", "technical-failure"}:
            raise ValueError("held-out failure has an unsupported decision")
        if _finite(value.get("selected_dependence_alpha"), name="failure alpha") != SELECTED_ALPHA:
            raise ValueError("held-out failure alpha changed")
        if value.get("source_calibration_id") != SOURCE_CALIBRATION_ID:
            raise ValueError("held-out failure source calibration changed")
        support_id = _require_hex(value.get("marker_support_id"), name="marker_support_id")
        if marker_support is not None and _verify_support(marker_support) != support_id:
            raise ValueError("held-out failure references another marker-support artifact")
        return {
            "decision": decision,
            "result_id": result_id,
            "selected_alpha": SELECTED_ALPHA,
            "confirmation_sequence_count": len(CONFIRMATION_SEQUENCES),
        }
    raise ValueError("unsupported held-out result schema")


def main() -> int:
    args = _parser().parse_args()
    value = _read_json(args.result)
    support = _read_json(args.marker_support) if args.marker_support is not None else None
    output = verify(value, support)
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
