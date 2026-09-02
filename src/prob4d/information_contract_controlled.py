"""Strict evaluator for the seven-system information-contract conformance suite.

This module intentionally evaluates a different artifact from the public sealed
challenge/submission protocol. The controlled suite compares complete toy systems
and cross-system failure cases; it is anti-gaming development evidence, not a
provider submission or public-data result.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ._atomic_file import atomic_write_text

CONTROLLED_SCHEMA: Final = "prob4d.information-contract-benchmark-suite"
CONTROLLED_VERSION: Final = 1
RESULT_SCHEMA: Final = "prob4d.information-contract-controlled-result"
RESULT_VERSION: Final = 1
CLAIM_BOUNDARY: Final = (
    "Deterministic anti-gaming and conformance evidence only. The suite opens no "
    "public dataset, invokes no learned provider, and establishes no calibration, "
    "physical correctness, state of the art, or deployment safety."
)
FloatArray = NDArray[np.float64]


def _object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: object, *, name: str, allow_empty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a nonempty list"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, unpadded string")
    return value


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _array(value: object, *, name: str, ndim: int) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return result


def _exact_fields(value: Mapping[str, Any], allowed: set[str], *, name: str) -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError(f"{name} contains unregistered fields: {sorted(unknown)}")


def _mean(records: Sequence[Mapping[str, float]], field: str) -> float:
    return float(np.mean([float(record[field]) for record in records]))


def _prediction_metrics(cases_value: object, *, system_id: str) -> dict[str, Any]:
    cases = _list(cases_value, name=f"{system_id}.prediction_cases")
    by_unit: dict[str, list[dict[str, float]]] = defaultdict(list)
    case_records: list[dict[str, Any]] = []
    for index, raw in enumerate(cases):
        case = _object(raw, name=f"{system_id}.prediction_cases[{index}]")
        _exact_fields(
            case,
            {
                "case_id",
                "statistical_unit_id",
                "truth",
                "mean",
                "covariance",
                "coverage_radius_squared",
            },
            name=f"{system_id}.prediction_cases[{index}]",
        )
        case_id = _string(case.get("case_id"), name="prediction case_id")
        unit = _string(
            case.get("statistical_unit_id"), name="prediction statistical_unit_id"
        )
        truth = _array(case.get("truth"), name=f"{case_id}.truth", ndim=1)
        mean = _array(case.get("mean"), name=f"{case_id}.mean", ndim=1)
        covariance = _array(
            case.get("covariance"), name=f"{case_id}.covariance", ndim=2
        )
        if truth.size < 1 or mean.shape != truth.shape or covariance.shape != (
            truth.size,
            truth.size,
        ):
            raise ValueError(f"{case_id}: incompatible prediction shapes")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise ValueError(f"{case_id}: covariance must be symmetric")
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{case_id}: covariance must be positive definite") from error
        radius = _number(
            case.get("coverage_radius_squared"),
            name=f"{case_id}.coverage_radius_squared",
            minimum=0.0,
        )
        error = truth - mean
        whitened = np.linalg.solve(cholesky, error)
        mahalanobis = float(whitened @ whitened)
        logdet = float(2.0 * np.log(np.diag(cholesky)).sum())
        dimension = int(truth.size)
        nll = 0.5 * (
            dimension * math.log(2.0 * math.pi) + logdet + mahalanobis
        )
        record = {
            "case_id": case_id,
            "statistical_unit_id": unit,
            "dimension": dimension,
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "joint_gaussian_nll": nll,
            "normalized_nees": mahalanobis / dimension,
            "registered_coverage": float(mahalanobis <= radius),
        }
        case_records.append(record)
        by_unit[unit].append(record)
    unit_records = []
    for unit, records in sorted(by_unit.items()):
        unit_records.append(
            {
                "statistical_unit_id": unit,
                "rmse": _mean(records, "rmse"),
                "joint_gaussian_nll": _mean(records, "joint_gaussian_nll"),
                "normalized_nees": _mean(records, "normalized_nees"),
                "registered_coverage": _mean(records, "registered_coverage"),
            }
        )
    return {
        "case_count": len(case_records),
        "independent_unit_count": len(unit_records),
        "accuracy": {"equal_unit_rmse": _mean(unit_records, "rmse")},
        "probabilistic": {
            "equal_unit_joint_gaussian_nll": _mean(
                unit_records, "joint_gaussian_nll"
            ),
            "equal_unit_normalized_nees": _mean(unit_records, "normalized_nees"),
            "equal_unit_registered_coverage": _mean(
                unit_records, "registered_coverage"
            ),
        },
        "cases": case_records,
    }


def _supported_classes(
    classes: Sequence[str],
    support: NDArray[np.bool_],
) -> tuple[str, ...]:
    return tuple(sorted({classes[index] for index in np.flatnonzero(support)}))


def _query_decision_metrics(cases_value: object, *, system_id: str) -> dict[str, Any]:
    cases = _list(cases_value, name=f"{system_id}.query_decision_cases")
    records: list[dict[str, Any]] = []
    for case_index, raw in enumerate(cases):
        case = _object(raw, name=f"{system_id}.query_decision_cases[{case_index}]")
        _exact_fields(
            case,
            {
                "case_id",
                "statistical_unit_id",
                "hypothesis_states",
                "prior_support",
                "quotient_class_ids",
                "quotient_masses",
                "queries",
                "decision",
            },
            name=f"{system_id}.query_decision_cases[{case_index}]",
        )
        case_id = _string(case.get("case_id"), name="query case_id")
        unit = _string(
            case.get("statistical_unit_id"), name="query statistical_unit_id"
        )
        states = _array(
            case.get("hypothesis_states"), name=f"{case_id}.hypothesis_states", ndim=2
        )
        if states.shape[0] < 1 or states.shape[1] < 1:
            raise ValueError(f"{case_id}: hypothesis_states must be nonempty")
        support_raw = _list(case.get("prior_support"), name=f"{case_id}.prior_support")
        if len(support_raw) != states.shape[0] or not all(
            type(value) is bool for value in support_raw
        ):
            raise ValueError(f"{case_id}: prior_support must be one Boolean per hypothesis")
        support = np.asarray(support_raw, dtype=np.bool_)
        if not np.any(support):
            raise ValueError(f"{case_id}: prior support must be nonempty")
        class_raw = _list(
            case.get("quotient_class_ids"), name=f"{case_id}.quotient_class_ids"
        )
        if len(class_raw) != states.shape[0]:
            raise ValueError(f"{case_id}: quotient_class_ids length mismatch")
        classes = tuple(
            _string(value, name=f"{case_id}.quotient class") for value in class_raw
        )
        masses_raw = _object(
            case.get("quotient_masses"), name=f"{case_id}.quotient_masses"
        )
        class_ids = _supported_classes(classes, support)
        if set(masses_raw) != set(class_ids):
            raise ValueError(f"{case_id}: quotient masses must match supported classes")
        masses = {
            class_id: _number(
                masses_raw[class_id],
                name=f"{case_id}.mass[{class_id}]",
                minimum=0.0,
            )
            for class_id in class_ids
        }
        if not math.isclose(sum(masses.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{case_id}: quotient masses must sum to one")

        identified_queries = 0
        rejected_queries = 0
        query_records = []
        for query_index, query_raw in enumerate(
            _list(case.get("queries"), name=f"{case_id}.queries")
        ):
            query = _object(query_raw, name=f"{case_id}.queries[{query_index}]")
            _exact_fields(
                query,
                {"query_id", "weights", "offset", "tolerance"},
                name=f"{case_id}.queries[{query_index}]",
            )
            query_id = _string(query.get("query_id"), name="query_id")
            weights = _array(query.get("weights"), name=f"{query_id}.weights", ndim=1)
            if weights.shape != (states.shape[1],):
                raise ValueError(f"{query_id}: weight dimension mismatch")
            offset = _number(query.get("offset"), name=f"{query_id}.offset")
            tolerance = _number(
                query.get("tolerance"), name=f"{query_id}.tolerance", minimum=0.0
            )
            values = states @ weights + offset
            widths = []
            for class_id in class_ids:
                selected = support & np.asarray(
                    [value == class_id for value in classes], dtype=np.bool_
                )
                widths.append(float(np.ptp(values[selected])))
            maximum_width = max(widths)
            identified = maximum_width <= tolerance + 1e-12
            identified_queries += int(identified)
            rejected_queries += int(not identified)
            query_records.append(
                {
                    "query_id": query_id,
                    "maximum_supported_class_width": maximum_width,
                    "identified": identified,
                }
            )

        decision = _object(case.get("decision"), name=f"{case_id}.decision")
        _exact_fields(
            decision,
            {
                "action_labels",
                "loss_by_hypothesis_action",
                "regret_budget",
                "fallback_action",
                "reported_output_action",
                "realized_loss_by_action",
            },
            name=f"{case_id}.decision",
        )
        labels = tuple(
            _string(value, name=f"{case_id}.action label")
            for value in _list(
                decision.get("action_labels"), name=f"{case_id}.action_labels"
            )
        )
        if len(labels) != len(set(labels)):
            raise ValueError(f"{case_id}: action labels must be unique")
        loss = _array(
            decision.get("loss_by_hypothesis_action"),
            name=f"{case_id}.loss_by_hypothesis_action",
            ndim=2,
        )
        if loss.shape != (states.shape[0], len(labels)):
            raise ValueError(f"{case_id}: decision loss shape mismatch")
        budget = _number(
            decision.get("regret_budget"),
            name=f"{case_id}.regret_budget",
            minimum=0.0,
        )
        fallback_label = _string(
            decision.get("fallback_action"), name=f"{case_id}.fallback_action"
        )
        reported_label = _string(
            decision.get("reported_output_action"),
            name=f"{case_id}.reported_output_action",
        )
        if fallback_label not in labels or reported_label not in labels:
            raise ValueError(f"{case_id}: unknown fallback or output action")
        realized = _array(
            decision.get("realized_loss_by_action"),
            name=f"{case_id}.realized_loss_by_action",
            ndim=1,
        )
        if realized.shape != (len(labels),):
            raise ValueError(f"{case_id}: realized loss shape mismatch")

        pairwise = np.zeros((len(labels), len(labels)), dtype=np.float64)
        for class_id in class_ids:
            selected = support & np.asarray(
                [value == class_id for value in classes], dtype=np.bool_
            )
            class_loss = loss[selected]
            pairwise += masses[class_id] * np.max(
                class_loss[:, :, None] - class_loss[:, None, :], axis=0
            )
        regret = np.max(pairwise, axis=1)
        minimax_index = int(np.argmin(regret))
        admitted = float(regret[minimax_index]) <= budget + 1e-12
        expected_label = labels[minimax_index] if admitted else fallback_label
        reported_index = labels.index(reported_label)
        fallback_index = labels.index(fallback_label)
        harmful_nonfallback = bool(
            reported_index != fallback_index
            and realized[reported_index] > realized[fallback_index] + 1e-12
        )
        records.append(
            {
                "case_id": case_id,
                "statistical_unit_id": unit,
                "query_identified_count": identified_queries,
                "query_rejected_count": rejected_queries,
                "queries": query_records,
                "worst_case_regret": regret.tolist(),
                "minimax_action": labels[minimax_index],
                "decision_admitted": admitted,
                "fallback_action": fallback_label,
                "expected_output_action": expected_label,
                "reported_output_action": reported_label,
                "output_contract_consistent": reported_label == expected_label,
                "fallback_used": reported_label == fallback_label,
                "harmful_nonfallback": harmful_nonfallback,
            }
        )
    return {
        "case_count": len(records),
        "independent_unit_count": len(
            {record["statistical_unit_id"] for record in records}
        ),
        "query_identified_count": sum(
            int(record["query_identified_count"]) for record in records
        ),
        "query_rejected_count": sum(
            int(record["query_rejected_count"]) for record in records
        ),
        "fallback_count": sum(int(record["fallback_used"]) for record in records),
        "harmful_nonfallback_count": sum(
            int(record["harmful_nonfallback"]) for record in records
        ),
        "output_contract_violation_count": sum(
            int(not record["output_contract_consistent"]) for record in records
        ),
        "cases": records,
    }


def _communication_metrics(cases_value: object, *, system_id: str) -> dict[str, Any]:
    cases = _list(cases_value, name=f"{system_id}.communication_cases")
    records = []
    for index, raw in enumerate(cases):
        case = _object(raw, name=f"{system_id}.communication_cases[{index}]")
        _exact_fields(
            case,
            {
                "case_id",
                "statistical_unit_id",
                "dense_payload_bytes",
                "contract_payload_bytes",
                "posterior_max_abs_error",
                "parity_tolerance",
            },
            name=f"{system_id}.communication_cases[{index}]",
        )
        dense = _number(
            case.get("dense_payload_bytes"), name="dense_payload_bytes", minimum=1.0
        )
        contract = _number(
            case.get("contract_payload_bytes"),
            name="contract_payload_bytes",
            minimum=1.0,
        )
        error = _number(
            case.get("posterior_max_abs_error"),
            name="posterior_max_abs_error",
            minimum=0.0,
        )
        tolerance = _number(
            case.get("parity_tolerance"), name="parity_tolerance", minimum=0.0
        )
        records.append(
            {
                "case_id": _string(case.get("case_id"), name="communication case_id"),
                "statistical_unit_id": _string(
                    case.get("statistical_unit_id"),
                    name="communication statistical_unit_id",
                ),
                "compression_ratio": dense / contract,
                "posterior_max_abs_error": error,
                "parity_tolerance": tolerance,
                "posterior_parity": error <= tolerance,
            }
        )
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_unit[str(record["statistical_unit_id"])].append(record)
    unit_ratios = [
        float(np.mean([record["compression_ratio"] for record in unit_records]))
        for unit_records in by_unit.values()
    ]
    return {
        "case_count": len(records),
        "independent_unit_count": len(by_unit),
        "equal_unit_compression_ratio": float(np.mean(unit_ratios)),
        "posterior_parity_fraction": float(
            np.mean([record["posterior_parity"] for record in records])
        ),
        "cases": records,
    }


def _cross_system_metrics(value: object) -> dict[str, Any]:
    cases = _list(value, name="cross_system_cases", allow_empty=True)
    records = []
    for index, raw in enumerate(cases):
        case = _object(raw, name=f"cross_system_cases[{index}]")
        _exact_fields(
            case,
            {
                "case_id",
                "statistical_unit_id",
                "first_system_id",
                "second_system_id",
                "truth",
                "first_mean",
                "second_mean",
                "corroboration_rmse_threshold",
                "inaccuracy_rmse_threshold",
                "shared_dependence_groups",
            },
            name=f"cross_system_cases[{index}]",
        )
        truth = _array(case.get("truth"), name="cross-system truth", ndim=1)
        first = _array(case.get("first_mean"), name="cross-system first_mean", ndim=1)
        second = _array(case.get("second_mean"), name="cross-system second_mean", ndim=1)
        if first.shape != truth.shape or second.shape != truth.shape or not truth.size:
            raise ValueError("cross-system vectors must share a nonempty shape")
        corroboration_threshold = _number(
            case.get("corroboration_rmse_threshold"),
            name="corroboration_rmse_threshold",
            minimum=0.0,
        )
        inaccuracy_threshold = _number(
            case.get("inaccuracy_rmse_threshold"),
            name="inaccuracy_rmse_threshold",
            minimum=0.0,
        )
        groups = tuple(
            _string(item, name="shared dependence group")
            for item in _list(
                case.get("shared_dependence_groups"),
                name="shared_dependence_groups",
                allow_empty=True,
            )
        )
        corroboration = float(np.sqrt(np.mean(np.square(first - second))))
        first_error = float(np.sqrt(np.mean(np.square(first - truth))))
        second_error = float(np.sqrt(np.mean(np.square(second - truth))))
        counterexample = bool(
            corroboration <= corroboration_threshold
            and first_error >= inaccuracy_threshold
            and second_error >= inaccuracy_threshold
            and groups
        )
        records.append(
            {
                "case_id": _string(case.get("case_id"), name="cross-system case_id"),
                "statistical_unit_id": _string(
                    case.get("statistical_unit_id"),
                    name="cross-system statistical_unit_id",
                ),
                "first_system_id": _string(
                    case.get("first_system_id"), name="first_system_id"
                ),
                "second_system_id": _string(
                    case.get("second_system_id"), name="second_system_id"
                ),
                "provider_corroboration_rmse": corroboration,
                "first_truth_rmse": first_error,
                "second_truth_rmse": second_error,
                "shared_dependence_groups": list(groups),
                "shared_bias_counterexample": counterexample,
            }
        )
    return {
        "case_count": len(records),
        "shared_bias_counterexample_count": sum(
            int(record["shared_bias_counterexample"]) for record in records
        ),
        "cases": records,
    }


def evaluate_controlled_suite(path: str | Path) -> dict[str, Any]:
    suite_path = Path(path)
    try:
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load controlled suite {suite_path}") from error
    suite = _object(suite, name="controlled suite")
    _exact_fields(
        suite,
        {
            "schema_name",
            "schema_version",
            "suite_id",
            "classification",
            "systems",
            "cross_system_cases",
            "information_boundary",
        },
        name="controlled suite",
    )
    if (
        suite.get("schema_name") != CONTROLLED_SCHEMA
        or suite.get("schema_version") != CONTROLLED_VERSION
    ):
        raise ValueError("unsupported controlled suite schema")
    systems_raw = _list(suite.get("systems"), name="systems")
    systems = []
    system_ids = set()
    for index, raw in enumerate(systems_raw):
        system = _object(raw, name=f"systems[{index}]")
        _exact_fields(
            system,
            {
                "system_id",
                "prediction_cases",
                "query_decision_cases",
                "communication_cases",
            },
            name=f"systems[{index}]",
        )
        system_id = _string(system.get("system_id"), name="system_id")
        if system_id in system_ids:
            raise ValueError(f"duplicate system_id {system_id!r}")
        system_ids.add(system_id)
        record: dict[str, Any] = {"system_id": system_id}
        if "prediction_cases" in system:
            record["prediction"] = _prediction_metrics(
                system["prediction_cases"], system_id=system_id
            )
        if "query_decision_cases" in system:
            record["query_decision"] = _query_decision_metrics(
                system["query_decision_cases"], system_id=system_id
            )
        if "communication_cases" in system:
            record["communication"] = _communication_metrics(
                system["communication_cases"], system_id=system_id
            )
        systems.append(record)
    cross_system = _cross_system_metrics(suite.get("cross_system_cases", []))
    for case in cross_system["cases"]:
        if case["first_system_id"] not in system_ids or case["second_system_id"] not in system_ids:
            raise ValueError("cross-system case references an unknown system")
    boundary = _object(suite.get("information_boundary"), name="information_boundary")
    required_boundary = {
        "public_dataset_records_accessed",
        "learned_provider_forward_calls",
        "protected_targets_opened",
        "physical_executions",
    }
    if set(boundary) != required_boundary:
        raise ValueError("information_boundary fields changed")
    normalized_boundary = {}
    for key in sorted(required_boundary):
        value = boundary[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"information_boundary.{key} must be a nonnegative integer")
        normalized_boundary[key] = value
    return {
        "schema_name": RESULT_SCHEMA,
        "schema_version": RESULT_VERSION,
        "suite_id": _string(suite.get("suite_id"), name="suite_id"),
        "classification": _string(
            suite.get("classification"), name="classification"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "systems": systems,
        "cross_system": cross_system,
        "completion": {
            "system_count": len(systems),
            "prediction_system_count": sum("prediction" in system for system in systems),
            "query_decision_system_count": sum(
                "query_decision" in system for system in systems
            ),
            "communication_system_count": sum(
                "communication" in system for system in systems
            ),
        },
        "information_boundary": normalized_boundary,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_controlled_suite(args.suite)
    atomic_write_text(
        args.output,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        overwrite=args.overwrite,
    )
    return 0


__all__ = [
    "CLAIM_BOUNDARY",
    "CONTROLLED_SCHEMA",
    "CONTROLLED_VERSION",
    "RESULT_SCHEMA",
    "RESULT_VERSION",
    "evaluate_controlled_suite",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
