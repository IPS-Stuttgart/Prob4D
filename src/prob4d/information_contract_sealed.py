"""Truth-separated evaluation for probabilistic 4-D information contracts.

The base evaluator accepts a self-contained suite for local replay. This module
adds the scientific information barrier required by a public benchmark: target
truth, queries, ambiguity sets, losses, and fallback are challenge-owned, while
provider means, uncertainty, admissions, certificates, and selected actions are
submission-owned. The two payloads are hash-bound and joined only inside a
temporary evaluator workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ._atomic_file import atomic_write_bytes, atomic_write_text
from .information_contract_benchmark import (
    _ALLOWED_TASKS,
    _aggregate,
    _case_tasks,
    _float_array,
    _hex_sha256,
    _load_json_object,
    _real,
    _relative_payload,
    _sha256_file,
    _string,
    CLAIM_BOUNDARY as CORE_CLAIM_BOUNDARY,
    LEADERBOARD_POLICY,
    RESULT_SCHEMA as CORE_RESULT_SCHEMA,
    RESULT_VERSION as CORE_RESULT_VERSION,
    SUITE_SCHEMA,
    SUITE_VERSION,
    evaluate_information_contract_suite,
)

CHALLENGE_SCHEMA: Final = "prob4d.information-contract-challenge"
CHALLENGE_VERSION: Final = 1
SUBMISSION_SCHEMA: Final = "prob4d.information-contract-submission"
SUBMISSION_VERSION: Final = 1
SEALED_RESULT_SCHEMA: Final = "prob4d.information-contract-sealed-result"
SEALED_RESULT_VERSION: Final = 1
FINITE_QUERY_TASK: Final = "finite_query"
SEALED_CLAIM_BOUNDARY: Final = (
    "This result validates one hash-bound challenge/submission pair. A prospective "
    "claim additionally requires the challenge information-order declaration and "
    "independent evidence that the submission was sealed before target opening. "
    "Retrospective replay remains diagnostic even when every numerical contract passes. "
    "No result proves that a declared quotient, gauge, loss, or fallback is physically "
    "correct outside the frozen challenge."
)
_INFORMATION_ORDERS: Final = frozenset(
    {"retrospective-open-target", "prospective-sealed-target"}
)
_SUBMISSION_MODES: Final = frozenset({"retrospective-replay", "prospective-sealed"})
_CHALLENGE_ARRAYS: Final = frozenset(
    {
        "truth_xyz_m",
        "fallback_mean_xyz_m",
        "fallback_conditional_covariance_m2",
        "fallback_shared_factor_m",
        "query_matrix",
        "nullspace_basis",
        "decision_loss_by_hypothesis",
        "hypothesis_prior",
        "quotient_class",
        "quotient_mass",
        "fallback_action",
        "regret_tolerance",
        "realized_action_loss",
        "finite_query_value_by_hypothesis",
        "finite_query_tolerance",
    }
)
_SUBMISSION_ARRAYS: Final = frozenset(
    {
        "prediction_mean_xyz_m",
        "conditional_covariance_m2",
        "shared_factor_m",
        "query_admitted",
        "reported_query_mean",
        "reported_query_variance",
        "reported_worst_case_regret",
        "selected_action",
        "decision_admitted",
        "finite_query_admitted",
    }
)
_FINITE_QUERY_ARRAYS: Final = frozenset(
    {
        "finite_query_value_by_hypothesis",
        "finite_query_tolerance",
        "finite_query_admitted",
    }
)
FloatArray = NDArray[np.float64]


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _deterministic_npz_bytes(arrays: Mapping[str, NDArray[Any]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            value = np.asarray(arrays[name])
            if value.dtype.kind == "O":
                raise ValueError(f"{name} must not use object dtype")
            payload = io.BytesIO()
            np.lib.format.write_array(payload, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())
    return stream.getvalue()


def _exact_fields(value: Mapping[str, Any], allowed: set[str], *, name: str) -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError(f"{name} contains unregistered fields: {sorted(unknown)}")


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return bool(value)


def _string_list(value: object, *, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a nonempty list"
        raise ValueError(f"{name} must be {qualifier} of strings")
    result = tuple(_string(item, name=f"{name} entry") for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _sealed_tasks(value: object, *, name: str) -> frozenset[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty task list")
    tasks = tuple(_string(item, name=f"{name} entry") for item in value)
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"{name} must not contain duplicates")
    allowed = set(_ALLOWED_TASKS)
    allowed.add(FINITE_QUERY_TASK)
    unknown = set(tasks).difference(allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown tasks: {sorted(unknown)}")
    core = [task for task in tasks if task != FINITE_QUERY_TASK]
    if core:
        _case_tasks(core, name=name)
    return frozenset(tasks)


def _required_split_arrays(tasks: frozenset[str]) -> tuple[set[str], set[str]]:
    challenge: set[str] = set()
    submission: set[str] = set()
    spatial = {
        "forecast",
        "calibration",
        "dependence",
        "query",
        "gauge",
        "fallback",
        "communication",
    }
    if tasks.intersection(spatial):
        challenge.add("truth_xyz_m")
        submission.add("prediction_mean_xyz_m")
    if tasks.intersection(
        {"calibration", "dependence", "query", "gauge", "fallback", "communication"}
    ):
        submission.update({"conditional_covariance_m2", "shared_factor_m"})
    if tasks.intersection({"query", "gauge", "fallback"}):
        challenge.add("query_matrix")
    if tasks.intersection({"gauge", "fallback"}):
        challenge.add("nullspace_basis")
        submission.add("query_admitted")
    if "fallback" in tasks:
        challenge.update(
            {
                "fallback_mean_xyz_m",
                "fallback_conditional_covariance_m2",
                "fallback_shared_factor_m",
            }
        )
        submission.update({"reported_query_mean", "reported_query_variance"})
    if "decision" in tasks:
        challenge.update(
            {
                "decision_loss_by_hypothesis",
                "hypothesis_prior",
                "quotient_class",
                "quotient_mass",
                "fallback_action",
                "regret_tolerance",
                "realized_action_loss",
            }
        )
        submission.update(
            {
                "reported_worst_case_regret",
                "selected_action",
                "decision_admitted",
            }
        )
    if FINITE_QUERY_TASK in tasks:
        challenge.update(
            {
                "finite_query_value_by_hypothesis",
                "finite_query_tolerance",
                "hypothesis_prior",
                "quotient_class",
                "quotient_mass",
            }
        )
        submission.add("finite_query_admitted")
    return challenge, submission


def _load_owned_payload(
    path: Path,
    expected_sha256: str,
    *,
    allowed: frozenset[str],
    forbidden: frozenset[str],
    required: set[str],
    owner: str,
) -> dict[str, NDArray[Any]]:
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{owner} payload SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            smuggled = names.intersection(forbidden)
            if smuggled:
                raise ValueError(
                    f"{owner} payload contains arrays owned by the other side: "
                    f"{sorted(smuggled)}"
                )
            unknown = names.difference(allowed)
            if unknown:
                raise ValueError(
                    f"{owner} payload contains unregistered arrays: {sorted(unknown)}"
                )
            missing = required.difference(names)
            extra = names.difference(required)
            if missing or extra:
                raise ValueError(
                    f"{owner} payload does not match declared tasks: "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            result = {name: np.array(archive[name], copy=True) for name in archive.files}
    except OSError as error:
        raise ValueError(f"cannot load {owner} payload {path}") from error
    for name, value in result.items():
        if value.dtype.kind == "O":
            raise ValueError(f"{owner} payload array {name} must not use object dtype")
    return result


def _validate_thresholds(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("thresholds must be an object")
    allowed = {
        "coverage_probability",
        "gauge_sensitivity_tolerance",
        "moment_atol",
        "relative_rank_tolerance",
    }
    _exact_fields(value, allowed, name="thresholds")
    coverage = _real(
        value.get("coverage_probability"),
        name="coverage_probability",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    if coverage == 0.0:
        raise ValueError("coverage_probability must be greater than zero")
    return {
        "coverage_probability": coverage,
        "gauge_sensitivity_tolerance": _real(
            value.get("gauge_sensitivity_tolerance"),
            name="gauge_sensitivity_tolerance",
            minimum=0.0,
            maximum=1.0,
        ),
        "moment_atol": _real(
            value.get("moment_atol"),
            name="moment_atol",
            minimum=0.0,
        ),
        "relative_rank_tolerance": _real(
            value.get("relative_rank_tolerance"),
            name="relative_rank_tolerance",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        ),
    }


def _validate_dataset(
    value: object,
    *,
    challenge_root: Path,
) -> tuple[dict[str, Any], Path]:
    if not isinstance(value, dict):
        raise ValueError("dataset must be an object")
    allowed = {
        "dataset_id",
        "dataset_version",
        "license_id",
        "public_data",
        "information_order",
        "manifest",
        "manifest_sha256",
    }
    _exact_fields(value, allowed, name="dataset")
    information_order = _string(value.get("information_order"), name="information_order")
    if information_order not in _INFORMATION_ORDERS:
        raise ValueError(
            f"information_order must be one of {sorted(_INFORMATION_ORDERS)}"
        )
    manifest = _relative_payload(
        challenge_root,
        value.get("manifest"),
        name="dataset.manifest",
    )
    expected = _hex_sha256(
        value.get("manifest_sha256"),
        name="dataset.manifest_sha256",
    )
    actual = _sha256_file(manifest)
    if actual != expected:
        raise ValueError(f"dataset manifest SHA-256 mismatch: {actual} != {expected}")
    return (
        {
            "dataset_id": _string(value.get("dataset_id"), name="dataset_id"),
            "dataset_version": _string(
                value.get("dataset_version"), name="dataset_version"
            ),
            "license_id": _string(value.get("license_id"), name="license_id"),
            "public_data": _boolean(value.get("public_data"), name="public_data"),
            "information_order": information_order,
            "manifest": manifest.relative_to(challenge_root.resolve()).as_posix(),
            "manifest_sha256": actual,
        },
        manifest,
    )


def _validate_producer(
    value: object,
    *,
    information_order: str,
) -> tuple[dict[str, Any], bool]:
    if not isinstance(value, dict):
        raise ValueError("producer must be an object")
    allowed = {
        "provider_name",
        "provider_contract",
        "implementation_revision",
        "model_revision",
        "calibration_revision",
        "output_coordinate_frame",
        "causal_cutoff",
        "dependence_group_ids",
        "submission_mode",
        "producer_output_manifest_sha256",
        "target_outcomes_used",
        "target_tuning",
        "prediction_sealed_before_truth",
    }
    _exact_fields(value, allowed, name="producer")
    mode = _string(value.get("submission_mode"), name="submission_mode")
    if mode not in _SUBMISSION_MODES:
        raise ValueError(f"submission_mode must be one of {sorted(_SUBMISSION_MODES)}")
    target_outcomes_used = _boolean(
        value.get("target_outcomes_used"),
        name="target_outcomes_used",
    )
    target_tuning = _boolean(value.get("target_tuning"), name="target_tuning")
    sealed_before_truth = _boolean(
        value.get("prediction_sealed_before_truth"),
        name="prediction_sealed_before_truth",
    )
    if target_outcomes_used or target_tuning:
        raise ValueError("claim-bearing submissions must not use target outcomes or tuning")
    prospective = information_order == "prospective-sealed-target"
    expected_mode = "prospective-sealed" if prospective else "retrospective-replay"
    if mode != expected_mode:
        raise ValueError(
            f"submission_mode {mode!r} is incompatible with {information_order!r}"
        )
    if prospective and not sealed_before_truth:
        raise ValueError(
            "prospective-sealed-target requires prediction_sealed_before_truth"
        )
    if not prospective and sealed_before_truth:
        raise ValueError(
            "retrospective-open-target must not be relabelled as sealed-before-truth"
        )
    producer = {
        key: _string(value.get(key), name=key)
        for key in (
            "provider_name",
            "provider_contract",
            "implementation_revision",
            "model_revision",
            "calibration_revision",
            "output_coordinate_frame",
            "causal_cutoff",
        )
    }
    producer.update(
        {
            "dependence_group_ids": list(
                _string_list(
                    value.get("dependence_group_ids"),
                    name="dependence_group_ids",
                )
            ),
            "submission_mode": mode,
            "producer_output_manifest_sha256": _hex_sha256(
                value.get("producer_output_manifest_sha256"),
                name="producer_output_manifest_sha256",
            ),
            "target_outcomes_used": target_outcomes_used,
            "target_tuning": target_tuning,
            "prediction_sealed_before_truth": sealed_before_truth,
        }
    )
    return producer, prospective


def _validate_case_record(
    value: object,
    *,
    root: Path,
    role: str,
    with_tasks: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{role} case must be an object")
    allowed = {"case_id", "payload", "payload_sha256", "metadata"}
    if with_tasks:
        allowed.update({"group_id", "tasks"})
    _exact_fields(value, allowed, name=f"{role} case")
    case_id = _string(value.get("case_id"), name=f"{role}.case_id")
    payload = _relative_payload(
        root,
        value.get("payload"),
        name=f"{role}.{case_id}.payload",
    )
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{role}.{case_id}.metadata must be an object")
    result: dict[str, Any] = {
        "case_id": case_id,
        "payload": payload,
        "payload_relative": payload.relative_to(root.resolve()).as_posix(),
        "payload_sha256": _hex_sha256(
            value.get("payload_sha256"),
            name=f"{role}.{case_id}.payload_sha256",
        ),
        "metadata": metadata,
    }
    if with_tasks:
        result["group_id"] = _string(
            value.get("group_id"), name=f"{role}.{case_id}.group_id"
        )
        result["tasks"] = _sealed_tasks(
            value.get("tasks"), name=f"{role}.{case_id}.tasks"
        )
    return result


def _records_by_id(records: object, *, role: str, root: Path, with_tasks: bool) -> dict[str, Any]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{role}.cases must be a nonempty list")
    result: dict[str, Any] = {}
    for value in records:
        record = _validate_case_record(
            value,
            root=root,
            role=role,
            with_tasks=with_tasks,
        )
        case_id = record["case_id"]
        if case_id in result:
            raise ValueError(f"duplicate {role} case_id {case_id!r}")
        result[case_id] = record
    return result


def _finite_query_metrics(
    challenge: Mapping[str, NDArray[Any]],
    submission: Mapping[str, NDArray[Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    values = _float_array(
        challenge["finite_query_value_by_hypothesis"],
        name="finite_query_value_by_hypothesis",
        ndim=2,
    )
    hypothesis_count, query_count = values.shape
    if hypothesis_count < 1 or query_count < 1:
        raise ValueError(
            "finite_query_value_by_hypothesis must have shape (H, Q), H,Q >= 1"
        )
    prior = _float_array(
        challenge["hypothesis_prior"],
        name="hypothesis_prior",
        ndim=1,
    )
    if (
        prior.shape != (hypothesis_count,)
        or np.any(prior < 0.0)
        or float(prior.sum()) <= 0.0
    ):
        raise ValueError(
            "hypothesis_prior must match finite queries and have positive mass"
        )
    classes = np.asarray(challenge["quotient_class"])
    if classes.shape != (hypothesis_count,) or classes.dtype.kind not in {"i", "u"}:
        raise ValueError(
            "quotient_class must be an integer vector matching finite hypotheses"
        )
    classes = classes.astype(np.int64, copy=False)
    class_count = int(classes.max(initial=-1)) + 1
    if class_count < 1 or set(classes.tolist()) != set(range(class_count)):
        raise ValueError("quotient_class labels must be contiguous from zero")
    mass = _float_array(challenge["quotient_mass"], name="quotient_mass", ndim=1)
    if (
        mass.shape != (class_count,)
        or np.any(mass < 0.0)
        or not math.isclose(float(mass.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("quotient_mass must have one nonnegative entry and sum to one")
    support = prior > 0.0
    widths = np.empty((class_count, query_count), dtype=np.float64)
    for class_index in range(class_count):
        selected = support & (classes == class_index)
        if not np.any(selected):
            raise ValueError("every finite-query class requires positive prior support")
        selected_values = values[selected]
        widths[class_index] = np.max(selected_values, axis=0) - np.min(
            selected_values,
            axis=0,
        )
    tolerance = _float_array(
        challenge["finite_query_tolerance"],
        name="finite_query_tolerance",
        ndim=1,
    )
    if tolerance.shape != (query_count,) or np.any(tolerance < 0.0):
        raise ValueError(
            "finite_query_tolerance must be nonnegative with one value per query"
        )
    admitted = np.asarray(submission["finite_query_admitted"])
    if admitted.shape != (query_count,) or admitted.dtype.kind != "b":
        raise ValueError(
            "finite_query_admitted must be Boolean with one entry per query"
        )
    maximum_width = np.max(widths, axis=0)
    expected = maximum_width <= tolerance
    false_accept = admitted & ~expected
    false_reject = ~admitted & expected
    return (
        {
            "hypothesis_count": hypothesis_count,
            "quotient_class_count": class_count,
            "query_count": query_count,
            "class_width": widths.tolist(),
            "maximum_class_width": maximum_width.tolist(),
            "mass_weighted_class_width": (mass @ widths).tolist(),
            "tolerance": tolerance.tolist(),
            "expected_admitted": expected.tolist(),
            "submitted_admitted": admitted.tolist(),
            "correct_accept_count": int(np.count_nonzero(admitted & expected)),
            "correct_reject_count": int(np.count_nonzero(~admitted & ~expected)),
            "false_accept_count": int(np.count_nonzero(false_accept)),
            "false_reject_count": int(np.count_nonzero(false_reject)),
            "false_accept_fraction": float(np.mean(false_accept)),
            "false_reject_fraction": float(np.mean(false_reject)),
        },
        {
            "finite_query_admission_consistent": bool(
                not np.any(false_accept) and not np.any(false_reject)
            )
        },
    )


def _finite_summary(case: Mapping[str, Any]) -> dict[str, float]:
    metrics = case["metrics"].get("finite_query")
    if not isinstance(metrics, dict):
        return {}
    return {
        "finite_query_false_accept_fraction": float(metrics["false_accept_fraction"]),
        "finite_query_false_reject_fraction": float(metrics["false_reject_fraction"]),
        "finite_query_mean_maximum_class_width": float(
            np.mean(metrics["maximum_class_width"])
        ),
        "local_admits_finite_rejects": float(
            metrics.get("local_admits_finite_rejects_count", 0)
        ),
    }


def _mean_records(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for record in records for key in record})
    return {
        key: float(np.mean([record[key] for record in records if key in record]))
        for key in keys
    }


def _enrich_finite_aggregate(
    aggregate: dict[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> None:
    selected = [case for case in cases if FINITE_QUERY_TASK in case["tasks"]]
    if not selected:
        aggregate["finite_query"] = {"status": "not_evaluated", "case_count": 0}
        return
    case_summaries = [_finite_summary(case) for case in selected]
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for case, summary in zip(selected, case_summaries, strict=True):
        grouped[str(case["group_id"])].append(summary)
    per_group = {
        group_id: _mean_records(records)
        for group_id, records in sorted(grouped.items())
    }
    aggregate["finite_query"] = {
        "status": "evaluated",
        "case_count": len(selected),
        "independent_group_count": len(grouped),
        "equal_case_mean": _mean_records(case_summaries),
        "per_group_mean": per_group,
        "equal_group_mean": _mean_records(list(per_group.values())),
        "local_admits_finite_rejects_count": sum(
            int(
                case["metrics"]["finite_query"].get(
                    "local_admits_finite_rejects_count",
                    0,
                )
            )
            for case in selected
        ),
    }


def _core_case_skeleton(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "group_id": case["group_id"],
        "tasks": [],
        "payload": None,
        "payload_sha256": None,
        "metadata": {},
        "metrics": {},
        "contract_checks": {},
        "contract_pass": True,
    }


def evaluate_sealed_information_contract(
    challenge_path: str | Path,
    submission_path: str | Path,
) -> dict[str, Any]:
    """Join one truth-separated challenge/submission pair and evaluate it."""

    challenge_file = Path(challenge_path).resolve()
    submission_file = Path(submission_path).resolve()
    challenge = _load_json_object(challenge_file)
    submission = _load_json_object(submission_file)
    _exact_fields(
        challenge,
        {
            "schema_name",
            "schema_version",
            "challenge_id",
            "aggregation_unit",
            "thresholds",
            "claim_boundary",
            "dataset",
            "cases",
        },
        name="challenge",
    )
    _exact_fields(
        submission,
        {
            "schema_name",
            "schema_version",
            "challenge_id",
            "submission_id",
            "producer",
            "claim_boundary",
            "cases",
        },
        name="submission",
    )
    if (
        challenge.get("schema_name") != CHALLENGE_SCHEMA
        or challenge.get("schema_version") != CHALLENGE_VERSION
    ):
        raise ValueError("unsupported information-contract challenge schema")
    if (
        submission.get("schema_name") != SUBMISSION_SCHEMA
        or submission.get("schema_version") != SUBMISSION_VERSION
    ):
        raise ValueError("unsupported information-contract submission schema")
    challenge_id = _string(challenge.get("challenge_id"), name="challenge_id")
    if challenge.get("aggregation_unit") != "group_id":
        raise ValueError("aggregation_unit must be 'group_id'")
    if submission.get("challenge_id") != challenge_id:
        raise ValueError("submission challenge_id does not match challenge")
    submission_id = _string(submission.get("submission_id"), name="submission_id")
    thresholds = _validate_thresholds(challenge.get("thresholds"))
    challenge_boundary = _string(
        challenge.get("claim_boundary"), name="challenge.claim_boundary"
    )
    submission_boundary = _string(
        submission.get("claim_boundary"), name="submission.claim_boundary"
    )
    dataset, _ = _validate_dataset(
        challenge.get("dataset"),
        challenge_root=challenge_file.parent.resolve(),
    )
    producer, prospective = _validate_producer(
        submission.get("producer"),
        information_order=dataset["information_order"],
    )
    challenge_cases = _records_by_id(
        challenge.get("cases"),
        role="challenge",
        root=challenge_file.parent.resolve(),
        with_tasks=True,
    )
    submission_cases = _records_by_id(
        submission.get("cases"),
        role="submission",
        root=submission_file.parent.resolve(),
        with_tasks=False,
    )
    if set(challenge_cases) != set(submission_cases):
        raise ValueError(
            "submission case roster must exactly match challenge: "
            f"missing={sorted(set(challenge_cases).difference(submission_cases))}, "
            f"extra={sorted(set(submission_cases).difference(challenge_cases))}"
        )

    loaded: dict[str, tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]]]] = {}
    with tempfile.TemporaryDirectory(prefix="prob4d-information-contract-") as temporary:
        temporary_root = Path(temporary)
        merged_cases: list[dict[str, Any]] = []
        for case_id in sorted(challenge_cases):
            challenge_case = challenge_cases[case_id]
            submission_case = submission_cases[case_id]
            tasks = challenge_case["tasks"]
            challenge_required, submission_required = _required_split_arrays(tasks)
            challenge_arrays = _load_owned_payload(
                challenge_case["payload"],
                challenge_case["payload_sha256"],
                allowed=_CHALLENGE_ARRAYS,
                forbidden=_SUBMISSION_ARRAYS,
                required=challenge_required,
                owner=f"challenge {case_id}",
            )
            submission_arrays = _load_owned_payload(
                submission_case["payload"],
                submission_case["payload_sha256"],
                allowed=_SUBMISSION_ARRAYS,
                forbidden=_CHALLENGE_ARRAYS,
                required=submission_required,
                owner=f"submission {case_id}",
            )
            loaded[case_id] = (challenge_arrays, submission_arrays)
            core_tasks = sorted(set(tasks).difference({FINITE_QUERY_TASK}))
            if not core_tasks:
                continue
            merged_arrays = {
                name: value
                for name, value in challenge_arrays.items()
                if name not in _FINITE_QUERY_ARRAYS
            }
            merged_arrays.update(
                {
                    name: value
                    for name, value in submission_arrays.items()
                    if name not in _FINITE_QUERY_ARRAYS
                }
            )
            merged_payload = temporary_root / f"{case_id}.npz"
            atomic_write_bytes(
                merged_payload,
                _deterministic_npz_bytes(merged_arrays),
                overwrite=False,
            )
            merged_cases.append(
                {
                    "case_id": case_id,
                    "group_id": challenge_case["group_id"],
                    "payload": merged_payload.name,
                    "payload_sha256": _sha256_file(merged_payload),
                    "tasks": core_tasks,
                    "metadata": {
                        "challenge": challenge_case["metadata"],
                        "submission": submission_case["metadata"],
                    },
                }
            )
        core_by_id: dict[str, dict[str, Any]] = {}
        if merged_cases:
            suite_path = temporary_root / "suite.json"
            atomic_write_text(
                suite_path,
                _canonical_json(
                    {
                        "schema_name": SUITE_SCHEMA,
                        "schema_version": SUITE_VERSION,
                        "suite_id": f"{challenge_id}/{submission_id}",
                        "aggregation_unit": "group_id",
                        "thresholds": thresholds,
                        "claim_boundary": (
                            "Temporary deterministic join of separately sealed challenge "
                            "and submission payloads."
                        ),
                        "cases": merged_cases,
                    }
                ),
                overwrite=False,
            )
            core_result = evaluate_information_contract_suite(suite_path)
            core_by_id = {
                str(case["case_id"]): case for case in core_result["cases"]
            }

    cases: list[dict[str, Any]] = []
    for case_id in sorted(challenge_cases):
        challenge_case = challenge_cases[case_id]
        submission_case = submission_cases[case_id]
        challenge_arrays, submission_arrays = loaded[case_id]
        case = dict(core_by_id.get(case_id, _core_case_skeleton(challenge_case)))
        case["tasks"] = sorted(challenge_case["tasks"])
        case["metadata"] = {
            "challenge": challenge_case["metadata"],
            "submission": submission_case["metadata"],
        }
        case["payloads"] = {
            "challenge": {
                "path": challenge_case["payload_relative"],
                "sha256": challenge_case["payload_sha256"],
                "size_bytes": challenge_case["payload"].stat().st_size,
            },
            "submission": {
                "path": submission_case["payload_relative"],
                "sha256": submission_case["payload_sha256"],
                "size_bytes": submission_case["payload"].stat().st_size,
            },
        }
        case.pop("payload", None)
        case.pop("payload_sha256", None)
        if "communication" in challenge_case["tasks"]:
            communication = case["metrics"]["communication"]
            communication["merged_evaluation_payload_file_bytes"] = communication[
                "payload_file_bytes"
            ]
            communication["challenge_payload_file_bytes"] = challenge_case[
                "payload"
            ].stat().st_size
            communication["submission_payload_file_bytes"] = submission_case[
                "payload"
            ].stat().st_size
            communication["payload_file_bytes"] = communication[
                "submission_payload_file_bytes"
            ]
        if FINITE_QUERY_TASK in challenge_case["tasks"]:
            finite_metrics, finite_checks = _finite_query_metrics(
                challenge_arrays,
                submission_arrays,
            )
            gauge = case["metrics"].get("gauge")
            if isinstance(gauge, dict):
                local = np.asarray(gauge["expected_admitted"], dtype=np.bool_)
                finite = np.asarray(
                    finite_metrics["expected_admitted"],
                    dtype=np.bool_,
                )
                if local.shape != finite.shape:
                    raise ValueError(
                        f"{case_id}: local and finite query counts must match"
                    )
                finite_metrics["local_admits_finite_rejects_count"] = int(
                    np.count_nonzero(local & ~finite)
                )
                finite_metrics["finite_admits_local_rejects_count"] = int(
                    np.count_nonzero(~local & finite)
                )
            case["metrics"]["finite_query"] = finite_metrics
            case["contract_checks"].update(finite_checks)
        case["contract_pass"] = all(case["contract_checks"].values())
        cases.append(case)

    aggregate = _aggregate(cases)
    _enrich_finite_aggregate(aggregate, cases)
    return {
        "schema_name": SEALED_RESULT_SCHEMA,
        "schema_version": SEALED_RESULT_VERSION,
        "challenge_id": challenge_id,
        "submission_id": submission_id,
        "challenge_sha256": _sha256_file(challenge_file),
        "submission_sha256": _sha256_file(submission_file),
        "dataset": dataset,
        "producer": producer,
        "information_order": {
            "mode": dataset["information_order"],
            "claim_class": (
                "prospective-heldout"
                if prospective
                else "retrospective-diagnostic"
            ),
            "prospective_claim_eligible": prospective,
        },
        "claim_boundary": SEALED_CLAIM_BOUNDARY,
        "core_claim_boundary": CORE_CLAIM_BOUNDARY,
        "challenge_claim_boundary": challenge_boundary,
        "submission_claim_boundary": submission_boundary,
        "leaderboard_policy": LEADERBOARD_POLICY,
        "core_evaluator": {
            "schema_name": CORE_RESULT_SCHEMA,
            "schema_version": CORE_RESULT_VERSION,
        },
        "thresholds": thresholds,
        "cases": cases,
        "aggregate": aggregate,
    }


def generate_sealed_smoke(directory: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Create the truth-separated deterministic smoke challenge and submission."""

    from .information_contract_sealed_smoke import generate_sealed_smoke as generate

    return generate(directory, overwrite=overwrite)


def _write_result(path: Path, result: Mapping[str, Any], *, overwrite: bool) -> None:
    atomic_write_text(path, _canonical_json(result), overwrite=overwrite)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate one truth-separated challenge/submission pair",
    )
    evaluate.add_argument("challenge", type=Path)
    evaluate.add_argument("submission", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--overwrite", action="store_true")
    smoke = subparsers.add_parser(
        "smoke",
        help="generate and evaluate a deterministic sealed smoke pair",
    )
    smoke.add_argument("directory", type=Path)
    smoke.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        result = evaluate_sealed_information_contract(
            args.challenge,
            args.submission,
        )
        _write_result(args.output, result, overwrite=args.overwrite)
        return 0
    challenge, submission = generate_sealed_smoke(
        args.directory,
        overwrite=args.overwrite,
    )
    result = evaluate_sealed_information_contract(challenge, submission)
    _write_result(
        args.directory / "result.json",
        result,
        overwrite=args.overwrite,
    )
    return 0


__all__ = [
    "CHALLENGE_SCHEMA",
    "CHALLENGE_VERSION",
    "FINITE_QUERY_TASK",
    "SEALED_CLAIM_BOUNDARY",
    "SEALED_RESULT_SCHEMA",
    "SEALED_RESULT_VERSION",
    "SUBMISSION_SCHEMA",
    "SUBMISSION_VERSION",
    "evaluate_sealed_information_contract",
    "generate_sealed_smoke",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
