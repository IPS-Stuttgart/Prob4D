"""Freeze and report outcome-blind diagnostic strata for CUT3R qualification.

The strata expose long-horizon, occlusion, deformation, viewpoint, and anchor-
conditioning failure modes without changing the source comparison or selecting a
method. Complete physical objects or acquisition sessions remain the independent
statistical units; frames are nested records.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from ._atomic_file import atomic_write_bytes
from .cut3r_comparison import (
    CUT3R_COMPARISON_GROUP_UNIT,
    load_cut3r_comparison_lock,
    validate_cut3r_comparison_lock,
)

CUT3R_STRATA_LOCK_SCHEMA: Final = "prob4d.cut3r-diagnostic-strata-lock"
CUT3R_STRATA_LOCK_VERSION: Final = 1
CUT3R_STRATA_RECORDS_SCHEMA: Final = "prob4d.cut3r-diagnostic-records"
CUT3R_STRATA_RECORDS_VERSION: Final = 1
CUT3R_STRATA_REPORT_SCHEMA: Final = "prob4d.cut3r-diagnostic-strata-report"
CUT3R_STRATA_REPORT_VERSION: Final = 1
CUT3R_STRATA_CLAIM_BOUNDARY: Final = (
    "This source-only report localizes CUT3R and Prob4D behavior across frozen "
    "reporting strata. It cannot select a provider, change a comparison arm, "
    "authorize target access, establish BayesianPhysTwin or Causal4D benefit, "
    "establish deployment safety, or establish state of the art."
)
CUT3R_STRATA_WEIGHTING: Final = {
    "within_group": "equal-seed-means-then-equal-case-means-v1",
    "across_groups": "equal-complete-group-mean-v1",
    "contrasts": "paired-complete-group-difference-v1",
}

# Metadata are canonical; only source-frozen bin edges are supplied by a study.
CUT3R_DIAGNOSTIC_STRATA: Final[tuple[tuple[str, str, str, str], ...]] = (
    (
        "absolute-prefix-age",
        "frames_since_sequence_start",
        "frames",
        "frozen-source-frame-index",
    ),
    (
        "frames-since-restart-boundary",
        "frames_since_restart_boundary",
        "frames",
        "frozen-window-schedule",
    ),
    (
        "occlusion-reappearance-gap",
        "occlusion_reappearance_gap_frames",
        "frames",
        "frozen-input-visibility",
    ),
    (
        "normalized-image-motion",
        "normalized_image_motion",
        "image-diagonal-fraction-per-frame",
        "frozen-input-image-motion",
    ),
    (
        "viewpoint-rotation-novelty",
        "viewpoint_rotation_novelty_deg",
        "degrees",
        "frozen-prefix-camera-geometry",
    ),
    (
        "metric-anchor-conditioning",
        "metric_anchor_log10_condition_number",
        "log10-condition-number",
        "frozen-prefix-metric-anchor-geometry",
    ),
)

_LOCK_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "comparison_lock_id",
        "comparison_protocol_name",
        "group_unit",
        "source_evaluation_groups",
        "random_seeds",
        "record_definition_sha256",
        "minimum_evaluable_groups_per_bin",
        "metric_names",
        "strata",
        "weighting",
        "selection_role",
        "source_access",
        "target_access",
        "claim_boundary",
        "strata_lock_id",
    }
)
_LOCK_SPEC_FIELDS: Final = frozenset(
    {
        "record_definition_sha256",
        "minimum_evaluable_groups_per_bin",
        "metric_names",
        "strata",
    }
)
_STRATUM_FIELDS: Final = frozenset(
    {
        "stratum_id",
        "feature_name",
        "unit",
        "bin_edges",
        "value_source",
        "uses_truth",
        "uses_downstream_physical_innovation",
        "uses_target_outcomes",
        "selection_role",
    }
)
_RECORDS_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "comparison_lock_id",
        "strata_lock_id",
        "record_definition_sha256",
        "source_truth_used",
        "target_payloads_opened",
        "target_outcomes_opened",
        "records",
    }
)
_RECORD_FIELDS: Final = frozenset(
    {
        "group_id",
        "case_id",
        "frame_index",
        "random_seed",
        "arm_id",
        "features",
        "metrics",
    }
)
_REPORT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "comparison_lock_id",
        "strata_lock_id",
        "records_id",
        "record_count",
        "nested_observation_count",
        "source_truth_used",
        "target_payloads_opened",
        "target_outcomes_opened",
        "group_unit",
        "random_seeds",
        "record_definition_sha256",
        "minimum_evaluable_groups_per_bin",
        "metric_names",
        "weighting",
        "strata_results",
        "selection_role",
        "source_access",
        "target_access",
        "claim_boundary",
        "report_id",
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _record_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with exact string keys")
    return cast(Mapping[str, Any], value)


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    name: str,
) -> None:
    keys = set(value)
    if keys != set(expected):
        missing = sorted(set(expected) - keys)
        extra = sorted(keys - set(expected))
        raise ValueError(f"{name} has noncanonical keys; missing={missing}, extra={extra}")


def _strict_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a nonempty exact string without surrounding whitespace"
        )
    return cast(str, value)


def _strict_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a genuine integer >= {minimum}")
    return cast(int, value)


def _strict_boolean(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a genuine Boolean")
    return cast(bool, value)


def _finite_number(value: Any, *, name: str, nonnegative: bool) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a genuine finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _sha256(value: Any, *, name: str) -> str:
    digest = _strict_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _strict_json(path: str | Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _canonical_metric_names(value: Any) -> list[str]:
    if type(value) is not list or not value:
        raise ValueError("metric_names must be a nonempty JSON array")
    names = [
        _strict_string(item, name=f"metric_names[{index}]")
        for index, item in enumerate(value)
    ]
    if len(names) != len(set(names)):
        raise ValueError("metric_names must be unique")
    return sorted(names)


def _canonical_random_seeds(value: Any, *, name: str) -> list[int]:
    if type(value) is not list or not value:
        raise ValueError(f"{name} must be a nonempty JSON array")
    seeds = [
        _strict_integer(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    ]
    if seeds != sorted(set(seeds)):
        raise ValueError(f"{name} must be sorted and unique")
    return seeds


def _canonical_bin_edges(value: Any, *, name: str) -> list[float]:
    if type(value) is not list or len(value) < 2:
        raise ValueError(f"{name} must contain at least two bin edges")
    edges = [
        _finite_number(item, name=f"{name}[{index}]", nonnegative=True)
        for index, item in enumerate(value)
    ]
    if edges[0] != 0.0:
        raise ValueError(f"{name} must start at zero")
    if any(right <= left for left, right in zip(edges, edges[1:], strict=False)):
        raise ValueError(f"{name} must be strictly increasing")
    return edges


def _canonical_strata(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise ValueError("strata must be a JSON array")
    expected = {
        stratum_id: (feature_name, unit, source)
        for stratum_id, feature_name, unit, source in CUT3R_DIAGNOSTIC_STRATA
    }
    if len(value) != len(expected):
        raise ValueError("strata must declare every canonical CUT3R diagnostic exactly once")

    normalized_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        item = _strict_mapping(raw, name=f"strata[{index}]")
        _exact_keys(item, _STRATUM_FIELDS, name=f"strata[{index}]")
        stratum_id = _strict_string(item["stratum_id"], name="stratum_id")
        if stratum_id not in expected:
            raise ValueError(f"unknown CUT3R diagnostic stratum: {stratum_id!r}")
        if stratum_id in normalized_by_id:
            raise ValueError(f"duplicate CUT3R diagnostic stratum: {stratum_id!r}")
        feature_name, unit, source = expected[stratum_id]
        if _strict_string(item["feature_name"], name="feature_name") != feature_name:
            raise ValueError(f"feature_name changed for stratum {stratum_id!r}")
        if _strict_string(item["unit"], name="unit") != unit:
            raise ValueError(f"unit changed for stratum {stratum_id!r}")
        if _strict_string(item["value_source"], name="value_source") != source:
            raise ValueError(f"value_source changed for stratum {stratum_id!r}")
        for field in (
            "uses_truth",
            "uses_downstream_physical_innovation",
            "uses_target_outcomes",
        ):
            if _strict_boolean(item[field], name=field):
                raise ValueError(f"{field} must be false for reporting strata")
        if _strict_string(item["selection_role"], name="selection_role") != "reporting-only":
            raise ValueError("diagnostic strata must remain reporting-only")
        normalized_by_id[stratum_id] = {
            "stratum_id": stratum_id,
            "feature_name": feature_name,
            "unit": unit,
            "bin_edges": _canonical_bin_edges(
                item["bin_edges"],
                name=f"{stratum_id}.bin_edges",
            ),
            "value_source": source,
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
            "uses_target_outcomes": False,
            "selection_role": "reporting-only",
        }

    return [normalized_by_id[stratum_id] for stratum_id, *_ in CUT3R_DIAGNOSTIC_STRATA]


def _comparison_context(comparison_lock: Any) -> dict[str, Any]:
    comparison = validate_cut3r_comparison_lock(comparison_lock)
    source_groups = list(comparison["group_roles"]["source_evaluation"])
    group_cases: dict[str, dict[str, tuple[int, int, int]]] = {}
    for group in comparison["groups"]:
        group_id = cast(str, group["group_id"])
        if group_id not in source_groups:
            continue
        group_cases[group_id] = {
            cast(str, case["case_id"]): (
                cast(int, case["frame_start"]),
                cast(int, case["evaluation_frame_start"]),
                cast(int, case["evaluation_frame_stop_exclusive"]),
            )
            for case in group["cases"]
        }
    enabled_arms = [
        cast(str, arm["arm_id"])
        for arm in comparison["arms"]
        if arm["enabled"] and arm["causal"] and arm["claim_eligible"]
    ]
    contrasts = [
        cast(dict[str, Any], contrast)
        for contrast in comparison["registered_contrasts"]
        if contrast["enabled"] and contrast["claim_eligible"]
    ]
    return {
        "comparison": comparison,
        "source_groups": source_groups,
        "group_cases": group_cases,
        "enabled_arms": enabled_arms,
        "contrasts": contrasts,
    }


def build_cut3r_diagnostic_strata_lock(
    comparison_lock: Any,
    specification: Any,
) -> dict[str, Any]:
    """Freeze source-only reporting strata against one CUT3R comparison lock."""

    context = _comparison_context(comparison_lock)
    comparison = cast(dict[str, Any], context["comparison"])
    source_groups = cast(list[str], context["source_groups"])
    spec = _strict_mapping(specification, name="CUT3R diagnostic strata specification")
    _exact_keys(spec, _LOCK_SPEC_FIELDS, name="CUT3R diagnostic strata specification")
    minimum_groups = _strict_integer(
        spec["minimum_evaluable_groups_per_bin"],
        name="minimum_evaluable_groups_per_bin",
        minimum=1,
    )
    if minimum_groups > len(source_groups):
        raise ValueError(
            "minimum_evaluable_groups_per_bin exceeds the frozen source-evaluation roster"
        )
    payload: dict[str, Any] = {
        "schema": CUT3R_STRATA_LOCK_SCHEMA,
        "schema_version": CUT3R_STRATA_LOCK_VERSION,
        "comparison_lock_id": comparison["lock_id"],
        "comparison_protocol_name": comparison["protocol_name"],
        "group_unit": CUT3R_COMPARISON_GROUP_UNIT,
        "source_evaluation_groups": source_groups,
        "random_seeds": comparison["random_seeds"],
        "record_definition_sha256": _sha256(
            spec["record_definition_sha256"],
            name="record_definition_sha256",
        ),
        "minimum_evaluable_groups_per_bin": minimum_groups,
        "metric_names": _canonical_metric_names(spec["metric_names"]),
        "strata": _canonical_strata(spec["strata"]),
        "weighting": dict(CUT3R_STRATA_WEIGHTING),
        "selection_role": "reporting-only",
        "source_access": "source-only",
        "target_access": "forbidden",
        "claim_boundary": CUT3R_STRATA_CLAIM_BOUNDARY,
    }
    payload["strata_lock_id"] = _record_id(payload)
    return validate_cut3r_diagnostic_strata_lock(comparison, payload)


def validate_cut3r_diagnostic_strata_lock(
    comparison_lock: Any,
    value: Any,
) -> dict[str, Any]:
    """Validate a strata lock and its exact comparison-lock binding."""

    context = _comparison_context(comparison_lock)
    comparison = cast(dict[str, Any], context["comparison"])
    source_groups = cast(list[str], context["source_groups"])
    payload = _strict_mapping(value, name="CUT3R diagnostic strata lock")
    _exact_keys(payload, _LOCK_FIELDS, name="CUT3R diagnostic strata lock")
    if _strict_string(payload["schema"], name="schema") != CUT3R_STRATA_LOCK_SCHEMA:
        raise ValueError("unsupported CUT3R diagnostic strata schema")
    if (
        _strict_integer(payload["schema_version"], name="schema_version", minimum=1)
        != CUT3R_STRATA_LOCK_VERSION
    ):
        raise ValueError("unsupported CUT3R diagnostic strata schema version")
    comparison_lock_id = _sha256(payload["comparison_lock_id"], name="comparison_lock_id")
    if comparison_lock_id != comparison["lock_id"]:
        raise ValueError("diagnostic strata lock is bound to a different comparison lock")
    protocol_name = _strict_string(
        payload["comparison_protocol_name"],
        name="comparison_protocol_name",
    )
    if protocol_name != comparison["protocol_name"]:
        raise ValueError("diagnostic strata protocol name does not match the comparison")
    if _strict_string(payload["group_unit"], name="group_unit") != CUT3R_COMPARISON_GROUP_UNIT:
        raise ValueError("diagnostic strata must use complete object/session groups")
    raw_source_groups = payload["source_evaluation_groups"]
    if type(raw_source_groups) is not list:
        raise ValueError("source_evaluation_groups must be a JSON array")
    normalized_source_groups = [
        _strict_string(item, name=f"source_evaluation_groups[{index}]")
        for index, item in enumerate(raw_source_groups)
    ]
    if normalized_source_groups != source_groups:
        raise ValueError("diagnostic strata source roster changed from the comparison lock")
    normalized_random_seeds = _canonical_random_seeds(
        payload["random_seeds"],
        name="random_seeds",
    )
    if normalized_random_seeds != comparison["random_seeds"]:
        raise ValueError("diagnostic strata random seeds changed from the comparison lock")
    record_definition_sha256 = _sha256(
        payload["record_definition_sha256"],
        name="record_definition_sha256",
    )
    minimum_groups = _strict_integer(
        payload["minimum_evaluable_groups_per_bin"],
        name="minimum_evaluable_groups_per_bin",
        minimum=1,
    )
    if minimum_groups > len(source_groups):
        raise ValueError(
            "minimum_evaluable_groups_per_bin exceeds the frozen source-evaluation roster"
        )
    if payload["weighting"] != CUT3R_STRATA_WEIGHTING:
        raise ValueError("diagnostic strata weighting changed from equal complete-group mass")
    if _strict_string(payload["selection_role"], name="selection_role") != "reporting-only":
        raise ValueError("diagnostic strata cannot select a method")
    if _strict_string(payload["source_access"], name="source_access") != "source-only":
        raise ValueError("diagnostic strata lock must remain source-only")
    if _strict_string(payload["target_access"], name="target_access") != "forbidden":
        raise ValueError("diagnostic strata lock cannot authorize target access")
    claim_boundary = _strict_string(payload["claim_boundary"], name="claim_boundary")
    if claim_boundary != CUT3R_STRATA_CLAIM_BOUNDARY:
        raise ValueError("diagnostic strata claim boundary changed")
    strata_lock_id = _sha256(payload["strata_lock_id"], name="strata_lock_id")
    normalized: dict[str, Any] = {
        "schema": CUT3R_STRATA_LOCK_SCHEMA,
        "schema_version": CUT3R_STRATA_LOCK_VERSION,
        "comparison_lock_id": comparison_lock_id,
        "comparison_protocol_name": protocol_name,
        "group_unit": CUT3R_COMPARISON_GROUP_UNIT,
        "source_evaluation_groups": normalized_source_groups,
        "random_seeds": normalized_random_seeds,
        "record_definition_sha256": record_definition_sha256,
        "minimum_evaluable_groups_per_bin": minimum_groups,
        "metric_names": _canonical_metric_names(payload["metric_names"]),
        "strata": _canonical_strata(payload["strata"]),
        "weighting": dict(CUT3R_STRATA_WEIGHTING),
        "selection_role": "reporting-only",
        "source_access": "source-only",
        "target_access": "forbidden",
        "claim_boundary": claim_boundary,
        "strata_lock_id": strata_lock_id,
    }
    unsigned = dict(normalized)
    unsigned.pop("strata_lock_id")
    if strata_lock_id != _record_id(unsigned):
        raise ValueError("strata_lock_id does not match canonical strata-lock content")
    return cast(dict[str, Any], json.loads(_canonical_json(normalized)))


def load_cut3r_diagnostic_strata_lock(
    comparison_lock: Any,
    path: str | Path,
) -> dict[str, Any]:
    return validate_cut3r_diagnostic_strata_lock(comparison_lock, _strict_json(path))


def _publish_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    load_existing: Any,
) -> dict[str, Any]:
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("artifact destination must not be a symbolic link")
    encoded = _canonical_json(payload) + b"\n"
    try:
        atomic_write_bytes(destination, encoded, overwrite=False)
    except FileExistsError:
        existing = load_existing(destination)
        if existing != payload:
            raise FileExistsError(
                f"refusing to replace a different artifact: {destination}"
            ) from None
        return existing
    return dict(payload)


def write_cut3r_diagnostic_strata_lock(
    comparison_lock: Any,
    path: str | Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    payload = validate_cut3r_diagnostic_strata_lock(comparison_lock, lock)
    return _publish_json(
        path,
        payload,
        load_existing=lambda existing: load_cut3r_diagnostic_strata_lock(
            comparison_lock,
            existing,
        ),
    )


def _normalize_records(
    comparison_lock: Any,
    strata_lock: Any,
    value: Any,
) -> dict[str, Any]:
    context = _comparison_context(comparison_lock)
    comparison = cast(dict[str, Any], context["comparison"])
    group_cases = cast(
        dict[str, dict[str, tuple[int, int, int]]],
        context["group_cases"],
    )
    enabled_arms = cast(list[str], context["enabled_arms"])
    lock = validate_cut3r_diagnostic_strata_lock(comparison, strata_lock)
    payload = _strict_mapping(value, name="CUT3R diagnostic records")
    _exact_keys(payload, _RECORDS_FIELDS, name="CUT3R diagnostic records")
    if _strict_string(payload["schema"], name="schema") != CUT3R_STRATA_RECORDS_SCHEMA:
        raise ValueError("unsupported CUT3R diagnostic records schema")
    if (
        _strict_integer(payload["schema_version"], name="schema_version", minimum=1)
        != CUT3R_STRATA_RECORDS_VERSION
    ):
        raise ValueError("unsupported CUT3R diagnostic records schema version")
    if _sha256(payload["comparison_lock_id"], name="comparison_lock_id") != comparison["lock_id"]:
        raise ValueError("diagnostic records are bound to a different comparison lock")
    if _sha256(payload["strata_lock_id"], name="strata_lock_id") != lock["strata_lock_id"]:
        raise ValueError("diagnostic records are bound to a different strata lock")
    if _sha256(
        payload["record_definition_sha256"],
        name="record_definition_sha256",
    ) != lock["record_definition_sha256"]:
        raise ValueError("diagnostic records use a different frozen record definition")
    source_truth_used = _strict_boolean(payload["source_truth_used"], name="source_truth_used")
    if _strict_boolean(payload["target_payloads_opened"], name="target_payloads_opened"):
        raise ValueError("diagnostic records may not open target payloads")
    if _strict_boolean(payload["target_outcomes_opened"], name="target_outcomes_opened"):
        raise ValueError("diagnostic records may not open target outcomes")
    raw_records = payload["records"]
    if type(raw_records) is not list or not raw_records:
        raise ValueError("diagnostic records must be a nonempty JSON array")

    feature_names = [cast(str, item["feature_name"]) for item in lock["strata"]]
    metric_names = cast(list[str], lock["metric_names"])
    normalized_records: list[dict[str, Any]] = []
    record_keys: set[tuple[str, str, int, int, str]] = set()
    observation_features: dict[tuple[str, str, int], dict[str, float]] = {}
    observation_arms: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    frame_seeds: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    random_seeds = cast(list[int], comparison["random_seeds"])
    seen_groups: set[str] = set()

    for index, raw_record in enumerate(raw_records):
        record = _strict_mapping(raw_record, name=f"records[{index}]")
        _exact_keys(record, _RECORD_FIELDS, name=f"records[{index}]")
        group_id = _strict_string(record["group_id"], name="group_id")
        if group_id not in group_cases:
            raise ValueError(f"record group {group_id!r} is not a source-evaluation group")
        case_id = _strict_string(record["case_id"], name="case_id")
        if case_id not in group_cases[group_id]:
            raise ValueError(f"record case {case_id!r} is not frozen inside group {group_id!r}")
        frame_index = _strict_integer(record["frame_index"], name="frame_index")
        source_start, evaluation_start, evaluation_stop = group_cases[group_id][case_id]
        if not evaluation_start <= frame_index < evaluation_stop:
            raise ValueError("record frame lies outside the frozen evaluation interval")
        random_seed = _strict_integer(record["random_seed"], name="random_seed")
        if random_seed not in random_seeds:
            raise ValueError(f"record random seed {random_seed} is not frozen by the comparison")
        arm_id = _strict_string(record["arm_id"], name="arm_id")
        if arm_id not in enabled_arms:
            raise ValueError(f"record arm {arm_id!r} is not an enabled causal claim arm")
        record_key = (group_id, case_id, frame_index, random_seed, arm_id)
        if record_key in record_keys:
            raise ValueError(f"duplicate diagnostic record: {record_key!r}")
        record_keys.add(record_key)

        features = _strict_mapping(record["features"], name="features")
        _exact_keys(features, set(feature_names), name="features")
        normalized_features = {
            name: _finite_number(features[name], name=f"features.{name}", nonnegative=True)
            for name in feature_names
        }
        expected_absolute_age = float(frame_index - source_start)
        if normalized_features["frames_since_sequence_start"] != expected_absolute_age:
            raise ValueError(
                "frames_since_sequence_start must match the frozen source frame index"
            )
        expected_restart_phase = float(
            (frame_index - source_start) % cast(int, comparison["windowing"]["stride"])
        )
        if (
            normalized_features["frames_since_restart_boundary"]
            != expected_restart_phase
        ):
            raise ValueError(
                "frames_since_restart_boundary must match the frozen window schedule"
            )
        metrics = _strict_mapping(record["metrics"], name="metrics")
        _exact_keys(metrics, set(metric_names), name="metrics")
        normalized_metrics = {
            name: _finite_number(metrics[name], name=f"metrics.{name}", nonnegative=False)
            for name in metric_names
        }
        frame_key = (group_id, case_id, frame_index)
        observation_key = (group_id, case_id, frame_index, random_seed)
        if frame_key in observation_features:
            if observation_features[frame_key] != normalized_features:
                raise ValueError(
                    "stratification features must be identical across arms and seeds"
                )
        else:
            observation_features[frame_key] = normalized_features
        observation_arms[observation_key].add(arm_id)
        frame_seeds[frame_key].add(random_seed)
        seen_groups.add(group_id)
        normalized_records.append(
            {
                "group_id": group_id,
                "case_id": case_id,
                "frame_index": frame_index,
                "random_seed": random_seed,
                "arm_id": arm_id,
                "features": normalized_features,
                "metrics": normalized_metrics,
            }
        )

    expected_arm_set = set(enabled_arms)
    for observation_key, arms in observation_arms.items():
        if arms != expected_arm_set:
            missing_arms = sorted(expected_arm_set - arms)
            extra_arms = sorted(arms - expected_arm_set)
            raise ValueError(
                "every nested observation must retain paired common support across arms; "
                f"observation={observation_key!r}, missing={missing_arms}, extra={extra_arms}"
            )
    expected_seed_set = set(random_seeds)
    for frame_key, seeds in frame_seeds.items():
        if seeds != expected_seed_set:
            missing_seeds = sorted(expected_seed_set - seeds)
            extra_seeds = sorted(seeds - expected_seed_set)
            raise ValueError(
                "every frame must retain the complete frozen random-seed roster; "
                f"frame={frame_key!r}, missing={missing_seeds}, extra={extra_seeds}"
            )
    expected_frame_keys = {
        (group_id, case_id, frame_index)
        for group_id, cases in group_cases.items()
        for case_id, (_, evaluation_start, evaluation_stop) in cases.items()
        for frame_index in range(evaluation_start, evaluation_stop)
    }
    observed_frame_keys = set(frame_seeds)
    if observed_frame_keys != expected_frame_keys:
        missing_frames = sorted(expected_frame_keys - observed_frame_keys)
        extra_frames = sorted(observed_frame_keys - expected_frame_keys)
        raise ValueError(
            "diagnostic records must retain every frozen evaluation frame; "
            f"missing={missing_frames[:10]}, extra={extra_frames[:10]}"
        )

    expected_groups = set(lock["source_evaluation_groups"])
    if seen_groups != expected_groups:
        missing_groups = sorted(expected_groups - seen_groups)
        raise ValueError(f"diagnostic records omit frozen source groups: {missing_groups}")

    normalized_records.sort(
        key=lambda item: (
            cast(str, item["group_id"]),
            cast(str, item["case_id"]),
            cast(int, item["frame_index"]),
            cast(int, item["random_seed"]),
            cast(str, item["arm_id"]),
        )
    )
    normalized = {
        "schema": CUT3R_STRATA_RECORDS_SCHEMA,
        "schema_version": CUT3R_STRATA_RECORDS_VERSION,
        "comparison_lock_id": comparison["lock_id"],
        "strata_lock_id": lock["strata_lock_id"],
        "record_definition_sha256": lock["record_definition_sha256"],
        "source_truth_used": source_truth_used,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "records": normalized_records,
    }
    return cast(dict[str, Any], json.loads(_canonical_json(normalized)))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty value sequence")
    return math.fsum(values) / len(values)


def _bin_index(value: float, edges: Sequence[float]) -> int:
    index = bisect.bisect_right(edges, value) - 1
    if index < 0:
        raise ValueError("diagnostic feature lies below the first frozen bin edge")
    return index


def _build_strata_results(
    comparison: Mapping[str, Any],
    lock: Mapping[str, Any],
    records: Mapping[str, Any],
) -> list[dict[str, Any]]:
    enabled_arms = [
        cast(str, arm["arm_id"])
        for arm in comparison["arms"]
        if arm["enabled"] and arm["causal"] and arm["claim_eligible"]
    ]
    contrasts = [
        cast(dict[str, Any], contrast)
        for contrast in comparison["registered_contrasts"]
        if contrast["enabled"] and contrast["claim_eligible"]
    ]
    metric_names = cast(list[str], lock["metric_names"])
    minimum_groups = cast(int, lock["minimum_evaluable_groups_per_bin"])
    normalized_records = cast(list[dict[str, Any]], records["records"])

    results: list[dict[str, Any]] = []
    for stratum in lock["strata"]:
        stratum_id = cast(str, stratum["stratum_id"])
        feature_name = cast(str, stratum["feature_name"])
        edges = cast(list[float], stratum["bin_edges"])
        values: dict[tuple[int, str, str, int, str, str], list[float]] = defaultdict(list)
        observation_keys: dict[tuple[int, str], set[tuple[str, str, int, int]]] = (
            defaultdict(set)
        )
        for record in normalized_records:
            bin_index = _bin_index(cast(float, record["features"][feature_name]), edges)
            group_id = cast(str, record["group_id"])
            case_id = cast(str, record["case_id"])
            random_seed = cast(int, record["random_seed"])
            arm_id = cast(str, record["arm_id"])
            observation_keys[(bin_index, group_id)].add(
                (
                    group_id,
                    case_id,
                    cast(int, record["frame_index"]),
                    random_seed,
                )
            )
            for metric_name in metric_names:
                values[
                    (bin_index, group_id, case_id, random_seed, arm_id, metric_name)
                ].append(
                    cast(float, record["metrics"][metric_name])
                )

        bins: list[dict[str, Any]] = []
        for bin_index, lower in enumerate(edges):
            upper = edges[bin_index + 1] if bin_index + 1 < len(edges) else None
            group_ids = sorted(
                group_id
                for candidate_bin, group_id in observation_keys
                if candidate_bin == bin_index
            )
            group_results: list[dict[str, Any]] = []
            group_metric_lookup: dict[tuple[str, str, str], float] = {}
            for group_id in group_ids:
                case_ids = sorted(
                    {
                        case_id
                        for _, case_id, _, _ in observation_keys[(bin_index, group_id)]
                    }
                )
                arm_metrics: list[dict[str, Any]] = []
                for arm_id in enabled_arms:
                    metric_means = {
                        metric_name: _mean(
                            [
                                _mean(
                                    [
                                        _mean(
                                            values[
                                                (
                                                    bin_index,
                                                    group_id,
                                                    case_id,
                                                    random_seed,
                                                    arm_id,
                                                    metric_name,
                                                )
                                            ]
                                        )
                                        for random_seed in comparison["random_seeds"]
                                    ]
                                )
                                for case_id in case_ids
                            ]
                        )
                        for metric_name in metric_names
                    }
                    for metric_name, metric_value in metric_means.items():
                        group_metric_lookup[(group_id, arm_id, metric_name)] = metric_value
                    arm_metrics.append({"arm_id": arm_id, "metrics": metric_means})
                group_results.append(
                    {
                        "group_id": group_id,
                        "nested_observation_count": len(
                            observation_keys[(bin_index, group_id)]
                        ),
                        "evaluable_case_count": len(case_ids),
                        "evaluable_seed_count": len(comparison["random_seeds"]),
                        "arm_metrics": arm_metrics,
                    }
                )

            arm_results: list[dict[str, Any]] = []
            contrast_results: list[dict[str, Any]] = []
            if group_ids:
                for arm_id in enabled_arms:
                    arm_results.append(
                        {
                            "arm_id": arm_id,
                            "evaluable_group_count": len(group_ids),
                            "equal_group_mean_metrics": {
                                metric_name: _mean(
                                    [
                                        group_metric_lookup[(group_id, arm_id, metric_name)]
                                        for group_id in group_ids
                                    ]
                                )
                                for metric_name in metric_names
                            },
                        }
                    )
                for contrast in contrasts:
                    treatment = cast(str, contrast["treatment_arm"])
                    control = cast(str, contrast["control_arm"])
                    contrast_results.append(
                        {
                            "contrast_id": contrast["contrast_id"],
                            "treatment_arm": treatment,
                            "control_arm": control,
                            "paired_group_count": len(group_ids),
                            "equal_group_mean_delta_metrics": {
                                metric_name: _mean(
                                    [
                                        group_metric_lookup[
                                            (group_id, treatment, metric_name)
                                        ]
                                        - group_metric_lookup[
                                            (group_id, control, metric_name)
                                        ]
                                        for group_id in group_ids
                                    ]
                                )
                                for metric_name in metric_names
                            },
                        }
                    )

            bins.append(
                {
                    "bin_index": bin_index,
                    "lower_inclusive": lower,
                    "upper_exclusive": upper,
                    "evaluable_group_count": len(group_ids),
                    "meets_minimum_evaluable_groups": len(group_ids) >= minimum_groups,
                    "nested_observation_count": sum(
                        len(observation_keys[(bin_index, group_id)])
                        for group_id in group_ids
                    ),
                    "group_results": group_results,
                    "arm_results": arm_results,
                    "contrast_results": contrast_results,
                }
            )
        results.append(
            {
                "stratum_id": stratum_id,
                "feature_name": feature_name,
                "unit": stratum["unit"],
                "bin_edges": edges,
                "bins": bins,
            }
        )
    return results


def build_cut3r_diagnostic_strata_report(
    comparison_lock: Any,
    strata_lock: Any,
    records: Any,
) -> dict[str, Any]:
    """Build an equal-group, paired-arm source diagnostic report."""

    comparison = validate_cut3r_comparison_lock(comparison_lock)
    lock = validate_cut3r_diagnostic_strata_lock(comparison, strata_lock)
    normalized_records = _normalize_records(comparison, lock, records)
    records_id = _record_id(normalized_records)
    observation_count = len(normalized_records["records"]) // len(
        [
            arm
            for arm in comparison["arms"]
            if arm["enabled"] and arm["causal"] and arm["claim_eligible"]
        ]
    )
    payload: dict[str, Any] = {
        "schema": CUT3R_STRATA_REPORT_SCHEMA,
        "schema_version": CUT3R_STRATA_REPORT_VERSION,
        "comparison_lock_id": comparison["lock_id"],
        "strata_lock_id": lock["strata_lock_id"],
        "records_id": records_id,
        "record_count": len(normalized_records["records"]),
        "nested_observation_count": observation_count,
        "source_truth_used": normalized_records["source_truth_used"],
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "group_unit": CUT3R_COMPARISON_GROUP_UNIT,
        "random_seeds": comparison["random_seeds"],
        "record_definition_sha256": lock["record_definition_sha256"],
        "minimum_evaluable_groups_per_bin": lock[
            "minimum_evaluable_groups_per_bin"
        ],
        "metric_names": lock["metric_names"],
        "weighting": dict(CUT3R_STRATA_WEIGHTING),
        "strata_results": _build_strata_results(
            comparison,
            lock,
            normalized_records,
        ),
        "selection_role": "reporting-only",
        "source_access": "source-only",
        "target_access": "forbidden",
        "claim_boundary": CUT3R_STRATA_CLAIM_BOUNDARY,
    }
    payload["report_id"] = _record_id(payload)
    return cast(dict[str, Any], json.loads(_canonical_json(payload)))


def validate_cut3r_diagnostic_strata_report(
    comparison_lock: Any,
    strata_lock: Any,
    records: Any,
    value: Any,
) -> dict[str, Any]:
    """Rebuild a report from records and require byte-equivalent semantics."""

    payload = _strict_mapping(value, name="CUT3R diagnostic strata report")
    _exact_keys(payload, _REPORT_FIELDS, name="CUT3R diagnostic strata report")
    if _strict_string(payload["schema"], name="schema") != CUT3R_STRATA_REPORT_SCHEMA:
        raise ValueError("unsupported CUT3R diagnostic strata report schema")
    if (
        _strict_integer(payload["schema_version"], name="schema_version", minimum=1)
        != CUT3R_STRATA_REPORT_VERSION
    ):
        raise ValueError("unsupported CUT3R diagnostic strata report version")
    expected = build_cut3r_diagnostic_strata_report(
        comparison_lock,
        strata_lock,
        records,
    )
    normalized = cast(dict[str, Any], json.loads(_canonical_json(payload)))
    if normalized != expected:
        raise ValueError("diagnostic strata report does not match the bound records")
    return expected


def load_cut3r_diagnostic_strata_report(
    comparison_lock: Any,
    strata_lock: Any,
    records: Any,
    path: str | Path,
) -> dict[str, Any]:
    return validate_cut3r_diagnostic_strata_report(
        comparison_lock,
        strata_lock,
        records,
        _strict_json(path),
    )


def write_cut3r_diagnostic_strata_report(
    comparison_lock: Any,
    strata_lock: Any,
    records: Any,
    path: str | Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    payload = validate_cut3r_diagnostic_strata_report(
        comparison_lock,
        strata_lock,
        records,
        report,
    )
    return _publish_json(
        path,
        payload,
        load_existing=lambda existing: load_cut3r_diagnostic_strata_report(
            comparison_lock,
            strata_lock,
            records,
            existing,
        ),
    )


def cut3r_diagnostic_strata_summary(report: Any) -> dict[str, Any]:
    payload = _strict_mapping(report, name="CUT3R diagnostic strata report")
    _exact_keys(payload, _REPORT_FIELDS, name="CUT3R diagnostic strata report")
    if _strict_string(payload["schema"], name="schema") != CUT3R_STRATA_REPORT_SCHEMA:
        raise ValueError("unsupported CUT3R diagnostic strata report schema")
    if (
        _strict_integer(payload["schema_version"], name="schema_version", minimum=1)
        != CUT3R_STRATA_REPORT_VERSION
    ):
        raise ValueError("unsupported CUT3R diagnostic strata report version")
    report_id = _sha256(payload["report_id"], name="report_id")
    unsigned = dict(payload)
    unsigned.pop("report_id")
    if report_id != _record_id(unsigned):
        raise ValueError("report_id does not match the diagnostic report content")
    if _strict_boolean(payload["target_payloads_opened"], name="target_payloads_opened"):
        raise ValueError("diagnostic report may not open target payloads")
    if _strict_boolean(payload["target_outcomes_opened"], name="target_outcomes_opened"):
        raise ValueError("diagnostic report may not open target outcomes")
    if _strict_string(payload["source_access"], name="source_access") != "source-only":
        raise ValueError("diagnostic report must remain source-only")
    if _strict_string(payload["target_access"], name="target_access") != "forbidden":
        raise ValueError("diagnostic report cannot authorize target access")
    if _strict_string(payload["selection_role"], name="selection_role") != "reporting-only":
        raise ValueError("diagnostic report cannot select a method")
    if _strict_string(payload["claim_boundary"], name="claim_boundary") != (
        CUT3R_STRATA_CLAIM_BOUNDARY
    ):
        raise ValueError("diagnostic report claim boundary changed")
    if _strict_string(payload["group_unit"], name="group_unit") != (
        CUT3R_COMPARISON_GROUP_UNIT
    ):
        raise ValueError("diagnostic report group unit changed")
    if payload["weighting"] != CUT3R_STRATA_WEIGHTING:
        raise ValueError("diagnostic report weighting changed")
    random_seeds = _canonical_random_seeds(
        payload["random_seeds"],
        name="random_seeds",
    )
    record_definition_sha256 = _sha256(
        payload["record_definition_sha256"],
        name="record_definition_sha256",
    )
    strata_results_value = payload["strata_results"]
    if type(strata_results_value) is not list:
        raise ValueError("strata_results must be a JSON array")
    strata_results = cast(list[dict[str, Any]], strata_results_value)
    passing_bins = sum(
        1
        for stratum in strata_results
        for bin_result in stratum["bins"]
        if bin_result["meets_minimum_evaluable_groups"]
    )
    populated_bins = sum(
        1
        for stratum in strata_results
        for bin_result in stratum["bins"]
        if bin_result["evaluable_group_count"] > 0
    )
    return {
        "report_id": report_id,
        "comparison_lock_id": _sha256(
            payload["comparison_lock_id"],
            name="comparison_lock_id",
        ),
        "strata_lock_id": _sha256(payload["strata_lock_id"], name="strata_lock_id"),
        "record_count": _strict_integer(payload["record_count"], name="record_count"),
        "nested_observation_count": _strict_integer(
            payload["nested_observation_count"],
            name="nested_observation_count",
        ),
        "record_definition_sha256": record_definition_sha256,
        "random_seed_count": len(random_seeds),
        "stratum_count": len(strata_results),
        "populated_bin_count": populated_bins,
        "adequately_supported_bin_count": passing_bins,
        "target_access": _strict_string(payload["target_access"], name="target_access"),
        "selection_role": _strict_string(
            payload["selection_role"],
            name="selection_role",
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction cut3r-strata",
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze diagnostic strata")
    freeze.add_argument("comparison_lock")
    freeze.add_argument("specification")
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(handler=_freeze_command)

    verify_lock = subparsers.add_parser("verify-lock", help="verify a strata lock")
    verify_lock.add_argument("comparison_lock")
    verify_lock.add_argument("strata_lock")
    verify_lock.set_defaults(handler=_verify_lock_command)

    report = subparsers.add_parser("report", help="build an equal-group strata report")
    report.add_argument("comparison_lock")
    report.add_argument("strata_lock")
    report.add_argument("records")
    report.add_argument("--output", required=True)
    report.set_defaults(handler=_report_command)

    verify_report = subparsers.add_parser("verify-report", help="rebuild and verify a report")
    verify_report.add_argument("comparison_lock")
    verify_report.add_argument("strata_lock")
    verify_report.add_argument("records")
    verify_report.add_argument("report")
    verify_report.set_defaults(handler=_verify_report_command)

    summarize = subparsers.add_parser("summarize", help="summarize a verified report")
    summarize.add_argument("comparison_lock")
    summarize.add_argument("strata_lock")
    summarize.add_argument("records")
    summarize.add_argument("report")
    summarize.add_argument("--json", action="store_true")
    summarize.set_defaults(handler=_summarize_command)
    return parser


def _load_comparison(path: str | Path) -> dict[str, Any]:
    return load_cut3r_comparison_lock(path)


def _freeze_command(arguments: argparse.Namespace) -> int:
    comparison = _load_comparison(arguments.comparison_lock)
    lock = build_cut3r_diagnostic_strata_lock(
        comparison,
        _strict_json(arguments.specification),
    )
    write_cut3r_diagnostic_strata_lock(comparison, arguments.output, lock)
    print(lock["strata_lock_id"])
    return 0


def _verify_lock_command(arguments: argparse.Namespace) -> int:
    comparison = _load_comparison(arguments.comparison_lock)
    lock = load_cut3r_diagnostic_strata_lock(comparison, arguments.strata_lock)
    print(lock["strata_lock_id"])
    return 0


def _report_command(arguments: argparse.Namespace) -> int:
    comparison = _load_comparison(arguments.comparison_lock)
    lock = load_cut3r_diagnostic_strata_lock(comparison, arguments.strata_lock)
    records = _strict_json(arguments.records)
    report = build_cut3r_diagnostic_strata_report(comparison, lock, records)
    write_cut3r_diagnostic_strata_report(
        comparison,
        lock,
        records,
        arguments.output,
        report,
    )
    print(report["report_id"])
    return 0


def _verify_report_command(arguments: argparse.Namespace) -> int:
    comparison = _load_comparison(arguments.comparison_lock)
    lock = load_cut3r_diagnostic_strata_lock(comparison, arguments.strata_lock)
    records = _strict_json(arguments.records)
    report = load_cut3r_diagnostic_strata_report(
        comparison,
        lock,
        records,
        arguments.report,
    )
    print(report["report_id"])
    return 0


def _summarize_command(arguments: argparse.Namespace) -> int:
    comparison = _load_comparison(arguments.comparison_lock)
    lock = load_cut3r_diagnostic_strata_lock(comparison, arguments.strata_lock)
    records = _strict_json(arguments.records)
    report = load_cut3r_diagnostic_strata_report(
        comparison,
        lock,
        records,
        arguments.report,
    )
    summary = cut3r_diagnostic_strata_summary(report)
    if arguments.json:
        print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    else:
        print(f"report_id: {summary['report_id']}")
        print(f"nested observations: {summary['nested_observation_count']}")
        print(f"populated bins: {summary['populated_bin_count']}")
        print(
            "adequately supported bins: "
            f"{summary['adequately_supported_bin_count']}"
        )
        print(f"selection role: {summary['selection_role']}")
        print(f"target access: {summary['target_access']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    return int(arguments.handler(arguments))


__all__ = [
    "CUT3R_DIAGNOSTIC_STRATA",
    "CUT3R_STRATA_CLAIM_BOUNDARY",
    "CUT3R_STRATA_LOCK_SCHEMA",
    "CUT3R_STRATA_LOCK_VERSION",
    "CUT3R_STRATA_RECORDS_SCHEMA",
    "CUT3R_STRATA_RECORDS_VERSION",
    "CUT3R_STRATA_REPORT_SCHEMA",
    "CUT3R_STRATA_REPORT_VERSION",
    "CUT3R_STRATA_WEIGHTING",
    "build_cut3r_diagnostic_strata_lock",
    "build_cut3r_diagnostic_strata_report",
    "cut3r_diagnostic_strata_summary",
    "load_cut3r_diagnostic_strata_lock",
    "load_cut3r_diagnostic_strata_report",
    "main",
    "validate_cut3r_diagnostic_strata_lock",
    "validate_cut3r_diagnostic_strata_report",
    "write_cut3r_diagnostic_strata_lock",
    "write_cut3r_diagnostic_strata_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
