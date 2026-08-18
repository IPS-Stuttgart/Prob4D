"""Freeze a group-aware CUT3R native-versus-Prob4D comparison.

The lock isolates the value of Prob4D fusion from CUT3R's own recurrent state.
It is source-only, outcome-blind, and treats complete physical objects or
acquisition sessions—not frames—as independent evidence units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from ._atomic_file import atomic_write_bytes
from .project_identity import PROB4D_PROJECT_ID

CUT3R_COMPARISON_SCHEMA: Final = "prob4d.cut3r-comparison-lock"
CUT3R_COMPARISON_VERSION: Final = 1
CUT3R_REPOSITORY: Final = "CUT3R/CUT3R"
CUT3R_COMPARISON_GROUP_UNIT: Final = "physical-object-or-acquisition-session"
CUT3R_COMPARISON_CLAIM_BOUNDARY: Final = (
    "This lock freezes a source-only provider comparison. It does not establish "
    "held-out provider competence, BayesianPhysTwin benefit, Causal4D intervention "
    "benefit, deployment safety, or state of the art. Frames, points, tracks, and "
    "views remain nested observations inside complete object/session groups."
)
CUT3R_PROVIDER_ENDPOINTS: Final[tuple[str, ...]] = (
    "support-rate",
    "technical-failure-rate",
    "point-or-track-error",
    "seam-error",
    "drift-error",
    "coverage-50",
    "coverage-90",
    "coverage-95",
    "proper-score",
    "normalized-nees",
    "full-covariance-width",
    "identity-retention",
    "selective-risk",
    "worst-group-coverage-shortfall",
)

_LOCK_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "protocol_name",
        "provider",
        "prob4d",
        "windowing",
        "group_unit",
        "groups",
        "group_roles",
        "random_seeds",
        "arms",
        "registered_contrasts",
        "provider_endpoints",
        "weighting",
        "source_access",
        "target_access",
        "claim_boundary",
        "lock_id",
    }
)
_SPEC_FIELDS: Final = frozenset(
    {
        "protocol_name",
        "provider_revision",
        "checkpoint_sha256",
        "prob4d_revision",
        "prob4d_distribution_sha256",
        "window_size",
        "overlap",
        "confidence_threshold",
        "storage_dtype",
        "random_seeds",
        "groups",
        "group_roles",
        "include_revisit_diagnostic",
    }
)
_PROVIDER_FIELDS: Final = frozenset(
    {"repository", "revision", "checkpoint_sha256"}
)
_PROB4D_FIELDS: Final = frozenset(
    {"project_id", "revision", "distribution_sha256"}
)
_WINDOW_FIELDS: Final = frozenset(
    {
        "window_size",
        "overlap",
        "stride",
        "confidence_threshold",
        "storage_dtype",
        "restart_policy",
    }
)
_GROUP_FIELDS: Final = frozenset({"group_id", "cases"})
_CASE_FIELDS: Final = frozenset(
    {
        "case_id",
        "input_video_sha256",
        "input_video_byte_count",
        "frame_start",
        "frame_stop_exclusive",
        "evaluation_frame_start",
        "evaluation_frame_stop_exclusive",
    }
)
_ROLE_NAMES: Final = ("development", "calibration", "source_evaluation")
_WEIGHTING: Final = {
    "within_group": "endpoint-specific-nested-observations-v1",
    "across_groups": "equal-complete-group-weight-v1",
}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
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


def _exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], *, name: str) -> None:
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


def _sha256(value: Any, *, name: str) -> str:
    digest = _strict_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _revision(value: Any, *, name: str) -> str:
    revision = _strict_string(value, name=name)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError(f"{name} must be an exact lowercase 40-character Git revision")
    return revision


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a genuine finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _canonical_arms(include_revisit: bool) -> list[dict[str, Any]]:
    return [
        {
            "arm_id": "native-continuous",
            "execution_mode": "recurrent-online-continuous-v1",
            "causal": True,
            "claim_eligible": True,
            "enabled": True,
        },
        {
            "arm_id": "restarted-newest",
            "execution_mode": "fresh-state-overlapping-windows-newest-only-v1",
            "causal": True,
            "claim_eligible": True,
            "enabled": True,
        },
        {
            "arm_id": "restarted-prob4d-fused",
            "execution_mode": "fresh-state-overlapping-windows-prob4d-fusion-v1",
            "causal": True,
            "claim_eligible": True,
            "enabled": True,
        },
        {
            "arm_id": "revisit-diagnostic",
            "execution_mode": "provider-revisit-noncausal-v1",
            "causal": False,
            "claim_eligible": False,
            "enabled": include_revisit,
        },
    ]


def _canonical_contrasts(include_revisit: bool) -> list[dict[str, Any]]:
    return [
        {
            "contrast_id": "prob4d-fusion-value",
            "treatment_arm": "restarted-prob4d-fused",
            "control_arm": "restarted-newest",
            "claim_eligible": True,
            "enabled": True,
        },
        {
            "contrast_id": "provider-recurrence-value",
            "treatment_arm": "native-continuous",
            "control_arm": "restarted-newest",
            "claim_eligible": True,
            "enabled": True,
        },
        {
            "contrast_id": "noncausal-revisit-upper-bound",
            "treatment_arm": "revisit-diagnostic",
            "control_arm": "native-continuous",
            "claim_eligible": False,
            "enabled": include_revisit,
        },
    ]


def _normalize_cases(value: Any, *, group_id: str) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise ValueError(f"group {group_id!r} must contain a nonempty cases array")
    cases: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, raw_case in enumerate(value):
        case = _strict_mapping(raw_case, name=f"groups[{group_id}].cases[{index}]")
        _exact_keys(case, _CASE_FIELDS, name=f"groups[{group_id}].cases[{index}]")
        case_id = _strict_string(case["case_id"], name="case_id")
        if case_id in identifiers:
            raise ValueError(f"group {group_id!r} repeats case_id {case_id!r}")
        identifiers.add(case_id)
        frame_start = _strict_integer(case["frame_start"], name=f"{case_id}.frame_start")
        frame_stop = _strict_integer(
            case["frame_stop_exclusive"],
            name=f"{case_id}.frame_stop_exclusive",
            minimum=1,
        )
        evaluation_start = _strict_integer(
            case["evaluation_frame_start"],
            name=f"{case_id}.evaluation_frame_start",
        )
        evaluation_stop = _strict_integer(
            case["evaluation_frame_stop_exclusive"],
            name=f"{case_id}.evaluation_frame_stop_exclusive",
            minimum=1,
        )
        if frame_stop <= frame_start:
            raise ValueError(f"case {case_id!r} has an empty source interval")
        if not (
            frame_start <= evaluation_start < evaluation_stop <= frame_stop
        ):
            raise ValueError(
                f"case {case_id!r} evaluation interval must lie inside its source interval"
            )
        cases.append(
            {
                "case_id": case_id,
                "input_video_sha256": _sha256(
                    case["input_video_sha256"],
                    name=f"{case_id}.input_video_sha256",
                ),
                "input_video_byte_count": _strict_integer(
                    case["input_video_byte_count"],
                    name=f"{case_id}.input_video_byte_count",
                    minimum=1,
                ),
                "frame_start": frame_start,
                "frame_stop_exclusive": frame_stop,
                "evaluation_frame_start": evaluation_start,
                "evaluation_frame_stop_exclusive": evaluation_stop,
            }
        )
    return sorted(cases, key=lambda item: cast(str, item["case_id"]))


def _normalize_groups(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) < 3:
        raise ValueError("groups must contain at least three independent complete groups")
    groups: list[dict[str, Any]] = []
    group_ids: set[str] = set()
    case_ids: set[str] = set()
    for index, raw_group in enumerate(value):
        group = _strict_mapping(raw_group, name=f"groups[{index}]")
        _exact_keys(group, _GROUP_FIELDS, name=f"groups[{index}]")
        group_id = _strict_string(group["group_id"], name=f"groups[{index}].group_id")
        if group_id in group_ids:
            raise ValueError(f"duplicate group_id {group_id!r}")
        group_ids.add(group_id)
        cases = _normalize_cases(group["cases"], group_id=group_id)
        for case in cases:
            case_id = cast(str, case["case_id"])
            if case_id in case_ids:
                raise ValueError(f"case_id {case_id!r} appears in multiple groups")
            case_ids.add(case_id)
        groups.append({"group_id": group_id, "cases": cases})
    return sorted(groups, key=lambda item: cast(str, item["group_id"]))


def _normalize_group_roles(value: Any, *, group_ids: set[str]) -> dict[str, list[str]]:
    roles = _strict_mapping(value, name="group_roles")
    _exact_keys(roles, set(_ROLE_NAMES), name="group_roles")
    normalized: dict[str, list[str]] = {}
    assigned: list[str] = []
    for role in _ROLE_NAMES:
        raw_ids = roles[role]
        if type(raw_ids) is not list or not raw_ids:
            raise ValueError(f"group_roles.{role} must be a nonempty JSON array")
        identifiers = [
            _strict_string(item, name=f"group_roles.{role}[{index}]")
            for index, item in enumerate(raw_ids)
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"group_roles.{role} must not repeat group IDs")
        normalized[role] = sorted(identifiers)
        assigned.extend(identifiers)
    if len(assigned) != len(set(assigned)):
        raise ValueError("development, calibration, and source-evaluation groups must be disjoint")
    if set(assigned) != group_ids:
        missing = sorted(group_ids - set(assigned))
        unknown = sorted(set(assigned) - group_ids)
        raise ValueError(
            f"group_roles must partition every group exactly once; missing={missing}, unknown={unknown}"
        )
    return normalized


def _normalize_seeds(value: Any) -> list[int]:
    if type(value) is not list or not value:
        raise ValueError("random_seeds must be a nonempty JSON array")
    seeds = [
        _strict_integer(item, name=f"random_seeds[{index}]")
        for index, item in enumerate(value)
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("random_seeds must be unique")
    return sorted(seeds)


def build_cut3r_comparison_lock(specification: Any) -> dict[str, Any]:
    """Build a canonical source-only comparison lock from an outcome-blind spec."""

    spec = _strict_mapping(specification, name="CUT3R comparison specification")
    _exact_keys(spec, _SPEC_FIELDS, name="CUT3R comparison specification")
    protocol_name = _strict_string(spec["protocol_name"], name="protocol_name")
    groups = _normalize_groups(spec["groups"])
    group_ids = {cast(str, group["group_id"]) for group in groups}
    group_roles = _normalize_group_roles(spec["group_roles"], group_ids=group_ids)
    random_seeds = _normalize_seeds(spec["random_seeds"])

    window_size = _strict_integer(spec["window_size"], name="window_size", minimum=2)
    overlap = _strict_integer(spec["overlap"], name="overlap", minimum=1)
    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size")
    storage_dtype = _strict_string(spec["storage_dtype"], name="storage_dtype")
    if storage_dtype not in {"float32", "float64"}:
        raise ValueError("storage_dtype must be float32 or float64")
    include_revisit = _strict_boolean(
        spec["include_revisit_diagnostic"],
        name="include_revisit_diagnostic",
    )

    payload: dict[str, Any] = {
        "schema": CUT3R_COMPARISON_SCHEMA,
        "schema_version": CUT3R_COMPARISON_VERSION,
        "protocol_name": protocol_name,
        "provider": {
            "repository": CUT3R_REPOSITORY,
            "revision": _revision(spec["provider_revision"], name="provider_revision"),
            "checkpoint_sha256": _sha256(
                spec["checkpoint_sha256"],
                name="checkpoint_sha256",
            ),
        },
        "prob4d": {
            "project_id": PROB4D_PROJECT_ID,
            "revision": _revision(spec["prob4d_revision"], name="prob4d_revision"),
            "distribution_sha256": _sha256(
                spec["prob4d_distribution_sha256"],
                name="prob4d_distribution_sha256",
            ),
        },
        "windowing": {
            "window_size": window_size,
            "overlap": overlap,
            "stride": window_size - overlap,
            "confidence_threshold": _finite_nonnegative(
                spec["confidence_threshold"],
                name="confidence_threshold",
            ),
            "storage_dtype": storage_dtype,
            "restart_policy": "fresh-state-per-window-v1",
        },
        "group_unit": CUT3R_COMPARISON_GROUP_UNIT,
        "groups": groups,
        "group_roles": group_roles,
        "random_seeds": random_seeds,
        "arms": _canonical_arms(include_revisit),
        "registered_contrasts": _canonical_contrasts(include_revisit),
        "provider_endpoints": list(CUT3R_PROVIDER_ENDPOINTS),
        "weighting": dict(_WEIGHTING),
        "source_access": "source-only",
        "target_access": "forbidden",
        "claim_boundary": CUT3R_COMPARISON_CLAIM_BOUNDARY,
    }
    payload["lock_id"] = _record_id(payload)
    return validate_cut3r_comparison_lock(payload)


def validate_cut3r_comparison_lock(value: Any) -> dict[str, Any]:
    """Strictly validate a persisted CUT3R comparison lock."""

    payload = _strict_mapping(value, name="CUT3R comparison lock")
    _exact_keys(payload, _LOCK_FIELDS, name="CUT3R comparison lock")
    if _strict_string(payload["schema"], name="schema") != CUT3R_COMPARISON_SCHEMA:
        raise ValueError("unsupported CUT3R comparison schema")
    if (
        _strict_integer(payload["schema_version"], name="schema_version", minimum=1)
        != CUT3R_COMPARISON_VERSION
    ):
        raise ValueError("unsupported CUT3R comparison schema version")
    protocol_name = _strict_string(payload["protocol_name"], name="protocol_name")

    provider = _strict_mapping(payload["provider"], name="provider")
    _exact_keys(provider, _PROVIDER_FIELDS, name="provider")
    if _strict_string(provider["repository"], name="provider.repository") != CUT3R_REPOSITORY:
        raise ValueError("CUT3R comparison provider repository is not canonical")
    normalized_provider = {
        "repository": CUT3R_REPOSITORY,
        "revision": _revision(provider["revision"], name="provider.revision"),
        "checkpoint_sha256": _sha256(
            provider["checkpoint_sha256"],
            name="provider.checkpoint_sha256",
        ),
    }

    prob4d = _strict_mapping(payload["prob4d"], name="prob4d")
    _exact_keys(prob4d, _PROB4D_FIELDS, name="prob4d")
    if _strict_string(prob4d["project_id"], name="prob4d.project_id") != PROB4D_PROJECT_ID:
        raise ValueError("CUT3R comparison Prob4D project identity changed")
    normalized_prob4d = {
        "project_id": PROB4D_PROJECT_ID,
        "revision": _revision(prob4d["revision"], name="prob4d.revision"),
        "distribution_sha256": _sha256(
            prob4d["distribution_sha256"],
            name="prob4d.distribution_sha256",
        ),
    }

    windowing = _strict_mapping(payload["windowing"], name="windowing")
    _exact_keys(windowing, _WINDOW_FIELDS, name="windowing")
    window_size = _strict_integer(windowing["window_size"], name="window_size", minimum=2)
    overlap = _strict_integer(windowing["overlap"], name="overlap", minimum=1)
    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size")
    stride = _strict_integer(windowing["stride"], name="stride", minimum=1)
    if stride != window_size - overlap:
        raise ValueError("stride must equal window_size minus overlap")
    storage_dtype = _strict_string(windowing["storage_dtype"], name="storage_dtype")
    if storage_dtype not in {"float32", "float64"}:
        raise ValueError("storage_dtype must be float32 or float64")
    restart_policy = _strict_string(windowing["restart_policy"], name="restart_policy")
    if restart_policy != "fresh-state-per-window-v1":
        raise ValueError("CUT3R restart policy is not canonical")
    normalized_windowing = {
        "window_size": window_size,
        "overlap": overlap,
        "stride": stride,
        "confidence_threshold": _finite_nonnegative(
            windowing["confidence_threshold"],
            name="confidence_threshold",
        ),
        "storage_dtype": storage_dtype,
        "restart_policy": restart_policy,
    }

    if _strict_string(payload["group_unit"], name="group_unit") != CUT3R_COMPARISON_GROUP_UNIT:
        raise ValueError("independent evidence unit must be the complete object/session group")
    groups = _normalize_groups(payload["groups"])
    group_ids = {cast(str, group["group_id"]) for group in groups}
    group_roles = _normalize_group_roles(payload["group_roles"], group_ids=group_ids)
    random_seeds = _normalize_seeds(payload["random_seeds"])

    raw_arms = payload["arms"]
    if type(raw_arms) is not list or len(raw_arms) != 4:
        raise ValueError("CUT3R comparison must retain exactly four declared arms")
    revisit = _strict_mapping(raw_arms[3], name="arms[3]")
    include_revisit = _strict_boolean(revisit.get("enabled"), name="revisit enabled")
    expected_arms = _canonical_arms(include_revisit)
    if raw_arms != expected_arms:
        raise ValueError("CUT3R comparison arms changed from the frozen causal contract")
    expected_contrasts = _canonical_contrasts(include_revisit)
    if payload["registered_contrasts"] != expected_contrasts:
        raise ValueError("CUT3R registered contrasts changed from the frozen contract")
    if payload["provider_endpoints"] != list(CUT3R_PROVIDER_ENDPOINTS):
        raise ValueError("CUT3R provider endpoints changed from the frozen contract")
    if payload["weighting"] != _WEIGHTING:
        raise ValueError("CUT3R weighting must give complete groups equal weight")
    if _strict_string(payload["source_access"], name="source_access") != "source-only":
        raise ValueError("CUT3R comparison lock must remain source-only")
    if _strict_string(payload["target_access"], name="target_access") != "forbidden":
        raise ValueError("CUT3R comparison lock may not authorize target access")
    claim_boundary = _strict_string(payload["claim_boundary"], name="claim_boundary")
    if claim_boundary != CUT3R_COMPARISON_CLAIM_BOUNDARY:
        raise ValueError("CUT3R comparison claim boundary is not canonical")

    lock_id = _sha256(payload["lock_id"], name="lock_id")
    normalized: dict[str, Any] = {
        "schema": CUT3R_COMPARISON_SCHEMA,
        "schema_version": CUT3R_COMPARISON_VERSION,
        "protocol_name": protocol_name,
        "provider": normalized_provider,
        "prob4d": normalized_prob4d,
        "windowing": normalized_windowing,
        "group_unit": CUT3R_COMPARISON_GROUP_UNIT,
        "groups": groups,
        "group_roles": group_roles,
        "random_seeds": random_seeds,
        "arms": expected_arms,
        "registered_contrasts": expected_contrasts,
        "provider_endpoints": list(CUT3R_PROVIDER_ENDPOINTS),
        "weighting": dict(_WEIGHTING),
        "source_access": "source-only",
        "target_access": "forbidden",
        "claim_boundary": claim_boundary,
        "lock_id": lock_id,
    }
    unsigned = dict(normalized)
    unsigned.pop("lock_id")
    if lock_id != _record_id(unsigned):
        raise ValueError("lock_id does not match the canonical comparison content")
    return cast(dict[str, Any], json.loads(_canonical_json(normalized)))


def load_cut3r_comparison_lock(path: str | Path) -> dict[str, Any]:
    """Load strict JSON and validate a CUT3R comparison lock."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    return validate_cut3r_comparison_lock(payload)


def write_cut3r_comparison_lock(
    path: str | Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish one no-clobber comparison lock, allowing idempotent writes."""

    payload = validate_cut3r_comparison_lock(lock)
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("CUT3R comparison lock destination must not be a symbolic link")
    encoded = _canonical_json(payload) + b"\n"
    try:
        atomic_write_bytes(destination, encoded, overwrite=False)
    except FileExistsError:
        existing = load_cut3r_comparison_lock(destination)
        if existing != payload:
            raise FileExistsError(
                f"refusing to replace a different CUT3R comparison lock: {destination}"
            ) from None
        return existing
    return payload


def cut3r_comparison_summary(lock: Any) -> dict[str, Any]:
    """Return a compact scientific-boundary summary."""

    payload = validate_cut3r_comparison_lock(lock)
    return {
        "lock_id": payload["lock_id"],
        "protocol_name": payload["protocol_name"],
        "independent_group_count": len(payload["groups"]),
        "case_count": sum(len(group["cases"]) for group in payload["groups"]),
        "group_role_counts": {
            role: len(payload["group_roles"][role]) for role in _ROLE_NAMES
        },
        "enabled_arms": [arm["arm_id"] for arm in payload["arms"] if arm["enabled"]],
        "claim_eligible_contrasts": [
            contrast["contrast_id"]
            for contrast in payload["registered_contrasts"]
            if contrast["enabled"] and contrast["claim_eligible"]
        ],
        "source_access": payload["source_access"],
        "target_access": payload["target_access"],
        "group_unit": payload["group_unit"],
    }


def _load_specification(path: str | Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    return _strict_mapping(payload, name="CUT3R comparison specification")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction cut3r-comparison",
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="freeze a source-only comparison")
    build.add_argument("specification")
    build.add_argument("--output", required=True)
    build.set_defaults(handler=_build_command)

    verify = subparsers.add_parser("verify", help="verify a frozen comparison lock")
    verify.add_argument("lock")
    verify.set_defaults(handler=_verify_command)

    summarize = subparsers.add_parser("summarize", help="summarize a comparison lock")
    summarize.add_argument("lock")
    summarize.add_argument("--json", action="store_true")
    summarize.set_defaults(handler=_summarize_command)
    return parser


def _build_command(arguments: argparse.Namespace) -> int:
    lock = build_cut3r_comparison_lock(_load_specification(arguments.specification))
    write_cut3r_comparison_lock(arguments.output, lock)
    print(lock["lock_id"])
    return 0


def _verify_command(arguments: argparse.Namespace) -> int:
    lock = load_cut3r_comparison_lock(arguments.lock)
    print(lock["lock_id"])
    return 0


def _summarize_command(arguments: argparse.Namespace) -> int:
    summary = cut3r_comparison_summary(load_cut3r_comparison_lock(arguments.lock))
    if arguments.json:
        print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    else:
        print(f"lock_id: {summary['lock_id']}")
        print(f"independent groups: {summary['independent_group_count']}")
        print(f"cases: {summary['case_count']}")
        print("enabled arms: " + ", ".join(summary["enabled_arms"]))
        print("claim contrasts: " + ", ".join(summary["claim_eligible_contrasts"]))
        print(f"target access: {summary['target_access']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CUT3R comparison-lock command."""

    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    return int(arguments.handler(arguments))


__all__ = [
    "CUT3R_COMPARISON_CLAIM_BOUNDARY",
    "CUT3R_COMPARISON_GROUP_UNIT",
    "CUT3R_COMPARISON_SCHEMA",
    "CUT3R_COMPARISON_VERSION",
    "CUT3R_PROVIDER_ENDPOINTS",
    "build_cut3r_comparison_lock",
    "cut3r_comparison_summary",
    "load_cut3r_comparison_lock",
    "main",
    "validate_cut3r_comparison_lock",
    "write_cut3r_comparison_lock",
]


if __name__ == "__main__":
    raise SystemExit(main())
