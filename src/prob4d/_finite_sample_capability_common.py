"""Shared validation and rank helpers for finite-sample capability reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any, cast

FINITE_SAMPLE_CAPABILITY_SCHEMA = "prob4d.finite-sample-capability"
FINITE_SAMPLE_CAPABILITY_VERSION = 1
FINITE_SAMPLE_CAPABILITY_CLAIM_BOUNDARY = (
    "This target-free report describes finite-sample rank availability and "
    "diagnostic resolution implied by the frozen independent group counts. It "
    "does not establish exchangeability, provider competence, calibrated target "
    "coverage, BayesianPhysTwin benefit, Causal4D benefit, safety, or state of "
    "the art."
)
DEFAULT_COVERAGE_LEVELS = (0.80, 0.90, 0.95)


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Encode one finite JSON mapping deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def load_json(path: Path) -> Mapping[str, Any]:
    """Load a strict finite-JSON object."""

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_pairs,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read finite-sample capability report {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError("finite-sample capability report root must be an object")
    return value


def exact_fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def strict_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def strict_digest(value: object, *, name: str) -> str:
    digest = strict_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def strict_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def coverage(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a real number")
    result = float(cast(int | float, value))
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    return result


def string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of strings")
    result = tuple(strict_string(item, name=f"{name}[{index}]") for index, item in enumerate(value))
    if not result or result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty, sorted, and unique")
    return result


def coverages(value: Sequence[float]) -> tuple[float, ...]:
    if not value:
        raise ValueError("requested coverages must not be empty")
    return tuple(sorted({coverage(item, name="requested coverage") for item in value}))


def _ceil(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def split_conformal_level(group_count: int, nominal_coverage: float) -> dict[str, object]:
    """Return exact one-sided split-conformal rank availability."""

    count = strict_integer(group_count, name="group_count", minimum=1)
    level = coverage(nominal_coverage, name="nominal_coverage")
    level_decimal = Decimal(str(level))
    rank = _ceil(Decimal(count + 1) * level_decimal)
    finite = rank <= count
    minimum_groups = _ceil(level_decimal / (Decimal(1) - level_decimal))
    return {
        "nominal_coverage": level,
        "alpha": float(Decimal(1) - level_decimal),
        "order_statistic_rank": rank,
        "finite_threshold": finite,
        "guaranteed_coverage_lower_bound": rank / (count + 1) if finite else None,
        "minimum_group_count_for_finite_threshold": minimum_groups,
    }


__all__ = [
    "DEFAULT_COVERAGE_LEVELS",
    "FINITE_SAMPLE_CAPABILITY_CLAIM_BOUNDARY",
    "FINITE_SAMPLE_CAPABILITY_SCHEMA",
    "FINITE_SAMPLE_CAPABILITY_VERSION",
    "canonical_json",
    "coverage",
    "coverages",
    "exact_fields",
    "load_json",
    "split_conformal_level",
    "strict_digest",
    "strict_integer",
    "strict_string",
    "string_tuple",
]
