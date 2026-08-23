"""Exact complete-group inference for CUT3R recurrent-state recovery v2."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

from ._cut3r_recovery_v2_spec import (
    CUT3R_RECURRENT_STATE_RECOVERY_V2_INTERVAL_METHOD,
)
from .cut3r_recurrent_state_recovery import _mean


def _count_vectors(total: int, length: int) -> Iterator[tuple[int, ...]]:
    if length == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for suffix in _count_vectors(total - first, length - 1):
            yield (first, *suffix)


def _multinomial_multiplicity(
    counts: Sequence[int],
    *,
    factorials: Sequence[int],
) -> int:
    denominator = math.prod(factorials[count] for count in counts)
    return factorials[sum(counts)] // denominator


def _weighted_mean(values: Sequence[float], counts: Sequence[int]) -> float:
    count = sum(counts)
    if count <= 0 or len(values) != len(counts):
        raise ValueError("weighted group mean requires matched nonempty inputs")
    try:
        result = math.fsum(
            value * multiplicity
            for value, multiplicity in zip(values, counts, strict=True)
        ) / count
    except OverflowError as error:
        raise ValueError("weighted group mean overflowed") from error
    if not math.isfinite(result):
        raise ValueError("weighted group mean must remain finite")
    return result


def _finite_difference(left: float, right: float, *, name: str) -> float:
    result = left - right
    if not math.isfinite(result):
        raise ValueError(f"{name} must remain finite")
    return result


def _triplet_v2(
    native: float,
    restarted: float,
    fused: float,
    *,
    denominator_threshold: float,
) -> dict[str, Any]:
    prob4d_gain = _finite_difference(restarted, fused, name="Prob4D gain")
    recurrence_gap = _finite_difference(restarted, native, name="recurrence gap")
    if recurrence_gap > denominator_threshold:
        recovery_fraction = prob4d_gain / recurrence_gap
        if not math.isfinite(recovery_fraction):
            raise ValueError("recovery fraction must remain finite")
        status = "defined"
    else:
        recovery_fraction = None
        status = "undefined-recurrence-gap-not-practically-separated"
    return {
        "native_continuous": native,
        "restarted_newest": restarted,
        "restarted_prob4d_fused": fused,
        "prob4d_gain": prob4d_gain,
        "recurrence_gap": recurrence_gap,
        "minimum_recurrence_gap": denominator_threshold,
        "recovery_fraction": recovery_fraction,
        "status": status,
    }


def _weighted_nearest_rank(
    weighted_values: Sequence[tuple[float, int]],
    *,
    total_weight: int,
    probability: float,
) -> float:
    if total_weight <= 0 or not weighted_values:
        raise ValueError("weighted quantile requires positive support")
    rank = max(1, math.ceil(probability * total_weight))
    cumulative = 0
    for value, weight in sorted(weighted_values, key=lambda item: item[0]):
        cumulative += weight
        if cumulative >= rank:
            return value
    raise RuntimeError("weighted quantile support did not reach its total weight")


def _weighted_interval(
    weighted_values: Sequence[tuple[float, int]],
    *,
    total_weight: int,
    confidence_level: float,
) -> dict[str, Any]:
    tail = (1.0 - confidence_level) / 2.0
    return {
        "method": CUT3R_RECURRENT_STATE_RECOVERY_V2_INTERVAL_METHOD,
        "quantile_rule": "weighted-nearest-rank-ceiling-v1",
        "confidence_level": confidence_level,
        "support_ordered_resample_count": total_weight,
        "lower": _weighted_nearest_rank(
            weighted_values,
            total_weight=total_weight,
            probability=tail,
        ),
        "upper": _weighted_nearest_rank(
            weighted_values,
            total_weight=total_weight,
            probability=1.0 - tail,
        ),
    }


def _exact_group_bootstrap(
    group_triplets: Sequence[tuple[float, float, float]],
    *,
    denominator_threshold: float,
    confidence_level: float,
    minimum_valid_denominator_probability: float,
    maximum_exact_group_count: int,
    point_status: str,
) -> dict[str, Any]:
    group_count = len(group_triplets)
    if group_count < 2:
        raise ValueError("v2 exact inference requires at least two evaluable groups")
    if group_count > maximum_exact_group_count:
        raise ValueError(
            "evaluable group count exceeds the prospectively frozen exact limit"
        )
    factorials = [math.factorial(index) for index in range(group_count + 1)]
    native_values = [item[0] for item in group_triplets]
    restarted_values = [item[1] for item in group_triplets]
    fused_values = [item[2] for item in group_triplets]
    gain_values: list[tuple[float, int]] = []
    gap_values: list[tuple[float, int]] = []
    ratio_values: list[tuple[float, int]] = []
    count_vector_count = 0
    total_ordered_resamples = 0
    valid_ordered_resamples = 0
    for counts in _count_vectors(group_count, group_count):
        multiplicity = _multinomial_multiplicity(counts, factorials=factorials)
        count_vector_count += 1
        total_ordered_resamples += multiplicity
        native = _weighted_mean(native_values, counts)
        restarted = _weighted_mean(restarted_values, counts)
        fused = _weighted_mean(fused_values, counts)
        triplet = _triplet_v2(
            native,
            restarted,
            fused,
            denominator_threshold=denominator_threshold,
        )
        gain_values.append((cast(float, triplet["prob4d_gain"]), multiplicity))
        gap_values.append((cast(float, triplet["recurrence_gap"]), multiplicity))
        if triplet["status"] == "defined":
            ratio_values.append(
                (cast(float, triplet["recovery_fraction"]), multiplicity)
            )
            valid_ordered_resamples += multiplicity

    expected_ordered_resamples = group_count**group_count
    expected_count_vectors = math.comb(2 * group_count - 1, group_count - 1)
    if total_ordered_resamples != expected_ordered_resamples:
        raise RuntimeError("exact bootstrap multiplicities do not sum to n**n")
    if count_vector_count != expected_count_vectors:
        raise RuntimeError("exact bootstrap count-vector enumeration is incomplete")
    valid_probability = valid_ordered_resamples / total_ordered_resamples
    if point_status != "defined":
        ratio_interval: dict[str, Any] = {
            "method": CUT3R_RECURRENT_STATE_RECOVERY_V2_INTERVAL_METHOD,
            "interval_status": "not-applicable-point-denominator",
            "lower": None,
            "upper": None,
        }
    elif (
        valid_ordered_resamples == 0
        or valid_probability < minimum_valid_denominator_probability
    ):
        ratio_interval = {
            "method": CUT3R_RECURRENT_STATE_RECOVERY_V2_INTERVAL_METHOD,
            "interval_status": "insufficient-valid-denominator-probability",
            "lower": None,
            "upper": None,
        }
    else:
        ratio_interval = {
            **_weighted_interval(
                ratio_values,
                total_weight=valid_ordered_resamples,
                confidence_level=confidence_level,
            ),
            "interval_status": "defined",
            "conditioning": "recurrence-gap-above-frozen-practical-floor",
        }
    return {
        "method": CUT3R_RECURRENT_STATE_RECOVERY_V2_INTERVAL_METHOD,
        "group_count": group_count,
        "count_vector_count": count_vector_count,
        "ordered_resample_count": total_ordered_resamples,
        "prob4d_gain_interval": _weighted_interval(
            gain_values,
            total_weight=total_ordered_resamples,
            confidence_level=confidence_level,
        ),
        "recurrence_gap_interval": _weighted_interval(
            gap_values,
            total_weight=total_ordered_resamples,
            confidence_level=confidence_level,
        ),
        "recovery_fraction_interval": ratio_interval,
        "valid_denominator_ordered_resample_count": valid_ordered_resamples,
        "invalid_denominator_ordered_resample_count": (
            total_ordered_resamples - valid_ordered_resamples
        ),
        "valid_denominator_probability": valid_probability,
        "minimum_valid_denominator_probability": (
            minimum_valid_denominator_probability
        ),
        "minimum_recurrence_gap": denominator_threshold,
    }


def _leave_one_group_out(
    group_ids: Sequence[str],
    group_triplets: Sequence[tuple[float, float, float]],
    *,
    denominator_threshold: float,
    point: Mapping[str, Any],
) -> dict[str, Any]:
    if len(group_ids) != len(group_triplets) or len(group_ids) < 2:
        raise ValueError("leave-one-group-out requires matched groups and triplets")
    rows: list[dict[str, Any]] = []
    for omitted_index, omitted_group_id in enumerate(group_ids):
        retained = [
            triplet
            for index, triplet in enumerate(group_triplets)
            if index != omitted_index
        ]
        row = _triplet_v2(
            _mean([item[0] for item in retained]),
            _mean([item[1] for item in retained]),
            _mean([item[2] for item in retained]),
            denominator_threshold=denominator_threshold,
        )
        rows.append({"omitted_group_id": omitted_group_id, **row})
    point_gain = cast(float, point["prob4d_gain"])
    gains = [cast(float, row["prob4d_gain"]) for row in rows]
    defined_recoveries = [
        cast(float, row["recovery_fraction"])
        for row in rows
        if row["status"] == "defined"
    ]
    return {
        "method": "leave-one-complete-source-group-out-v1",
        "omission_count": len(rows),
        "rows": rows,
        "summary": {
            "minimum_prob4d_gain": min(gains),
            "maximum_prob4d_gain": max(gains),
            "prob4d_gain_sign_reversal": any(
                point_gain * gain < 0.0 for gain in gains
            ),
            "point_denominator_status": point["status"],
            "denominator_status_changed": any(
                row["status"] != point["status"] for row in rows
            ),
            "defined_recovery_count": len(defined_recoveries),
            "minimum_defined_recovery_fraction": (
                min(defined_recoveries) if defined_recoveries else None
            ),
            "maximum_defined_recovery_fraction": (
                max(defined_recoveries) if defined_recoveries else None
            ),
        },
    }


__all__ = [
    "_count_vectors",
    "_exact_group_bootstrap",
    "_leave_one_group_out",
    "_triplet_v2",
]
