"""Finite-sample source-only calibration for uncertainty-normalized cycle guards."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from .uncertainty_normalized_cycles import UncertaintyNormalizedCycleAudit

SOURCE_UNCERTAINTY_FEATURE: Final = "median_cycle_minkowski_uncertainty_scale"
SOURCE_UNCERTAINTY_STRATA_SEMANTICS: Final = (
    "development-frozen-tertiles-of-observable-cycle-minkowski-scale-v1"
)
GROUP_ROBUST_THRESHOLD_SEMANTICS: Final = (
    "maximum-of-source-stratum-split-conformal-thresholds-v1"
)
_STRATUM_IDS: Final = ("low", "medium", "high")


@dataclass(frozen=True)
class ObservableUncertaintyStrata:
    """Development-frozen bins defined only from observable source uncertainty."""

    boundaries: tuple[float, float]
    feature_name: str = SOURCE_UNCERTAINTY_FEATURE
    semantics: str = SOURCE_UNCERTAINTY_STRATA_SEMANTICS
    stratum_ids: tuple[str, str, str] = _STRATUM_IDS

    def __post_init__(self) -> None:
        boundaries = tuple(float(value) for value in self.boundaries)
        if len(boundaries) != 2:
            raise ValueError("observable uncertainty strata require two boundaries")
        if (
            not np.all(np.isfinite(boundaries))
            or boundaries[0] <= 0.0
            or boundaries[0] >= boundaries[1]
        ):
            raise ValueError(
                "observable uncertainty boundaries must be finite, positive, and increasing"
            )
        if self.feature_name != SOURCE_UNCERTAINTY_FEATURE:
            raise ValueError("observable uncertainty feature semantics changed")
        if self.semantics != SOURCE_UNCERTAINTY_STRATA_SEMANTICS:
            raise ValueError("observable uncertainty stratum semantics changed")
        if tuple(self.stratum_ids) != _STRATUM_IDS:
            raise ValueError("observable uncertainty stratum IDs changed")
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "stratum_ids", _STRATUM_IDS)

    def assign(self, feature: float) -> str:
        """Assign one finite positive feature without using hidden scenario labels."""

        value = float(feature)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("source uncertainty feature must be finite and positive")
        index = int(np.searchsorted(self.boundaries, value, side="right"))
        return self.stratum_ids[index]

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "semantics": self.semantics,
            "boundaries": list(self.boundaries),
            "stratum_ids": list(self.stratum_ids),
        }


@dataclass(frozen=True)
class FiniteSampleConformalThreshold:
    """One nonrandomized split-conformal upper threshold."""

    threshold: float
    alpha: float
    sample_count: int
    order_index: int
    quantile_semantics: str = (
        "sorted_score_at_ceil((n+1)*(1-alpha)); nonrandomized ties"
    )

    def __post_init__(self) -> None:
        threshold = float(self.threshold)
        alpha = float(self.alpha)
        sample_count = int(self.sample_count)
        order_index = int(self.order_index)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("conformal threshold must be finite and positive")
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("conformal alpha must lie in (0, 1)")
        if sample_count < 1:
            raise ValueError("conformal sample_count must be positive")
        expected = math.ceil((sample_count + 1) * (1.0 - alpha))
        if order_index != expected or not 1 <= order_index <= sample_count:
            raise ValueError("conformal order index differs from the declared rule")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "order_index", order_index)

    @property
    def finite_sample_coverage_level(self) -> float:
        return self.order_index / (self.sample_count + 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "alpha": self.alpha,
            "sample_count": self.sample_count,
            "order_index": self.order_index,
            "finite_sample_coverage_level": self.finite_sample_coverage_level,
            "quantile_semantics": self.quantile_semantics,
        }


@dataclass(frozen=True)
class GroupRobustCycleCalibration:
    """Pooled, stratum-specific, and worst-stratum source-only thresholds."""

    strata: ObservableUncertaintyStrata
    pooled_threshold: FiniteSampleConformalThreshold
    stratum_thresholds: tuple[
        tuple[str, FiniteSampleConformalThreshold],
        tuple[str, FiniteSampleConformalThreshold],
        tuple[str, FiniteSampleConformalThreshold],
    ]
    worst_group_threshold: float
    minimum_calibration_per_stratum: int
    threshold_semantics: str = GROUP_ROBUST_THRESHOLD_SEMANTICS

    def __post_init__(self) -> None:
        pairs = tuple(self.stratum_thresholds)
        if tuple(name for name, _ in pairs) != self.strata.stratum_ids:
            raise ValueError("stratum thresholds must follow canonical stratum order")
        minimum = int(self.minimum_calibration_per_stratum)
        if minimum < 1:
            raise ValueError("minimum_calibration_per_stratum must be positive")
        if any(item.sample_count < minimum for _, item in pairs):
            raise ValueError("one observable source stratum lacks calibration support")
        expected = max(item.threshold for _, item in pairs)
        threshold = float(self.worst_group_threshold)
        if not math.isfinite(threshold) or threshold != expected:
            raise ValueError("worst_group_threshold must equal the maximum stratum threshold")
        if self.threshold_semantics != GROUP_ROBUST_THRESHOLD_SEMANTICS:
            raise ValueError("group-robust threshold semantics changed")
        object.__setattr__(self, "stratum_thresholds", pairs)
        object.__setattr__(self, "minimum_calibration_per_stratum", minimum)
        object.__setattr__(self, "worst_group_threshold", threshold)

    @property
    def threshold_by_stratum(self) -> Mapping[str, float]:
        return {name: item.threshold for name, item in self.stratum_thresholds}

    def threshold_for_feature(self, feature: float) -> float:
        return self.threshold_by_stratum[self.strata.assign(feature)]

    def to_dict(self) -> dict[str, object]:
        return {
            "strata": self.strata.to_dict(),
            "pooled_threshold": self.pooled_threshold.to_dict(),
            "stratum_thresholds": {
                name: threshold.to_dict()
                for name, threshold in self.stratum_thresholds
            },
            "worst_group_threshold": self.worst_group_threshold,
            "minimum_calibration_per_stratum": (
                self.minimum_calibration_per_stratum
            ),
            "threshold_semantics": self.threshold_semantics,
        }


def cycle_source_uncertainty_feature(
    audit: UncertaintyNormalizedCycleAudit,
) -> float:
    """Return a robust observable scale used only for source-side stratification."""

    if audit.cycle_count < 1:
        raise ValueError("source uncertainty feature requires at least one cycle")
    values = np.asarray(
        [cycle.minkowski_uncertainty_scale for cycle in audit.cycles],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("cycle uncertainty scales must be finite and positive")
    return float(np.median(values))


def fit_observable_uncertainty_strata(
    development_features: Sequence[float],
) -> ObservableUncertaintyStrata:
    """Freeze tertile boundaries on a development split without target access."""

    values = np.asarray(development_features, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 12
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError(
            "development source features must contain at least 12 finite positive values"
        )
    boundaries = tuple(
        float(value)
        for value in np.quantile(
            values,
            [1.0 / 3.0, 2.0 / 3.0],
            method="linear",
        )
    )
    if boundaries[0] >= boundaries[1]:
        raise ValueError("development features do not define distinct uncertainty strata")
    return ObservableUncertaintyStrata(boundaries=boundaries)


def finite_sample_conformal_threshold(
    scores: Sequence[float],
    *,
    alpha: float,
) -> FiniteSampleConformalThreshold:
    """Fit the nonrandomized split-conformal upper order statistic."""

    values = np.asarray(scores, dtype=np.float64)
    alpha_value = float(alpha)
    if (
        values.ndim != 1
        or values.size < 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("conformal scores must be a nonempty finite nonnegative vector")
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    order_index = math.ceil((values.size + 1) * (1.0 - alpha_value))
    if order_index > values.size:
        raise ValueError(
            "calibration sample is too small for a finite conformal threshold"
        )
    threshold = max(
        float(np.sort(values)[order_index - 1]),
        float(np.nextafter(0.0, 1.0)),
    )
    return FiniteSampleConformalThreshold(
        threshold=threshold,
        alpha=alpha_value,
        sample_count=int(values.size),
        order_index=order_index,
    )


def fit_group_robust_cycle_calibration(
    development_features: Sequence[float],
    calibration_features: Sequence[float],
    calibration_scores: Sequence[float],
    *,
    alpha: float = 0.05,
    minimum_calibration_per_stratum: int = 20,
) -> GroupRobustCycleCalibration:
    """Fit observable strata and a worst-stratum finite-sample threshold."""

    features = np.asarray(calibration_features, dtype=np.float64)
    scores = np.asarray(calibration_scores, dtype=np.float64)
    if features.ndim != 1 or scores.ndim != 1 or features.shape != scores.shape:
        raise ValueError("calibration features and scores must be aligned vectors")
    if not np.all(np.isfinite(features)) or np.any(features <= 0.0):
        raise ValueError("calibration source features must be finite and positive")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError("calibration cycle scores must be finite and nonnegative")
    if isinstance(minimum_calibration_per_stratum, bool):
        raise ValueError("minimum_calibration_per_stratum must be a positive integer")
    minimum = int(minimum_calibration_per_stratum)
    if minimum != minimum_calibration_per_stratum or minimum < 1:
        raise ValueError("minimum_calibration_per_stratum must be a positive integer")

    strata = fit_observable_uncertainty_strata(development_features)
    pooled = finite_sample_conformal_threshold(scores, alpha=alpha)
    pairs: list[tuple[str, FiniteSampleConformalThreshold]] = []
    for stratum_id in strata.stratum_ids:
        selected = np.asarray(
            [
                score
                for feature, score in zip(features, scores, strict=True)
                if strata.assign(float(feature)) == stratum_id
            ],
            dtype=np.float64,
        )
        if selected.size < minimum:
            raise ValueError(
                f"observable source stratum {stratum_id!r} has only "
                f"{selected.size} calibration cases"
            )
        pairs.append(
            (
                stratum_id,
                finite_sample_conformal_threshold(selected, alpha=alpha),
            )
        )
    typed_pairs = tuple(pairs)
    return GroupRobustCycleCalibration(
        strata=strata,
        pooled_threshold=pooled,
        stratum_thresholds=typed_pairs,  # type: ignore[arg-type]
        worst_group_threshold=max(item.threshold for _, item in typed_pairs),
        minimum_calibration_per_stratum=minimum,
    )


__all__ = [
    "GROUP_ROBUST_THRESHOLD_SEMANTICS",
    "SOURCE_UNCERTAINTY_FEATURE",
    "SOURCE_UNCERTAINTY_STRATA_SEMANTICS",
    "FiniteSampleConformalThreshold",
    "GroupRobustCycleCalibration",
    "ObservableUncertaintyStrata",
    "cycle_source_uncertainty_feature",
    "finite_sample_conformal_threshold",
    "fit_group_robust_cycle_calibration",
    "fit_observable_uncertainty_strata",
]
