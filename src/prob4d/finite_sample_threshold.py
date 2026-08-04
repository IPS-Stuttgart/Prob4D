"""Finite-sample upper thresholds for exchangeable source-only scores."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
FINITE_SAMPLE_UPPER_THRESHOLD_SEMANTICS: Final = (
    "split-conformal-upper-order-statistic-v1"
)


def _canonical_score_digest(scores: FloatArray) -> str:
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    canonical = np.ascontiguousarray(ordered, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"prob4d.finite-sample-upper-threshold.scores.v1\0")
    digest.update(struct.pack("<Q", canonical.size))
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _validated_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _order_statistic_rank(calibration_count: int, miscoverage: float) -> int:
    rank = int(math.ceil((calibration_count + 1) * (1.0 - miscoverage)))
    if rank > calibration_count:
        raise ValueError(
            "miscoverage is below the finite calibration resolution; increase "
            "calibration_count or miscoverage"
        )
    return rank


@dataclass(frozen=True)
class FiniteSampleUpperThreshold:
    """One split-conformal upper order statistic and its finite-sample bound."""

    miscoverage: float
    calibration_count: int
    order_statistic_rank: int
    threshold: float
    guaranteed_miscoverage_upper_bound: float
    canonical_scores_sha256: str
    semantics: str = FINITE_SAMPLE_UPPER_THRESHOLD_SEMANTICS

    def __post_init__(self) -> None:
        miscoverage = _validated_real(self.miscoverage, name="miscoverage")
        if not 0.0 < miscoverage < 1.0:
            raise ValueError("miscoverage must lie strictly between zero and one")
        if isinstance(self.calibration_count, bool) or not isinstance(
            self.calibration_count, int
        ):
            raise ValueError("calibration_count must be a positive integer")
        calibration_count = self.calibration_count
        if calibration_count < 1:
            raise ValueError("calibration_count must be a positive integer")
        if isinstance(self.order_statistic_rank, bool) or not isinstance(
            self.order_statistic_rank, int
        ):
            raise ValueError("order_statistic_rank must be a positive integer")
        rank = self.order_statistic_rank
        if not 1 <= rank <= calibration_count:
            raise ValueError("order_statistic_rank must lie within calibration support")
        expected_rank = _order_statistic_rank(calibration_count, miscoverage)
        if rank != expected_rank:
            raise ValueError("order_statistic_rank differs from the registered formula")
        threshold = _validated_real(self.threshold, name="threshold")
        if threshold < 0.0:
            raise ValueError("threshold must be finite and nonnegative")
        bound = _validated_real(
            self.guaranteed_miscoverage_upper_bound,
            name="guaranteed_miscoverage_upper_bound",
        )
        if not 0.0 <= bound <= miscoverage:
            raise ValueError(
                "guaranteed_miscoverage_upper_bound must lie in [0, miscoverage]"
            )
        if not isinstance(self.canonical_scores_sha256, str):
            raise ValueError("canonical_scores_sha256 must be a string")
        digest = self.canonical_scores_sha256
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("canonical_scores_sha256 must be a lowercase SHA-256 digest")
        if self.semantics != FINITE_SAMPLE_UPPER_THRESHOLD_SEMANTICS:
            raise ValueError("finite-sample upper-threshold semantics changed")
        expected_bound = (calibration_count + 1 - rank) / (calibration_count + 1)
        if not math.isclose(bound, expected_bound, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("guaranteed miscoverage bound differs from the order rank")
        object.__setattr__(self, "miscoverage", miscoverage)
        object.__setattr__(self, "calibration_count", calibration_count)
        object.__setattr__(self, "order_statistic_rank", rank)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "guaranteed_miscoverage_upper_bound", bound)
        object.__setattr__(self, "canonical_scores_sha256", digest)

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": self.semantics,
            "miscoverage": self.miscoverage,
            "calibration_count": self.calibration_count,
            "order_statistic_rank": self.order_statistic_rank,
            "threshold": self.threshold,
            "guaranteed_miscoverage_upper_bound": (
                self.guaranteed_miscoverage_upper_bound
            ),
            "canonical_scores_sha256": self.canonical_scores_sha256,
            "exchangeability_boundary": (
                "The finite-sample bound requires exchangeability between the clean "
                "calibration scores and the future clean source score. It is marginal, "
                "not conditional on a source-noise subgroup."
            ),
        }


def fit_finite_sample_upper_threshold(
    scores: FloatArray,
    *,
    miscoverage: float,
) -> FiniteSampleUpperThreshold:
    """Fit the standard split-conformal upper order statistic.

    For ``n`` clean calibration scores and requested miscoverage ``alpha``, the
    threshold is the ``ceil((n + 1) * (1 - alpha))``-th smallest calibration
    score. The function fails closed when the requested miscoverage is below the
    finite resolution ``1 / (n + 1)`` because a finite calibration threshold
    cannot then provide the requested bound.
    """

    raw = np.asarray(scores)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("scores must be a nonempty one-dimensional vector")
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("scores must contain real numeric values")
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("scores must be finite and nonnegative")
    alpha = _validated_real(miscoverage, name="miscoverage")
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage must lie strictly between zero and one")
    count = int(values.size)
    rank = _order_statistic_rank(count, alpha)
    ordered = np.sort(values)
    threshold = float(ordered[rank - 1])
    bound = (count + 1 - rank) / (count + 1)
    return FiniteSampleUpperThreshold(
        miscoverage=alpha,
        calibration_count=count,
        order_statistic_rank=rank,
        threshold=threshold,
        guaranteed_miscoverage_upper_bound=bound,
        canonical_scores_sha256=_canonical_score_digest(values),
    )


__all__ = [
    "FINITE_SAMPLE_UPPER_THRESHOLD_SEMANTICS",
    "FiniteSampleUpperThreshold",
    "fit_finite_sample_upper_threshold",
]
