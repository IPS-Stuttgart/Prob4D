from __future__ import annotations

import numpy as np
import pytest

from prob4d.finite_sample_threshold import (
    FINITE_SAMPLE_UPPER_THRESHOLD_SEMANTICS,
    FiniteSampleUpperThreshold,
    fit_finite_sample_upper_threshold,
)


def test_finite_sample_upper_threshold_uses_registered_order_statistic() -> None:
    result = fit_finite_sample_upper_threshold(
        np.arange(1.0, 20.0),
        miscoverage=0.10,
    )

    assert result.semantics == FINITE_SAMPLE_UPPER_THRESHOLD_SEMANTICS
    assert result.calibration_count == 19
    assert result.order_statistic_rank == 18
    assert result.threshold == 18.0
    assert result.guaranteed_miscoverage_upper_bound == pytest.approx(0.10)
    assert result.to_dict()["threshold"] == 18.0


def test_finite_sample_threshold_is_row_order_invariant() -> None:
    scores = np.asarray([0.2, 0.7, 0.1, 0.4, 0.3, 0.6, 0.5])
    first = fit_finite_sample_upper_threshold(scores, miscoverage=0.25)
    second = fit_finite_sample_upper_threshold(scores[::-1], miscoverage=0.25)

    assert first == second
    assert first.canonical_scores_sha256 == second.canonical_scores_sha256


def test_finite_sample_threshold_digest_is_canonical_and_versioned() -> None:
    result = fit_finite_sample_upper_threshold(
        np.asarray([3.0, 1.0, 2.0]),
        miscoverage=0.25,
    )

    assert result.canonical_scores_sha256 == (
        "e5245f81c591c1ddada50243e6f8abf442d2c549b42ee5343da4bdf40f26c36e"
    )


def test_finite_sample_threshold_retains_ties_deterministically() -> None:
    result = fit_finite_sample_upper_threshold(
        np.asarray([0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0]),
        miscoverage=0.25,
    )

    assert result.order_statistic_rank == 6
    assert result.threshold == 3.0


@pytest.mark.parametrize(
    "scores,miscoverage,match",
    [
        (np.asarray([]), 0.10, "nonempty"),
        (np.asarray([[1.0]]), 0.10, "one-dimensional"),
        (np.asarray([1.0, np.nan]), 0.10, "finite"),
        (np.asarray([1.0, -1.0]), 0.10, "nonnegative"),
        (np.asarray([1.0, 2.0]), 0.0, "strictly between"),
        (np.asarray([1.0, 2.0]), 1.0, "strictly between"),
        (np.asarray([1.0, 2.0]), True, "real number"),
        (np.asarray(["1.0", "2.0"]), 0.50, "real numeric"),
        (np.asarray([1.0, 2.0]), 0.10, "finite calibration resolution"),
    ],
)
def test_finite_sample_threshold_fails_closed(
    scores: np.ndarray,
    miscoverage: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        fit_finite_sample_upper_threshold(scores, miscoverage=miscoverage)


def test_threshold_contract_rejects_inconsistent_bound() -> None:
    fitted = fit_finite_sample_upper_threshold(
        np.arange(1.0, 10.0),
        miscoverage=0.20,
    )
    with pytest.raises(ValueError, match="differs from the order rank"):
        FiniteSampleUpperThreshold(
            miscoverage=fitted.miscoverage,
            calibration_count=fitted.calibration_count,
            order_statistic_rank=fitted.order_statistic_rank,
            threshold=fitted.threshold,
            guaranteed_miscoverage_upper_bound=0.0,
            canonical_scores_sha256=fitted.canonical_scores_sha256,
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("calibration_count", 9.0, "positive integer"),
        ("order_statistic_rank", 8.0, "positive integer"),
        ("threshold", False, "real number"),
        ("guaranteed_miscoverage_upper_bound", False, "real number"),
        ("canonical_scores_sha256", 1, "must be a string"),
    ],
)
def test_threshold_contract_rejects_noncanonical_scalar_types(
    field: str,
    value: object,
    match: str,
) -> None:
    fitted = fit_finite_sample_upper_threshold(
        np.arange(1.0, 10.0),
        miscoverage=0.20,
    )
    arguments = {
        "miscoverage": fitted.miscoverage,
        "calibration_count": fitted.calibration_count,
        "order_statistic_rank": fitted.order_statistic_rank,
        "threshold": fitted.threshold,
        "guaranteed_miscoverage_upper_bound": (
            fitted.guaranteed_miscoverage_upper_bound
        ),
        "canonical_scores_sha256": fitted.canonical_scores_sha256,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=match):
        FiniteSampleUpperThreshold(**arguments)  # type: ignore[arg-type]


def test_threshold_contract_rejects_a_rank_from_another_miscoverage() -> None:
    fitted = fit_finite_sample_upper_threshold(
        np.arange(1.0, 10.0),
        miscoverage=0.20,
    )

    with pytest.raises(ValueError, match="registered formula"):
        FiniteSampleUpperThreshold(
            miscoverage=fitted.miscoverage,
            calibration_count=fitted.calibration_count,
            order_statistic_rank=fitted.order_statistic_rank + 1,
            threshold=fitted.threshold,
            guaranteed_miscoverage_upper_bound=0.0,
            canonical_scores_sha256=fitted.canonical_scores_sha256,
        )
