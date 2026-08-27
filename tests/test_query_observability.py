from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from prob4d.observable_gauge import ObservableGaugeFactor
from prob4d.query_observability import (
    QueryObservabilityGate,
    evaluate_query_observability,
    point_position_query_jacobian,
)
from prob4d.query_observability_study import run_query_observability_study
from prob4d.sim3 import Sim3


@dataclass(frozen=True)
class _Chart:
    linearization: Sim3
    reference_centroid: np.ndarray
    cloud_scale: float


@dataclass(frozen=True)
class _PartialFactor:
    observable_basis: np.ndarray
    nullspace_basis: np.ndarray
    observable_information: np.ndarray
    chart: _Chart

    @property
    def rank(self) -> int:
        return int(self.observable_basis.shape[1])

    @property
    def information_matrix(self) -> np.ndarray:
        return (
            self.observable_basis
            @ self.observable_information
            @ self.observable_basis.T
        )


def _twist_ambiguous_factor() -> ObservableGaugeFactor:
    identity = np.eye(7)
    observable = np.delete(identity, 1, axis=1)
    nullspace = identity[:, 1:2]
    factor = _PartialFactor(
        observable_basis=observable,
        nullspace_basis=nullspace,
        observable_information=10.0 * np.eye(6),
        chart=_Chart(
            linearization=Sim3.identity(),
            reference_centroid=np.zeros(3),
            cloud_scale=1.0,
        ),
    )
    return cast(ObservableGaugeFactor, factor)


def test_query_conditioning_distinguishes_support_from_off_support() -> None:
    factor = _twist_ambiguous_factor()
    prior = np.eye(7)
    supported = evaluate_query_observability(
        factor,
        prior_covariance_local=prior,
        query_jacobian_local=point_position_query_jacobian(
            factor,
            np.array([1.0, 0.0, 0.0]),
        ),
    )
    off_support = evaluate_query_observability(
        factor,
        prior_covariance_local=prior,
        query_jacobian_local=point_position_query_jacobian(
            factor,
            np.array([0.0, 5.0, 0.0]),
        ),
    )

    assert supported.direct_observability_fraction == pytest.approx(1.0)
    assert supported.nullspace_sensitivity_fraction == pytest.approx(0.0)
    assert supported.metric_variance_reduction_fraction > 0.90
    assert supported.worst_supported_variance_ratio < 0.10

    assert off_support.direct_observability_fraction < 0.70
    assert off_support.nullspace_sensitivity_fraction > 0.30
    assert off_support.metric_variance_reduction_fraction < 0.65
    assert off_support.worst_supported_variance_ratio > 0.95

    gate = QueryObservabilityGate(
        minimum_direct_observability_fraction=0.80,
        minimum_metric_variance_reduction_fraction=0.80,
        maximum_worst_supported_variance_ratio=0.50,
    )
    assert gate.evaluate(supported).admitted
    decision = gate.evaluate(off_support)
    assert not decision.admitted
    assert decision.reason_codes == (
        "insufficient-direct-query-observability",
        "insufficient-query-variance-reduction",
        "excessive-worst-direction-variance-ratio",
    )


def test_prior_mediated_reduction_is_not_called_direct_observability() -> None:
    factor = _twist_ambiguous_factor()
    prior = np.eye(7)
    prior[1, 2] = 0.8
    prior[2, 1] = 0.8
    nullspace_query = factor.nullspace_basis.T

    report = evaluate_query_observability(
        factor,
        prior_covariance_local=prior,
        query_jacobian_local=nullspace_query,
    )

    assert report.direct_observability_fraction == pytest.approx(0.0)
    assert report.nullspace_sensitivity_fraction == pytest.approx(1.0)
    assert report.metric_variance_reduction_fraction > 0.50
    assert report.worst_supported_variance_ratio < 0.50


def test_declared_query_metric_controls_multioutput_weighting() -> None:
    factor = _twist_ambiguous_factor()
    observable_query = factor.observable_basis[:, 0]
    nullspace_query = factor.nullspace_basis[:, 0]
    jacobian = np.vstack((observable_query, nullspace_query))

    identity_metric = evaluate_query_observability(
        factor,
        prior_covariance_local=np.eye(7),
        query_jacobian_local=jacobian,
    )
    nullspace_weighted = evaluate_query_observability(
        factor,
        prior_covariance_local=np.eye(7),
        query_jacobian_local=jacobian,
        query_metric=np.diag([1.0, 4.0]),
    )

    assert identity_metric.direct_observability_fraction == pytest.approx(0.5)
    assert nullspace_weighted.direct_observability_fraction == pytest.approx(0.2)
    assert (
        nullspace_weighted.metric_variance_reduction_fraction
        < identity_metric.metric_variance_reduction_fraction
    )


def test_full_rank_factor_has_no_direct_nullspace_sensitivity() -> None:
    identity = np.eye(7)
    factor = cast(
        ObservableGaugeFactor,
        _PartialFactor(
            observable_basis=identity,
            nullspace_basis=np.empty((7, 0)),
            observable_information=4.0 * identity,
            chart=_Chart(
                linearization=Sim3.identity(),
                reference_centroid=np.zeros(3),
                cloud_scale=1.0,
            ),
        ),
    )
    jacobian = np.array(
        [
            [1.0, 0.2, 0.0, -0.1, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.1, 0.0, 1.0, 0.0],
        ]
    )

    report = evaluate_query_observability(
        factor,
        prior_covariance_local=np.eye(7),
        query_jacobian_local=jacobian,
    )

    assert report.direct_observability_fraction == pytest.approx(1.0)
    assert report.nullspace_sensitivity_fraction == pytest.approx(0.0)
    assert report.metric_variance_reduction_fraction == pytest.approx(0.8)
    assert report.worst_supported_variance_ratio == pytest.approx(0.2)


def test_point_query_jacobian_is_read_only_and_validated() -> None:
    factor = _twist_ambiguous_factor()

    jacobian = point_position_query_jacobian(
        factor,
        np.array([1.0, 2.0, 3.0]),
    )

    assert jacobian.shape == (3, 7)
    assert not jacobian.flags.writeable
    with pytest.raises(ValueError, match="shape"):
        point_position_query_jacobian(factor, np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        point_position_query_jacobian(
            factor,
            np.array([0.0, np.nan, 0.0]),
        )


def test_invalid_query_metric_and_prior_fail_closed() -> None:
    factor = _twist_ambiguous_factor()
    jacobian = np.eye(7)[0:1]

    with pytest.raises(ValueError, match="positive definite"):
        evaluate_query_observability(
            factor,
            prior_covariance_local=np.eye(7),
            query_jacobian_local=jacobian,
            query_metric=np.zeros((1, 1)),
        )
    with pytest.raises(ValueError, match="positive definite"):
        evaluate_query_observability(
            factor,
            prior_covariance_local=np.zeros((7, 7)),
            query_jacobian_local=jacobian,
        )
    with pytest.raises(ValueError, match=r"shape \(Q, 7\)"):
        evaluate_query_observability(
            factor,
            prior_covariance_local=np.eye(7),
            query_jacobian_local=np.ones((2, 6)),
        )


def test_analytic_control_exposes_fabricated_full_rank_information() -> None:
    result = run_query_observability_study()
    supported = result["results"]["rank_six_supported_query"]
    off_support = result["results"]["rank_six_off_support_query"]
    invalid_completion = result["results"][
        "invalid_full_rank_off_support_query"
    ]

    assert supported["admitted"]
    assert not off_support["admitted"]
    assert invalid_completion["admitted"]
    assert off_support["worst_supported_variance_ratio"] > 0.95
    assert invalid_completion["worst_supported_variance_ratio"] < 0.10


def test_checked_control_evidence_matches_implementation() -> None:
    path = (
        Path(__file__).parents[1]
        / "evidence"
        / "query-observability-control-v1"
        / "result.json"
    )

    checked = json.loads(path.read_text(encoding="utf-8"))

    assert checked == run_query_observability_study()
