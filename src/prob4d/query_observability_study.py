"""Analytic control for query-conditioned observability.

The control isolates one unobservable twist direction.  A rank-six factor
correctly supports a point on the observed line and rejects a distant off-axis
probe under a frozen query gate.  An intentionally invalid full-rank completion
passes the same off-axis query because it fabricates information in the missing
direction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .observable_gauge import (
    IID_OBSERVABLE_INFORMATION,
    CentroidGaugeChart,
    ObservableGaugeFactor,
)
from .query_observability import (
    QueryObservabilityDecision,
    QueryObservabilityGate,
    QueryObservabilityReport,
    evaluate_query_observability,
    point_position_query_jacobian,
)
from .sim3 import Sim3

SCHEMA_NAME = "prob4d.query-conditioned-observability-control"
SCHEMA_VERSION = 1


def _controlled_factor(*, complete_nullspace: bool) -> ObservableGaugeFactor:
    identity = np.eye(7)
    if complete_nullspace:
        observable_basis = identity
        nullspace_basis = np.empty((7, 0))
        observable_information = 10.0 * identity
        spectrum = np.ones(7)
    else:
        observable_basis = np.delete(identity, 1, axis=1)
        nullspace_basis = identity[:, 1:2]
        observable_information = 10.0 * np.eye(6)
        spectrum = np.concatenate((np.ones(6), np.zeros(1)))
    return ObservableGaugeFactor(
        chart=CentroidGaugeChart(
            linearization=Sim3.identity(),
            source_centroid=np.zeros(3),
            cloud_scale=1.0,
        ),
        observable_basis=observable_basis,
        nullspace_basis=nullspace_basis,
        observable_information=observable_information,
        normalized_geometry_spectrum=spectrum,
        rank_threshold=1e-8,
        residual_rms=0.01,
        residual_variance=0.01,
        inlier_fraction=1.0,
        num_correspondences=48,
        covariance_method=IID_OBSERVABLE_INFORMATION,
    )


def _report_payload(
    report: QueryObservabilityReport,
    decision: QueryObservabilityDecision,
) -> dict[str, Any]:
    return {
        "factor_rank": report.factor_rank,
        "query_dimension": report.query_dimension,
        "direct_observability_fraction": report.direct_observability_fraction,
        "nullspace_sensitivity_fraction": report.nullspace_sensitivity_fraction,
        "metric_variance_reduction_fraction": (
            report.metric_variance_reduction_fraction
        ),
        "worst_supported_variance_ratio": (
            report.worst_supported_variance_ratio
        ),
        "prior_metric_variance": report.prior_metric_variance,
        "posterior_metric_variance": report.posterior_metric_variance,
        "admitted": decision.admitted,
        "reason_codes": list(decision.reason_codes),
    }


def run_query_observability_study() -> dict[str, Any]:
    """Run the deterministic partial-observability control."""

    partial = _controlled_factor(complete_nullspace=False)
    invalid_completion = _controlled_factor(complete_nullspace=True)
    prior_covariance = np.eye(7)
    supported_point = np.array([1.0, 0.0, 0.0])
    off_support_point = np.array([0.0, 5.0, 0.0])
    gate = QueryObservabilityGate(
        minimum_direct_observability_fraction=0.80,
        minimum_metric_variance_reduction_fraction=0.80,
        maximum_worst_supported_variance_ratio=0.50,
    )

    partial_supported = evaluate_query_observability(
        partial,
        prior_covariance_local=prior_covariance,
        query_jacobian_local=point_position_query_jacobian(
            partial,
            supported_point,
        ),
    )
    partial_off_support = evaluate_query_observability(
        partial,
        prior_covariance_local=prior_covariance,
        query_jacobian_local=point_position_query_jacobian(
            partial,
            off_support_point,
        ),
    )
    completion_off_support = evaluate_query_observability(
        invalid_completion,
        prior_covariance_local=prior_covariance,
        query_jacobian_local=point_position_query_jacobian(
            invalid_completion,
            off_support_point,
        ),
    )

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "Deterministic mechanism evidence only; no real-provider accuracy, "
            "calibration, BayesianPhysTwin benefit, Causal4D benefit, or "
            "deployment claim."
        ),
        "chart_coordinates": (
            "[log-scale, left-rotation(3), centroid-translation/cloud-scale]"
        ),
        "controlled_missing_direction": "rotation-x",
        "factor_information_precision": 10.0,
        "prior_covariance": "identity-7",
        "queries": {
            "supported_source_point": supported_point.tolist(),
            "off_support_source_point": off_support_point.tolist(),
            "output_metric": "identity-3",
        },
        "gate": {
            "minimum_direct_observability_fraction": (
                gate.minimum_direct_observability_fraction
            ),
            "minimum_metric_variance_reduction_fraction": (
                gate.minimum_metric_variance_reduction_fraction
            ),
            "maximum_worst_supported_variance_ratio": (
                gate.maximum_worst_supported_variance_ratio
            ),
        },
        "results": {
            "rank_six_supported_query": _report_payload(
                partial_supported,
                gate.evaluate(partial_supported),
            ),
            "rank_six_off_support_query": _report_payload(
                partial_off_support,
                gate.evaluate(partial_off_support),
            ),
            "invalid_full_rank_off_support_query": _report_payload(
                completion_off_support,
                gate.evaluate(completion_off_support),
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the query-conditioned observability analytic control."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_query_observability_study()
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
