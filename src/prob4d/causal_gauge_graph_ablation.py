"""Held-out diagnostic comparing causal gauge graph alternatives.

This runner is intentionally separate from the frozen seven-row ablations and
from claim-bearing provider-v2 export. It compares the production causal spanning
tree, marginal multi-parent CI initialization, full-joint multi-edge CI graph,
and fixed-lag reconstruction control under one calibrated dense uncertainty
model and one paired evaluation contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .causal_gauge_graph import estimate_causal_multi_edge_gauge_graph
from .guarded_causal_gauge_graph import (
    estimate_guarded_causal_multi_edge_gauge_graph,
)
from .experiments import (
    AblationRow,
    _dataset_calibration,
    _evaluate,
    _uncertainties,
    _window_truth_gauge,
    _write_results,
)
from .fusion import fuse_windows
from .gauge import (
    FixedLagGaugeSmoother,
    RelativeGaugeConstraint,
    SequentialGaugeEstimator,
)
from .io import load_prediction_bundle, load_truth
from .observation_export import _build_alignments
from .provider_v2_gauge_ablation import (
    estimate_provider_v2_gauge_backend,
    marginal_gauge_estimates,
)
from .sim3 import Sim3
from .uncertainty import CalibrationReport, DepthDisagreementModel


def _transform_and_covariance_maps(estimates):
    return (
        {
            window_id: estimate.global_from_local
            for window_id, estimate in estimates.items()
        },
        {
            window_id: estimate.covariance
            for window_id, estimate in estimates.items()
        },
    )


def run_causal_gauge_graph_ablation(
    *,
    predictions: str | Path,
    truth_path: str | Path,
    calibration_predictions: str | Path,
    calibration_truth_path: str | Path,
    fixed_lag: int = 4,
    minimum_edge_weight: float = 0.0,
    maximum_cycle_displacement: float | None = None,
    cycle_representative_radius: float = 1.0,
    minimum_cycles_per_multi_edge_child: int = 1,
) -> tuple[list[AblationRow], CalibrationReport, dict[str, Any]]:
    """Compare gauge estimators with paired CI dense fusion."""

    if fixed_lag < 2:
        raise ValueError("fixed_lag must be at least two")
    guard_enabled = maximum_cycle_displacement is not None
    if not guard_enabled and (
        cycle_representative_radius != 1.0
        or minimum_cycles_per_multi_edge_child != 1
    ):
        raise ValueError("cycle guard settings require maximum_cycle_displacement")
    test_bundle = load_prediction_bundle(predictions)
    truth = load_truth(truth_path)
    calibration_bundle = load_prediction_bundle(calibration_predictions)
    calibration_truth = load_truth(calibration_truth_path)
    model, report, _ = _dataset_calibration(
        calibration_bundle,
        calibration_truth,
        DepthDisagreementModel(),
    )
    windows = test_bundle.overlap_windows
    alignments = _build_alignments(windows)
    constraints = [
        RelativeGaugeConstraint.from_window_alignment(alignment)
        for alignment in alignments
    ]
    ordered_ids = [window.window_id for window in windows]
    anchor_covariance = np.diag(np.full(7, 1e-10, dtype=np.float64))

    tree_posterior, tree = estimate_provider_v2_gauge_backend(
        windows,
        alignments,
        initial_transform=Sim3.identity(),
        initial_covariance=anchor_covariance,
    )
    marginal_ci = SequentialGaugeEstimator().estimate(
        ordered_ids,
        constraints,
        initial_transform=Sim3.identity(),
        initial_covariance=anchor_covariance,
    )
    graph_posterior, graph_report = estimate_causal_multi_edge_gauge_graph(
        windows,
        alignments,
        initial_transform=Sim3.identity(),
        initial_covariance=anchor_covariance,
        minimum_edge_weight=minimum_edge_weight,
    )
    graph = marginal_gauge_estimates(graph_posterior)
    guarded_graph = None
    guarded_graph_report = None
    if maximum_cycle_displacement is not None:
        guarded_posterior, guarded_graph_report = (
            estimate_guarded_causal_multi_edge_gauge_graph(
                windows,
                alignments,
                initial_transform=Sim3.identity(),
                initial_covariance=anchor_covariance,
                maximum_cycle_displacement=maximum_cycle_displacement,
                representative_radius=cycle_representative_radius,
                minimum_cycles_per_multi_edge_child=(
                    minimum_cycles_per_multi_edge_child
                ),
                minimum_edge_weight=minimum_edge_weight,
            )
        )
        guarded_graph = marginal_gauge_estimates(guarded_posterior)
    fixed_lag_estimates = FixedLagGaugeSmoother(lag=fixed_lag).smooth(
        ordered_ids,
        marginal_ci,
        constraints,
    )

    uncertainties = _uncertainties(windows, alignments, model)
    estimators = {
        "tree_ci": tree,
        "sequential_multi_parent_ci": marginal_ci,
        "causal_graph_ci": graph,
    }
    if guarded_graph is not None:
        estimators["guarded_causal_graph_ci"] = guarded_graph
    estimators["fixed_lag_ci"] = fixed_lag_estimates
    sequences = {}
    for key, estimates in estimators.items():
        transforms, gauge_covariances = _transform_and_covariance_maps(estimates)
        sequences[key] = fuse_windows(
            windows,
            transforms,
            uncertainties,
            method="covariance_intersection",
            gauge_covariances=gauge_covariances,
        )

    true_gauges = {
        window.window_id: _window_truth_gauge(window, truth)
        for window in windows
    }
    boundaries = [window.start_frame for window in windows[1:]]
    labels = {
        "tree_ci": "Causal single-parent spanning tree + CI",
        "sequential_multi_parent_ci": "Marginal multi-parent gauge CI + dense CI",
        "causal_graph_ci": "Full-joint causal multi-edge graph CI + dense CI",
    }
    if guarded_graph is not None:
        labels["guarded_causal_graph_ci"] = (
            "Source-cycle-guarded causal graph CI + exact tree fallback"
        )
    labels["fixed_lag_ci"] = "Fixed-lag gauge reconstruction control + dense CI"
    rows = [
        _evaluate(
            key,
            labels[key],
            sequences[key],
            truth,
            gauges=estimators[key],
            true_gauges=true_gauges,
            boundary_frames=boundaries,
            baseline_source="prob4d_causal_gauge_graph_diagnostic",
        )
        for key in labels
    ]
    metadata: dict[str, Any] = {
        "benchmark": "prob4d_causal_gauge_graph_ablation_v1",
        "predictions": str(Path(predictions).resolve()),
        "truth": str(Path(truth_path).resolve()),
        "calibration_predictions": str(Path(calibration_predictions).resolve()),
        "calibration_truth": str(Path(calibration_truth_path).resolve()),
        "motioncrafter_commit": test_bundle.metadata.get("motioncrafter_commit"),
        "paired_method_order": list(labels),
        "dense_fusion_method": "covariance_intersection",
        "fixed_lag": fixed_lag,
        "minimum_edge_weight": minimum_edge_weight,
        "tree_posterior_mode": tree_posterior.mode,
        "graph": graph_report.to_dict(),
        "graph_joint_covariance_dimension": int(
            graph_posterior.joint_covariance.shape[0]
        ),
        "provider_v2_claim_bearing_export": False,
        "target_truth_used_only_for_evaluation": True,
        "promotion_rule": (
            "retain the production tree unless a frozen held-out grouped result "
            "improves seam, drift, or point error without material coverage or "
            "harmful-update regression"
        ),
    }
    if maximum_cycle_displacement is not None:
        if guarded_graph_report is None:
            raise RuntimeError("enabled cycle guard did not produce an audit report")
        metadata.update(
            {
                "benchmark": "prob4d_causal_gauge_graph_ablation_v2",
                "cycle_guard_enabled": True,
                "maximum_cycle_displacement": maximum_cycle_displacement,
                "cycle_representative_radius": cycle_representative_radius,
                "minimum_cycles_per_multi_edge_child": (
                    minimum_cycles_per_multi_edge_child
                ),
                "guarded_graph": guarded_graph_report.to_dict(),
            }
        )
    return rows, report, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--calibration-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-lag", type=int, default=4)
    parser.add_argument("--minimum-edge-weight", type=float, default=0.0)
    parser.add_argument(
        "--maximum-cycle-displacement",
        type=float,
        help=(
            "source/calibration-frozen representative displacement threshold; "
            "a failed cycle gate returns the exact provider-v2 tree"
        ),
    )
    parser.add_argument(
        "--cycle-representative-radius",
        type=float,
        default=1.0,
    )
    parser.add_argument("--minimum-cycles-per-multi-edge-child", type=int, default=1)
    arguments = parser.parse_args(argv)
    try:
        rows, calibration, metadata = run_causal_gauge_graph_ablation(
            predictions=arguments.predictions,
            truth_path=arguments.truth,
            calibration_predictions=arguments.calibration_predictions,
            calibration_truth_path=arguments.calibration_truth,
            fixed_lag=arguments.fixed_lag,
            minimum_edge_weight=arguments.minimum_edge_weight,
            maximum_cycle_displacement=arguments.maximum_cycle_displacement,
            cycle_representative_radius=arguments.cycle_representative_radius,
            minimum_cycles_per_multi_edge_child=(
                arguments.minimum_cycles_per_multi_edge_child
            ),
        )
    except ValueError as error:
        parser.error(str(error))
    _write_results(arguments.output_dir, rows, calibration, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
