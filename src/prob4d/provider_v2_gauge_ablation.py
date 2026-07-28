"""Ablation runner using the exact provider-v2 sequential gauge backend.

The historical :mod:`prob4d.experiments` runner is retained for reproduction. This
module reuses its seven-row evaluation contract while replacing the multi-parent
initial gauge estimator with the causal spanning-tree posterior used by provider v2.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import WindowAlignment
from .composition_jacobian import composition_jacobian_mode
from .data import PredictionWindow
from .experiments import (
    AblationRow,
    _baseline_sequence,
    _dataset_calibration,
    _evaluate,
    _simulated_scale_anchors,
    _uncertainties,
    _window_truth_gauge,
    _write_results,
)
from .fusion import fuse_windows
from .gauge import FixedLagGaugeSmoother, GaugeEstimate, RelativeGaugeConstraint
from .io import load_prediction_bundle, load_truth
from .observation_export import (
    JointGaugePosterior,
    _build_alignments as _build_provider_alignments,
    estimate_joint_gauge_tree,
)
from .sim3 import Sim3
from .uncertainty import CalibrationReport, DepthDisagreementModel

PROVIDER_V2_GAUGE_POSTERIOR_MODE = "sequential_joint_spanning_tree_v1"
PROVIDER_V2_COMPOSITION_JACOBIAN_MODE = "analytic"


def _default_anchor_covariance() -> np.ndarray:
    """Return the frozen near-deterministic anchor used by the legacy benchmark."""

    return np.diag(np.full(7, 1e-10, dtype=np.float64))


def marginal_gauge_estimates(
    posterior: JointGaugePosterior,
) -> dict[str, GaugeEstimate]:
    """Convert a joint posterior to the marginal format expected by fusion code."""

    estimates: dict[str, GaugeEstimate] = {}
    for index, window_id in enumerate(posterior.window_ids):
        block = slice(7 * index, 7 * (index + 1))
        estimates[window_id] = GaugeEstimate(
            window_id=window_id,
            global_from_local=posterior.estimates[window_id],
            covariance=posterior.joint_covariance[block, block],
        )
    return estimates


def build_provider_v2_alignments(
    windows: Sequence[PredictionWindow],
) -> list[WindowAlignment]:
    """Build overlap alignments with the provider export's deterministic seeds."""

    return _build_provider_alignments(windows)


def estimate_provider_v2_gauge_backend(
    windows: Sequence[PredictionWindow],
    alignments: Sequence[WindowAlignment],
    *,
    initial_transform: Sim3 | None = None,
    initial_covariance: np.ndarray | None = None,
) -> tuple[JointGaugePosterior, dict[str, GaugeEstimate]]:
    """Run the provider-v2 causal tree with analytic composition Jacobians.

    This function deliberately uses the same ``estimate_joint_gauge_tree`` routine
    selected by claim-bearing provider-v2 exports. It returns both the full joint
    posterior and its per-window marginal adapter for the legacy fusion interfaces.
    """

    transform = Sim3.identity() if initial_transform is None else initial_transform
    covariance = (
        _default_anchor_covariance()
        if initial_covariance is None
        else np.asarray(initial_covariance, dtype=np.float64)
    )
    with composition_jacobian_mode(PROVIDER_V2_COMPOSITION_JACOBIAN_MODE):
        posterior = estimate_joint_gauge_tree(
            windows,
            alignments,
            initial_transform=transform,
            initial_covariance=covariance,
        )
    if posterior.mode != PROVIDER_V2_GAUGE_POSTERIOR_MODE:
        raise RuntimeError("provider-v2 gauge posterior mode changed unexpectedly")
    if not posterior.cross_window_covariance_preserved:
        raise RuntimeError("provider-v2 gauge posterior lost cross-window covariance")
    return posterior, marginal_gauge_estimates(posterior)


def run_provider_v2_gauge_manifest_ablation(
    *,
    predictions: str | Path,
    truth_path: str | Path,
    calibration_predictions: str | Path,
    calibration_truth_path: str | Path,
    metric_anchor_every: int = 2,
    metric_anchor_standard_deviation: float = 0.02,
) -> tuple[list[AblationRow], CalibrationReport, dict[str, Any]]:
    """Run the seven real-data variants with provider-v2 gauge-tree semantics."""

    test_bundle = load_prediction_bundle(predictions)
    truth = load_truth(truth_path)
    calibration_bundle = load_prediction_bundle(calibration_predictions)
    calibration_truth = load_truth(calibration_truth_path)
    model, report, _ = _dataset_calibration(
        calibration_bundle,
        calibration_truth,
        DepthDisagreementModel(),
    )

    alignments = build_provider_v2_alignments(test_bundle.overlap_windows)
    constraints = [
        RelativeGaugeConstraint.from_window_alignment(alignment)
        for alignment in alignments
    ]
    ordered_ids = [window.window_id for window in test_bundle.overlap_windows]
    sequential_posterior, sequential = estimate_provider_v2_gauge_backend(
        test_bundle.overlap_windows,
        alignments,
    )
    smoothed = FixedLagGaugeSmoother(lag=4).smooth(
        ordered_ids,
        sequential,
        constraints,
    )

    anchors = _simulated_scale_anchors(
        test_bundle.overlap_windows,
        truth,
        every=metric_anchor_every,
        standard_deviation=metric_anchor_standard_deviation,
    )
    anchored_posterior, anchored_initial = estimate_provider_v2_gauge_backend(
        test_bundle.overlap_windows,
        alignments,
        initial_transform=Sim3(scale=anchors[0].scale),
    )
    anchored = FixedLagGaugeSmoother(lag=4).smooth(
        ordered_ids,
        anchored_initial,
        constraints,
        scale_anchors=anchors,
    )

    true_gauges = {
        window.window_id: _window_truth_gauge(window, truth)
        for window in test_bundle.overlap_windows
    }
    uncertainties = _uncertainties(test_bundle.overlap_windows, alignments, model)
    boundaries = [window.start_frame for window in test_bundle.overlap_windows[1:]]
    transforms = {
        "sequential": {
            key: value.global_from_local for key, value in sequential.items()
        },
        "smoothed": {key: value.global_from_local for key, value in smoothed.items()},
        "anchored": {key: value.global_from_local for key, value in anchored.items()},
    }
    gauge_covariances = {
        "sequential": {key: value.covariance for key, value in sequential.items()},
        "smoothed": {key: value.covariance for key, value in smoothed.items()},
        "anchored": {key: value.covariance for key, value in anchored.items()},
    }
    sequences = {
        "decoded_uniform": fuse_windows(
            test_bundle.overlap_windows,
            transforms["sequential"],
            uncertainties,
            method="uniform",
            gauge_covariances=gauge_covariances["sequential"],
        ),
        "precision": fuse_windows(
            test_bundle.overlap_windows,
            transforms["sequential"],
            uncertainties,
            method="precision",
            gauge_covariances=gauge_covariances["sequential"],
        ),
        "ci": fuse_windows(
            test_bundle.overlap_windows,
            transforms["sequential"],
            uncertainties,
            method="covariance_intersection",
            gauge_covariances=gauge_covariances["sequential"],
        ),
        "ci_smoothed": fuse_windows(
            test_bundle.overlap_windows,
            transforms["smoothed"],
            uncertainties,
            method="covariance_intersection",
            gauge_covariances=gauge_covariances["smoothed"],
        ),
        "ci_smoothed_anchored": fuse_windows(
            test_bundle.overlap_windows,
            transforms["anchored"],
            uncertainties,
            method="covariance_intersection",
            gauge_covariances=gauge_covariances["anchored"],
        ),
    }
    definitions = [
        (
            "disjoint",
            "Disjoint 25-frame windows",
            _baseline_sequence(test_bundle.disjoint_baseline, model),
            None,
            "upstream_motioncrafter",
        ),
        (
            "latent_linear",
            "Latent-space linear overlap blend",
            _baseline_sequence(test_bundle.latent_linear_baseline, model),
            None,
            "upstream_motioncrafter",
        ),
        (
            "decoded_uniform",
            "Decoded Sim(3) alignment + uniform fusion",
            sequences["decoded_uniform"],
            sequential,
            "prob4d_provider_v2_gauge",
        ),
        (
            "precision",
            "Naive precision-weighted fusion",
            sequences["precision"],
            sequential,
            "prob4d_provider_v2_gauge",
        ),
        (
            "ci",
            "Covariance intersection",
            sequences["ci"],
            sequential,
            "prob4d_provider_v2_gauge",
        ),
        (
            "ci_smoothed",
            "Covariance intersection + fixed-lag gauge smoothing",
            sequences["ci_smoothed"],
            smoothed,
            "prob4d_provider_v2_gauge",
        ),
        (
            "ci_smoothed_anchored",
            "Smoothed covariance intersection + sparse metric anchors",
            sequences["ci_smoothed_anchored"],
            anchored,
            "prob4d_provider_v2_gauge",
        ),
    ]
    rows = [
        _evaluate(
            key,
            label,
            sequence,
            truth,
            gauges=gauges,
            true_gauges=true_gauges,
            boundary_frames=boundaries,
            baseline_source=source,
        )
        for key, label, sequence, gauges, source in definitions
    ]
    metadata = {
        "benchmark": "motioncrafter_manifest_provider_v2_gauge",
        "predictions": str(Path(predictions).resolve()),
        "truth": str(Path(truth_path).resolve()),
        "calibration_predictions": str(Path(calibration_predictions).resolve()),
        "calibration_truth": str(Path(calibration_truth_path).resolve()),
        "motioncrafter_commit": test_bundle.metadata.get("motioncrafter_commit"),
        "metric_anchor_source": "simulated_from_ground_truth",
        "metric_anchor_every": metric_anchor_every,
        "metric_anchor_standard_deviation": metric_anchor_standard_deviation,
        "gauge_backend": {
            "provider_api_version": 2,
            "posterior_mode": sequential_posterior.mode,
            "anchored_posterior_mode": anchored_posterior.mode,
            "composition_jacobian_mode": PROVIDER_V2_COMPOSITION_JACOBIAN_MODE,
            "cross_window_covariance_preserved": (
                sequential_posterior.cross_window_covariance_preserved
            ),
            "legacy_ablation_runner_unchanged": True,
        },
    }
    return rows, report, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--calibration-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric-anchor-every", type=int, default=2)
    parser.add_argument("--metric-anchor-std", type=float, default=0.02)
    arguments = parser.parse_args(argv)

    rows, calibration, metadata = run_provider_v2_gauge_manifest_ablation(
        predictions=arguments.predictions,
        truth_path=arguments.truth,
        calibration_predictions=arguments.calibration_predictions,
        calibration_truth_path=arguments.calibration_truth,
        metric_anchor_every=arguments.metric_anchor_every,
        metric_anchor_standard_deviation=arguments.metric_anchor_std,
    )
    _write_results(arguments.output_dir, rows, calibration, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
