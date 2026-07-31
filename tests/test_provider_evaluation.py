import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.fusion import FusedSequence
from prob4d.io import pack_symmetric_covariance, save_fused_prediction, save_truth
from prob4d.metrics import TruthSequence
from prob4d.provider_evaluation import run_provider_evaluation


def _truth() -> TruthSequence:
    points = np.array(
        [
            [[[0.0, 0.0, 1.0]]],
            [[[1.0, 0.0, 1.0]]],
        ]
    )
    return TruthSequence(
        frame_indices=np.array([0, 1]),
        point_map=points,
        valid_mask=np.ones((2, 1, 1), dtype=bool),
    )


def _prediction(truth: TruthSequence, error: float) -> FusedSequence:
    points = truth.point_map.copy()
    points[..., 0] += error
    covariance = np.broadcast_to(np.eye(3), points.shape + (3,)).copy()
    return FusedSequence(
        frame_indices=truth.frame_indices,
        point_map=points,
        valid_mask=truth.valid_mask,
        point_covariance=covariance,
        contributors=np.ones(truth.valid_mask.shape, dtype=np.uint16),
    )


def _artifact_metadata(*, model_set_sha256: str = "c" * 64) -> dict[str, object]:
    return {
        "prob4d_revision": "a" * 40,
        "motioncrafter_revision": "b" * 40,
        "motioncrafter_seed_policy": "derived-per-call",
        "motioncrafter_model_set_sha256": model_set_sha256,
        "prediction_manifest_sha256": "d" * 64,
        "includes_covariance": True,
        "gauge_estimator": "sequential",
        "uncertainty_calibration": "held_out",
    }


def _write_case(
    root: Path,
    case_id: str,
    group_id: str,
    *,
    uniform_error: float,
    ci_error: float,
) -> dict[str, object]:
    truth = _truth()
    truth_path = root / f"{case_id}-truth.npz"
    uniform_path = root / f"{case_id}-uniform.npz"
    ci_path = root / f"{case_id}-ci.npz"
    save_truth(truth_path, truth)
    save_fused_prediction(
        uniform_path,
        _prediction(truth, uniform_error),
        method_id="uniform",
        fusion_method="uniform",
        metadata=_artifact_metadata(),
    )
    save_fused_prediction(
        ci_path,
        _prediction(truth, ci_error),
        method_id="ci",
        fusion_method="covariance_intersection",
        metadata=_artifact_metadata(),
    )
    return {
        "case_id": case_id,
        "group_id": group_id,
        "truth": truth_path.name,
        "predictions": {
            "uniform": uniform_path.name,
            "ci": ci_path.name,
        },
        "boundary_frames": [1],
        "prefix_frame_stop_exclusive": 1,
    }


def test_provider_evaluation_uses_equal_group_aggregation(tmp_path: Path) -> None:
    cases = [
        _write_case(
            tmp_path,
            "c1",
            "g1",
            uniform_error=1.0,
            ci_error=0.5,
        ),
        _write_case(
            tmp_path,
            "c2",
            "g1",
            uniform_error=3.0,
            ci_error=1.0,
        ),
        _write_case(
            tmp_path,
            "c3",
            "g2",
            uniform_error=5.0,
            ci_error=2.0,
        ),
    ]
    manifest = {
        "schema_name": "prob4d.provider-evaluation",
        "schema_version": 1,
        "primary_mode": "metric",
        "reference_method": "uniform",
        "cases": cases,
        "metadata": {"split": "held-out-objects"},
    }
    manifest_path = tmp_path / "evaluation.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "report"

    report = run_provider_evaluation(
        manifest_path,
        output,
        bootstrap_resamples=200,
        seed=17,
    )

    uniform = report["aggregate"]["uniform"]
    ci = report["aggregate"]["ci"]
    uniform_metric = uniform["metrics"]["metric.metrics.metric_point_rmse"]
    ci_metric = ci["metrics"]["metric.metrics.metric_point_rmse"]
    assert uniform["group_count"] == 2
    assert uniform["case_count"] == 3
    assert uniform_metric["mean"] == pytest.approx(3.5)
    assert ci_metric["mean"] == pytest.approx(1.375)
    difference = report["comparisons"]["ci"]["metrics"][
        "metric.metrics.metric_point_rmse"
    ]
    assert difference["mean"] == pytest.approx(-2.125)
    assert output.joinpath("provider_evaluation.json").exists()
    assert output.joinpath("provider_evaluation.csv").exists()
    markdown = output.joinpath("provider_evaluation.md").read_text(encoding="utf-8")
    assert "equal aggregate mass" in markdown
    assert "unknown_correlation_consistency_bound" in json.dumps(report)


def test_provider_evaluation_rejects_method_relabelling(tmp_path: Path) -> None:
    case = _write_case(
        tmp_path,
        "c1",
        "g1",
        uniform_error=1.0,
        ci_error=0.5,
    )
    case["predictions"] = {"renamed": case["predictions"]["uniform"]}
    manifest = {
        "schema_name": "prob4d.provider-evaluation",
        "schema_version": 1,
        "primary_mode": "metric",
        "reference_method": "renamed",
        "cases": [case],
        "metadata": {},
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="method label"):
        run_provider_evaluation(path, tmp_path / "output")


def test_provider_evaluation_rejects_model_set_drift(tmp_path: Path) -> None:
    first = _write_case(
        tmp_path,
        "c1",
        "g1",
        uniform_error=1.0,
        ci_error=0.5,
    )
    second = _write_case(
        tmp_path,
        "c2",
        "g2",
        uniform_error=2.0,
        ci_error=1.0,
    )
    changed_path = tmp_path / "c2-uniform.npz"
    truth = _truth()
    save_fused_prediction(
        changed_path,
        _prediction(truth, 2.0),
        method_id="uniform",
        fusion_method="uniform",
        metadata=_artifact_metadata(model_set_sha256="e" * 64),
    )
    manifest = {
        "schema_name": "prob4d.provider-evaluation",
        "schema_version": 1,
        "primary_mode": "metric",
        "reference_method": "uniform",
        "cases": [first, second],
        "metadata": {},
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="mixes covariance, model"):
        run_provider_evaluation(path, tmp_path / "output")


def test_provider_evaluation_rejects_legacy_unspecified_artifacts(
    tmp_path: Path,
) -> None:
    truth = _truth()
    truth_path = tmp_path / "truth.npz"
    prediction_path = tmp_path / "legacy.npz"
    save_truth(truth_path, truth)
    sequence = _prediction(truth, 1.0)
    np.savez(
        prediction_path,
        frame_indices=sequence.frame_indices,
        point_map=sequence.point_map,
        valid_mask=sequence.valid_mask,
        point_covariance_packed=pack_symmetric_covariance(sequence.point_covariance),
        contributors=sequence.contributors,
    )
    manifest = {
        "schema_name": "prob4d.provider-evaluation",
        "schema_version": 1,
        "primary_mode": "metric",
        "reference_method": "legacy",
        "cases": [
            {
                "case_id": "c1",
                "group_id": "g1",
                "truth": truth_path.name,
                "predictions": {"legacy": prediction_path.name},
                "boundary_frames": [],
                "prefix_frame_stop_exclusive": None,
            }
        ],
        "metadata": {},
    }
    manifest_path = tmp_path / "evaluation.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy unspecified"):
        run_provider_evaluation(manifest_path, tmp_path / "output")
