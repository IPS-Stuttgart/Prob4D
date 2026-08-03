from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d._provider_evaluation_decision import evaluate_provider_decision_policy
from prob4d._provider_evaluation_manifest import (
    ProviderEvaluationDecisionPolicy,
    ProviderEvaluationDecisionRule,
    load_provider_evaluation_plan,
)
from prob4d.fusion import FusedSequence
from prob4d.io import save_fused_prediction, save_truth
from prob4d.metrics import TruthSequence
from prob4d.provider_evaluation import main, run_provider_evaluation


def _rule(
    *,
    rule_id: str = "rmse-superiority",
    metric: str = "point_rmse",
    direction: str = "lower",
    criterion: str = "superiority",
    margin: float = 0.05,
) -> ProviderEvaluationDecisionRule:
    return ProviderEvaluationDecisionRule(
        rule_id=rule_id,
        candidate_method="ci",
        metric=metric,
        direction=direction,  # type: ignore[arg-type]
        criterion=criterion,  # type: ignore[arg-type]
        margin=margin,
    )


def _comparison_summary() -> tuple[dict[str, object], dict[str, object]]:
    aggregate: dict[str, object] = {
        "uniform": {"group_count": 9},
        "ci": {"group_count": 9},
    }
    comparisons: dict[str, object] = {
        "ci": {
            "reference_method": "uniform",
            "group_count": 9,
            "metrics": {
                "metric.metrics.point_rmse": {
                    "mean": -0.20,
                    "ci95_lower": -0.30,
                    "ci95_upper": -0.10,
                    "group_count": 9,
                },
                "metric.metrics.coverage_95": {
                    "mean": -0.01,
                    "ci95_lower": -0.02,
                    "ci95_upper": 0.01,
                    "group_count": 9,
                },
            },
        }
    }
    return aggregate, comparisons


def test_decision_policy_applies_directional_bootstrap_bounds() -> None:
    aggregate, comparisons = _comparison_summary()
    policy = ProviderEvaluationDecisionPolicy(
        policy_id="provider-gate-v1",
        minimum_group_count=9,
        rules=(
            _rule(),
            _rule(
                rule_id="coverage-noninferiority",
                metric="coverage_95",
                direction="higher",
                criterion="noninferiority",
                margin=0.03,
            ),
        ),
    )

    result = evaluate_provider_decision_policy(
        policy,
        aggregate=aggregate,
        comparisons=comparisons,
        primary_mode="metric",
        reference_method="uniform",
    )

    assert result["overall_passed"] is True
    assert result["group_count_passed"] is True
    assert result["passed_rule_count"] == 2
    first, second = result["rules"]
    assert first["decision_bound"] == "ci95_upper"
    assert first["pass_threshold"] == pytest.approx(-0.05)
    assert second["decision_bound"] == "ci95_lower"
    assert second["pass_threshold"] == pytest.approx(-0.03)


def test_decision_policy_retains_failed_independent_group_gate() -> None:
    aggregate, comparisons = _comparison_summary()
    policy = ProviderEvaluationDecisionPolicy(
        policy_id="provider-gate-v1",
        minimum_group_count=10,
        rules=(_rule(),),
    )

    result = evaluate_provider_decision_policy(
        policy,
        aggregate=aggregate,
        comparisons=comparisons,
        primary_mode="metric",
        reference_method="uniform",
    )

    assert result["rules"][0]["passed"] is True
    assert result["group_count_passed"] is False
    assert result["overall_passed"] is False


def test_decision_policy_rejects_unavailable_registered_metric() -> None:
    aggregate, comparisons = _comparison_summary()
    policy = ProviderEvaluationDecisionPolicy(
        policy_id="provider-gate-v1",
        minimum_group_count=9,
        rules=(_rule(metric="not_a_metric"),),
    )

    with pytest.raises(ValueError, match="unavailable metric"):
        evaluate_provider_decision_policy(
            policy,
            aggregate=aggregate,
            comparisons=comparisons,
            primary_mode="metric",
            reference_method="uniform",
        )


def _manifest_case(tmp_path: Path) -> dict[str, object]:
    truth = tmp_path / "truth.npz"
    uniform = tmp_path / "uniform.npz"
    ci = tmp_path / "ci.npz"
    for path in (truth, uniform, ci):
        path.write_bytes(b"placeholder")
    return {
        "case_id": "case-1",
        "group_id": "group-1",
        "truth": truth.name,
        "predictions": {"uniform": uniform.name, "ci": ci.name},
        "boundary_frames": [],
        "prefix_frame_stop_exclusive": None,
    }


def test_manifest_v2_loads_strict_decision_policy(tmp_path: Path) -> None:
    manifest = {
        "schema_name": "prob4d.provider-evaluation",
        "schema_version": 2,
        "primary_mode": "metric",
        "reference_method": "uniform",
        "cases": [_manifest_case(tmp_path)],
        "metadata": {"split_id": "held-out-v1"},
        "decision_policy": {
            "policy_id": "provider-gate-v1",
            "minimum_group_count": 9,
            "rules": [
                {
                    "rule_id": "rmse-superiority",
                    "candidate_method": "ci",
                    "metric": "point_rmse",
                    "direction": "lower",
                    "criterion": "superiority",
                    "margin": 0.05,
                }
            ],
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    _, mode, reference, metadata, policy = load_provider_evaluation_plan(path)

    assert mode == "metric"
    assert reference == "uniform"
    assert metadata["split_id"] == "held-out-v1"
    assert policy is not None
    assert policy.policy_id == "provider-gate-v1"
    assert policy.rules[0].metric == "point_rmse"


def test_manifest_v2_rejects_oracle_decision_mode(tmp_path: Path) -> None:
    manifest = {
        "schema_name": "prob4d.provider-evaluation",
        "schema_version": 2,
        "primary_mode": "oracle_aligned",
        "reference_method": "uniform",
        "cases": [_manifest_case(tmp_path)],
        "metadata": {},
        "decision_policy": {
            "policy_id": "provider-gate-v1",
            "minimum_group_count": 1,
            "rules": [
                {
                    "rule_id": "rmse-superiority",
                    "candidate_method": "ci",
                    "metric": "point_rmse",
                    "direction": "lower",
                    "criterion": "superiority",
                    "margin": 0.0,
                }
            ],
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot use oracle_aligned"):
        load_provider_evaluation_plan(path)


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


def _artifact_metadata() -> dict[str, object]:
    return {
        "prob4d_revision": "a" * 40,
        "motioncrafter_revision": "b" * 40,
        "motioncrafter_seed_policy": "derived-per-call",
        "motioncrafter_model_set_sha256": "c" * 64,
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
        "predictions": {"uniform": uniform_path.name, "ci": ci_path.name},
        "boundary_frames": [1],
        "prefix_frame_stop_exclusive": 1,
    }


def _evaluation_manifest(tmp_path: Path, *, minimum_group_count: int) -> Path:
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
        "schema_version": 2,
        "primary_mode": "metric",
        "reference_method": "uniform",
        "cases": cases,
        "metadata": {"split": "held-out-objects"},
        "decision_policy": {
            "policy_id": "provider-gate-v1",
            "minimum_group_count": minimum_group_count,
            "rules": [
                {
                    "rule_id": "rmse-superiority",
                    "candidate_method": "ci",
                    "metric": "metric_point_rmse",
                    "direction": "lower",
                    "criterion": "superiority",
                    "margin": 0.5,
                }
            ],
        },
    }
    path = tmp_path / f"evaluation-{minimum_group_count}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_provider_evaluation_v2_reports_registered_decision(tmp_path: Path) -> None:
    path = _evaluation_manifest(tmp_path, minimum_group_count=2)
    output = tmp_path / "report"

    report = run_provider_evaluation(
        path,
        output,
        bootstrap_resamples=200,
        seed=17,
    )

    assert report["schema_version"] == 3
    assert report["decision"]["overall_passed"] is True
    assert report["decision"]["rules"][0]["decision_bound"] == "ci95_upper"
    markdown = output.joinpath("provider_evaluation.md").read_text(encoding="utf-8")
    assert "Preregistered provider decision" in markdown
    assert "Overall result: **PASS**" in markdown


def test_cli_returns_three_after_writing_failed_registered_decision(
    tmp_path: Path,
) -> None:
    path = _evaluation_manifest(tmp_path, minimum_group_count=3)
    output = tmp_path / "report-failed"

    assert (
        main(
            [
                str(path),
                "--output-dir",
                str(output),
                "--bootstrap-resamples",
                "100",
                "--require-decision-pass",
            ]
        )
        == 3
    )
    report = json.loads(
        output.joinpath("provider_evaluation.json").read_text(encoding="utf-8")
    )
    assert report["decision"]["group_count_passed"] is False
    assert report["decision"]["overall_passed"] is False
