from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d._provider_evaluation_compute import (
    evaluate_provider_cases as evaluate_legacy_provider_cases,
)
from prob4d._provider_evaluation_manifest import ProviderEvaluationCase
from prob4d._provider_evaluation_provider_neutral import (
    evaluate_provider_cases as evaluate_provider_neutral_cases,
)
from prob4d.fusion import FusedSequence
from prob4d.io import save_fused_prediction, save_truth
from prob4d.metrics import TruthSequence
from prob4d.prediction_provider_manifest import SOURCE_DEPENDENCY_SEMANTICS
from prob4d.provider_evaluation import run_provider_evaluation
from prob4d.provider_evaluation_identity import (
    PROVIDER_EVALUATION_IDENTITY_SCHEMA,
    PROVIDER_EVALUATION_IDENTITY_VERSION,
)


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


def _metadata(
    *,
    provider_manifest_id: str,
    provider_manifest_sha256: str,
    provider_run_id: str,
    loader_id: str = "6" * 64,
) -> dict[str, object]:
    return {
        "prob4d_revision": "a" * 40,
        "provider_identity": {
            "schema_name": PROVIDER_EVALUATION_IDENTITY_SCHEMA,
            "schema_version": PROVIDER_EVALUATION_IDENTITY_VERSION,
            "provider_manifest_id": provider_manifest_id,
            "provider_manifest_sha256": provider_manifest_sha256,
            "provider_family": "cut3r",
            "provider_repository": "naver/CUT3R",
            "provider_revision": "b" * 40,
            "provider_run_id": provider_run_id,
            "model_set_id": "c" * 64,
            "loader_id": loader_id,
            "coordinate_semantics": "sequence-local-sim3",
            "point_semantics": "dense-point-map",
            "flow_semantics": "absent",
            "ray_semantics": "absent",
            "source_dependency_semantics": SOURCE_DEPENDENCY_SEMANTICS,
        },
        "includes_covariance": True,
        "gauge_estimator": "sequential",
        "uncertainty_calibration": "held_out",
    }


def _legacy_motioncrafter_metadata() -> dict[str, object]:
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


def _case(
    root: Path,
    case_id: str,
    group_id: str,
    *,
    identity_digit: str,
    loader_id: str = "6" * 64,
) -> dict[str, object]:
    truth = _truth()
    truth_path = root / f"{case_id}-truth.npz"
    prediction_path = root / f"{case_id}-cut3r.npz"
    save_truth(truth_path, truth)
    save_fused_prediction(
        prediction_path,
        _prediction(truth, 0.25),
        method_id="cut3r",
        fusion_method="covariance_intersection",
        metadata=_metadata(
            provider_manifest_id=identity_digit * 64,
            provider_manifest_sha256=chr(ord(identity_digit) + 1) * 64,
            provider_run_id=chr(ord(identity_digit) + 2) * 64,
            loader_id=loader_id,
        ),
    )
    return {
        "case_id": case_id,
        "group_id": group_id,
        "truth": truth_path.name,
        "predictions": {"cut3r": prediction_path.name},
        "boundary_frames": [],
        "prefix_frame_stop_exclusive": None,
    }


def _manifest(root: Path, cases: list[dict[str, object]]) -> Path:
    path = root / "evaluation.json"
    path.write_text(
        json.dumps(
            {
                "schema_name": "prob4d.provider-evaluation",
                "schema_version": 1,
                "primary_mode": "metric",
                "reference_method": "cut3r",
                "cases": cases,
                "metadata": {"split": "fresh-provider"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cut3r_evaluates_without_motioncrafter_metadata(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        [
            _case(tmp_path, "case-1", "group-1", identity_digit="1"),
            _case(tmp_path, "case-2", "group-2", identity_digit="4"),
        ],
    )

    report = run_provider_evaluation(
        path,
        tmp_path / "output",
        bootstrap_resamples=10,
    )

    contract = report["method_metadata"]["cut3r"]["provider_contract"]
    assert contract["provider_family"] == "cut3r"
    assert contract["provider_repository"] == "naver/CUT3R"
    assert contract["coordinate_semantics"] == "sequence-local-sim3"
    assert contract["identity_format"] == "prediction-provider-manifest-v1"
    assert report["aggregate"]["cut3r"]["group_count"] == 2
    artifact_metadata = report["cases"][0]["artifact"]["metadata"]
    assert "motioncrafter_revision" not in artifact_metadata
    assert "provider_identity" not in report["method_metadata"]["cut3r"]["metadata"]


def test_case_local_manifest_and_run_id_changes_are_not_contract_drift(
    tmp_path: Path,
) -> None:
    path = _manifest(
        tmp_path,
        [
            _case(tmp_path, "case-1", "group-1", identity_digit="1"),
            _case(tmp_path, "case-2", "group-2", identity_digit="4"),
        ],
    )

    report = run_provider_evaluation(path, tmp_path / "output", bootstrap_resamples=10)

    first_identity = report["cases"][0]["provider_identity"]
    second_identity = report["cases"][1]["provider_identity"]
    assert first_identity["provider_manifest_id"] != second_identity["provider_manifest_id"]
    assert first_identity["provider_run_id"] != second_identity["provider_run_id"]


def test_generic_provider_contract_drift_is_rejected(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        [
            _case(tmp_path, "case-1", "group-1", identity_digit="1"),
            _case(
                tmp_path,
                "case-2",
                "group-2",
                identity_digit="4",
                loader_id="7" * 64,
            ),
        ],
    )

    with pytest.raises(ValueError, match="provider semantics"):
        run_provider_evaluation(path, tmp_path / "output", bootstrap_resamples=10)


def test_generic_provider_identity_rejects_unknown_fields(tmp_path: Path) -> None:
    truth = _truth()
    truth_path = tmp_path / "truth.npz"
    prediction_path = tmp_path / "prediction.npz"
    save_truth(truth_path, truth)
    metadata = _metadata(
        provider_manifest_id="1" * 64,
        provider_manifest_sha256="2" * 64,
        provider_run_id="3" * 64,
    )
    identity = metadata["provider_identity"]
    assert isinstance(identity, dict)
    identity["unregistered_field"] = "not admitted"
    save_fused_prediction(
        prediction_path,
        _prediction(truth, 0.25),
        method_id="cut3r",
        fusion_method="covariance_intersection",
        metadata=metadata,
    )
    path = _manifest(
        tmp_path,
        [
            {
                "case_id": "case-1",
                "group_id": "group-1",
                "truth": truth_path.name,
                "predictions": {"cut3r": prediction_path.name},
                "boundary_frames": [],
                "prefix_frame_stop_exclusive": None,
            }
        ],
    )

    with pytest.raises(ValueError, match="provider identity fields changed"):
        run_provider_evaluation(path, tmp_path / "output", bootstrap_resamples=10)


def test_generic_identity_rejects_partial_motioncrafter_mirror(tmp_path: Path) -> None:
    truth = _truth()
    truth_path = tmp_path / "truth.npz"
    prediction_path = tmp_path / "prediction.npz"
    save_truth(truth_path, truth)
    metadata = _metadata(
        provider_manifest_id="1" * 64,
        provider_manifest_sha256="2" * 64,
        provider_run_id="3" * 64,
    )
    metadata["motioncrafter_revision"] = "b" * 40
    save_fused_prediction(
        prediction_path,
        _prediction(truth, 0.25),
        method_id="cut3r",
        fusion_method="covariance_intersection",
        metadata=metadata,
    )
    path = _manifest(
        tmp_path,
        [
            {
                "case_id": "case-1",
                "group_id": "group-1",
                "truth": truth_path.name,
                "predictions": {"cut3r": prediction_path.name},
                "boundary_frames": [],
                "prefix_frame_stop_exclusive": None,
            }
        ],
    )

    with pytest.raises(ValueError, match="partial MotionCrafter identity"):
        run_provider_evaluation(path, tmp_path / "output", bootstrap_resamples=10)


def test_historical_motioncrafter_evaluation_records_remain_identical(
    tmp_path: Path,
) -> None:
    truth = _truth()
    truth_path = tmp_path / "truth.npz"
    prediction_path = tmp_path / "motioncrafter.npz"
    save_truth(truth_path, truth)
    save_fused_prediction(
        prediction_path,
        _prediction(truth, 0.25),
        method_id="motioncrafter",
        fusion_method="uniform",
        metadata=_legacy_motioncrafter_metadata(),
    )
    case = ProviderEvaluationCase(
        case_id="case-1",
        group_id="group-1",
        truth_path=truth_path,
        predictions={"motioncrafter": prediction_path},
    )

    historical = evaluate_legacy_provider_cases(
        [case],
        allow_legacy_artifacts=False,
        evaluation_chunk_size=1,
    )
    provider_neutral = evaluate_provider_neutral_cases(
        [case],
        allow_legacy_artifacts=False,
        evaluation_chunk_size=1,
    )

    assert provider_neutral == historical
