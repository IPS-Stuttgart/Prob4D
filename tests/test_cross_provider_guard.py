from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import prob4d.cross_provider_guard as cross_provider
from prob4d.cli import main as grouped_main
from prob4d.cross_provider_guard import (
    apply_cross_provider_calibration,
    compute_cross_provider_score,
    evaluate_cross_provider_panel,
    fit_cross_provider_calibration,
    load_cross_provider_calibration,
    load_cross_provider_decision,
    run_cross_provider_guard_stress,
    save_cross_provider_calibration,
    save_cross_provider_decision,
)
from prob4d.data import PredictionWindow
from prob4d.prediction_provider_manifest import (
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    save_prediction_provider_manifest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _window(path: Path, *, window_id: str, frames: tuple[int, ...] = (0, 1)) -> None:
    points = np.zeros((len(frames), 1, 2, 3), dtype=np.float32)
    points[..., 2] = 1.0
    valid = np.ones(points.shape[:-1], dtype=bool)
    PredictionWindow(
        window_id=window_id,
        frame_indices=np.asarray(frames, dtype=np.int64),
        point_map=points,
        valid_mask=valid,
        dense_storage_dtype="float32",
    ).to_npz(path)


def _provider_manifest(
    root: Path,
    *,
    name: str,
    family: str,
    repository: str,
    revision: str,
    run_id: str,
    model_set_id: str,
    loader_id: str,
    video_sha256: str,
    coordinate_semantics: str,
    sequence_id: str = "case-a",
) -> tuple[Path, PredictionProviderManifestV1]:
    payload_path = root / f"{name}.npz"
    window_id = f"{name}-window"
    _window(payload_path, window_id=window_id)
    descriptor = PredictionPayloadDescriptorV1(
        product_role="external-sequence",
        window_id=window_id,
        path=payload_path.name,
        sha256=_sha(payload_path),
        byte_count=payload_path.stat().st_size,
        view_id="camera-0",
        stochastic_member_id=f"member:{name}",
        dependence_group_ids=(
            f"model-set:{model_set_id}",
            f"input-video:{video_sha256}",
            f"provider-run:{run_id}",
        ),
        dense_storage_dtype="float32",
        has_scene_flow=False,
        has_ray_directions=False,
        frame_lineage=tuple(
            PredictionFrameLineageV1(
                output_frame_id=frame,
                source_frame_start=0,
                source_frame_stop_exclusive=2,
                contributor_ids=(window_id,),
            )
            for frame in (0, 1)
        ),
    )
    manifest = PredictionProviderManifestV1(
        sequence_id=sequence_id,
        provider_family=family,
        provider_repository=repository,
        provider_revision=revision,
        provider_run_id=run_id,
        model_set_id=model_set_id,
        loader_id=loader_id,
        coordinate_semantics=coordinate_semantics,
        point_semantics="dense-point-map",
        flow_semantics="absent",
        ray_semantics="absent",
        payloads=(descriptor,),
        metadata={
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
        },
    )
    manifest_path = root / f"{name}.json"
    save_prediction_provider_manifest(manifest_path, manifest)
    return manifest_path, manifest


def _matched(
    path: Path,
    *,
    offset_m: float = 0.0,
    valid_mask: np.ndarray | None = None,
    explicit_cross_covariance: bool = False,
) -> None:
    rows = 8
    first = np.zeros((rows, 3), dtype=np.float64)
    second = np.zeros((rows, 3), dtype=np.float64)
    second[:, 0] = offset_m
    covariance = np.repeat(
        (0.002**2 * np.eye(3, dtype=np.float64))[None],
        rows,
        axis=0,
    )
    valid = np.ones(rows, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    arrays: dict[str, np.ndarray] = {
        "first_points_m": first,
        "second_points_m": second,
        "first_covariance_m2": covariance,
        "second_covariance_m2": covariance,
        "valid_mask": valid,
        "alignment_artifact_id": np.asarray("a" * 64),
        "row_identity_sha256": np.asarray("b" * 64),
        "coordinate_frame_id": np.asarray("registered-world-case-a"),
    }
    if explicit_cross_covariance:
        arrays["cross_covariance_m2"] = np.zeros_like(covariance)
    np.savez(path, **arrays)


def _panel(
    path: Path,
    *,
    purpose: str,
    first_manifest: Path,
    second_manifest: Path,
    first_payload_id: str,
    second_payload_id: str,
    matched: Path,
) -> None:
    record = {
        "schema": "prob4d.cross-provider-panel",
        "schema_version": 1,
        "purpose": purpose,
        "cases": [
            {
                "case_id": "case-a",
                "first_manifest": first_manifest.name,
                "second_manifest": second_manifest.name,
                "first_payload_ids": [first_payload_id],
                "second_payload_ids": [second_payload_id],
                "matched_observations": matched.name,
                "matched_observations_sha256": _sha(matched),
                "alignment_artifact_id": "a" * 64,
                "row_identity_sha256": "b" * 64,
                "coordinate_frame_id": "registered-world-case-a",
            }
        ],
        "metadata": {
            "uses_truth": False,
            "uses_target_outcomes": False,
            "uses_downstream_physical_innovation": False,
            "alignment_uses_truth": False,
            "alignment_uses_downstream_physical_innovation": False,
        },
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def _provider_pair(
    root: Path,
    *,
    second_video_sha256: str | None = None,
    same_contract: bool = False,
) -> tuple[
    Path,
    PredictionProviderManifestV1,
    Path,
    PredictionProviderManifestV1,
]:
    video_sha = "1" * 64
    first_path, first = _provider_manifest(
        root,
        name="motioncrafter",
        family="MotionCrafter",
        repository="TencentARC/MotionCrafter",
        revision="2" * 40,
        run_id="3" * 64,
        model_set_id="4" * 64,
        loader_id="5" * 64,
        video_sha256=video_sha,
        coordinate_semantics="window-local-sim3",
    )
    second_path, second = _provider_manifest(
        root,
        name="vggt",
        family="MotionCrafter" if same_contract else "VGGT",
        repository=("TencentARC/MotionCrafter" if same_contract else "facebookresearch/vggt"),
        revision="2" * 40 if same_contract else "6" * 40,
        run_id="7" * 64,
        model_set_id="4" * 64 if same_contract else "8" * 64,
        loader_id="5" * 64 if same_contract else "9" * 64,
        video_sha256=(video_sha if second_video_sha256 is None else second_video_sha256),
        coordinate_semantics=("window-local-sim3" if same_contract else "sequence-local-sim3"),
    )
    return first_path, first, second_path, second


def test_unknown_dependence_bound_is_conservative() -> None:
    sigma = 0.01
    first = np.zeros((1, 3), dtype=np.float64)
    second = np.asarray([[2.0 * sigma, 0.0, 0.0]], dtype=np.float64)
    covariance = np.asarray([sigma**2 * np.eye(3)], dtype=np.float64)
    valid = np.ones(1, dtype=bool)

    unknown = compute_cross_provider_score(
        first,
        second,
        covariance,
        covariance,
        valid,
        row_quantile=0.5,
    )
    explicit = compute_cross_provider_score(
        first,
        second,
        covariance,
        covariance,
        valid,
        row_quantile=0.5,
        cross_covariance_m2=np.zeros_like(covariance),
    )

    assert unknown.case_score == pytest.approx(np.sqrt(1.0 / 3.0))
    assert explicit.case_score == pytest.approx(np.sqrt(2.0 / 3.0))
    assert unknown.case_score < explicit.case_score


def test_calibration_and_target_rejection_roundtrip(tmp_path: Path) -> None:
    first_path, first, second_path, second = _provider_pair(tmp_path)
    calibration_npz = tmp_path / "calibration.npz"
    _matched(calibration_npz)
    calibration_panel_path = tmp_path / "calibration-panel.json"
    _panel(
        calibration_panel_path,
        purpose="calibration",
        first_manifest=first_path,
        second_manifest=second_path,
        first_payload_id=str(first.payloads[0].payload_id),
        second_payload_id=str(second.payloads[0].payload_id),
        matched=calibration_npz,
    )
    calibration_panel = evaluate_cross_provider_panel(
        calibration_panel_path,
        row_quantile=0.95,
        expected_purpose="calibration",
    )
    calibration = fit_cross_provider_calibration(
        calibration_panel,
        miscoverage=0.5,
        row_quantile=0.95,
        minimum_support_fraction=0.75,
    )
    calibration_path = tmp_path / "guard.json"
    save_cross_provider_calibration(calibration_path, calibration)
    loaded_calibration = load_cross_provider_calibration(calibration_path)
    assert loaded_calibration.artifact_id == calibration.artifact_id

    target_npz = tmp_path / "target.npz"
    _matched(target_npz, offset_m=0.05)
    target_panel_path = tmp_path / "target-panel.json"
    _panel(
        target_panel_path,
        purpose="target",
        first_manifest=first_path,
        second_manifest=second_path,
        first_payload_id=str(first.payloads[0].payload_id),
        second_payload_id=str(second.payloads[0].payload_id),
        matched=target_npz,
    )
    target_panel = evaluate_cross_provider_panel(
        target_panel_path,
        row_quantile=loaded_calibration.row_quantile,
        expected_purpose="target",
    )
    decision = apply_cross_provider_calibration(loaded_calibration, target_panel)
    assert decision.accepted_count == 0
    assert decision.rejected_count == 1
    assert decision.cases[0].rejection_reasons == ("cross-provider-disagreement",)
    decision_path = tmp_path / "decision.json"
    save_cross_provider_decision(decision_path, decision)
    assert load_cross_provider_decision(decision_path).artifact_id == decision.artifact_id


def test_low_common_support_is_rejected(tmp_path: Path) -> None:
    first_path, first, second_path, second = _provider_pair(tmp_path)
    calibration_npz = tmp_path / "calibration.npz"
    _matched(calibration_npz)
    calibration_panel_path = tmp_path / "calibration-panel.json"
    _panel(
        calibration_panel_path,
        purpose="calibration",
        first_manifest=first_path,
        second_manifest=second_path,
        first_payload_id=str(first.payloads[0].payload_id),
        second_payload_id=str(second.payloads[0].payload_id),
        matched=calibration_npz,
    )
    calibration = fit_cross_provider_calibration(
        evaluate_cross_provider_panel(
            calibration_panel_path,
            row_quantile=0.95,
            expected_purpose="calibration",
        ),
        miscoverage=0.5,
        row_quantile=0.95,
        minimum_support_fraction=0.75,
    )

    target_npz = tmp_path / "target.npz"
    _matched(
        target_npz,
        valid_mask=np.asarray([True, True, False, False, False, False, False, False]),
    )
    target_panel_path = tmp_path / "target-panel.json"
    _panel(
        target_panel_path,
        purpose="target",
        first_manifest=first_path,
        second_manifest=second_path,
        first_payload_id=str(first.payloads[0].payload_id),
        second_payload_id=str(second.payloads[0].payload_id),
        matched=target_npz,
    )
    decision = apply_cross_provider_calibration(
        calibration,
        evaluate_cross_provider_panel(
            target_panel_path,
            row_quantile=calibration.row_quantile,
            expected_purpose="target",
        ),
    )
    assert decision.cases[0].rejection_reasons == ("insufficient-common-support",)


def test_alternative_constructions_from_one_provider_are_not_two_votes(
    tmp_path: Path,
) -> None:
    first_path, first, second_path, second = _provider_pair(
        tmp_path,
        same_contract=True,
    )
    matched = tmp_path / "matched.npz"
    _matched(matched)
    panel_path = tmp_path / "panel.json"
    _panel(
        panel_path,
        purpose="calibration",
        first_manifest=first_path,
        second_manifest=second_path,
        first_payload_id=str(first.payloads[0].payload_id),
        second_payload_id=str(second.payloads[0].payload_id),
        matched=matched,
    )
    with pytest.raises(ValueError, match="must not be presented as distinct providers"):
        evaluate_cross_provider_panel(
            panel_path,
            row_quantile=0.95,
            expected_purpose="calibration",
        )


def test_provider_pair_must_share_the_exact_input_video(tmp_path: Path) -> None:
    first_path, first, second_path, second = _provider_pair(
        tmp_path,
        second_video_sha256="f" * 64,
    )
    matched = tmp_path / "matched.npz"
    _matched(matched)
    panel_path = tmp_path / "panel.json"
    _panel(
        panel_path,
        purpose="calibration",
        first_manifest=first_path,
        second_manifest=second_path,
        first_payload_id=str(first.payloads[0].payload_id),
        second_payload_id=str(second.payloads[0].payload_id),
        matched=matched,
    )
    with pytest.raises(ValueError, match="share exactly one input-video"):
        evaluate_cross_provider_panel(
            panel_path,
            row_quantile=0.95,
            expected_purpose="calibration",
        )


def test_resigned_threshold_tamper_is_rejected(tmp_path: Path) -> None:
    first_path, first, second_path, second = _provider_pair(tmp_path)
    matched = tmp_path / "matched.npz"
    _matched(matched)
    panel_path = tmp_path / "panel.json"
    _panel(
        panel_path,
        purpose="calibration",
        first_manifest=first_path,
        second_manifest=second_path,
        first_payload_id=str(first.payloads[0].payload_id),
        second_payload_id=str(second.payloads[0].payload_id),
        matched=matched,
    )
    calibration = fit_cross_provider_calibration(
        evaluate_cross_provider_panel(
            panel_path,
            row_quantile=0.95,
            expected_purpose="calibration",
        ),
        miscoverage=0.5,
        row_quantile=0.95,
        minimum_support_fraction=0.75,
    )
    record = calibration.to_record()
    threshold = record["finite_sample_threshold"]
    assert isinstance(threshold, dict)
    threshold["threshold"] = float(threshold["threshold"]) + 1.0
    descriptor = {key: value for key, value in record.items() if key != "artifact_id"}
    record["artifact_id"] = cross_provider._sha256_json(descriptor)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from calibration cases"):
        load_cross_provider_calibration(path)


def test_calibration_rejects_a_different_declared_row_quantile(
    tmp_path: Path,
) -> None:
    first_path, first, second_path, second = _provider_pair(tmp_path)
    matched = tmp_path / "matched.npz"
    _matched(matched)
    panel_path = tmp_path / "panel.json"
    _panel(
        panel_path,
        purpose="calibration",
        first_manifest=first_path,
        second_manifest=second_path,
        first_payload_id=str(first.payloads[0].payload_id),
        second_payload_id=str(second.payloads[0].payload_id),
        matched=matched,
    )
    panel = evaluate_cross_provider_panel(
        panel_path,
        row_quantile=0.9,
        expected_purpose="calibration",
    )
    with pytest.raises(ValueError, match="row_quantile differs"):
        fit_cross_provider_calibration(
            panel,
            miscoverage=0.5,
            row_quantile=0.95,
            minimum_support_fraction=0.75,
        )


def test_resigned_provider_contract_tamper_is_rejected(tmp_path: Path) -> None:
    first_path, first, second_path, second = _provider_pair(tmp_path)
    matched = tmp_path / "matched.npz"
    _matched(matched)
    panel_path = tmp_path / "panel.json"
    _panel(
        panel_path,
        purpose="calibration",
        first_manifest=first_path,
        second_manifest=second_path,
        first_payload_id=str(first.payloads[0].payload_id),
        second_payload_id=str(second.payloads[0].payload_id),
        matched=matched,
    )
    calibration = fit_cross_provider_calibration(
        evaluate_cross_provider_panel(
            panel_path,
            row_quantile=0.95,
            expected_purpose="calibration",
        ),
        miscoverage=0.5,
        row_quantile=0.95,
        minimum_support_fraction=0.75,
    )
    record = calibration.to_record()
    provider = record["first_provider"]
    assert isinstance(provider, dict)
    provider["provider_family"] = "DifferentProvider"
    contract_descriptor = {key: value for key, value in provider.items() if key != "contract_id"}
    provider["contract_id"] = cross_provider._sha256_json(contract_descriptor)
    descriptor = {key: value for key, value in record.items() if key != "artifact_id"}
    record["artifact_id"] = cross_provider._sha256_json(descriptor)
    path = tmp_path / "provider-tampered.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the provider contracts"):
        load_cross_provider_calibration(path)


def test_controlled_stress_exposes_provider_specific_but_not_shared_bias() -> None:
    report = run_cross_provider_guard_stress(
        calibration_cases=120,
        clean_target_cases=200,
        corrupted_target_cases=200,
        shared_bias_target_cases=200,
        rows_per_case=64,
        miscoverage=0.1,
        seed=20260806,
    )
    assert all(report["gates"].values())
    results = report["results"]
    assert results["provider_specific_corruption_detection_rate"] >= 0.95
    assert results["shared_common_bias_rejection_rate"] <= 0.13


def test_grouped_cli_exposes_cross_provider_guard(capsys: Any) -> None:
    assert grouped_main(["diagnostic", "--help"]) == 0
    output = capsys.readouterr().out
    assert "cross-provider-guard" in output
