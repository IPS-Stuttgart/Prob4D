from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from prob4d.cut3r_source_comparison_execution import (
    build_native_product,
    build_restarted_comparison_products,
    causal_window_schedule,
    newest_eligible_windows,
)
from prob4d.cut3r_source_comparison_verifier import (
    content_id,
    path_identity_sha256,
    validate_case_artifact,
    validate_shard_artifact,
    write_custody_receipt,
)
from prob4d.data import PredictionWindow
from prob4d.sim3 import Sim3

ROOT = Path(__file__).resolve().parents[1]
V1_2_RESULT = ROOT / "evidence" / "cut3r-source-comparison-smoke-v1-2" / "summary.json"


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts/science/run_cut3r_source_comparison.py"
    spec = importlib.util.spec_from_file_location("cut3r_source_comparison_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _window(
    window_id: str,
    frames: list[int],
    *,
    local_from_global: Sim3 | None = None,
    invalid: dict[int, tuple[int, int]] | None = None,
) -> PredictionWindow:
    height, width = 3, 4
    rows, columns = np.indices((height, width), dtype=np.float64)
    points = []
    rays = []
    for frame in frames:
        global_points = np.stack(
            (
                0.02 * columns + 0.001 * frame,
                0.02 * rows,
                np.ones_like(rows) + 0.002 * frame,
            ),
            axis=-1,
        )
        local_points = (
            global_points
            if local_from_global is None
            else local_from_global.transform_points(global_points)
        )
        points.append(local_points)
        rays.append(local_points / np.linalg.norm(local_points, axis=-1, keepdims=True))
    point_map = np.asarray(points, dtype=np.float32)
    valid = np.ones(point_map.shape[:-1], dtype=bool)
    for frame, pixel in (invalid or {}).items():
        valid[frames.index(frame)][pixel] = False
    return PredictionWindow(
        window_id=window_id,
        frame_indices=np.asarray(frames),
        point_map=point_map,
        valid_mask=valid,
        ray_directions=np.asarray(rays, dtype=np.float32),
        dense_storage_dtype="float32",
    )


def test_end_anchored_window_schedule_is_exact() -> None:
    schedule = causal_window_schedule(0, 58, window_size=25, overlap=8)

    assert [(span.start, span.stop) for span in schedule] == [
        (0, 25),
        (17, 42),
        (33, 58),
    ]
    assert [span.window_id for span in schedule] == [
        "window-000000-000025",
        "window-000017-000042",
        "window-000033-000058",
    ]


def test_newest_eligible_falls_back_per_pixel() -> None:
    older = _window("older", [0, 1, 2])
    newer = _window("newer", [2, 3, 4], invalid={2: (1, 2)})

    owned = newest_eligible_windows([older, newer])
    older_owned, newer_owned = owned

    assert np.all(older_owned.valid_mask[:2])
    assert np.count_nonzero(older_owned.valid_mask[2]) == 1
    assert older_owned.valid_mask[2, 1, 2]
    assert np.count_nonzero(newer_owned.valid_mask[0]) == 11
    assert not newer_owned.valid_mask[0, 1, 2]
    assert np.all(newer_owned.valid_mask[1:])


def test_restarted_arms_share_gauges_and_newest_has_one_contributor() -> None:
    first = _window("w0", [0, 1, 2])
    second = _window(
        "w1",
        [2, 3, 4],
        local_from_global=Sim3(scale=1.2, translation=np.asarray([0.1, -0.05, 0.2])),
    )

    products = build_restarted_comparison_products([first, second], random_seed=7)

    assert products.newest.frame_indices.tolist() == [0, 1, 2, 3, 4]
    assert products.fused.frame_indices.tolist() == [0, 1, 2, 3, 4]
    assert np.max(products.newest.contributors) == 1
    assert np.max(products.fused.contributors) == 2
    assert products.alignments[0].reference_id == "w0"
    assert products.alignments[0].moving_id == "w1"
    assert [estimate.window_id for estimate in products.gauges] == ["w0", "w1"]
    overlap_index = products.fused.frame_indices.tolist().index(2)
    np.testing.assert_allclose(
        products.newest.point_map[overlap_index],
        products.fused.point_map[overlap_index],
        atol=1e-5,
    )


def test_native_product_preserves_point_mean_and_support() -> None:
    window = _window("continuous", [0, 1, 2])

    native = build_native_product(window)

    np.testing.assert_allclose(native.point_map, window.point_map)
    np.testing.assert_array_equal(native.valid_mask, window.valid_mask)
    assert np.all(native.contributors[native.valid_mask] == 1)


def test_v1_2_smoke_result_is_content_bound_terminal_and_target_closed() -> None:
    result = json.loads(V1_2_RESULT.read_text(encoding="utf-8"))
    unsigned = dict(result)
    artifact_id = unsigned.pop("artifact_id")

    assert content_id(unsigned) == artifact_id
    assert result["result"] == "retained-technical-failure-no-retry"
    assert result["attempt"]["attempt_number"] == 1
    assert result["attempt"]["retry_authorized"] is False
    assert result["source_shards_authorized"] is False
    assert result["execution"]["ordinary_success_count"] == 0
    assert result["execution"]["source_predictions_written"] is False
    assert result["custody"]["publication_authorized"] is False
    assert not any(result["information_boundary"].values())


def test_cut3r_runtime_exposes_repository_and_internal_package(tmp_path: Path, monkeypatch) -> None:
    module = _load_runner_module()
    checkout = tmp_path / "CUT3R"
    (checkout / "src").mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))

    module._prepend_cut3r_import_paths(checkout)

    assert sys.path[:2] == [str(checkout / "src"), str(checkout)]


def test_runtime_bootstrap_failure_is_retained_without_progress(tmp_path: Path) -> None:
    module = _load_runner_module()
    processed = tmp_path / "processed"
    output = tmp_path / "output"
    checkout = tmp_path / "CUT3R"
    checkpoint = tmp_path / "model.pth"
    processed.mkdir()
    checkout.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    case = {
        "case_id": "development-case",
        "group_id": "development-group",
        "role": "development",
    }

    manifest = module._retain_runtime_failure(
        error=ModuleNotFoundError(f"missing package below {checkout}"),
        traceback_text=f"trace below {processed}",
        case=case,
        plan={"plan_id": "frozen-plan"},
        processed_root=processed,
        output_root=output,
        cut3r_checkout=checkout,
        checkpoint=checkpoint,
        shard_index=0,
    )

    assert manifest["status"] == "retained-technical-failure"
    assert manifest["source_rgb_frames_decoded"] is False
    assert manifest["cut3r_inference_executed"] is False
    assert manifest["source_predictions_written"] is False
    assert str(checkout) not in manifest["failure"]
    assert (output / "cases/development-case/case_manifest.json").is_file()


def test_provider_failure_after_decode_removes_source_frames(tmp_path: Path, monkeypatch) -> None:
    module = _load_runner_module()
    processed = tmp_path / "processed"
    output = tmp_path / "output"
    checkout = tmp_path / "CUT3R"
    checkpoint = tmp_path / "model.pth"
    processed.mkdir()
    checkout.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    video = processed / "video.mp4"
    video.write_bytes(b"video")

    class FailingRuntime:
        def __init__(self) -> None:
            self.checkout = checkout
            self.checkpoint = checkpoint

        def infer_to_direct_tree(self, frames: list[Path], output_root: Path) -> None:
            assert len(frames) == 1
            raise RuntimeError("provider inference failed")

    def decode_frames(
        source: Path,
        decoded_root: Path,
        *,
        frame_start: int,
        frame_stop_exclusive: int,
    ) -> list[Path]:
        assert source == video
        assert (frame_start, frame_stop_exclusive) == (0, 1)
        decoded_root.mkdir(parents=True)
        frame = decoded_root / "000000.png"
        frame.write_bytes(b"source-rgb")
        return [frame]

    monkeypatch.setattr(module, "_verify_video", lambda *_args: video)
    monkeypatch.setattr(module, "_decode_frames", decode_frames)
    case = {
        "case_id": "development-case",
        "group_id": "development-group",
        "role": "development",
        "frame_start": 0,
        "frame_stop_exclusive": 1,
    }

    manifest = module._execute_case(
        FailingRuntime(),
        case=case,
        plan={"plan_id": "a" * 64},
        processed_root=processed,
        output_root=output,
        shard_index=0,
    )

    case_root = output / "cases" / "development-case"
    assert manifest["status"] == "retained-technical-failure"
    assert manifest["source_rgb_frames_decoded"] is True
    assert manifest["cut3r_inference_executed"] is False
    assert not (case_root / "decoded").exists()
    assert [member["path"] for member in manifest["members"]] == ["failure_traceback.txt"]
    validated = validate_case_artifact(case_root, expected_plan_id="a" * 64)
    assert validated["artifact_id"] == manifest["artifact_id"]


def test_amended_plan_selects_only_the_registered_replacement_smoke() -> None:
    module = _load_runner_module()
    plan = {
        "schema_version": 3,
        "execution": {
            "shard_count": 2,
            "smoke_policy": {"registered_case_id": "development-new"},
        },
        "cases": [
            {"case_id": "development-new", "role": "development"},
            {"case_id": "development-old", "role": "development"},
        ],
    }

    selected = module._selected_cases(
        plan,
        shard_index=0,
        shard_count=2,
        smoke_case_id="development-new",
    )
    assert [case["case_id"] for case in selected] == ["development-new"]

    with pytest.raises(ValueError, match="amended registered case"):
        module._selected_cases(
            plan,
            shard_index=0,
            shard_count=2,
            smoke_case_id="development-old",
        )


def _amended_gate_plan(output_root: Path, ledger: Path) -> dict[str, object]:
    case_id = "development-new"
    return {
        "schema_version": 3,
        "plan_id": "a" * 64,
        "execution": {
            "smoke_policy": {
                "registered_case_id": case_id,
                "registered_case_id_sha256": hashlib.sha256(case_id.encode("utf-8")).hexdigest(),
                "registered_output_root_path_sha256": path_identity_sha256(output_root),
                "registered_attempt_ledger_path_sha256": path_identity_sha256(ledger),
            }
        },
    }


def _write_successful_smoke(output_root: Path, plan: dict[str, object]) -> Path:
    case_id = "development-new"
    case_root = output_root / "cases" / case_id
    member = case_root / "predictions" / "result.bin"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"prediction")
    manifest: dict[str, object] = {
        "schema": "prob4d.cut3r-source-comparison-case",
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "case_id": case_id,
        "group_id": "development-group",
        "role": "development",
        "status": "ordinary-success",
        "elapsed_seconds": 1.0,
        "failure": None,
        "members": [
            {
                "path": "predictions/result.bin",
                "sha256": hashlib.sha256(b"prediction").hexdigest(),
                "byte_count": len(b"prediction"),
            }
        ],
        "source_rgb_frames_decoded": True,
        "cut3r_inference_executed": True,
        "source_predictions_written": True,
        "source_residuals_or_truth_opened": False,
        "candidate_reference_file_contents_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
    }
    manifest["artifact_id"] = content_id(manifest)
    (case_root / "case_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "schema": "prob4d.cut3r-source-comparison-shard",
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "scope": "development-smoke",
        "shard_index": 0,
        "shard_count": 2,
        "case_count": 1,
        "ordinary_success_count": 1,
        "retained_technical_failure_count": 0,
        "case_artifact_ids": [manifest["artifact_id"]],
        "source_residuals_or_truth_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
    }
    report["artifact_id"] = content_id(report)
    report_path = output_root / "shards" / "smoke.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return report_path


def test_amended_attempt_is_path_bound_and_source_shards_require_custody(
    tmp_path: Path,
) -> None:
    module = _load_runner_module()
    output_root = tmp_path / "registered-smoke"
    ledger = tmp_path / "attempts" / "smoke.json"
    plan = _amended_gate_plan(output_root, ledger)

    with pytest.raises(ValueError, match="require the smoke artifact"):
        module._validate_amended_shard_authorization(
            plan,
            smoke_output_root=None,
            smoke_shard_report=None,
            smoke_custody_receipt=None,
            smoke_attempt_ledger=None,
        )

    module._claim_registered_smoke_attempt(
        plan,
        smoke_case_id="development-new",
        output_root=output_root,
        attempt_ledger=ledger,
    )
    with pytest.raises(FileExistsError, match="already consumed"):
        module._claim_registered_smoke_attempt(
            plan,
            smoke_case_id="development-new",
            output_root=output_root,
            attempt_ledger=ledger,
        )
    with pytest.raises(ValueError, match="output root differs"):
        module._claim_registered_smoke_attempt(
            plan,
            smoke_case_id="development-new",
            output_root=tmp_path / "fresh-unregistered-output",
            attempt_ledger=tmp_path / "fresh-unregistered-ledger.json",
        )

    report_path = _write_successful_smoke(output_root, plan)
    receipt = validate_shard_artifact(
        output_root,
        report_path,
        expected_plan_id=str(plan["plan_id"]),
        require_success=True,
    )
    receipt_path = tmp_path / "custody" / "smoke.json"
    write_custody_receipt(receipt_path, receipt)

    validated = module._validate_amended_shard_authorization(
        plan,
        smoke_output_root=output_root,
        smoke_shard_report=report_path,
        smoke_custody_receipt=receipt_path,
        smoke_attempt_ledger=ledger,
    )
    assert validated == receipt
