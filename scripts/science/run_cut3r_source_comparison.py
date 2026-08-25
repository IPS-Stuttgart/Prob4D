#!/usr/bin/env python3
"""Execute one shard of the frozen source-only recurrent CUT3R comparison."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from prob4d._atomic_file import atomic_write_bytes
from prob4d.cut3r_direct_provider_adapter import import_cut3r_direct_prediction_manifest
from prob4d.cut3r_source_comparison_execution import (
    SOURCE_COMPARISON_METHOD_ID,
    build_native_product,
    build_restarted_comparison_products,
)
from prob4d.cut3r_source_comparison_plan import (
    AMENDED_EXECUTION_PLAN_VERSION,
    REVOKED_AMENDED_EXECUTION_PLAN_VERSION,
    _content_id,
    _file_sha256,
    _runtime_inventory,
    load_execution_plan,
)
from prob4d.cut3r_source_comparison_verifier import (
    claim_smoke_attempt,
    path_identity_sha256,
    validate_custody_receipt,
    validate_shard_artifact,
    validate_smoke_attempt,
)
from prob4d.data import PredictionWindow
from prob4d.io import save_fused_prediction


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--cut3r-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--smoke-case-id")
    parser.add_argument("--smoke-attempt-ledger", type=Path)
    parser.add_argument("--required-smoke-output-root", type=Path)
    parser.add_argument("--required-smoke-shard-report", type=Path)
    parser.add_argument("--required-smoke-custody-receipt", type=Path)
    return parser


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_bytes(path, encoded, overwrite=False)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise


def _safe_error(error: BaseException, redactions: Mapping[str, str]) -> str:
    message = f"{type(error).__name__}: {error}"
    for raw, replacement in sorted(redactions.items(), key=lambda item: -len(item[0])):
        message = message.replace(raw, replacement)
    return " ".join(message.split())[:2000]


def _verify_video(processed_root: Path, case: Mapping[str, Any]) -> Path:
    relative = cast(str, case["relative_video_path"])
    if "\\" in relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("case video path is not canonical")
    candidate = processed_root
    for part in Path(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("case video path traverses a symbolic link")
    video = candidate.resolve(strict=True)
    video.relative_to(processed_root.resolve(strict=True))
    if not video.is_file():
        raise ValueError("case video is not a regular file")
    if int(video.stat().st_size) != case["video_byte_count"]:
        raise ValueError("case video byte count changed")
    if _file_sha256(video) != case["video_sha256"]:
        raise ValueError("case video digest changed")
    return video


def _decode_frames(
    video: Path,
    destination: Path,
    *,
    frame_start: int,
    frame_stop_exclusive: int,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=False)
    expression = f"select=between(n\\,{frame_start}\\,{frame_stop_exclusive - 1})"
    command = (
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        os.fspath(video),
        "-vf",
        expression,
        "-vsync",
        "0",
        "-start_number",
        "0",
        os.fspath(destination / "%06d.png"),
    )
    subprocess.run(command, check=True, capture_output=True)
    frames = sorted(destination.glob("*.png"))
    expected_count = frame_stop_exclusive - frame_start
    expected_names = [f"{index:06d}.png" for index in range(expected_count)]
    if [path.name for path in frames] != expected_names:
        raise ValueError("ffmpeg did not emit the exact frozen frame interval")
    return frames


def _prepend_cut3r_import_paths(checkout: Path) -> None:
    """Expose CUT3R's repository modules and its internal ``dust3r`` package."""
    for candidate in (checkout, checkout / "src"):
        value = os.fspath(candidate)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


class _Cut3RRuntime:
    def __init__(self, checkout: Path, checkpoint: Path, *, device: str) -> None:
        self.checkout = checkout
        self.checkpoint = checkpoint
        self.device = device
        _prepend_cut3r_import_paths(checkout)
        from add_ckpt_path import add_path_to_dust3r

        add_path_to_dust3r(os.fspath(checkpoint))
        import demo
        import torch
        from dust3r.inference import inference
        from dust3r.model import ARCroco3DStereo
        from dust3r.post_process import estimate_focal_knowing_depth
        from dust3r.utils.camera import pose_encoding_to_camera

        self.demo = demo
        self.torch = torch
        self.inference = inference
        self.estimate_focal = estimate_focal_knowing_depth
        self.pose_encoding_to_camera = pose_encoding_to_camera
        self._reset_seed()
        self.model = ARCroco3DStereo.from_pretrained(os.fspath(checkpoint)).to(device)
        self.model.eval()

    def _reset_seed(self) -> None:
        random.seed(42)
        np.random.seed(42)
        self.torch.manual_seed(42)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(42)
        self.torch.backends.cudnn.benchmark = False

    def infer_to_direct_tree(self, frame_paths: list[Path], output: Path) -> None:
        if not frame_paths:
            raise ValueError("CUT3R inference requires at least one frame")
        self._reset_seed()
        views = self.demo.prepare_input(
            img_paths=[os.fspath(path) for path in frame_paths],
            img_mask=[True] * len(frame_paths),
            size=512,
            revisit=1,
            update=True,
        )
        outputs, _ = self.inference(views, self.model, self.device, verbose=False)
        predictions = outputs["pred"]
        if len(predictions) != len(frame_paths):
            raise ValueError("CUT3R returned a noncanonical prediction count")

        points = self.torch.cat(
            [prediction["pts3d_in_self_view"].cpu() for prediction in predictions],
            dim=0,
        )
        confidence = self.torch.cat(
            [prediction["conf_self"].cpu() for prediction in predictions],
            dim=0,
        )
        poses = self.torch.cat(
            [
                self.pose_encoding_to_camera(prediction["camera_pose"].clone()).cpu()
                for prediction in predictions
            ],
            dim=0,
        )
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("CUT3R direct point maps changed shape")
        if confidence.shape != points.shape[:-1] or poses.shape != (len(frame_paths), 4, 4):
            raise ValueError("CUT3R confidence or pose output changed shape")
        _, height, width, _ = points.shape
        principal = self.torch.tensor(
            [width // 2, height // 2], dtype=points.dtype, device=points.device
        ).repeat(len(frame_paths), 1)
        focal = self.estimate_focal(points, principal, focal_mode="weiszfeld").cpu()
        intrinsics = self.torch.eye(3).repeat(len(frame_paths), 1, 1)
        intrinsics[:, 0, 0] = focal
        intrinsics[:, 1, 1] = focal
        intrinsics[:, 0, 2] = principal[:, 0].cpu()
        intrinsics[:, 1, 2] = principal[:, 1].cpu()

        for directory in ("points", "conf", "camera"):
            (output / directory).mkdir(parents=True, exist_ok=False)
        for index in range(len(frame_paths)):
            stem = f"{index:06d}"
            np.save(
                output / "points" / f"{stem}.npy",
                points[index].numpy().astype(np.float32, copy=False),
                allow_pickle=False,
            )
            np.save(
                output / "conf" / f"{stem}.npy",
                confidence[index].numpy().astype(np.float32, copy=False),
                allow_pickle=False,
            )
            np.savez(
                output / "camera" / f"{stem}.npz",
                pose=poses[index].numpy().astype(np.float64, copy=False),
                intrinsics=intrinsics[index].numpy().astype(np.float64, copy=False),
            )


def _load_imported_window(manifest_path: Path) -> PredictionWindow:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payloads = manifest.get("payloads")
    if type(payloads) is not list or len(payloads) != 1:
        raise ValueError("direct provider manifest must contain exactly one payload")
    payload = cast(Mapping[str, Any], payloads[0])
    return PredictionWindow.from_npz(
        manifest_path.parent / cast(str, payload["path"]),
        dense_storage_dtype=cast(Any, payload["dense_storage_dtype"]),
    )


def _import_direct_window(
    raw_root: Path,
    manifest_path: Path,
    *,
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
    frame_start: int,
    window_id: str,
) -> PredictionWindow:
    provider = cast(Mapping[str, Any], plan["provider"])
    method = cast(Mapping[str, Any], plan["method"])
    import_cut3r_direct_prediction_manifest(
        raw_root,
        manifest_path,
        sequence_id=f"{case['case_id']}:{window_id}",
        cut3r_revision=cast(str, provider["revision"]),
        checkpoint_sha256=cast(str, provider["checkpoint_sha256"]),
        input_video_sha256=cast(str, case["video_sha256"]),
        input_video_byte_count=cast(int, case["video_byte_count"]),
        frame_start=frame_start,
        view_id=cast(str, case["case_id"]),
        window_id=window_id,
        confidence_threshold=float(method["confidence_threshold"]),
        storage_dtype=cast(Any, method["storage_dtype"]),
    )
    return _load_imported_window(manifest_path)


def _alignment_record(products: Any) -> list[dict[str, object]]:
    result = []
    for alignment in products.alignments:
        fitted = alignment.result
        result.append(
            {
                "reference_id": alignment.reference_id,
                "moving_id": alignment.moving_id,
                "common_frames": alignment.common_frames.tolist(),
                "sim3_reference_from_moving": fitted.transform.as_vector().tolist(),
                "covariance": fitted.covariance.tolist(),
                "residual_rms": fitted.residual_rms,
                "inlier_fraction": fitted.inlier_fraction,
                "num_correspondences": fitted.num_correspondences,
                "covariance_method": fitted.covariance_method,
                "num_covariance_clusters": fitted.num_covariance_clusters,
                "information_rank": fitted.information_rank,
                "information_condition": fitted.information_condition,
                "covariance_fallback": fitted.covariance_fallback,
            }
        )
    return result


def _members(root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "case_manifest.json":
            continue
        if path.is_symlink():
            raise ValueError("case artifact contains a symbolic link")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _file_sha256(path),
                "byte_count": int(path.stat().st_size),
            }
        )
    return records


def _publish_case_manifest(
    staging: Path,
    *,
    plan: Mapping[str, Any],
    case: Mapping[str, Any],
    status: str,
    elapsed_seconds: float,
    failure: str | None,
    progress: Mapping[str, bool],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": "prob4d.cut3r-source-comparison-case",
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "case_id": case["case_id"],
        "group_id": case["group_id"],
        "role": case["role"],
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "failure": failure,
        "members": _members(staging),
        "source_rgb_frames_decoded": progress["source_rgb_frames_decoded"],
        "cut3r_inference_executed": progress["cut3r_inference_executed"],
        "source_predictions_written": progress["source_predictions_written"],
        "source_residuals_or_truth_opened": False,
        "candidate_reference_file_contents_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
    }
    manifest["artifact_id"] = _content_id(manifest)
    _write_json_no_clobber(staging / "case_manifest.json", manifest)
    return manifest


def _execute_case(
    runtime: _Cut3RRuntime,
    *,
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
    processed_root: Path,
    output_root: Path,
    shard_index: int,
) -> dict[str, Any]:
    case_id = cast(str, case["case_id"])
    final = output_root / "cases" / case_id
    if final.exists():
        manifest = json.loads((final / "case_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("plan_id") != plan["plan_id"] or manifest.get("case_id") != case_id:
            raise FileExistsError(f"different retained case artifact exists: {case_id}")
        return cast(dict[str, Any], manifest)
    staging = output_root / "staging" / f"{case_id}-shard-{shard_index}-{os.getpid()}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    redactions = {
        os.fspath(processed_root): "<DEFORM360_PROCESSED_ROOT>",
        os.fspath(output_root): "<OUTPUT_ROOT>",
        os.fspath(runtime.checkout): "<CUT3R_CHECKOUT>",
        os.fspath(runtime.checkpoint): "<CUT3R_CHECKPOINT>",
    }
    status = "ordinary-success"
    failure = None
    progress = {
        "source_rgb_frames_decoded": False,
        "cut3r_inference_executed": False,
        "source_predictions_written": False,
    }
    try:
        video = _verify_video(processed_root, case)
        frames = _decode_frames(
            video,
            staging / "decoded",
            frame_start=cast(int, case["frame_start"]),
            frame_stop_exclusive=cast(int, case["frame_stop_exclusive"]),
        )
        progress["source_rgb_frames_decoded"] = True
        provider_root = staging / "providers"
        native_raw = staging / "raw" / "native-continuous"
        runtime.infer_to_direct_tree(frames, native_raw)
        progress["cut3r_inference_executed"] = True
        native_window = _import_direct_window(
            native_raw,
            provider_root / "native-continuous.json",
            case=case,
            plan=plan,
            frame_start=cast(int, case["frame_start"]),
            window_id="native-continuous",
        )
        windowed = []
        method = cast(Mapping[str, Any], plan["method"])
        for raw_span in cast(list[Mapping[str, Any]], method["window_schedule"]):
            start = cast(int, raw_span["start"])
            stop = cast(int, raw_span["stop"])
            window_id = cast(str, raw_span["window_id"])
            raw_root = staging / "raw" / window_id
            runtime.infer_to_direct_tree(frames[start:stop], raw_root)
            windowed.append(
                _import_direct_window(
                    raw_root,
                    provider_root / f"{window_id}.json",
                    case=case,
                    plan=plan,
                    frame_start=start,
                    window_id=window_id,
                )
            )
        predictions = staging / "predictions"
        save_fused_prediction(
            predictions / "native-continuous.npz",
            build_native_product(native_window),
            method_id=f"{SOURCE_COMPARISON_METHOD_ID}:native-continuous",
            fusion_method="uniform",
            metadata={"plan_id": plan["plan_id"], "case_id": case_id},
            compressed=True,
        )
        diagnostics = {}
        for seed in cast(list[int], method["random_seeds"]):
            products = build_restarted_comparison_products(windowed, random_seed=seed)
            seed_root = predictions / f"seed-{seed}"
            save_fused_prediction(
                seed_root / "restarted-newest.npz",
                products.newest,
                method_id=f"{SOURCE_COMPARISON_METHOD_ID}:restarted-newest",
                fusion_method="uniform",
                metadata={"plan_id": plan["plan_id"], "case_id": case_id, "seed": seed},
                compressed=True,
            )
            save_fused_prediction(
                seed_root / "restarted-prob4d-fused.npz",
                products.fused,
                method_id=f"{SOURCE_COMPARISON_METHOD_ID}:restarted-prob4d-fused",
                fusion_method="uniform",
                metadata={"plan_id": plan["plan_id"], "case_id": case_id, "seed": seed},
                compressed=True,
            )
            diagnostics[str(seed)] = {
                "alignments": _alignment_record(products),
                "gauges": [
                    {
                        "window_id": estimate.window_id,
                        "sim3_global_from_local": estimate.global_from_local.as_vector().tolist(),
                        "covariance": estimate.covariance.tolist(),
                    }
                    for estimate in products.gauges
                ],
            }
        _write_json_no_clobber(staging / "gauge_diagnostics.json", diagnostics)
        progress["source_predictions_written"] = True
        shutil.rmtree(staging / "decoded")
    except Exception as error:
        status = "retained-technical-failure"
        failure = _safe_error(error, redactions)
        (staging / "failure_traceback.txt").write_text(
            _safe_error(RuntimeError(traceback.format_exc()), redactions) + "\n",
            encoding="utf-8",
        )
    manifest = _publish_case_manifest(
        staging,
        plan=plan,
        case=case,
        status=status,
        elapsed_seconds=time.monotonic() - started,
        failure=failure,
        progress=progress,
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(final)
    return manifest


def _retain_runtime_failure(
    *,
    error: BaseException,
    traceback_text: str,
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
    processed_root: Path,
    output_root: Path,
    cut3r_checkout: Path,
    checkpoint: Path,
    shard_index: int,
) -> dict[str, Any]:
    case_id = cast(str, case["case_id"])
    final = output_root / "cases" / case_id
    if final.exists():
        manifest = json.loads((final / "case_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("plan_id") != plan["plan_id"] or manifest.get("case_id") != case_id:
            raise FileExistsError(f"different retained case artifact exists: {case_id}")
        return cast(dict[str, Any], manifest)
    staging = output_root / "staging" / f"{case_id}-shard-{shard_index}-{os.getpid()}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    redactions = {
        os.fspath(processed_root): "<DEFORM360_PROCESSED_ROOT>",
        os.fspath(output_root): "<OUTPUT_ROOT>",
        os.fspath(cut3r_checkout): "<CUT3R_CHECKOUT>",
        os.fspath(checkpoint): "<CUT3R_CHECKPOINT>",
    }
    failure = _safe_error(error, redactions)
    (staging / "failure_traceback.txt").write_text(
        _safe_error(RuntimeError(traceback_text), redactions) + "\n",
        encoding="utf-8",
    )
    manifest = _publish_case_manifest(
        staging,
        plan=plan,
        case=case,
        status="retained-technical-failure",
        elapsed_seconds=0.0,
        failure=failure,
        progress={
            "source_rgb_frames_decoded": False,
            "cut3r_inference_executed": False,
            "source_predictions_written": False,
        },
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(final)
    return manifest


def _selected_cases(
    plan: Mapping[str, Any],
    *,
    shard_index: int,
    shard_count: int,
    smoke_case_id: str | None,
) -> list[Mapping[str, Any]]:
    cases = cast(list[Mapping[str, Any]], plan["cases"])
    expected_shards = cast(Mapping[str, Any], plan["execution"])["shard_count"]
    if shard_count != expected_shards or not 0 <= shard_index < shard_count:
        raise ValueError("shard selection differs from the frozen execution plan")
    if smoke_case_id is not None:
        if plan["schema_version"] == AMENDED_EXECUTION_PLAN_VERSION:
            execution = cast(Mapping[str, Any], plan["execution"])
            policy = cast(Mapping[str, Any], execution["smoke_policy"])
            if smoke_case_id != policy["registered_case_id"]:
                raise ValueError("smoke case differs from the amended registered case")
        selected = [case for case in cases if case["case_id"] == smoke_case_id]
        if len(selected) != 1 or selected[0]["role"] != "development":
            raise ValueError("smoke case must be one frozen development case")
        return selected
    return [case for index, case in enumerate(cases) if index % shard_count == shard_index]


def _amended_smoke_policy(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    if plan["schema_version"] != AMENDED_EXECUTION_PLAN_VERSION:
        raise ValueError("execution does not use the fail-closed amended plan")
    execution = cast(Mapping[str, Any], plan["execution"])
    return cast(Mapping[str, Any], execution["smoke_policy"])


def _validate_registered_path(
    path: Path,
    *,
    expected_sha256: object,
    name: str,
) -> None:
    if path_identity_sha256(path) != expected_sha256:
        raise ValueError(f"{name} differs from the registered path")


def _claim_registered_smoke_attempt(
    plan: Mapping[str, Any],
    *,
    smoke_case_id: str,
    output_root: Path,
    attempt_ledger: Path | None,
) -> dict[str, Any]:
    policy = _amended_smoke_policy(plan)
    if attempt_ledger is None:
        raise ValueError("amended smoke requires its registered attempt ledger")
    _validate_registered_path(
        output_root,
        expected_sha256=policy["registered_output_root_path_sha256"],
        name="smoke output root",
    )
    _validate_registered_path(
        attempt_ledger,
        expected_sha256=policy["registered_attempt_ledger_path_sha256"],
        name="smoke attempt ledger",
    )
    if output_root.is_symlink() or output_root.exists():
        raise FileExistsError("registered smoke output root is no longer fresh")
    if attempt_ledger.is_symlink() or attempt_ledger.exists():
        raise FileExistsError("the registered CUT3R smoke attempt is already consumed")
    if smoke_case_id != policy["registered_case_id"]:
        raise ValueError("smoke case differs from the amended registered case")
    record = claim_smoke_attempt(
        attempt_ledger,
        plan_id=cast(str, plan["plan_id"]),
        case_id_sha256=cast(str, policy["registered_case_id_sha256"]),
        output_root=output_root,
    )
    validated = validate_smoke_attempt(
        attempt_ledger,
        expected_plan_id=cast(str, plan["plan_id"]),
        expected_case_id_sha256=cast(str, policy["registered_case_id_sha256"]),
        expected_output_root=output_root,
    )
    if validated != record:
        raise ValueError("published smoke attempt record changed after atomic creation")
    return record


def _validate_amended_shard_authorization(
    plan: Mapping[str, Any],
    *,
    smoke_output_root: Path | None,
    smoke_shard_report: Path | None,
    smoke_custody_receipt: Path | None,
    smoke_attempt_ledger: Path | None,
) -> dict[str, Any]:
    policy = _amended_smoke_policy(plan)
    if any(
        path is None
        for path in (
            smoke_output_root,
            smoke_shard_report,
            smoke_custody_receipt,
            smoke_attempt_ledger,
        )
    ):
        raise ValueError(
            "amended source shards require the smoke artifact, report, custody "
            "receipt, and attempt ledger"
        )
    assert smoke_output_root is not None
    assert smoke_shard_report is not None
    assert smoke_custody_receipt is not None
    assert smoke_attempt_ledger is not None
    _validate_registered_path(
        smoke_output_root,
        expected_sha256=policy["registered_output_root_path_sha256"],
        name="smoke output root",
    )
    _validate_registered_path(
        smoke_attempt_ledger,
        expected_sha256=policy["registered_attempt_ledger_path_sha256"],
        name="smoke attempt ledger",
    )
    validate_smoke_attempt(
        smoke_attempt_ledger,
        expected_plan_id=cast(str, plan["plan_id"]),
        expected_case_id_sha256=cast(str, policy["registered_case_id_sha256"]),
        expected_output_root=smoke_output_root,
    )
    recomputed = validate_shard_artifact(
        smoke_output_root,
        smoke_shard_report,
        expected_plan_id=cast(str, plan["plan_id"]),
        require_success=True,
        forbid_decoded_frames=True,
    )
    published = validate_custody_receipt(
        smoke_custody_receipt,
        expected_plan_id=cast(str, plan["plan_id"]),
        expected_scope="development-smoke",
        require_success=True,
    )
    if recomputed != published:
        raise ValueError("published smoke custody receipt differs from recomputed custody")
    if published["case_ids"] != [policy["registered_case_id"]]:
        raise ValueError("smoke custody receipt belongs to a different development case")
    return published


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve(strict=True)
    cut3r = args.cut3r_checkout.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    processed_root = args.processed_root.resolve(strict=True)
    plan = load_execution_plan(
        args.plan,
        repository=repository,
        cut3r_checkout=cut3r,
        checkpoint=checkpoint,
    )
    if _runtime_inventory() != plan["runtime"]:
        raise ValueError("runtime inventory changed from the frozen execution plan")
    if plan["schema_version"] == REVOKED_AMENDED_EXECUTION_PLAN_VERSION:
        raise ValueError("the declarative v1.1 execution plan was independently revoked")
    selected = _selected_cases(
        plan,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        smoke_case_id=args.smoke_case_id,
    )
    output_root = args.output_root.resolve(strict=False)
    if plan["schema_version"] == AMENDED_EXECUTION_PLAN_VERSION:
        if args.smoke_case_id is not None:
            if any(
                value is not None
                for value in (
                    args.required_smoke_output_root,
                    args.required_smoke_shard_report,
                    args.required_smoke_custody_receipt,
                )
            ):
                raise ValueError("smoke execution must not supply source-shard custody inputs")
            _claim_registered_smoke_attempt(
                plan,
                smoke_case_id=args.smoke_case_id,
                output_root=output_root,
                attempt_ledger=args.smoke_attempt_ledger,
            )
        else:
            _validate_amended_shard_authorization(
                plan,
                smoke_output_root=args.required_smoke_output_root,
                smoke_shard_report=args.required_smoke_shard_report,
                smoke_custody_receipt=args.required_smoke_custody_receipt,
                smoke_attempt_ledger=args.smoke_attempt_ledger,
            )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        runtime = _Cut3RRuntime(cut3r, checkpoint, device=args.device)
    except Exception as error:
        traceback_text = traceback.format_exc()
        records = [
            _retain_runtime_failure(
                error=error,
                traceback_text=traceback_text,
                case=case,
                plan=plan,
                processed_root=processed_root,
                output_root=output_root,
                cut3r_checkout=cut3r,
                checkpoint=checkpoint,
                shard_index=args.shard_index,
            )
            for case in selected
        ]
    else:
        records = [
            _execute_case(
                runtime,
                case=case,
                plan=plan,
                processed_root=processed_root,
                output_root=output_root,
                shard_index=args.shard_index,
            )
            for case in selected
        ]
    report: dict[str, Any] = {
        "schema": "prob4d.cut3r-source-comparison-shard",
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "scope": "development-smoke" if args.smoke_case_id else "frozen-source-shard",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "case_count": len(records),
        "ordinary_success_count": sum(
            record["status"] == "ordinary-success" for record in records
        ),
        "retained_technical_failure_count": sum(
            record["status"] == "retained-technical-failure" for record in records
        ),
        "case_artifact_ids": [record["artifact_id"] for record in records],
        "source_residuals_or_truth_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
    }
    report["artifact_id"] = _content_id(report)
    report_name = (
        f"smoke-{args.smoke_case_id}.json"
        if args.smoke_case_id
        else f"shard-{args.shard_index:02d}-of-{args.shard_count:02d}.json"
    )
    _write_json_no_clobber(output_root / "shards" / report_name, report)
    print(_canonical_json(report).decode("utf-8"))
    return 0 if report["retained_technical_failure_count"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
