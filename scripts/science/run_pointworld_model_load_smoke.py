#!/usr/bin/env python3
"""Validate and execute the dataset-free PointWorld model-load smoke.

The smoke is deliberately narrower than provider evaluation. It verifies the
pinned PointWorld and DINOv3 sources, hashes the exact checkpoint and DINOv3
weights before import, initializes the complete PointWorld model on CUDA, and
records a sanitized machine-readable result. It never opens a dataset or runs a
prediction forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA = "prob4d.pointworld-model-load-smoke-protocol"
REQUEST_SCHEMA = "prob4d.pointworld-model-load-smoke-request"
PREFLIGHT_SCHEMA = "prob4d.pointworld-model-load-smoke-preflight"
LOAD_SCHEMA = "prob4d.pointworld-model-load-smoke-load"
RESULT_SCHEMA = "prob4d.pointworld-model-load-smoke-result"
SCHEMA_VERSION = 1
PROFILE = "pointworld-model-load-smoke-v1"
ISSUE_NUMBER = 333
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _strict_json(path: Path) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON number {token!r}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except OSError as error:
        raise ValueError(f"cannot read {path.name}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_id(value: Mapping[str, Any]) -> str:
    """Return a canonical SHA-256 identifier for one finite JSON mapping."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed; missing={missing}, extra={extra}")


def _require_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _require_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be Boolean")
    return value


def _require_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _require_sha(value: object, *, name: str, length: int) -> str:
    text = _require_string(value, name=name)
    pattern = _HEX40 if length == 40 else _HEX64
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{name} must be lowercase {length}-hex")
    return text


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema",
        "schema_version",
        "protocol_id",
        "issue_number",
        "profile",
        "pointworld_repository",
        "pointworld_revision",
        "dinov3_repository",
        "dinov3_revision",
        "asset_root_alias",
        "pointworld_checkout_relative_path",
        "checkpoint_relative_path",
        "checkpoint_hash_policy",
        "dinov3_weights_relative_path",
        "dinov3_weights_size_bytes",
        "dinov3_weights_sha256",
        "runtime_policy",
        "dataset_access_authorized",
        "prediction_execution_authorized",
        "provider_residuals_authorized",
        "target_outcomes_authorized",
        "required_result_fields",
        "claim_boundary",
    }
    _require_exact_fields(protocol, expected, name="protocol")
    if protocol["schema"] != PROTOCOL_SCHEMA or protocol["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported PointWorld model-load protocol")
    if protocol["profile"] != PROFILE or protocol["issue_number"] != ISSUE_NUMBER:
        raise ValueError("protocol profile or issue changed")
    _require_string(protocol["protocol_id"], name="protocol_id")
    if protocol["pointworld_repository"] != "NVlabs/PointWorld":
        raise ValueError("unexpected PointWorld repository")
    if protocol["dinov3_repository"] != "facebookresearch/dinov3":
        raise ValueError("unexpected DINOv3 repository")
    _require_sha(protocol["pointworld_revision"], name="pointworld_revision", length=40)
    _require_sha(protocol["dinov3_revision"], name="dinov3_revision", length=40)
    _require_sha(protocol["dinov3_weights_sha256"], name="dinov3_weights_sha256", length=64)
    _require_int(protocol["dinov3_weights_size_bytes"], name="dinov3_weights_size_bytes", minimum=1)
    for field in (
        "asset_root_alias",
        "pointworld_checkout_relative_path",
        "checkpoint_relative_path",
        "checkpoint_hash_policy",
        "dinov3_weights_relative_path",
        "runtime_policy",
        "claim_boundary",
    ):
        _require_string(protocol[field], name=field)
    if protocol["checkpoint_hash_policy"] != "measure-and-record-before-load":
        raise ValueError("checkpoint hash policy changed")
    for field in (
        "dataset_access_authorized",
        "prediction_execution_authorized",
        "provider_residuals_authorized",
        "target_outcomes_authorized",
    ):
        if _require_bool(protocol[field], name=field):
            raise ValueError(f"{field} must remain false")
    result_fields = protocol["required_result_fields"]
    if not isinstance(result_fields, list) or not result_fields:
        raise ValueError("required_result_fields must be one nonempty array")
    if any(type(item) is not str or not item for item in result_fields):
        raise ValueError("required_result_fields contains an invalid entry")
    if result_fields != sorted(set(result_fields)):
        raise ValueError("required_result_fields must be sorted and unique")
    return dict(protocol)


def validate_request(
    request: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    source_protocol_git_blob_sha: str,
) -> str:
    validate_protocol(protocol)
    expected = {
        "schema",
        "schema_version",
        "request_id",
        "issue_number",
        "profile",
        "source_protocol_path",
        "source_protocol_git_blob_sha",
        "execution_authorized",
        "dataset_access_authorized",
        "prediction_execution_authorized",
        "provider_residuals_authorized",
        "target_outcomes_authorized",
        "claim_boundary",
    }
    _require_exact_fields(request, expected, name="execution request")
    if request["schema"] != REQUEST_SCHEMA or request["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported PointWorld model-load request")
    if request["issue_number"] != ISSUE_NUMBER or request["profile"] != PROFILE:
        raise ValueError("execution request profile or issue changed")
    if request["source_protocol_path"] != "protocols/pointworld-model-load-smoke-v1.json":
        raise ValueError("execution request source protocol path changed")
    expected_blob = _require_sha(
        source_protocol_git_blob_sha,
        name="source_protocol_git_blob_sha argument",
        length=40,
    )
    if request["source_protocol_git_blob_sha"] != expected_blob:
        raise ValueError("execution request does not bind the checked protocol blob")
    if not _require_bool(request["execution_authorized"], name="execution_authorized"):
        raise ValueError("execution_authorized must be true")
    for field in (
        "dataset_access_authorized",
        "prediction_execution_authorized",
        "provider_residuals_authorized",
        "target_outcomes_authorized",
    ):
        if _require_bool(request[field], name=field):
            raise ValueError(f"{field} must remain false")
    if request["claim_boundary"] != protocol["claim_boundary"]:
        raise ValueError("execution request claim boundary changed")
    request_id = _require_sha(request["request_id"], name="request_id", length=64)
    identity = dict(request)
    identity.pop("request_id")
    if canonical_id(identity) != request_id:
        raise ValueError("execution request ID mismatch")
    return request_id


def _sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash required file {path.name}") from error
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot identify Git revision for {path.name}") from error
    revision = completed.stdout.strip()
    _require_sha(revision, name=f"{path.name} revision", length=40)
    return revision


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _sanitized_error(error: BaseException, replacements: Mapping[str, str]) -> dict[str, str]:
    message = f"{type(error).__name__}: {error}"
    tail = "".join(traceback.format_exception(type(error), error, error.__traceback__))[-6000:]
    for source, replacement in sorted(replacements.items(), key=lambda item: -len(item[0])):
        if source:
            message = message.replace(source, replacement)
            tail = tail.replace(source, replacement)
    return {"message": message, "traceback_tail": tail}


def run_preflight(
    *,
    protocol_path: Path,
    pointworld_checkout: Path,
    checkpoint: Path,
    dinov3_weights: Path,
    request_id: str,
    prob4d_revision: str,
    output: Path,
) -> int:
    started = {
        "schema": PREFLIGHT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": _require_sha(request_id, name="request_id", length=64),
        "prob4d_revision": _require_sha(
            prob4d_revision,
            name="prob4d_revision",
            length=40,
        ),
        "decision": "fail",
        "failure_stage": "preflight",
        "dataset_opened": False,
        "prediction_executed": False,
        "provider_residuals_opened": False,
        "target_outcomes_opened": False,
    }
    replacements = {
        str(pointworld_checkout): "<POINTWORLD_CHECKOUT>",
        str(checkpoint): "<POINTWORLD_CHECKPOINT>",
        str(dinov3_weights): "<DINOV3_WEIGHTS>",
    }
    try:
        protocol = validate_protocol(_strict_json(protocol_path))
        if not pointworld_checkout.is_dir() or pointworld_checkout.is_symlink():
            raise ValueError("PointWorld checkout is missing or is a symbolic link")
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise ValueError("PointWorld checkpoint is missing or is a symbolic link")
        if not dinov3_weights.is_file() or dinov3_weights.is_symlink():
            raise ValueError("DINOv3 weights are missing or are a symbolic link")
        pointworld_revision = _git_revision(pointworld_checkout)
        if pointworld_revision != protocol["pointworld_revision"]:
            raise ValueError("PointWorld checkout revision mismatch")
        dinov3_checkout = pointworld_checkout / "third_party" / "dinov3"
        if not (dinov3_checkout / "hubconf.py").is_file():
            raise ValueError("pinned DINOv3 checkout or hubconf.py is missing")
        dinov3_revision = _git_revision(dinov3_checkout)
        if dinov3_revision != protocol["dinov3_revision"]:
            raise ValueError("DINOv3 checkout revision mismatch")
        dinov3_size = dinov3_weights.stat().st_size
        if dinov3_size != protocol["dinov3_weights_size_bytes"]:
            raise ValueError("DINOv3 weights byte count mismatch")
        dinov3_sha256 = _sha256_file(dinov3_weights)
        if dinov3_sha256 != protocol["dinov3_weights_sha256"]:
            raise ValueError("DINOv3 weights SHA-256 mismatch")
        checkpoint_size = checkpoint.stat().st_size
        if checkpoint_size <= 0:
            raise ValueError("PointWorld checkpoint is empty")
        checkpoint_sha256 = _sha256_file(checkpoint)
        result = {
            **started,
            "decision": "pass",
            "failure_stage": None,
            "protocol_id": protocol["protocol_id"],
            "pointworld_revision": pointworld_revision,
            "dinov3_revision": dinov3_revision,
            "checkpoint": {
                "logical_path": protocol["checkpoint_relative_path"],
                "size_bytes": checkpoint_size,
                "sha256": checkpoint_sha256,
            },
            "dinov3_weights": {
                "logical_path": protocol["dinov3_weights_relative_path"],
                "size_bytes": dinov3_size,
                "sha256": dinov3_sha256,
            },
            "claim_boundary": protocol["claim_boundary"],
        }
    except Exception as error:
        result = {**started, "error": _sanitized_error(error, replacements)}
    _atomic_json(output, result)
    return 0 if result["decision"] == "pass" else 3


def _finite_number(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite")
    return value


def run_load(
    *,
    protocol_path: Path,
    preflight_path: Path,
    pointworld_checkout: Path,
    checkpoint: Path,
    dinov3_weights: Path,
    output: Path,
    log_dir: Path,
) -> int:
    protocol = validate_protocol(_strict_json(protocol_path))
    preflight = _strict_json(preflight_path)
    started = {
        "schema": LOAD_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": preflight.get("request_id"),
        "prob4d_revision": preflight.get("prob4d_revision"),
        "decision": "fail",
        "failure_stage": "model_load",
        "dataset_opened": False,
        "prediction_executed": False,
        "provider_residuals_opened": False,
        "target_outcomes_opened": False,
        "checkpoint": preflight.get("checkpoint"),
        "dinov3_weights": preflight.get("dinov3_weights"),
        "pointworld_revision": preflight.get("pointworld_revision"),
        "dinov3_revision": preflight.get("dinov3_revision"),
        "claim_boundary": protocol["claim_boundary"],
    }
    replacements = {
        str(pointworld_checkout): "<POINTWORLD_CHECKOUT>",
        str(checkpoint): "<POINTWORLD_CHECKPOINT>",
        str(dinov3_weights): "<DINOV3_WEIGHTS>",
        str(log_dir): "<POINTWORLD_LOG_DIR>",
    }
    try:
        if preflight.get("decision") != "pass":
            raise ValueError("model load requires a passing preflight artifact")
        if _sha256_file(checkpoint) != preflight["checkpoint"]["sha256"]:
            raise ValueError("PointWorld checkpoint changed after preflight")
        if _sha256_file(dinov3_weights) != preflight["dinov3_weights"]["sha256"]:
            raise ValueError("DINOv3 weights changed after preflight")

        expected_weights = (
            pointworld_checkout
            / "third_party"
            / "dinov3"
            / "checkpoints"
            / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )
        expected_weights.parent.mkdir(parents=True, exist_ok=True)
        if expected_weights.exists() or expected_weights.is_symlink():
            expected_weights.unlink()
        expected_weights.symlink_to(dinov3_weights)

        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.chdir(pointworld_checkout)
        sys.path.insert(0, str(pointworld_checkout))

        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.reset_peak_memory_stats()

        from arguments import parse_args  # type: ignore[import-not-found]
        from training.trainer import Trainer  # type: ignore[import-not-found]

        args = parse_args(skip_command_line=True)
        args.model_path = str(checkpoint)
        args.device = "cuda"
        args.distributed = False
        args.disable_compile = True
        args.deterministic_train = False
        args.deterministic_algorithms = False
        args.seed = 42
        args.log_dir = str(log_dir)
        args.exp_name = "pointworld-model-load-smoke-v1"
        args.norm_stats_path = str(pointworld_checkout / "stats" / "droid_behavior")
        args.og_args = vars(args).copy()

        trainer = Trainer(args, inference_only=True, data_info_dict=None)
        model = trainer.model
        parameters = list(model.parameters())
        parameter_count = sum(parameter.numel() for parameter in parameters)
        trainable_parameter_count = sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        )
        devices = sorted({str(parameter.device) for parameter in parameters})
        if parameter_count <= 0:
            raise RuntimeError("loaded PointWorld model has no parameters")
        if not devices or any(not device.startswith("cuda") for device in devices):
            raise RuntimeError("loaded PointWorld parameters are not entirely on CUDA")
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        result = {
            **started,
            "decision": "pass",
            "failure_stage": None,
            "runtime": {
                "python": sys.version.split()[0],
                "torch": str(torch.__version__),
                "torch_cuda": str(torch.version.cuda),
                "cuda_device": str(torch.cuda.get_device_name(0)),
                "cuda_capability": list(torch.cuda.get_device_capability(0)),
                "amp_dtype": str(trainer.amp_dtype),
            },
            "model": {
                "class": f"{type(model).__module__}.{type(model).__name__}",
                "parameter_count": int(parameter_count),
                "trainable_parameter_count": int(trainable_parameter_count),
                "parameter_devices": devices,
                "peak_cuda_memory_allocated_bytes": peak_allocated,
                "peak_cuda_memory_reserved_bytes": peak_reserved,
            },
            "complete_checkpoint_state_loaded": True,
            "complete_dinov3_backbone_loaded": True,
            "forward_pass_executed": False,
        }
        for key in (
            "peak_cuda_memory_allocated_bytes",
            "peak_cuda_memory_reserved_bytes",
        ):
            _finite_number(float(result["model"][key]), name=key)
    except Exception as error:
        result = {**started, "error": _sanitized_error(error, replacements)}
    _atomic_json(output, result)
    return 0 if result["decision"] == "pass" else 3


def summarize(
    *,
    protocol_path: Path,
    request_id: str,
    prob4d_revision: str,
    preflight_path: Path,
    load_path: Path,
    runtime_install_outcome: str,
    output: Path,
) -> int:
    protocol = validate_protocol(_strict_json(protocol_path))
    preflight = _strict_json(preflight_path) if preflight_path.is_file() else None
    load = _strict_json(load_path) if load_path.is_file() else None
    if load is not None:
        decision = load.get("decision", "fail")
        failure_stage = load.get("failure_stage")
        detail = load
    elif preflight is not None and preflight.get("decision") != "pass":
        decision = "fail"
        failure_stage = preflight.get("failure_stage", "preflight")
        detail = preflight
    else:
        decision = "fail"
        failure_stage = "runtime_install"
        detail = {
            "error": {
                "message": f"runtime installation outcome was {runtime_install_outcome}",
                "traceback_tail": "",
            }
        }
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": _require_sha(request_id, name="request_id", length=64),
        "prob4d_revision": _require_sha(
            prob4d_revision,
            name="prob4d_revision",
            length=40,
        ),
        "decision": decision,
        "failure_stage": failure_stage,
        "runtime_install_outcome": runtime_install_outcome,
        "preflight": preflight,
        "load": load,
        "detail": detail,
        "dataset_opened": False,
        "prediction_executed": False,
        "provider_residuals_opened": False,
        "target_outcomes_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    _atomic_json(output, result)
    return 0 if decision == "pass" else 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-request")
    validate.add_argument("--request", required=True, type=Path)
    validate.add_argument("--protocol", required=True, type=Path)
    validate.add_argument("--source-protocol-git-blob-sha", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--protocol", required=True, type=Path)
    preflight.add_argument("--pointworld-checkout", required=True, type=Path)
    preflight.add_argument("--checkpoint", required=True, type=Path)
    preflight.add_argument("--dinov3-weights", required=True, type=Path)
    preflight.add_argument("--request-id", required=True)
    preflight.add_argument("--prob4d-revision", required=True)
    preflight.add_argument("--output", required=True, type=Path)

    load = subparsers.add_parser("load")
    load.add_argument("--protocol", required=True, type=Path)
    load.add_argument("--preflight", required=True, type=Path)
    load.add_argument("--pointworld-checkout", required=True, type=Path)
    load.add_argument("--checkpoint", required=True, type=Path)
    load.add_argument("--dinov3-weights", required=True, type=Path)
    load.add_argument("--output", required=True, type=Path)
    load.add_argument("--log-dir", required=True, type=Path)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--protocol", required=True, type=Path)
    summary.add_argument("--request-id", required=True)
    summary.add_argument("--prob4d-revision", required=True)
    summary.add_argument("--preflight", required=True, type=Path)
    summary.add_argument("--load", required=True, type=Path)
    summary.add_argument("--runtime-install-outcome", required=True)
    summary.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    if arguments.command == "validate-request":
        protocol = _strict_json(arguments.protocol)
        request = _strict_json(arguments.request)
        request_id = validate_request(
            request,
            protocol,
            source_protocol_git_blob_sha=arguments.source_protocol_git_blob_sha,
        )
        print(json.dumps({"request_id": request_id}, sort_keys=True))
        return 0
    if arguments.command == "preflight":
        return run_preflight(
            protocol_path=arguments.protocol,
            pointworld_checkout=arguments.pointworld_checkout,
            checkpoint=arguments.checkpoint,
            dinov3_weights=arguments.dinov3_weights,
            request_id=arguments.request_id,
            prob4d_revision=arguments.prob4d_revision,
            output=arguments.output,
        )
    if arguments.command == "load":
        return run_load(
            protocol_path=arguments.protocol,
            preflight_path=arguments.preflight,
            pointworld_checkout=arguments.pointworld_checkout,
            checkpoint=arguments.checkpoint,
            dinov3_weights=arguments.dinov3_weights,
            output=arguments.output,
            log_dir=arguments.log_dir,
        )
    return summarize(
        protocol_path=arguments.protocol,
        request_id=arguments.request_id,
        prob4d_revision=arguments.prob4d_revision,
        preflight_path=arguments.preflight,
        load_path=arguments.load,
        runtime_install_outcome=arguments.runtime_install_outcome,
        output=arguments.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
