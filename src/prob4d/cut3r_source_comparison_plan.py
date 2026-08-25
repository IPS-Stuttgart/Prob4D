"""Outcome-blind execution plan for the recurrent CUT3R source comparison."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from ._atomic_file import atomic_write_bytes
from .cut3r_source_comparison_execution import (
    ALIGNMENT_HUBER_MULTIPLIER,
    ALIGNMENT_MAX_CORRESPONDENCES,
    ALIGNMENT_MAX_ITERATIONS,
    ALIGNMENT_TOLERANCE,
    GAUGE_COVARIANCE_INTERSECTION_GRID_SIZE,
    SOURCE_COMPARISON_METHOD_ID,
    causal_window_schedule,
)

EXECUTION_PLAN_SCHEMA: Final = "prob4d.cut3r-source-comparison-execution-plan"
EXECUTION_PLAN_VERSION: Final = 1
AMENDED_EXECUTION_PLAN_VERSION: Final = 2
EXECUTION_DECISION: Final = "source-comparison-execution-authorized"
RUNTIME_AMENDMENT_SCHEMA: Final = "prob4d.cut3r-source-comparison-runtime-amendment"
RUNTIME_AMENDMENT_VERSION: Final = 1
PREFLIGHT_SCHEMA: Final = "prob4d.cut3r-deform360-source-comparison-preflight"
PREFLIGHT_VERSION: Final = 1
PREFLIGHT_DECISION: Final = "source-comparison-preflight-ready"
IMPLEMENTATION_FILES: Final = (
    "scripts/science/build_cut3r_source_comparison_execution_plan.py",
    "scripts/science/run_cut3r_source_comparison.py",
    "src/prob4d/cut3r_source_comparison_execution.py",
    "src/prob4d/cut3r_source_comparison_plan.py",
)
AMENDMENT_IMPLEMENTATION_FILES: Final = (
    "scripts/science/build_cut3r_source_comparison_execution_amendment.py",
    "scripts/science/verify_cut3r_source_comparison_artifacts.py",
    "src/prob4d/cut3r_source_comparison_verifier.py",
)
PROVIDER_FILES: Final = (
    "add_ckpt_path.py",
    "demo.py",
    "src/dust3r/inference.py",
    "src/dust3r/model.py",
    "src/dust3r/post_process.py",
    "src/dust3r/utils/camera.py",
)
FALSE_BOUNDARY_FIELDS: Final = (
    "source_residuals_or_truth_opened",
    "candidate_reference_file_contents_opened",
    "target_payloads_opened",
    "target_outcomes_opened",
    "bayesian_phystwin_executed",
    "causal4d_executed",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"refusing symbolic link: {path}")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    def identity(item: os.stat_result) -> tuple[int, int, int, int]:
        return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after):
        raise ValueError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {name}: {value}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load {name}: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _revision(value: object, *, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{name} must be an exact lowercase Git revision")
    return value


def _regular_file(root: Path, relative: str, *, name: str) -> Path:
    if "\\" in relative:
        raise ValueError(f"{name} must use POSIX separators")
    parts = PurePosixPath(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{name} is not a canonical relative path")
    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"{name} traverses a symbolic link")
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", os.fspath(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runtime_inventory() -> dict[str, object]:
    import importlib.metadata

    packages: dict[str, str | None] = {}
    for name in ("numpy", "pillow", "opencv-python", "torch", "torchvision"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch

        torch_cuda = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        torch_cuda = None
        cuda_available = False
    return {
        "python_executable_filename": Path(sys.executable).name,
        "python_version": ".".join(str(item) for item in sys.version_info[:3]),
        "packages": packages,
        "torch_cuda_version": torch_cuda,
        "cuda_available": cuda_available,
    }


def _validate_preflight(value: Mapping[str, Any]) -> None:
    if value.get("schema") != PREFLIGHT_SCHEMA or value.get("schema_version") != PREFLIGHT_VERSION:
        raise ValueError("unsupported source-comparison preflight")
    if value.get("decision") != PREFLIGHT_DECISION:
        raise ValueError("source-comparison preflight did not pass")
    recorded = _sha256(value.get("artifact_id"), name="preflight artifact_id")
    unsigned = dict(value)
    unsigned.pop("artifact_id")
    if recorded != _content_id(unsigned):
        raise ValueError("preflight artifact identity is invalid")
    for name in (
        "source_rgb_frames_decoded",
        "cut3r_inference_executed",
        "source_prediction_payloads_opened",
        "source_residuals_or_truth_opened",
        "candidate_reference_file_contents_opened",
        "target_payloads_opened",
        "target_outcomes_opened",
        "comparison_execution_authorized",
    ):
        if value.get(name) is not False:
            raise ValueError(f"preflight boundary was exceeded: {name}")
    if value.get("resolved_case_count") != 40 or value.get("resolved_group_count") != 10:
        raise ValueError("preflight no longer contains the frozen 40-case/10-group source roster")


def _case_id_sha256(case_id: str) -> str:
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def _validate_parent_smoke_result(
    value: Mapping[str, Any], *, parent_plan_id: str
) -> dict[str, Any]:
    if value.get("schema") != "prob4d.cut3r-source-comparison-smoke-result":
        raise ValueError("unsupported parent smoke result")
    if value.get("schema_version") != 1:
        raise ValueError("unsupported parent smoke result version")
    recorded = _sha256(value.get("artifact_id"), name="parent smoke artifact_id")
    unsigned = dict(value)
    unsigned.pop("artifact_id")
    if recorded != _content_id(unsigned):
        raise ValueError("parent smoke artifact identity is invalid")
    if value.get("execution_plan_id") != parent_plan_id:
        raise ValueError("parent smoke result belongs to a different execution plan")
    expected = {
        "decision": "pre-science-technical-failure-no-retry",
        "attempt_count": 1,
        "ordinary_success_count": 0,
        "retained_technical_failure_count": 1,
        "output_file_count_after_failure": 0,
        "retry_authorized": False,
        "retry_performed": False,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value or type(value.get(name)) is not type(expected_value):
            raise ValueError(f"parent smoke result changed: {name}")
    failure = value.get("failure")
    if type(failure) is not dict or failure.get("terminal_stage") != "initialize-cut3r-runtime":
        raise ValueError("parent smoke was not the registered pre-science runtime failure")
    boundary = value.get("information_boundary")
    if (
        type(boundary) is not dict
        or not boundary
        or any(item is not False for item in boundary.values())
    ):
        raise ValueError("parent smoke exceeded its information boundary")
    _sha256(value.get("case_id_sha256"), name="parent case_id_sha256")
    return cast(dict[str, Any], value)


def build_execution_plan(
    *,
    repository: Path,
    preflight_path: Path,
    cut3r_checkout: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    """Build one source-only plan without decoding RGB or opening outcomes."""

    repository = repository.resolve(strict=True)
    checkout = cut3r_checkout.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    preflight = _load_json(preflight_path.resolve(strict=True), name="source preflight")
    _validate_preflight(preflight)
    repository_revision = _revision(
        _git(repository, "rev-parse", "HEAD"), name="Prob4D revision"
    )
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Prob4D implementation checkout must be clean including untracked files")
    implementation = {
        relative: _file_sha256(_regular_file(repository, relative, name=relative))
        for relative in IMPLEMENTATION_FILES
    }
    cut3r_revision = _revision(_git(checkout, "rev-parse", "HEAD"), name="CUT3R revision")
    if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("CUT3R checkout must be clean including untracked files")
    cut3r = cast(Mapping[str, Any], preflight["cut3r"])
    if cut3r_revision != cut3r.get("checkout_revision"):
        raise ValueError("CUT3R checkout revision changed from preflight")
    if checkpoint.name != cut3r.get("checkpoint_filename"):
        raise ValueError("CUT3R checkpoint filename changed from preflight")
    if _file_sha256(checkpoint) != cut3r.get("checkpoint_sha256"):
        raise ValueError("CUT3R checkpoint bytes changed from preflight")
    provider_files = {
        relative: _file_sha256(_regular_file(checkout, relative, name=relative))
        for relative in PROVIDER_FILES
    }
    comparison_lock_id = _sha256(
        preflight.get("comparison_lock_id"), name="comparison_lock_id"
    )
    cases = []
    for raw in cast(list[Mapping[str, Any]], preflight["cases"]):
        case_id = cast(str, raw["case_id"])
        cases.append(
            {
                "case_id": case_id,
                "group_id": raw["group_id"],
                "role": raw["role"],
                "relative_video_path": raw["relative_video_path"],
                "video_sha256": raw["video_sha256"],
                "video_byte_count": raw["video_byte_count"],
                "frame_start": 0,
                "frame_stop_exclusive": 58,
                "evaluation_frame_start": 24,
                "evaluation_frame_stop_exclusive": 58,
            }
        )
    schedule = causal_window_schedule(0, 58, window_size=25, overlap=8)
    plan: dict[str, Any] = {
        "schema": EXECUTION_PLAN_SCHEMA,
        "schema_version": EXECUTION_PLAN_VERSION,
        "decision": EXECUTION_DECISION,
        "preflight_artifact_id": preflight["artifact_id"],
        "source_freeze_id": preflight["source_freeze_id"],
        "comparison_lock_id": comparison_lock_id,
        "implementation": {
            "repository": "IPS-Stuttgart/Prob4D",
            "revision": repository_revision,
            "source_file_sha256": implementation,
        },
        "provider": {
            "repository": "CUT3R/CUT3R",
            "revision": cut3r_revision,
            "checkpoint_filename": checkpoint.name,
            "checkpoint_sha256": _file_sha256(checkpoint),
            "source_file_sha256": provider_files,
            "callable": "dust3r.inference.inference",
            "input_size": 512,
            "raw_inference_seed": 42,
            "revisit_count": 1,
            "global_alignment": False,
            "second_pass_allowed": False,
        },
        "runtime": _runtime_inventory(),
        "method": {
            "method_id": SOURCE_COMPARISON_METHOD_ID,
            "geometry_source": "pts3d-in-self-view-direct-v1",
            "window_schedule": [
                {"window_id": span.window_id, "start": span.start, "stop": span.stop}
                for span in schedule
            ],
            "window_size": 25,
            "overlap": 8,
            "stride": 17,
            "end_anchored_tail": True,
            "alignment_topology": "adjacent-window-chain-v1",
            "alignment_correspondence": "same-frame-same-pixel-valid-intersection-v1",
            "alignment_max_correspondences": ALIGNMENT_MAX_CORRESPONDENCES,
            "alignment_covariance_cluster_size": 32,
            "alignment_max_iterations": ALIGNMENT_MAX_ITERATIONS,
            "alignment_huber_multiplier": ALIGNMENT_HUBER_MULTIPLIER,
            "alignment_tolerance": ALIGNMENT_TOLERANCE,
            "gauge_estimator": "sequential-sim3-covariance-intersection-v1",
            "gauge_covariance_intersection_grid_size": (
                GAUGE_COVARIANCE_INTERSECTION_GRID_SIZE
            ),
            "point_uncertainty": "camera-relative-depth-plus-overlap-disagreement-v1",
            "fused_mean": "decoded-uniform-gaussian-mixture-second-moment-v1",
            "control_mean": "latest-starting-valid-window-per-frame-pixel-v1",
            "control_uses_same_windows_gauges_and_uncertainty": True,
            "confidence_threshold": 1.5,
            "storage_dtype": "float32",
            "random_seeds": [7, 11, 19],
        },
        "execution": {
            "case_count": 40,
            "group_count": 10,
            "shard_count": 2,
            "case_order": "lexicographic-case-id-v1",
            "failure_policy": "retain-once-no-replacement-no-retry-v1",
            "publication": "case-directory-atomic-rename-plus-content-manifest-v1",
        },
        "cases": sorted(cases, key=lambda item: cast(str, item["case_id"])),
        "information_boundary": {
            "source_rgb_decode_authorized": True,
            "source_cut3r_inference_authorized": True,
            "source_predictions_authorized": True,
            "source_outcomes_authorized": False,
            "source_residuals_or_truth_opened": False,
            "candidate_reference_file_contents_opened": False,
            "target_payloads_opened": False,
            "target_outcomes_opened": False,
            "bayesian_phystwin_executed": False,
            "causal4d_executed": False,
        },
        "claim_boundary": (
            "This plan authorizes only the frozen 40-case source RGB decode and "
            "three-arm CUT3R prediction comparison. It opens no source truth or "
            "residual, no target payload or outcome, and establishes neither held-out "
            "competence nor BayesianPhysTwin, Causal4D, deployment, or SOTA benefit."
        ),
    }
    plan["plan_id"] = _content_id(plan)
    return plan


def build_amended_execution_plan(
    *,
    repository: Path,
    preflight_path: Path,
    cut3r_checkout: Path,
    checkpoint: Path,
    parent_plan_path: Path,
    parent_smoke_result_path: Path,
) -> dict[str, Any]:
    """Build the one-attempt replacement smoke after a zero-progress runtime failure."""

    parent_plan = validate_execution_plan(
        _load_json(parent_plan_path.resolve(strict=True), name="parent execution plan")
    )
    if parent_plan["schema_version"] != EXECUTION_PLAN_VERSION:
        raise ValueError("runtime amendment requires the original v1 execution plan")
    parent_result = _validate_parent_smoke_result(
        _load_json(parent_smoke_result_path.resolve(strict=True), name="parent smoke result"),
        parent_plan_id=cast(str, parent_plan["plan_id"]),
    )
    plan = build_execution_plan(
        repository=repository,
        preflight_path=preflight_path,
        cut3r_checkout=cut3r_checkout,
        checkpoint=checkpoint,
    )
    for field in ("preflight_artifact_id", "source_freeze_id", "comparison_lock_id"):
        if plan[field] != parent_plan[field]:
            raise ValueError(f"runtime amendment changed {field}")
    if plan["method"] != parent_plan["method"] or plan["cases"] != parent_plan["cases"]:
        raise ValueError("runtime amendment changed the frozen method or source roster")
    parent_provider = cast(Mapping[str, Any], parent_plan["provider"])
    provider = cast(Mapping[str, Any], plan["provider"])
    for field in (
        "revision",
        "checkpoint_filename",
        "checkpoint_sha256",
        "source_file_sha256",
        "input_size",
        "raw_inference_seed",
        "revisit_count",
        "global_alignment",
        "second_pass_allowed",
    ):
        if provider[field] != parent_provider[field]:
            raise ValueError(f"runtime amendment changed provider field {field}")
    if parent_provider["callable"] != "src.dust3r.inference.inference":
        raise ValueError("parent plan no longer has the registered import spelling")
    if provider["callable"] != "dust3r.inference.inference":
        raise ValueError("amended plan does not bind the repaired import spelling")

    repository_root = repository.resolve(strict=True)
    implementation = cast(dict[str, Any], plan["implementation"])
    source_hashes = cast(dict[str, str], implementation["source_file_sha256"])
    for relative in AMENDMENT_IMPLEMENTATION_FILES:
        source_hashes[relative] = _file_sha256(
            _regular_file(repository_root, relative, name=relative)
        )

    prior_case_sha256 = cast(str, parent_result["case_id_sha256"])
    development_cases = [
        cast(Mapping[str, Any], case)
        for case in cast(list[Mapping[str, Any]], plan["cases"])
        if case["role"] == "development"
    ]
    replacements = [
        case
        for case in development_cases
        if _case_id_sha256(cast(str, case["case_id"])) != prior_case_sha256
    ]
    if not replacements:
        raise ValueError("no distinct frozen development case remains for the amendment")
    replacement_case_id = cast(str, replacements[0]["case_id"])
    replacement_case_sha256 = _case_id_sha256(replacement_case_id)

    plan["schema_version"] = AMENDED_EXECUTION_PLAN_VERSION
    execution = cast(dict[str, Any], plan["execution"])
    execution["smoke_policy"] = {
        "registered_case_id": replacement_case_id,
        "registered_case_id_sha256": replacement_case_sha256,
        "attempt_limit": 1,
        "prior_case_retry_authorized": False,
        "source_shards_require_ordinary_success_custody": True,
    }
    plan["amendment"] = {
        "schema": RUNTIME_AMENDMENT_SCHEMA,
        "schema_version": RUNTIME_AMENDMENT_VERSION,
        "reason": "pre-science-runtime-import-repair-v1",
        "parent_plan_id": parent_plan["plan_id"],
        "parent_smoke_result_id": parent_result["artifact_id"],
        "prior_case_id_sha256": prior_case_sha256,
        "prior_failure_terminal_stage": "initialize-cut3r-runtime",
        "prior_attempt_count": 1,
        "prior_retry_authorized": False,
        "method_changed": False,
        "provider_weights_or_sources_changed": False,
        "source_outcomes_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
    }
    plan["claim_boundary"] = (
        "This amendment authorizes exactly one different frozen development smoke "
        "after the retained zero-progress runtime import failure. It changes only "
        "the import/bootstrap implementation and custody binding, keeps the frozen "
        "method and provider bytes unchanged, forbids retry of the prior case, and "
        "does not authorize source shards until an ordinary-success custody receipt."
    )
    plan.pop("plan_id")
    plan["plan_id"] = _content_id(plan)
    return plan


def validate_execution_plan(
    value: Mapping[str, Any],
    *,
    repository: Path | None = None,
    cut3r_checkout: Path | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Strictly validate plan identity and optionally all executable bytes."""

    if value.get("schema") != EXECUTION_PLAN_SCHEMA or value.get("schema_version") not in {
        EXECUTION_PLAN_VERSION,
        AMENDED_EXECUTION_PLAN_VERSION,
    }:
        raise ValueError("unsupported CUT3R source-comparison execution plan")
    if value.get("decision") != EXECUTION_DECISION:
        raise ValueError("CUT3R source-comparison execution is not authorized")
    recorded = _sha256(value.get("plan_id"), name="plan_id")
    plan = dict(value)
    plan.pop("plan_id")
    if recorded != _content_id(plan):
        raise ValueError("execution plan content identity is invalid")
    boundary = value.get("information_boundary")
    if type(boundary) is not dict:
        raise ValueError("execution information_boundary must be a JSON object")
    if boundary.get("source_rgb_decode_authorized") is not True:
        raise ValueError("execution plan does not authorize source RGB decode")
    if boundary.get("source_cut3r_inference_authorized") is not True:
        raise ValueError("execution plan does not authorize CUT3R inference")
    for name in FALSE_BOUNDARY_FIELDS:
        if boundary.get(name) is not False:
            raise ValueError(f"execution plan exceeds boundary: {name}")
    cases = value.get("cases")
    if type(cases) is not list or len(cases) != 40:
        raise ValueError("execution plan must contain the frozen 40 source cases")
    case_ids = []
    for index, item in enumerate(cases):
        if type(item) is not dict or type(item.get("case_id")) is not str:
            raise ValueError(f"execution case {index} has no exact case_id")
        case_ids.append(cast(str, item["case_id"]))
    if case_ids != sorted(case_ids) or len(set(case_ids)) != 40:
        raise ValueError("execution cases must have unique lexicographically sorted IDs")

    if value["schema_version"] == AMENDED_EXECUTION_PLAN_VERSION:
        amendment = value.get("amendment")
        if type(amendment) is not dict or set(amendment) != {
            "schema",
            "schema_version",
            "reason",
            "parent_plan_id",
            "parent_smoke_result_id",
            "prior_case_id_sha256",
            "prior_failure_terminal_stage",
            "prior_attempt_count",
            "prior_retry_authorized",
            "method_changed",
            "provider_weights_or_sources_changed",
            "source_outcomes_opened",
            "target_payloads_opened",
            "target_outcomes_opened",
        }:
            raise ValueError("amended execution plan has an invalid amendment record")
        if amendment["schema"] != RUNTIME_AMENDMENT_SCHEMA:
            raise ValueError("amended execution plan has an unknown amendment schema")
        if amendment["schema_version"] != RUNTIME_AMENDMENT_VERSION:
            raise ValueError("amended execution plan has an unknown amendment version")
        for name in ("parent_plan_id", "parent_smoke_result_id", "prior_case_id_sha256"):
            _sha256(amendment[name], name=f"amendment {name}")
        expected_amendment = {
            "reason": "pre-science-runtime-import-repair-v1",
            "prior_failure_terminal_stage": "initialize-cut3r-runtime",
            "prior_attempt_count": 1,
            "prior_retry_authorized": False,
            "method_changed": False,
            "provider_weights_or_sources_changed": False,
            "source_outcomes_opened": False,
            "target_payloads_opened": False,
            "target_outcomes_opened": False,
        }
        for name, expected_value in expected_amendment.items():
            if amendment[name] != expected_value or type(amendment[name]) is not type(
                expected_value
            ):
                raise ValueError(f"amended execution plan changed {name}")
        execution = value.get("execution")
        smoke_policy = execution.get("smoke_policy") if type(execution) is dict else None
        if type(smoke_policy) is not dict or set(smoke_policy) != {
            "registered_case_id",
            "registered_case_id_sha256",
            "attempt_limit",
            "prior_case_retry_authorized",
            "source_shards_require_ordinary_success_custody",
        }:
            raise ValueError("amended execution plan has an invalid smoke policy")
        registered_case_id = smoke_policy["registered_case_id"]
        if type(registered_case_id) is not str or registered_case_id not in case_ids:
            raise ValueError("amended smoke case is not in the frozen source roster")
        registered_case = cast(Mapping[str, Any], cases[case_ids.index(registered_case_id)])
        if registered_case["role"] != "development":
            raise ValueError("amended smoke case is not a development case")
        if _sha256(
            smoke_policy["registered_case_id_sha256"],
            name="registered_case_id_sha256",
        ) != _case_id_sha256(registered_case_id):
            raise ValueError("amended smoke case hash is invalid")
        if smoke_policy["registered_case_id_sha256"] == amendment["prior_case_id_sha256"]:
            raise ValueError("amended smoke attempts to reuse the failed case")
        if smoke_policy["attempt_limit"] != 1:
            raise ValueError("amended smoke attempt limit changed")
        if smoke_policy["prior_case_retry_authorized"] is not False:
            raise ValueError("amended smoke authorizes retry of the failed case")
        if smoke_policy["source_shards_require_ordinary_success_custody"] is not True:
            raise ValueError("amended source shards no longer require smoke custody")
        provider = value.get("provider")
        if type(provider) is not dict or provider.get("callable") != "dust3r.inference.inference":
            raise ValueError("amended plan does not use the repaired CUT3R import surface")

    if repository is not None:
        root = repository.resolve(strict=True)
        implementation = cast(Mapping[str, Any], value["implementation"])
        if _git(root, "rev-parse", "HEAD") != implementation["revision"]:
            raise ValueError("Prob4D revision changed from execution plan")
        if _git(root, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("Prob4D execution checkout is not clean")
        source_hashes = cast(Mapping[str, str], implementation["source_file_sha256"])
        for relative, expected in source_hashes.items():
            actual = _file_sha256(_regular_file(root, relative, name=relative))
            if actual != expected:
                raise ValueError(f"implementation source changed: {relative}")
    if cut3r_checkout is not None:
        root = cut3r_checkout.resolve(strict=True)
        provider = cast(Mapping[str, Any], value["provider"])
        if _git(root, "rev-parse", "HEAD") != provider["revision"]:
            raise ValueError("CUT3R revision changed from execution plan")
        if _git(root, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("CUT3R checkout is not clean")
        for relative, expected in cast(
            Mapping[str, str], provider["source_file_sha256"]
        ).items():
            actual = _file_sha256(_regular_file(root, relative, name=relative))
            if actual != expected:
                raise ValueError(f"CUT3R source changed: {relative}")
    if checkpoint is not None:
        provider = cast(Mapping[str, Any], value["provider"])
        resolved = checkpoint.resolve(strict=True)
        if resolved.name != provider["checkpoint_filename"]:
            raise ValueError("checkpoint filename changed from execution plan")
        if _file_sha256(resolved) != provider["checkpoint_sha256"]:
            raise ValueError("checkpoint bytes changed from execution plan")
    return cast(dict[str, Any], value)


def load_execution_plan(
    path: Path,
    *,
    repository: Path | None = None,
    cut3r_checkout: Path | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    return validate_execution_plan(
        _load_json(path.resolve(strict=True), name="execution plan"),
        repository=repository,
        cut3r_checkout=cut3r_checkout,
        checkpoint=checkpoint,
    )


def save_execution_plan(path: Path, plan: Mapping[str, Any]) -> None:
    validate_execution_plan(plan)
    encoded = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    try:
        atomic_write_bytes(path, encoded, overwrite=False)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise


__all__ = [
    "AMENDED_EXECUTION_PLAN_VERSION",
    "AMENDMENT_IMPLEMENTATION_FILES",
    "EXECUTION_DECISION",
    "EXECUTION_PLAN_SCHEMA",
    "EXECUTION_PLAN_VERSION",
    "IMPLEMENTATION_FILES",
    "PROVIDER_FILES",
    "RUNTIME_AMENDMENT_SCHEMA",
    "RUNTIME_AMENDMENT_VERSION",
    "build_amended_execution_plan",
    "build_execution_plan",
    "load_execution_plan",
    "save_execution_plan",
    "validate_execution_plan",
]
