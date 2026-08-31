#!/usr/bin/env python3
"""Run the frozen R04--R10 DOT/CUT3R held-out dependence confirmation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.dependence_tempering import temper_shared_dependence
from prob4d.dot_rope_cut3r_study import content_id

PROTOCOL_SCHEMA = "prob4d.dot-rope-cut3r-heldout-confirmation-protocol"
REQUEST_SCHEMA = "prob4d.dot-rope-cut3r-heldout-confirmation-request"
RESULT_SCHEMA = "prob4d.dot-rope-cut3r-heldout-confirmation"
SUPPORT_SCHEMA = "prob4d.dot-rope-cut3r-heldout-support"
FAILURE_SCHEMA = "prob4d.dot-rope-cut3r-heldout-failure"
SCHEMA_VERSION = 1

CONFIRMATION_SEQUENCES = [f"R{index:02d}" for index in range(4, 11)]
RESERVED_SEQUENCES = "R11-R70"
ARCHIVE = "R01-10.zip"
CAMERA = "cam001"
FRAMES = list(range(1, 8))
SELECTED_ALPHA = 0.85
SELECTED_METHOD = "dependence_alpha_0850"
SOURCE_CALIBRATION_ID = "943339ac864fda04cc59081bc81a605576b3c90bf0aa996aea00b00335cfc0c7"
CUT3R_REVISION = "8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
BASE_PROVIDER_BLOB = "612c8ae61b0a64d464256a11992b46c486c88012"
POOLED_EVALUATOR_BLOB = "6195e70997f0e9582251c08772b1e423a3062ad6"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-request")
    validate.add_argument("--request", type=Path, required=True)
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--protocol-git-blob-sha", required=True)

    predict = commands.add_parser("predict")
    _execution_arguments(predict)
    predict.add_argument("--dataset-root", type=Path, required=True)
    predict.add_argument("--cut3r-checkout", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--runtime-receipt", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)

    evaluate = commands.add_parser("evaluate")
    _execution_arguments(evaluate)
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--provider-bundle", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser


def _execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--prob4d-revision", required=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _hex(value: object, *, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must have {length} hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return value


def _git_blob_sha1(payload: bytes) -> str:
    import hashlib

    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported held-out confirmation protocol")
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id", None)
    _hex(protocol_id, name="protocol_id", length=64)
    if content_id(unsigned) != protocol_id:
        raise ValueError("held-out confirmation protocol identity mismatch")
    if protocol.get("confirmation_sequences") != CONFIRMATION_SEQUENCES:
        raise ValueError("confirmation sequence roster changed")
    if protocol.get("reserved_sequences") != RESERVED_SEQUENCES:
        raise ValueError("reserved sequence boundary changed")
    if protocol.get("archive") != ARCHIVE or protocol.get("camera") != CAMERA:
        raise ValueError("archive or camera changed")
    if protocol.get("frames") != FRAMES:
        raise ValueError("frame roster changed")
    if protocol.get("windows") != {
        "continuous": list(range(1, 8)),
        "window_a": list(range(1, 6)),
        "window_b": list(range(3, 8)),
    }:
        raise ValueError("provider windows changed")
    provider = protocol.get("provider") or {}
    if provider.get("cut3r_revision") != CUT3R_REVISION:
        raise ValueError("CUT3R revision changed")
    if provider.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("CUT3R checkpoint changed")
    calibration = protocol.get("source_calibration") or {}
    if calibration.get("calibration_id") != SOURCE_CALIBRATION_ID:
        raise ValueError("source calibration identity changed")
    if float(calibration.get("selected_alpha", math.nan)) != SELECTED_ALPHA:
        raise ValueError("source-frozen alpha changed")
    uncertainty = protocol.get("uncertainty") or {}
    if float(uncertainty.get("selected_dependence_alpha", math.nan)) != SELECTED_ALPHA:
        raise ValueError("confirmation alpha changed")
    if uncertainty.get("means_held_fixed") is not True:
        raise ValueError("provider means must remain fixed")
    marker = protocol.get("marker_sampling") or {}
    if marker.get("coordinate_columns") != [0, 1]:
        raise ValueError("coordinate columns changed")
    if marker.get("coordinate_mode") != "pixel-zero-based":
        raise ValueError("coordinate mode changed")
    if marker.get("selected_coordinate_candidate") != "columns-0-1:pixel-zero-based":
        raise ValueError("coordinate candidate changed")
    boundary = protocol.get("information_boundary") or {}
    if boundary.get("source_calibration_frozen") is not True:
        raise ValueError("source calibration must remain frozen")
    if boundary.get("confirmation_tuning_authorized") is not False:
        raise ValueError("confirmation tuning must remain disabled")
    return protocol


def _base_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(protocol)
    value["source_sequences"] = list(protocol["confirmation_sequences"])
    value["reserved_sequences"] = protocol["reserved_sequences"]
    return value


def validate_request(
    request_path: Path,
    protocol_path: Path,
    protocol_git_blob_sha: str,
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    request = _read_json(request_path)
    expected = {
        "bayesian_phystwin_executed",
        "causal4d_executed",
        "claim_boundary",
        "confirmation_sequences",
        "marker_evaluation_authorized",
        "normal_view_prediction_authorized",
        "protocol_git_blob_sha",
        "protocol_path",
        "request_id",
        "reserved_sequences",
        "schema",
        "schema_version",
        "selected_alpha",
        "source_calibration_id",
    }
    if set(request) != expected:
        raise ValueError("held-out confirmation request fields changed")
    if request["schema"] != REQUEST_SCHEMA or request["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported held-out confirmation request schema")
    _hex(protocol_git_blob_sha, name="protocol Git blob", length=40)
    if request["protocol_path"] != protocol_path.as_posix():
        raise ValueError("confirmation protocol path changed")
    if request["protocol_git_blob_sha"] != protocol_git_blob_sha:
        raise ValueError("confirmation request does not bind the reviewed protocol blob")
    if request["confirmation_sequences"] != CONFIRMATION_SEQUENCES:
        raise ValueError("confirmation request sequence roster changed")
    if request["reserved_sequences"] != RESERVED_SEQUENCES:
        raise ValueError("confirmation request reserved boundary changed")
    if request["normal_view_prediction_authorized"] is not True:
        raise ValueError("normal-view confirmation prediction must be explicitly authorized")
    if request["marker_evaluation_authorized"] is not True:
        raise ValueError("marker evaluation must be explicitly authorized")
    if request["bayesian_phystwin_executed"] is not False:
        raise ValueError("BayesianPhysTwin exceeds this confirmation boundary")
    if request["causal4d_executed"] is not False:
        raise ValueError("Causal4D exceeds this confirmation boundary")
    if request["source_calibration_id"] != protocol["source_calibration"]["calibration_id"]:
        raise ValueError("confirmation request source calibration changed")
    if float(request["selected_alpha"]) != SELECTED_ALPHA:
        raise ValueError("confirmation request alpha changed")
    unsigned = dict(request)
    request_id = unsigned.pop("request_id", None)
    _hex(request_id, name="request_id", length=64)
    if content_id(unsigned) != request_id:
        raise ValueError("held-out confirmation request identity mismatch")
    return {
        "request_id": request_id,
        "protocol_id": protocol["protocol_id"],
        "confirmation_sequences": CONFIRMATION_SEQUENCES,
        "reserved_sequences": RESERVED_SEQUENCES,
    }


def _require_execution_identity(request_id: str, revision: str) -> None:
    _hex(request_id, name="request_id", length=64)
    _hex(revision, name="Prob4D revision", length=40)


def _load_script(filename: str, name: str, expected_blob: str) -> Any:
    path = Path(__file__).with_name(filename)
    source = path.read_bytes()
    if _git_blob_sha1(source) != expected_blob:
        raise RuntimeError(f"{filename} source bytes changed")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict(args: argparse.Namespace) -> int:
    protocol = _load_protocol(args.protocol)
    _require_execution_identity(args.request_id, args.prob4d_revision)
    base = _load_script(
        "run_dot_rope_cut3r_native_provider.py",
        "dot_cut3r_confirmation_provider",
        BASE_PROVIDER_BLOB,
    )
    adapted = _base_protocol(protocol)
    base._load_protocol = lambda _path: adapted
    return int(
        base.predict(
            argparse.Namespace(
                protocol=args.protocol,
                request_id=args.request_id,
                prob4d_revision=args.prob4d_revision,
                dataset_root=args.dataset_root,
                cut3r_checkout=args.cut3r_checkout,
                checkpoint=args.checkpoint,
                runtime_receipt=args.runtime_receipt,
                output_dir=args.output_dir,
            )
        )
    )


def _paired_difference(
    rows: Sequence[Mapping[str, Any]],
    selected: str,
    comparator: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    selected_values = {
        str(row["sequence"]): float(row["normalized_nll_per_dimension"])
        for row in rows
        if row.get("method") == selected
    }
    comparator_values = {
        str(row["sequence"]): float(row["normalized_nll_per_dimension"])
        for row in rows
        if row.get("method") == comparator
    }
    if set(selected_values) != set(CONFIRMATION_SEQUENCES):
        raise ValueError(f"selected method sequence roster is incomplete against {comparator}")
    if set(comparator_values) != set(CONFIRMATION_SEQUENCES):
        raise ValueError(f"comparator sequence roster is incomplete for {comparator}")
    differences = np.asarray(
        [
            selected_values[sequence] - comparator_values[sequence]
            for sequence in CONFIRMATION_SEQUENCES
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(replicates, differences.size))
    bootstrap = np.mean(differences[indices], axis=1)
    return {
        "comparator": comparator,
        "mean_selected_minus_comparator": float(np.mean(differences)),
        "lower_95": float(np.quantile(bootstrap, 0.025)),
        "upper_95": float(np.quantile(bootstrap, 0.975)),
        "sequence_wins": int(np.count_nonzero(differences < 0.0)),
        "sequence_count": int(differences.size),
        "per_sequence": {
            sequence: float(difference)
            for sequence, difference in zip(
                CONFIRMATION_SEQUENCES,
                differences,
                strict=True,
            )
        },
    }


def _marker_support_payload(
    pooled: Any,
    protocol: Mapping[str, Any],
    request_id: str,
    revision: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SUPPORT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "repository_revision": revision,
        "confirmation_sequences": CONFIRMATION_SEQUENCES,
        "reserved_sequences": RESERVED_SEQUENCES,
        "coordinate_columns": protocol["marker_sampling"]["coordinate_columns"],
        "coordinate_mode": protocol["marker_sampling"]["coordinate_mode"],
        "preprocessing_transform": protocol["marker_sampling"]["preprocessing_transform"],
        "support_rule": protocol["marker_sampling"]["support_rule"],
        "marker_frames": sorted(
            pooled._MARKER_DIAGNOSTICS.values(),
            key=lambda row: (row["sequence"], row["run"], row["frame"]),
        ),
        "collections": list(pooled._COLLECTION_DIAGNOSTICS),
        "information_boundary": {
            "sealed_confirmation_provider_predictions_opened": True,
            "confirmation_2d_markers_opened_after_provider_seal": True,
            "confirmation_3d_markers_opened_after_provider_seal": True,
            "opened_sequences": CONFIRMATION_SEQUENCES,
            "reserved_sequences": RESERVED_SEQUENCES,
            "bayesian_phystwin_executed": False,
            "causal4d_executed": False,
        },
    }
    payload["support_id"] = content_id(payload)
    return payload


def _classification(comparisons: Mapping[str, Mapping[str, Any]]) -> str:
    pointwise = comparisons["pointwise_quadratic"]
    shared = comparisons["shared_quadratic_curvature"]
    local = comparisons["local_first_order"]
    means_negative = all(
        comparison["mean_selected_minus_comparator"] < 0.0
        for comparison in (pointwise, shared, local)
    )
    if means_negative and pointwise["upper_95"] < 0.0 and shared["upper_95"] < 0.0:
        return "heldout-strong-positive"
    if means_negative:
        return "heldout-directional-positive"
    return "heldout-mixed-or-negative"


def evaluate(args: argparse.Namespace) -> int:
    protocol = _load_protocol(args.protocol)
    _require_execution_identity(args.request_id, args.prob4d_revision)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)

    base = _load_script(
        "run_dot_rope_cut3r_native_provider.py",
        "dot_cut3r_confirmation_evaluator",
        BASE_PROVIDER_BLOB,
    )
    pooled = _load_script(
        "evaluate_dot_rope_cut3r_pooled.py",
        "dot_cut3r_confirmation_marker_mapping",
        POOLED_EVALUATOR_BLOB,
    )
    adapted = _base_protocol(protocol)
    base._load_protocol = lambda _path: adapted

    pooled.SEQUENCES = list(CONFIRMATION_SEQUENCES)
    pooled.RESERVED = RESERVED_SEQUENCES
    pooled.SUPPORT_RULE = dict(protocol["marker_sampling"]["support_rule"])
    pooled.PREPROCESSING_TRANSFORM = protocol["marker_sampling"]["preprocessing_transform"]
    pooled._ACTIVE_COORDINATE_COLUMNS = tuple(protocol["marker_sampling"]["coordinate_columns"])
    pooled._ACTIVE_COORDINATE_MODE = protocol["marker_sampling"]["coordinate_mode"]
    pooled._MARKER_DIAGNOSTICS.clear()
    pooled._COLLECTION_DIAGNOSTICS.clear()

    base._ORIGINAL_LOAD_RUN = base._load_run

    def load_run(bundle: Path, record: Mapping[str, Any]) -> dict[str, Any]:
        return pooled._load_run_with_metadata(base, bundle, record)

    base._load_run = load_run
    base.parse_coordinate_text = pooled._parse_coordinate_text
    base._sample_markers = pooled._sample_markers
    base._collect_pair = pooled._collect_pair
    base._collect_provider_truth = pooled._collect_provider_truth

    original_closures = base.covariance_closures

    def confirmation_closures(*closure_args, **closure_kwargs):
        closures = original_closures(*closure_args, **closure_kwargs)
        closures[SELECTED_METHOD] = temper_shared_dependence(
            closures["pointwise_quadratic"],
            closures["shared_quadratic_curvature"],
            SELECTED_ALPHA,
        )
        return closures

    base.covariance_closures = confirmation_closures

    try:
        status = int(
            base.evaluate(
                argparse.Namespace(
                    protocol=args.protocol,
                    request_id=args.request_id,
                    prob4d_revision=args.prob4d_revision,
                    dataset_root=args.dataset_root,
                    provider_bundle=args.provider_bundle,
                    output_dir=output / "base",
                )
            )
        )
        if status != 0:
            raise RuntimeError(f"registered evaluator returned status {status}")
        raw = _read_json(output / "base" / "result.json")
        rows = raw["method_rows"]
        statistics = protocol["statistics"]
        comparisons = {
            comparator: _paired_difference(
                rows,
                SELECTED_METHOD,
                comparator,
                replicates=int(statistics["paired_bootstrap_replicates"]),
                seed=int(statistics["paired_bootstrap_seed"]) + index,
            )
            for index, comparator in enumerate(statistics["primary_comparators"])
        }
        classification = _classification(comparisons)
        support = _marker_support_payload(
            pooled,
            protocol,
            args.request_id,
            args.prob4d_revision,
        )
        result = dict(raw)
        predecessor = result.pop("evaluation_id")
        result["schema"] = RESULT_SCHEMA
        result["schema_version"] = SCHEMA_VERSION
        result["decision"] = classification
        result["predecessor_evaluation_id"] = predecessor
        result["source_calibration"] = protocol["source_calibration"]
        result["selected_dependence_alpha"] = SELECTED_ALPHA
        result["selected_dependence_method"] = SELECTED_METHOD
        result["heldout_statistics"] = {
            "independent_unit": statistics["independent_unit"],
            "primary_metric": statistics["primary_metric"],
            "comparisons": comparisons,
            "classification": classification,
        }
        result["marker_support_id"] = support["support_id"]
        result["information_boundary"] = {
            "normal_view_images_opened_by_sealed_provider_stage": True,
            "markers_opened_only_after_provider_seal": True,
            "opened_sequences": CONFIRMATION_SEQUENCES,
            "reserved_sequences": RESERVED_SEQUENCES,
            "source_alpha_refit": False,
            "provider_means_changed": False,
            "bayesian_phystwin_executed": False,
            "causal4d_executed": False,
        }
        result["claim_boundary"] = protocol["claim_boundary"]
        result["evaluation_id"] = content_id(result)
        _write_json(output / "marker-support.json", support)
        _write_json(output / "result.json", result)

        aggregate = {
            row["method"]: row
            for row in result["aggregate_methods"]
            if row["method"]
            in {
                SELECTED_METHOD,
                "pointwise_quadratic",
                "shared_quadratic_curvature",
                "local_first_order",
                "cluster_bootstrap_fallback",
            }
        }
        lines = [
            "# Held-out DOT R04-R10 CUT3R dependence confirmation",
            "",
            f"Decision: **{classification}**",
            "",
            f"Evaluation ID: `{result['evaluation_id']}`",
            "",
            "| Method | mean NLL/dim | 95% covered | mean SD/span |",
            "|---|---:|---:|---:|",
        ]
        for method in (
            SELECTED_METHOD,
            "pointwise_quadratic",
            "shared_quadratic_curvature",
            "local_first_order",
            "cluster_bootstrap_fallback",
        ):
            row = aggregate[method]
            lines.append(
                f"| {method} | {row['mean_normalized_nll_per_dimension']:.6f} | "
                f"{row['covered_95_count']}/{row['sequence_count']} | "
                f"{row['mean_predictive_sd_fraction_of_span']:.6f} |"
            )
        lines.extend(["", "## Paired complete-sequence comparisons", ""])
        for comparator in statistics["primary_comparators"]:
            value = comparisons[comparator]
            lines.append(
                f"- selected - {comparator}: "
                f"{value['mean_selected_minus_comparator']:.6f} "
                f"[{value['lower_95']:.6f}, {value['upper_95']:.6f}], "
                f"wins {value['sequence_wins']}/{value['sequence_count']}."
            )
        lines.extend(
            [
                "",
                "No confirmation-side retuning was performed. R11-R70 remained unopened.",
            ]
        )
        (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "decision": classification,
                    "evaluation_id": result["evaluation_id"],
                    "selected_alpha": SELECTED_ALPHA,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        support = _marker_support_payload(
            pooled,
            protocol,
            args.request_id,
            args.prob4d_revision,
        )
        _write_json(output / "marker-support.json", support)
        message = f"{type(error).__name__}: {' '.join(str(error).split())}"
        is_support = any(
            token in message
            for token in (
                "pooled overlap has",
                "pooled overlap spans",
                "pooled provider/truth support",
            )
        )
        decision = "heldout-support-negative" if is_support else "technical-failure"
        failure: dict[str, Any] = {
            "schema": FAILURE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "request_id": args.request_id,
            "repository_revision": args.prob4d_revision,
            "decision": decision,
            "failure": message[:2000],
            "traceback_tail": traceback.format_exc().splitlines()[-20:],
            "marker_support_id": support["support_id"],
            "selected_dependence_alpha": SELECTED_ALPHA,
            "source_calibration_id": SOURCE_CALIBRATION_ID,
            "information_boundary": support["information_boundary"],
            "claim_boundary": protocol["claim_boundary"],
        }
        failure["result_id"] = content_id(failure)
        _write_json(output / "failure.json", failure)
        print(
            json.dumps(
                {
                    "decision": decision,
                    "result_id": failure["result_id"],
                    "failure": failure["failure"],
                },
                sort_keys=True,
            )
        )
        return 4 if is_support else 3


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-request":
        print(
            json.dumps(
                validate_request(
                    args.request,
                    args.protocol,
                    args.protocol_git_blob_sha,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "predict":
        return predict(args)
    if args.command == "evaluate":
        return evaluate(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
