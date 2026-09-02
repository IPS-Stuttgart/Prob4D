#!/usr/bin/env python3
"""Source-select a dependence-sensitive DEFORM query and evaluate it held out."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from prob4d.information_contract_witness import (
    evaluate_frozen_witness,
    select_falsification_witness,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
DLOS: Final = ("DLO4", "DLO5")
SPLIT_DOMAIN: Final = "prob4d-deform-dlo45-dependence-witness-v1"
SOURCE_ID: Final = "deform-dlo45-dependence-witness-source-v1"
HELD_ID: Final = "deform-dlo45-dependence-witness-held-v1"
QUERY_FAMILY_ID: Final = "deform-summary-linear-query-r12-v1"
SUMMARY_COMPONENTS: Final = (
    "terminal internal-node centroid x/y/z",
    "horizon-average internal-node centroid x/y/z",
    "terminal right-half minus left-half centroid x/y/z",
    "terminal minus initial internal-node centroid x/y/z",
)


def _adapter() -> dict[str, Any]:
    return runpy.run_path(
        "benchmarks/information_contract_v1/adapters/"
        "deform_dlo45_retrospective_v1.py"
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary_vector(residual_rows: FloatArray, *, horizon_frames: int) -> FloatArray:
    """Map one 25x8x3 forecast residual to twelve linear physical summaries."""

    values = np.asarray(residual_rows, dtype=np.float64)
    if values.shape != (horizon_frames * 8, 3) or not np.all(np.isfinite(values)):
        raise ValueError("residual_rows must contain one finite 8-node forecast")
    shaped = values.reshape(horizon_frames, 8, 3)
    terminal_centroid = np.mean(shaped[-1], axis=0)
    horizon_centroid = np.mean(shaped, axis=(0, 1))
    terminal_half_span = np.mean(shaped[-1, 4:], axis=0) - np.mean(
        shaped[-1, :4], axis=0
    )
    centroid_change = terminal_centroid - np.mean(shaped[0], axis=0)
    return np.concatenate(
        (terminal_centroid, horizon_centroid, terminal_half_span, centroid_change)
    )


def _ordered(paths: tuple[Path, ...], dlo: str) -> tuple[Path, ...]:
    def key(path: Path) -> tuple[bytes, str]:
        digest = hashlib.sha256(
            SPLIT_DOMAIN.encode("utf-8")
            + b"\0"
            + dlo.encode("utf-8")
            + b"\0"
            + path.name.encode("utf-8")
        ).digest()
        return digest, path.name

    return tuple(sorted(paths, key=key))


def _collect(
    paths: tuple[Path, ...],
    *,
    adapter: dict[str, Any],
    spec: Any,
) -> tuple[FloatArray, IntArray]:
    rows: list[FloatArray] = []
    group_index: list[int] = []
    for group, path in enumerate(paths):
        trajectory = adapter["_load_trajectory"](path)
        for current in adapter["_window_starts"](spec):
            prediction, truth = adapter["_prediction_and_truth"](
                trajectory, current, spec
            )
            rows.append(
                summary_vector(truth - prediction, horizon_frames=spec.horizon_frames)
            )
            group_index.append(group)
    return np.asarray(rows, dtype=np.float64), np.asarray(group_index, dtype=np.int64)


def _equal_group_second_moment(
    values: FloatArray, group_index: IntArray, group_count: int
) -> FloatArray:
    moments = []
    for group in range(group_count):
        selected = values[group_index == group]
        if not len(selected):
            raise ValueError("empty source group")
        moments.append(selected.T @ selected / len(selected))
    result = np.mean(moments, axis=0)
    result = 0.5 * (result + result.T)
    scale = max(float(np.trace(result)) / len(result), 1e-12)
    result += (1e-8 * scale + 1e-12) * np.eye(len(result))
    np.linalg.cholesky(result)
    return result


def _model_arrays(models: dict[str, FloatArray]) -> dict[str, FloatArray]:
    return {f"{dlo.lower()}_full_covariance": value for dlo, value in models.items()}


def source(protocol: Path, dataset_root: Path, output_root: Path) -> dict[str, Any]:
    adapter = _adapter()
    _, spec = adapter["_load_protocol"](protocol)
    output_root.mkdir(parents=True, exist_ok=False)
    source_values: list[FloatArray] = []
    source_covariance: list[FloatArray] = []
    source_index: list[IntArray] = []
    group_ids: list[str] = []
    full_covariances: dict[str, FloatArray] = {}
    partitions: dict[str, dict[str, list[str]]] = {}
    offset = 0
    for dlo in DLOS:
        paths = _ordered(adapter["_trajectory_paths"](dataset_root, dlo, "train"), dlo)
        fit_paths = paths[:40]
        witness_paths = paths[40:]
        fit_values, fit_index = _collect(fit_paths, adapter=adapter, spec=spec)
        full = _equal_group_second_moment(fit_values, fit_index, len(fit_paths))
        diagonal = np.diag(np.diag(full))
        witness_values, witness_index = _collect(
            witness_paths, adapter=adapter, spec=spec
        )
        source_values.append(witness_values)
        source_covariance.append(
            np.repeat(diagonal[None, :, :], len(witness_values), axis=0)
        )
        source_index.append(witness_index + offset)
        group_ids.extend(f"{dlo}/{path.name}" for path in witness_paths)
        offset += len(witness_paths)
        full_covariances[dlo] = full
        partitions[dlo] = {
            "fit": [path.name for path in fit_paths],
            "witness": [path.name for path in witness_paths],
        }

    payload = output_root / "source_payload.npz"
    adapter["_deterministic_npz"](
        payload,
        {
            "residual_vectors": np.concatenate(source_values),
            "reported_covariance": np.concatenate(source_covariance),
            "group_index": np.concatenate(source_index),
            "query_basis": np.eye(12, dtype=np.float64),
        },
    )
    manifest = {
        "schema_name": "prob4d.information-contract-witness-source",
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "audited_submission_id": "marginal-matched-diagonal",
        "aggregation_unit": "group_index",
        "group_ids": group_ids,
        "payload": payload.name,
        "payload_sha256": _sha256(payload),
        "query_family": {
            "query_family_id": QUERY_FAMILY_ID,
            "semantic_label": (
                "linear combinations of terminal, horizon-average, bending, "
                "and temporal-change 3-D residual summaries"
            ),
            "units": "metres",
            "basis_frozen_before_source_outcomes": True,
        },
        "coverage_probability": 0.9,
        "information_order": "source-only",
        "claim_boundary": (
            "The dependence witness uses disjoint official training trajectories. "
            "Its later DLO4/DLO5 evaluation is retrospective because those public "
            "targets were historically open."
        ),
    }
    manifest_path = output_root / "source_manifest.json"
    _write_json(manifest_path, manifest)
    witness = select_falsification_witness(manifest_path)
    _write_json(output_root / "witness.json", witness)
    np.savez_compressed(output_root / "source_model.npz", **_model_arrays(full_covariances))
    receipt = {
        "schema_name": "prob4d.deform-dlo45-dependence-witness-source-receipt",
        "schema_version": 1,
        "source_fit_trajectory_count": 80,
        "source_witness_trajectory_count": 32,
        "held_trajectory_count_opened": 0,
        "source_case_count": int(sum(len(value) for value in source_values)),
        "summary_components": list(SUMMARY_COMPONENTS),
        "partitions": partitions,
        "witness_id": witness["witness_id"],
        "query_vector": witness["query_vector"],
        "source_max_normalized_error_ratio": witness[
            "source_max_normalized_error_ratio"
        ],
        "source_payload_sha256": _sha256(payload),
    }
    _write_json(output_root / "source_receipt.json", receipt)
    return receipt


def held(
    protocol: Path,
    dataset_root: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    adapter = _adapter()
    _, spec = adapter["_load_protocol"](protocol)
    output_root.mkdir(parents=True, exist_ok=False)
    witness_path = source_root / "witness.json"
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    receipt = json.loads(
        (source_root / "source_receipt.json").read_text(encoding="utf-8")
    )
    if receipt["held_trajectory_count_opened"] != 0:
        raise ValueError("source stage opened held trajectories")
    with np.load(source_root / "source_model.npz", allow_pickle=False) as archive:
        full_covariances = {
            dlo: np.array(archive[f"{dlo.lower()}_full_covariance"], copy=True)
            for dlo in DLOS
        }

    values: list[FloatArray] = []
    full_blocks: list[FloatArray] = []
    diagonal_blocks: list[FloatArray] = []
    overconfident_blocks: list[FloatArray] = []
    group_index: list[int] = []
    group_ids: list[str] = []
    group = 0
    for dlo in DLOS:
        full = full_covariances[dlo]
        diagonal = np.diag(np.diag(full))
        for path in adapter["_trajectory_paths"](dataset_root, dlo, "eval"):
            group_ids.append(f"{dlo}/{path.name}")
            trajectory = adapter["_load_trajectory"](path)
            for current in adapter["_window_starts"](spec):
                prediction, truth = adapter["_prediction_and_truth"](
                    trajectory, current, spec
                )
                values.append(
                    summary_vector(
                        truth - prediction, horizon_frames=spec.horizon_frames
                    )
                )
                full_blocks.append(full)
                diagonal_blocks.append(diagonal)
                overconfident_blocks.append(0.05 * full)
                group_index.append(group)
            group += 1

    residual = np.asarray(values, dtype=np.float64)
    index = np.asarray(group_index, dtype=np.int64)
    covariance_by_id = {
        "full-source-fitted-dependence": np.asarray(full_blocks),
        "marginal-matched-diagonal": np.asarray(diagonal_blocks),
        "overconfident-full-scale-0.05": np.asarray(overconfident_blocks),
    }
    submissions = []
    for identifier, covariance in covariance_by_id.items():
        payload = output_root / f"{identifier}.npz"
        adapter["_deterministic_npz"](
            payload,
            {
                "residual_vectors": residual,
                "reported_covariance": covariance,
                "group_index": index,
            },
        )
        submissions.append(
            {
                "submission_id": identifier,
                "payload": payload.name,
                "payload_sha256": _sha256(payload),
            }
        )

    manifest = {
        "schema_name": "prob4d.information-contract-witness-held",
        "schema_version": 1,
        "held_id": HELD_ID,
        "source_witness_id": witness["witness_id"],
        "aggregation_unit": "group_index",
        "group_ids": group_ids,
        "information_order": "retrospective-open-target",
        "submissions": submissions,
        "claim_boundary": (
            "This is a retrospective dependence-witness diagnostic on public "
            "DLO4/DLO5 evaluation trajectories; no target query is selected."
        ),
    }
    manifest_path = output_root / "held_manifest.json"
    _write_json(manifest_path, manifest)
    result = evaluate_frozen_witness(manifest_path, witness_path)
    _write_json(output_root / "result.json", result)
    metrics = result["submissions"]
    full = metrics["full-source-fitted-dependence"]
    diagonal = metrics["marginal-matched-diagonal"]
    comparison = {
        "schema_name": "prob4d.deform-dlo45-dependence-witness-result",
        "schema_version": 1,
        "classification": "retrospective public-data dependence witness",
        "witness_id": witness["witness_id"],
        "query_vector": witness["query_vector"],
        "query_components": list(SUMMARY_COMPONENTS),
        "source_max_normalized_error_ratio": witness[
            "source_max_normalized_error_ratio"
        ],
        "held_case_count": int(len(residual)),
        "held_independent_trajectory_count": len(group_ids),
        "same_residuals_across_covariance_submissions": True,
        "full_nll_gain_over_diagonal": float(
            diagonal["equal_group_query_gaussian_nll"]
            - full["equal_group_query_gaussian_nll"]
        ),
        "full_calibration_error_gain_over_diagonal": float(
            diagonal["equal_group_query_absolute_log_calibration_error"]
            - full["equal_group_query_absolute_log_calibration_error"]
        ),
        "full_query_normalized_error_ratio": full[
            "equal_group_query_normalized_error_ratio"
        ],
        "diagonal_query_normalized_error_ratio": diagonal[
            "equal_group_query_normalized_error_ratio"
        ],
        "full_query_coverage": full["equal_group_query_coverage"],
        "diagonal_query_coverage": diagonal["equal_group_query_coverage"],
        "dependence_witness_holds_on_held_data": bool(
            diagonal["equal_group_query_gaussian_nll"]
            > full["equal_group_query_gaussian_nll"]
            and diagonal["equal_group_query_absolute_log_calibration_error"]
            > full["equal_group_query_absolute_log_calibration_error"]
        ),
        "target_query_reselection": result["target_query_reselection"],
        "submissions": metrics,
        "claim_boundary": (
            "The query was source-selected against a marginal-matched diagonal "
            "control. The held cohort was historically open, so the result is "
            "not an independent prospective confirmation."
        ),
    }
    _write_json(output_root / "comparison.json", comparison)
    return comparison


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("source", "held"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "source":
        result = source(args.protocol, args.dataset_root, args.output_root)
    else:
        if args.source_root is None:
            raise ValueError("held command requires --source-root")
        result = held(
            args.protocol,
            args.dataset_root,
            args.source_root,
            args.output_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
