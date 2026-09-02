"""Generate a truth-separated retrospective DEFORM DLO4/DLO5 scorecard pair."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prob4d._atomic_file import atomic_write_bytes, atomic_write_text
from prob4d.information_contract_sealed import (
    CHALLENGE_SCHEMA,
    CHALLENGE_VERSION,
    SUBMISSION_SCHEMA,
    SUBMISSION_VERSION,
    _canonical_json,
    _deterministic_npz_bytes,
)

FloatArray = NDArray[np.float64]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def npz_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            value = np.ascontiguousarray(archive[name])
            header = json.dumps(
                {"name": name, "dtype": value.dtype.str, "shape": list(value.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            data = value.tobytes()
            digest.update(len(header).to_bytes(8, "big") + header)
            digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_bpt(root: Path) -> tuple[Any, Any, Any]:
    sys.path[:0] = [str(root.resolve()), str((root / "src").resolve())]
    prefix = "experiments.deform_dlo45_decision_identifiability_v1"
    return (
        importlib.import_module(f"{prefix}._common"),
        importlib.import_module(f"{prefix}._evaluation"),
        importlib.import_module(f"{prefix}._model"),
    )


def trace_decision(
    feature: FloatArray,
    fitted: Any,
    protocol: Any,
    model_module: Any,
    *,
    atol: float,
) -> tuple[Any, dict[str, NDArray[Any]]]:
    decision = model_module.decide(feature, fitted, protocol)
    query = (feature - fitted.feature_mean) / fitted.feature_scale
    pool = (fitted.features - fitted.feature_mean) / fitted.feature_scale
    distance = np.mean(np.square(pool - query[None]), axis=1)
    count = min(fitted.neighbors, len(distance))
    selected = np.argpartition(distance, count - 1)[:count]
    selected = selected[np.lexsort((selected, distance[selected]))]
    d = distance[selected]
    positive = d[d > 0]
    base = float(np.median(positive)) if len(positive) else max(float(np.mean(d)), 1e-12)
    bandwidth = max(base * fitted.temperature_scale, 1e-12)
    weights = np.exp(-(d - float(np.min(d))) / bandwidth)
    weights /= weights.sum()
    global_classes = fitted.class_labels[selected]
    unique = np.unique(global_classes)
    remap = {int(value): index for index, value in enumerate(unique)}
    classes = np.asarray([remap[int(value)] for value in global_classes], dtype=np.int64)
    quotient = np.bincount(classes, weights=weights, minlength=len(unique)).astype(float)
    sizes = np.bincount(classes, minlength=len(unique)).astype(float)
    jeffrey = quotient[classes] / sizes[classes]
    residuals = fitted.residuals[selected]
    correction = np.einsum("i,id->d", jeffrey, residuals)
    actions = fitted.action_scales[:, None] * correction[None]
    raw_loss = np.mean(np.square(residuals[:, None] - actions[None]), axis=2)
    relative_loss = raw_loss / (raw_loss[:, :1] + fitted.loss_floor)
    pairwise = np.zeros((len(actions), len(actions)), dtype=float)
    for class_index in range(len(unique)):
        loss = relative_loss[classes == class_index]
        pairwise += quotient[class_index] * np.max(
            loss[:, :, None] - loss[:, None, :], axis=0
        )
    regret = np.max(pairwise, axis=1)
    np.testing.assert_allclose(regret, decision.worst_case_regret, rtol=0, atol=atol)
    np.testing.assert_allclose(correction, decision.correction, rtol=0, atol=atol)
    minimax = int(np.argmin(regret))
    admitted = float(regret[minimax]) <= fitted.regret_tolerance + atol
    if int(decision.certificate_action) != (minimax if admitted else 0):
        raise ValueError("independent trace does not reproduce the certificate action")
    return decision, {
        "decision_loss_by_hypothesis": relative_loss,
        "hypothesis_prior": np.full(count, 1.0 / count),
        "quotient_class": classes,
        "quotient_mass": quotient,
        "reported_worst_case_regret": regret,
        "decision_admitted": np.asarray(admitted, dtype=np.bool_),
    }


def write_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> str:
    atomic_write_bytes(path, _deterministic_npz_bytes(arrays), overwrite=False)
    return sha256_file(path)


def build(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    common, evaluation, model_module = load_bpt(args.bpt_root)
    protocol = common.load_protocol(args.protocol)
    models = evaluation.load_models(args.source_model)
    seal = load_json(args.source_seal)
    source_model_sha = sha256_file(args.source_model)
    if seal.get("source_model_sha256") != source_model_sha:
        raise ValueError("source model is not bound by the source seal")
    if seal.get("protocol_sha256") != sha256_file(args.protocol):
        raise ValueError("protocol is not bound by the source seal")

    root = args.output_root
    challenge_root = root / "challenge"
    submission_root = root / "submission"
    challenge_root.mkdir(parents=True)
    submission_root.mkdir(parents=True)
    challenge_cases: list[dict[str, Any]] = []
    submission_cases: list[dict[str, Any]] = []
    eval_manifest: dict[str, Any] = {}
    trajectory_stats: list[dict[str, Any]] = []
    baseline_sse = 0.0
    selected_sse = 0.0
    action_counts = np.zeros(3, dtype=np.int64)
    harmful = 0

    for dlo in common.DLOS:
        for path in common.trajectory_paths(args.dataset_root, dlo, "eval"):
            relative = path.relative_to(args.dataset_root).as_posix()
            eval_manifest[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            trajectory = common.load_trajectory(path)
            local_fallback: list[float] = []
            local_selected: list[float] = []
            for ordinal, current in enumerate(common.window_starts(protocol)):
                observation = common.extract_observation(trajectory, current, protocol)
                decision, trace = trace_decision(
                    observation.feature,
                    models[dlo],
                    protocol,
                    model_module,
                    atol=float(common.ATOL),
                )
                truth = trajectory[
                    current + 1 : current + 1 + protocol.horizon_frames,
                    common.INTERNAL,
                ].copy()
                actual = (
                    (truth - observation.baseline).reshape(-1)
                    / observation.length_scale
                )
                actions = models[dlo].action_scales[:, None] * decision.correction[None]
                normalized_loss = np.mean(np.square(actual[None] - actions), axis=1)
                physical_loss = normalized_loss * observation.length_scale**2
                selected = int(decision.certificate_action)
                prediction = (
                    observation.baseline
                    + actions[selected].reshape(truth.shape) * observation.length_scale
                )
                identifier = f"{dlo.lower()}-{path.stem}-window-{ordinal:02d}"
                challenge_payload = challenge_root / f"{identifier}.npz"
                submission_payload = submission_root / f"{identifier}.npz"
                challenge_sha = write_npz(
                    challenge_payload,
                    {
                        "truth_xyz_m": truth.reshape(-1, 3),
                        "decision_loss_by_hypothesis": trace[
                            "decision_loss_by_hypothesis"
                        ],
                        "hypothesis_prior": trace["hypothesis_prior"],
                        "quotient_class": trace["quotient_class"],
                        "quotient_mass": trace["quotient_mass"],
                        "fallback_action": np.asarray(0, dtype=np.int64),
                        "regret_tolerance": np.asarray(models[dlo].regret_tolerance),
                        "realized_action_loss": normalized_loss,
                    },
                )
                submission_sha = write_npz(
                    submission_payload,
                    {
                        "prediction_mean_xyz_m": prediction.reshape(-1, 3),
                        "reported_worst_case_regret": trace[
                            "reported_worst_case_regret"
                        ],
                        "selected_action": np.asarray(selected, dtype=np.int64),
                        "decision_admitted": trace["decision_admitted"],
                    },
                )
                case_id = f"{dlo}/{path.stem}/window-{ordinal:02d}"
                challenge_cases.append(
                    {
                        "case_id": case_id,
                        "group_id": f"{dlo}/{path.name}",
                        "payload": challenge_payload.name,
                        "payload_sha256": challenge_sha,
                        "tasks": ["forecast", "decision"],
                        "metadata": {
                            "dlo": dlo,
                            "trajectory": path.name,
                            "window_ordinal": ordinal,
                            "current_frame": current,
                            "loss": "fallback-normalized-trajectory-mse",
                        },
                    }
                )
                submission_cases.append(
                    {
                        "case_id": case_id,
                        "payload": submission_payload.name,
                        "payload_sha256": submission_sha,
                    }
                )
                fallback = float(physical_loss[0])
                chosen = float(physical_loss[selected])
                local_fallback.append(fallback)
                local_selected.append(chosen)
                baseline_sse += fallback
                selected_sse += chosen
                action_counts[selected] += 1
                harmful += int(chosen > fallback + float(common.ATOL))
            fallback_rmse = math.sqrt(float(np.mean(local_fallback)))
            selected_rmse = math.sqrt(float(np.mean(local_selected)))
            trajectory_stats.append(
                {
                    "group_id": f"{dlo}/{path.name}",
                    "fallback_rmse_mm": 1000 * fallback_rmse,
                    "certificate_rmse_mm": 1000 * selected_rmse,
                    "certificate_ratio": selected_rmse / max(fallback_rmse, 1e-12),
                }
            )

    dataset_manifest = {
        "schema_name": "prob4d.information-contract-dataset-manifest",
        "schema_version": 1,
        "dataset_id": "roahmlab-DEFORM-DLO4-DLO5-eval",
        "dataset_repository": "roahmlab/DEFORM",
        "dataset_revision": args.dataset_revision,
        "license_id": "NOASSERTION",
        "information_order": "retrospective-open-target",
        "aggregation_unit": "complete evaluation trajectory",
        "evaluation_files": eval_manifest,
    }
    dataset_manifest_path = challenge_root / "dataset-manifest.json"
    atomic_write_text(
        dataset_manifest_path, _canonical_json(dataset_manifest), overwrite=False
    )
    challenge = {
        "schema_name": CHALLENGE_SCHEMA,
        "schema_version": CHALLENGE_VERSION,
        "challenge_id": "deform-dlo45-decision-information-contract-v1",
        "aggregation_unit": "group_id",
        "thresholds": {
            "coverage_probability": 0.9,
            "gauge_sensitivity_tolerance": 1e-12,
            "moment_atol": 1e-12,
            "relative_rank_tolerance": 1e-10,
        },
        "claim_boundary": "Retrospective adapter parity only; no new held-out claim.",
        "dataset": {
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_version": args.dataset_revision,
            "license_id": "NOASSERTION",
            "public_data": True,
            "information_order": "retrospective-open-target",
            "manifest": dataset_manifest_path.name,
            "manifest_sha256": sha256_file(dataset_manifest_path),
        },
        "cases": challenge_cases,
    }
    challenge_path = challenge_root / "challenge.json"
    atomic_write_text(challenge_path, _canonical_json(challenge), overwrite=False)

    model_content_sha = npz_content_sha256(args.source_model)
    producer_manifest = {
        "schema_name": "prob4d.information-contract-producer-output-manifest",
        "schema_version": 1,
        "implementation_revision": args.bpt_revision,
        "model_content_sha256": model_content_sha,
        "source_result_sha256": sha256_file(args.source_result),
        "protocol_sha256": sha256_file(args.protocol),
        "case_payloads": {
            case["case_id"]: case["payload_sha256"] for case in submission_cases
        },
    }
    producer_manifest_path = submission_root / "producer-output-manifest.json"
    atomic_write_text(
        producer_manifest_path, _canonical_json(producer_manifest), overwrite=False
    )
    submission = {
        "schema_name": SUBMISSION_SCHEMA,
        "schema_version": SUBMISSION_VERSION,
        "challenge_id": challenge["challenge_id"],
        "submission_id": "bayesian-phystwin-dlo45-certificate-retrospective-v1",
        "producer": {
            "provider_name": "BayesianPhysTwin finite-support decision certificate",
            "provider_contract": "deform-dlo45-decision-identifiability-v1",
            "implementation_revision": args.bpt_revision,
            "model_revision": model_content_sha,
            "calibration_revision": sha256_file(args.source_result),
            "output_coordinate_frame": "DEFORM motion-capture metric frame",
            "causal_cutoff": (
                "five prefix frames plus registered future endpoint path"
            ),
            "dependence_group_ids": [
                "dataset:roahmlab-DEFORM",
                f"model:{model_content_sha}",
            ],
            "submission_mode": "retrospective-replay",
            "producer_output_manifest_sha256": sha256_file(producer_manifest_path),
            "target_outcomes_used": False,
            "target_tuning": False,
            "prediction_sealed_before_truth": False,
        },
        "claim_boundary": "Retrospective diagnostic replay only.",
        "cases": submission_cases,
    }
    submission_path = submission_root / "submission.json"
    atomic_write_text(submission_path, _canonical_json(submission), overwrite=False)

    ratios = np.asarray([row["certificate_ratio"] for row in trajectory_stats])
    stats = {
        "schema_name": "prob4d.deform-dlo45-information-contract-adapter",
        "schema_version": 1,
        "information_order": "retrospective-open-target",
        "prospective_claim_eligible": False,
        "decision_count": len(challenge_cases),
        "independent_trajectory_count": len(trajectory_stats),
        "action_counts": action_counts.tolist(),
        "nonfallback_count": int(action_counts[1:].sum()),
        "harmful_nonfallback_count": harmful,
        "certificate_rmse_ratio": math.sqrt(selected_sse / baseline_sse),
        "mean_trajectory_improvement": float(np.mean(1 - ratios)),
        "trajectory_wins_ties_losses": [
            int(np.count_nonzero(ratios < 1 - 1e-12)),
            int(np.count_nonzero(np.abs(ratios - 1) <= 1e-12)),
            int(np.count_nonzero(ratios > 1 + 1e-12)),
        ],
        "source_model_file_sha256": source_model_sha,
        "source_model_content_sha256": model_content_sha,
        "source_result_sha256": sha256_file(args.source_result),
        "challenge_sha256": sha256_file(challenge_path),
        "submission_sha256": sha256_file(submission_path),
        "trajectory_stats": trajectory_stats,
        "claim_boundary": (
            "Adapter-format validation only; target outcomes were already open."
        ),
    }
    stats_path = root / "adapter-stats.json"
    atomic_write_text(stats_path, _canonical_json(stats), overwrite=False)
    return challenge_path, submission_path, stats_path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "bpt-root",
        "dataset-root",
        "protocol",
        "source-model",
        "source-seal",
        "source-result",
        "output-root",
    ):
        result.add_argument(f"--{flag}", type=Path, required=True)
    result.add_argument("--bpt-revision", required=True)
    result.add_argument("--dataset-revision", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    challenge, submission, stats = build(args)
    print(
        json.dumps(
            {
                "challenge": str(challenge),
                "submission": str(submission),
                "stats": str(stats),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
