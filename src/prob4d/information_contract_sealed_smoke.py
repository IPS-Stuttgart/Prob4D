"""Deterministic truth-separated fixture for the information-contract benchmark."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ._atomic_file import atomic_write_bytes, atomic_write_text
from .information_contract_benchmark import _sha256_file
from .information_contract_benchmark_smoke import generate_smoke_suite
from .information_contract_sealed import (
    _CHALLENGE_ARRAYS,
    _SUBMISSION_ARRAYS,
    CHALLENGE_SCHEMA,
    CHALLENGE_VERSION,
    FINITE_QUERY_TASK,
    SUBMISSION_SCHEMA,
    SUBMISSION_VERSION,
    _deterministic_npz_bytes,
)


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _finite_query_arrays(query_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if query_count != 3:
        raise ValueError("the deterministic smoke fixture requires exactly three queries")
    values = np.array(
        [
            [0.0, 0.0, 0.3],
            [0.2, 1.0, 0.3],
            [0.0, 0.0, 0.4],
            [0.2, 1.0, 0.4],
        ],
        dtype=np.float64,
    )
    tolerance = np.array([0.05, 0.05, 0.0], dtype=np.float64)
    admitted = np.array([False, False, True], dtype=np.bool_)
    return values, tolerance, admitted


def generate_sealed_smoke(
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Create one deterministic retrospective challenge/submission pair."""

    root = Path(directory)
    challenge_root = root / "challenge"
    submission_root = root / "submission"
    challenge_cases_root = challenge_root / "cases"
    submission_cases_root = submission_root / "cases"
    challenge_cases_root.mkdir(parents=True, exist_ok=True)
    submission_cases_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="prob4d-legacy-smoke-") as temporary:
        suite_path = generate_smoke_suite(Path(temporary) / "suite")
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        challenge_cases: list[dict[str, Any]] = []
        submission_cases: list[dict[str, Any]] = []
        dataset_cases: list[dict[str, str]] = []
        producer_rows: list[dict[str, str]] = []
        for case in suite["cases"]:
            case_id = str(case["case_id"])
            source_payload = suite_path.parent / case["payload"]
            with np.load(source_payload, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            finite_values, finite_tolerance, finite_admitted = _finite_query_arrays(
                int(arrays["query_matrix"].shape[0])
            )
            challenge_arrays = {
                name: value
                for name, value in arrays.items()
                if name in _CHALLENGE_ARRAYS
            }
            challenge_arrays["finite_query_value_by_hypothesis"] = finite_values
            challenge_arrays["finite_query_tolerance"] = finite_tolerance
            submission_arrays = {
                name: value
                for name, value in arrays.items()
                if name in _SUBMISSION_ARRAYS
            }
            submission_arrays["finite_query_admitted"] = finite_admitted

            challenge_payload = challenge_cases_root / f"{case_id}.npz"
            submission_payload = submission_cases_root / f"{case_id}.npz"
            atomic_write_bytes(
                challenge_payload,
                _deterministic_npz_bytes(challenge_arrays),
                overwrite=overwrite,
            )
            atomic_write_bytes(
                submission_payload,
                _deterministic_npz_bytes(submission_arrays),
                overwrite=overwrite,
            )
            challenge_sha = _sha256_file(challenge_payload)
            submission_sha = _sha256_file(submission_payload)
            challenge_cases.append(
                {
                    "case_id": case_id,
                    "group_id": str(case["group_id"]),
                    "payload": f"cases/{case_id}.npz",
                    "payload_sha256": challenge_sha,
                    "tasks": sorted([*case["tasks"], FINITE_QUERY_TASK]),
                    "metadata": {
                        "classification": "deterministic conformance control",
                        "original_case_id": case_id,
                    },
                }
            )
            submission_cases.append(
                {
                    "case_id": case_id,
                    "payload": f"cases/{case_id}.npz",
                    "payload_sha256": submission_sha,
                    "metadata": {
                        "classification": "deterministic provider control"
                    },
                }
            )
            dataset_cases.append(
                {
                    "case_id": case_id,
                    "group_id": str(case["group_id"]),
                    "challenge_payload_sha256": challenge_sha,
                }
            )
            producer_rows.append(
                {"case_id": case_id, "submission_payload_sha256": submission_sha}
            )

    dataset_manifest = {
        "schema_name": "prob4d.information-contract-controlled-dataset-manifest",
        "schema_version": 1,
        "classification": "deterministic synthetic development control",
        "public_data": False,
        "cases": dataset_cases,
    }
    dataset_manifest_path = challenge_root / "dataset-manifest.json"
    atomic_write_text(
        dataset_manifest_path,
        _canonical_json(dataset_manifest),
        overwrite=overwrite,
    )
    challenge = {
        "schema_name": CHALLENGE_SCHEMA,
        "schema_version": CHALLENGE_VERSION,
        "challenge_id": "prob4d-information-contract-sealed-smoke-v1",
        "aggregation_unit": "group_id",
        "thresholds": suite["thresholds"],
        "claim_boundary": (
            "Deterministic retrospective development control; no public data, "
            "learned provider, calibration claim, or physical benefit."
        ),
        "dataset": {
            "dataset_id": "prob4d-controlled-information-contract-smoke",
            "dataset_version": "1",
            "license_id": "MIT-generated-control",
            "public_data": False,
            "information_order": "retrospective-open-target",
            "manifest": "dataset-manifest.json",
            "manifest_sha256": _sha256_file(dataset_manifest_path),
        },
        "cases": challenge_cases,
    }
    challenge_path = challenge_root / "challenge.json"
    atomic_write_text(challenge_path, _canonical_json(challenge), overwrite=overwrite)

    producer_manifest_sha256 = hashlib.sha256(
        json.dumps(
            producer_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    submission = {
        "schema_name": SUBMISSION_SCHEMA,
        "schema_version": SUBMISSION_VERSION,
        "challenge_id": challenge["challenge_id"],
        "submission_id": "prob4d-deterministic-provider-control-v1",
        "producer": {
            "provider_name": "Prob4D deterministic sealed smoke provider",
            "provider_contract": "prob4d-controlled-provider-v1",
            "implementation_revision": "generated-with-current-checkout",
            "model_revision": "no-learned-model",
            "calibration_revision": "deterministic-analytic-covariance",
            "output_coordinate_frame": "controlled-metric-frame",
            "causal_cutoff": "complete-controlled-input",
            "dependence_group_ids": ["controlled-input:sealed-smoke-v1"],
            "submission_mode": "retrospective-replay",
            "producer_output_manifest_sha256": producer_manifest_sha256,
            "target_outcomes_used": False,
            "target_tuning": False,
            "prediction_sealed_before_truth": False,
        },
        "claim_boundary": (
            "Submission serialization and contract semantics only; the generated "
            "provider is not an empirical method."
        ),
        "cases": submission_cases,
    }
    submission_path = submission_root / "submission.json"
    atomic_write_text(submission_path, _canonical_json(submission), overwrite=overwrite)
    return challenge_path, submission_path


__all__ = ["generate_sealed_smoke"]
