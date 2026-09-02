"""Deterministic truth-separated fixture for the information-contract benchmark."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._atomic_file import atomic_write_bytes, atomic_write_text
from .information_contract_benchmark import generate_smoke_suite
from .information_contract_sealed import (
    _CHALLENGE_ARRAYS,
    _SUBMISSION_ARRAYS,
    _canonical_json,
    _deterministic_npz_bytes,
    _sha256_file,
    CHALLENGE_SCHEMA,
    CHALLENGE_VERSION,
    SUBMISSION_SCHEMA,
    SUBMISSION_VERSION,
)

FloatArray = NDArray[np.float64]


def _finite_query_control(
    arrays: dict[str, NDArray[Any]],
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    prior = np.asarray(arrays["hypothesis_prior"], dtype=np.float64)
    query = np.asarray(arrays["query_matrix"], dtype=np.float64)
    hypothesis_count = int(prior.size)
    query_count = int(query.shape[0])
    if hypothesis_count != 4 or query_count != 3:
        raise ValueError("legacy smoke fixture changed unexpectedly")
    values = np.array(
        [
            [0.0, 0.0, 0.30],
            [0.2, 1.0, 0.30],
            [0.0, 0.0, 0.40],
            [0.2, 1.0, 0.40],
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
    """Generate one deterministic retrospective challenge/submission pair."""

    root = Path(directory)
    challenge_root = root / "challenge"
    submission_root = root / "submission"
    challenge_root.mkdir(parents=True, exist_ok=True)
    submission_root.mkdir(parents=True, exist_ok=True)

    challenge_cases: list[dict[str, Any]] = []
    submission_cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="prob4d-information-contract-legacy-smoke-"
    ) as temporary:
        legacy_suite_path = generate_smoke_suite(Path(temporary) / "legacy")
        legacy_suite = json.loads(
            legacy_suite_path.read_text(encoding="utf-8")
        )
        for case in legacy_suite["cases"]:
            case_id = str(case["case_id"])
            payload_path = legacy_suite_path.parent / str(case["payload"])
            with np.load(payload_path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            values, tolerance, finite_admitted = _finite_query_control(arrays)
            challenge_arrays = {
                name: value
                for name, value in arrays.items()
                if name in _CHALLENGE_ARRAYS
            }
            challenge_arrays.update(
                {
                    "finite_query_value_by_hypothesis": values,
                    "finite_query_tolerance": tolerance,
                }
            )
            submission_arrays = {
                name: value
                for name, value in arrays.items()
                if name in _SUBMISSION_ARRAYS
            }
            submission_arrays["finite_query_admitted"] = finite_admitted

            challenge_payload = challenge_root / f"{case_id}.npz"
            submission_payload = submission_root / f"{case_id}.npz"
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
            challenge_cases.append(
                {
                    "case_id": case_id,
                    "group_id": str(case["group_id"]),
                    "payload": challenge_payload.name,
                    "payload_sha256": _sha256_file(challenge_payload),
                    "tasks": sorted(
                        set(case["tasks"]).union({"finite_query"})
                    ),
                    "metadata": {
                        "classification": "deterministic truth-side control",
                        "legacy_case_id": case_id,
                    },
                }
            )
            submission_cases.append(
                {
                    "case_id": case_id,
                    "payload": submission_payload.name,
                    "payload_sha256": _sha256_file(submission_payload),
                    "metadata": {
                        "classification": "deterministic provider-side control",
                    },
                }
            )

    dataset_manifest = {
        "schema_name": "prob4d.information-contract-dataset-manifest",
        "schema_version": 1,
        "dataset_id": "prob4d-controlled-information-contract-smoke",
        "case_ids": sorted(case["case_id"] for case in challenge_cases),
        "classification": "deterministic development fixture",
        "public_data_records": 0,
        "target_outcomes_were_open": True,
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
        "thresholds": {
            "coverage_probability": 0.9,
            "gauge_sensitivity_tolerance": 1e-12,
            "moment_atol": 1e-12,
            "relative_rank_tolerance": 1e-10,
        },
        "claim_boundary": (
            "Deterministic retrospective control only; the target values are "
            "constructed and open, so no held-out or provider claim is permitted."
        ),
        "dataset": {
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_version": "1",
            "license_id": "CC0-1.0",
            "public_data": False,
            "information_order": "retrospective-open-target",
            "manifest": dataset_manifest_path.name,
            "manifest_sha256": _sha256_file(dataset_manifest_path),
        },
        "cases": challenge_cases,
    }
    challenge_path = challenge_root / "challenge.json"
    atomic_write_text(
        challenge_path,
        _canonical_json(challenge),
        overwrite=overwrite,
    )

    producer_identity = b"prob4d-deterministic-sealed-smoke-provider-v1"
    submission = {
        "schema_name": SUBMISSION_SCHEMA,
        "schema_version": SUBMISSION_VERSION,
        "challenge_id": challenge["challenge_id"],
        "submission_id": "prob4d-deterministic-sealed-smoke-provider-v1",
        "producer": {
            "provider_name": "Prob4D deterministic smoke provider",
            "provider_contract": "prob4d.controlled-provider-v1",
            "implementation_revision": "deterministic-source-tree",
            "model_revision": "no-learned-model",
            "calibration_revision": "constructed-control",
            "output_coordinate_frame": "registered-control-frame",
            "causal_cutoff": "all constructed source inputs",
            "dependence_group_ids": ["controlled-generator-v1"],
            "submission_mode": "retrospective-replay",
            "producer_output_manifest_sha256": hashlib.sha256(
                producer_identity
            ).hexdigest(),
            "target_outcomes_used": False,
            "target_tuning": False,
            "prediction_sealed_before_truth": False,
        },
        "claim_boundary": (
            "Deterministic provider-side conformance control; no empirical "
            "accuracy, calibration, or safety claim."
        ),
        "cases": submission_cases,
    }
    submission_path = submission_root / "submission.json"
    atomic_write_text(
        submission_path,
        _canonical_json(submission),
        overwrite=overwrite,
    )
    return challenge_path, submission_path


__all__ = ["generate_sealed_smoke"]
