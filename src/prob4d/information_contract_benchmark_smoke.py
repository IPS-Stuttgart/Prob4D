"""Deterministic development fixture for the information-contract benchmark."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._atomic_file import atomic_write_bytes, atomic_write_text
from .information_contract_benchmark import (
    _ALLOWED_TASKS,
    SUITE_SCHEMA,
    SUITE_VERSION,
    _query_moments,
    _sha256_file,
)

FloatArray = NDArray[np.float64]


def _deterministic_npz_bytes(arrays: Mapping[str, NDArray[Any]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            value = np.asarray(arrays[name])
            if value.dtype.kind == "O":
                raise ValueError("smoke arrays must not use object dtype")
            payload = io.BytesIO()
            np.lib.format.write_array(payload, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())
    return stream.getvalue()


def _smoke_covariance(sample_count: int) -> tuple[FloatArray, FloatArray]:
    local = np.repeat(
        np.diag(np.square([0.02, 0.02, 0.02]))[None, :, :],
        sample_count,
        axis=0,
    )
    factor = np.zeros((sample_count, 3, 1), dtype=np.float64)
    factor[:, 0, 0] = 0.05
    factor[:, 1, 0] = 0.08
    return local, factor


def _smoke_fallback_covariance(sample_count: int) -> tuple[FloatArray, FloatArray]:
    local = np.repeat(
        np.diag(np.square([0.04, 0.04, 0.04]))[None, :, :],
        sample_count,
        axis=0,
    )
    return local, np.empty((sample_count, 3, 0), dtype=np.float64)


def _reported_query_branch(
    prediction: FloatArray,
    local: FloatArray,
    factor: FloatArray,
    fallback: FloatArray,
    fallback_local: FloatArray,
    fallback_factor: FloatArray,
    query: FloatArray,
    admitted: NDArray[np.bool_],
) -> tuple[FloatArray, FloatArray]:
    candidate_mean, candidate_variance = _query_moments(
        prediction, local, factor, query
    )
    fallback_mean, fallback_variance = _query_moments(
        fallback, fallback_local, fallback_factor, query
    )
    return (
        np.where(admitted, candidate_mean, fallback_mean),
        np.where(admitted, candidate_variance, fallback_variance),
    )


def _smoke_case(
    *,
    reject_decision: bool,
) -> dict[str, NDArray[Any]]:
    truth = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    prediction = truth + np.array([0.05, 0.10, 0.0])
    fallback = truth + np.array([0.10, 0.0, 0.0])
    local, factor = _smoke_covariance(len(truth))
    fallback_local, fallback_factor = _smoke_fallback_covariance(len(truth))
    query = np.zeros((3, truth.size), dtype=np.float64)
    query[0, 0] = -1.0
    query[0, 6] = 1.0
    query[1, 1::3] = 1.0 / 3.0
    query[2, 0::3] = 1.0 / 3.0
    nullspace = np.zeros((truth.size, 1), dtype=np.float64)
    nullspace[1::3, 0] = 1.0
    admitted = np.array([True, False, True])
    reported_mean, reported_variance = _reported_query_branch(
        prediction,
        local,
        factor,
        fallback,
        fallback_local,
        fallback_factor,
        query,
        admitted,
    )

    if reject_decision:
        decision_losses = np.array(
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        )
        prior = np.full(4, 0.25)
        classes = np.array([0, 0, 1, 1], dtype=np.int64)
        mass = np.array([0.5, 0.5])
        reported_regret = np.array([1.0, 1.0])
        selected_action = np.array(1, dtype=np.int64)
        fallback_action = np.array(1, dtype=np.int64)
        decision_admitted = np.array(False)
        tolerance = np.array(0.1)
        realized = np.array([0.8, 0.2])
    else:
        decision_losses = np.array(
            [
                [0.0, 0.4, 0.6],
                [0.02, 0.3, 0.7],
                [0.01, 0.5, 0.2],
                [0.0, 0.6, 0.25],
            ]
        )
        prior = np.full(4, 0.25)
        classes = np.array([0, 0, 1, 1], dtype=np.int64)
        mass = np.array([0.6, 0.4])
        # Filled below from the exact evaluator formula for readability.
        differences = np.zeros((3, 3), dtype=np.float64)
        for class_index in range(2):
            mask = classes == class_index
            differences += mass[class_index] * np.max(
                decision_losses[mask, :, None] - decision_losses[mask, None, :],
                axis=0,
            )
        reported_regret = np.max(differences, axis=1)
        selected_action = np.array(0, dtype=np.int64)
        fallback_action = np.array(2, dtype=np.int64)
        decision_admitted = np.array(True)
        tolerance = np.array(0.05)
        realized = np.array([0.05, 0.4, 0.3])

    return {
        "truth_xyz_m": truth,
        "prediction_mean_xyz_m": prediction,
        "conditional_covariance_m2": local,
        "shared_factor_m": factor,
        "fallback_mean_xyz_m": fallback,
        "fallback_conditional_covariance_m2": fallback_local,
        "fallback_shared_factor_m": fallback_factor,
        "query_matrix": query,
        "nullspace_basis": nullspace,
        "query_admitted": admitted,
        "reported_query_mean": reported_mean,
        "reported_query_variance": reported_variance,
        "decision_loss_by_hypothesis": decision_losses,
        "hypothesis_prior": prior,
        "quotient_class": classes,
        "quotient_mass": mass,
        "reported_worst_case_regret": reported_regret,
        "selected_action": selected_action,
        "fallback_action": fallback_action,
        "decision_admitted": decision_admitted,
        "regret_tolerance": tolerance,
        "realized_action_loss": realized,
    }


def generate_smoke_suite(directory: str | Path, *, overwrite: bool = False) -> Path:
    """Create a deterministic two-group benchmark fixture."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    tasks = sorted(_ALLOWED_TASKS)
    for case_id, group_id, reject in (
        ("admissible-shared-dependence", "object-a", False),
        ("ambiguous-exact-fallback", "object-b", True),
    ):
        payload_path = root / f"{case_id}.npz"
        atomic_write_bytes(
            payload_path,
            _deterministic_npz_bytes(_smoke_case(reject_decision=reject)),
            overwrite=overwrite,
        )
        cases.append(
            {
                "case_id": case_id,
                "group_id": group_id,
                "payload": payload_path.name,
                "payload_sha256": _sha256_file(payload_path),
                "tasks": tasks,
                "metadata": {
                    "classification": "deterministic smoke control",
                    "uses_public_data": False,
                },
            }
        )
    suite = {
        "schema_name": SUITE_SCHEMA,
        "schema_version": SUITE_VERSION,
        "suite_id": "prob4d-information-contract-smoke-v1",
        "aggregation_unit": "group_id",
        "thresholds": {
            "coverage_probability": 0.9,
            "gauge_sensitivity_tolerance": 1e-12,
            "moment_atol": 1e-12,
            "relative_rank_tolerance": 1e-10,
        },
        "claim_boundary": (
            "Deterministic development control only; no learned provider, public "
            "dataset, calibration claim, physical benefit, or deployment result."
        ),
        "cases": cases,
    }
    suite_path = root / "suite.json"
    atomic_write_text(
        suite_path,
        json.dumps(suite, indent=2, sort_keys=True, allow_nan=False) + "\n",
        overwrite=overwrite,
    )
    return suite_path


__all__ = ["generate_smoke_suite"]
