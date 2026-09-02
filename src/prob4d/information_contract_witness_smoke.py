"""Deterministic source/held control for falsification-witness evaluation."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .information_contract_witness import (
    HELD_SCHEMA,
    SOURCE_SCHEMA,
    _sha256_bytes,
    _write_bytes,
    _write_json,
    evaluate_frozen_witness,
    select_falsification_witness,
)


def _deterministic_npz(arrays: Mapping[str, NDArray[Any]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o600 << 16
            stream = io.BytesIO()
            np.save(stream, np.asarray(arrays[name]), allow_pickle=False)
            archive.writestr(member, stream.getvalue())
    return output.getvalue()


def _axis_cases(
    group_count: int,
    amplitudes: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    groups: list[int] = []
    for group in range(group_count):
        group_scale = 1.0 + 0.01 * (group - (group_count - 1) / 2.0)
        for axis, amplitude in enumerate(amplitudes):
            for sign in (-1.0, 1.0):
                value = np.zeros(3, dtype=np.float64)
                value[axis] = sign * amplitude * group_scale
                rows.append(value)
                groups.append(group)
    return np.asarray(rows), np.asarray(groups, dtype=np.int64)


def _write_npz(
    path: Path,
    arrays: Mapping[str, NDArray[Any]],
    *,
    overwrite: bool,
) -> str:
    payload = _deterministic_npz(arrays)
    _write_bytes(path, payload, overwrite=overwrite)
    return _sha256_bytes(payload)


def generate_witness_smoke(directory: str | Path, *, overwrite: bool = False) -> Path:
    """Generate a controlled point-accuracy/query-calibration ranking reversal."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    source_residual, source_groups = _axis_cases(4, (0.020, 0.004, 0.003))
    source_empirical = np.mean(
        source_residual.reshape(4, 6, 3)[:, :, :, None]
        * source_residual.reshape(4, 6, 3)[:, :, None, :],
        axis=(0, 1),
    )
    source_covariance = source_empirical.copy()
    source_covariance[0, 0] /= 10.0
    source_covariances = np.repeat(source_covariance[None, :, :], len(source_residual), axis=0)
    source_payload = root / "source-provider-a.npz"
    source_sha = _write_npz(
        source_payload,
        {
            "group_index": source_groups,
            "query_basis": np.eye(3, dtype=np.float64),
            "reported_covariance": source_covariances,
            "residual_vectors": source_residual,
        },
        overwrite=overwrite,
    )
    source_manifest = {
        "schema_name": SOURCE_SCHEMA,
        "schema_version": 1,
        "source_id": "controlled-source-v1",
        "audited_submission_id": "provider-a-accurate-overconfident",
        "aggregation_unit": "group_index",
        "group_ids": [f"source-object-{index}" for index in range(4)],
        "payload": source_payload.name,
        "payload_sha256": source_sha,
        "query_family": {
            "query_family_id": "registered-regional-displacement-basis-v1",
            "semantic_label": "three orthogonal regional displacement coordinates",
            "units": "metres",
            "basis_frozen_before_source_outcomes": True,
        },
        "coverage_probability": 0.9,
        "information_order": "source-only",
        "claim_boundary": "Deterministic conformance control; no empirical provider claim.",
    }
    source_manifest_path = root / "source.json"
    _write_json(source_manifest_path, source_manifest, overwrite=overwrite)
    witness = select_falsification_witness(source_manifest_path)
    witness_path = root / "witness.json"
    _write_json(witness_path, witness, overwrite=overwrite)

    held_a, held_groups = _axis_cases(6, (0.018, 0.004, 0.003))
    held_b, held_groups_b = _axis_cases(6, (0.014, 0.014, 0.014))
    if not np.array_equal(held_groups, held_groups_b):
        raise RuntimeError("controlled held rosters diverged")
    held_a_empirical = np.mean(
        held_a.reshape(6, 6, 3)[:, :, :, None]
        * held_a.reshape(6, 6, 3)[:, :, None, :],
        axis=(0, 1),
    )
    held_a_covariance = held_a_empirical.copy()
    held_a_covariance[0, 0] /= 10.0
    held_b_covariance = np.mean(
        held_b.reshape(6, 6, 3)[:, :, :, None]
        * held_b.reshape(6, 6, 3)[:, :, None, :],
        axis=(0, 1),
    )
    submissions: list[dict[str, str]] = []
    for identifier, residual, covariance in (
        ("provider-a-accurate-overconfident", held_a, held_a_covariance),
        ("provider-b-less-accurate-calibrated", held_b, held_b_covariance),
    ):
        payload_path = root / f"held-{identifier}.npz"
        payload_sha = _write_npz(
            payload_path,
            {
                "group_index": held_groups,
                "reported_covariance": np.repeat(
                    covariance[None, :, :], len(residual), axis=0
                ),
                "residual_vectors": residual,
            },
            overwrite=overwrite,
        )
        submissions.append(
            {
                "submission_id": identifier,
                "payload": payload_path.name,
                "payload_sha256": payload_sha,
            }
        )
    held_manifest = {
        "schema_name": HELD_SCHEMA,
        "schema_version": 1,
        "held_id": "controlled-held-ranking-reversal-v1",
        "source_witness_id": witness["witness_id"],
        "aggregation_unit": "group_index",
        "group_ids": [f"held-object-{index}" for index in range(6)],
        "information_order": "retrospective-open-target",
        "submissions": submissions,
        "claim_boundary": "Deterministic conformance control; no empirical provider claim.",
    }
    held_path = root / "held.json"
    _write_json(held_path, held_manifest, overwrite=overwrite)
    result = evaluate_frozen_witness(held_path, witness_path)
    result_path = root / "result.json"
    _write_json(result_path, result, overwrite=overwrite)
    manifest = {
        "schema_name": "prob4d.information-contract-witness-smoke-manifest",
        "schema_version": 1,
        "files": {
            path.name: _sha256_bytes(path.read_bytes())
            for path in sorted(root.iterdir())
            if path.is_file()
        },
    }
    _write_json(root / "SHA256SUMS.json", manifest, overwrite=overwrite)
    return result_path


__all__ = ["generate_witness_smoke"]
