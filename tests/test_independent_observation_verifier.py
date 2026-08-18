from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from prob4d.observation_contract import (
    ObservationBeliefExportV1,
    save_observation_belief_export,
)
from prob4d.observation_contract_bundle import (
    invalid_observation_contract_vector,
    observation_contract_artifact_id,
    observation_contract_invalid_cases,
    observation_contract_vector,
)
from prob4d_independent_verifier import (
    VERIFIER_IMPLEMENTATION,
    verify_observation_belief,
    write_verification_report,
)


def _artifact_from_vector(name: str) -> ObservationBeliefExportV1:
    vector = observation_contract_vector(name)
    descriptor = dict(vector.descriptor)
    return ObservationBeliefExportV1(
        case_id=descriptor["case_id"],
        stream_id=descriptor["stream_id"],
        causal_frame_stop=descriptor["causal_frame_stop"],
        view_names=tuple(descriptor["view_names"]),
        window_names=tuple(descriptor["window_names"]),
        factor_names=tuple(descriptor["factor_names"]),
        source_repository=descriptor["source_repository"],
        source_revision=descriptor["source_revision"],
        source_artifact_sha256=descriptor["source_artifact_sha256"],
        metadata=descriptor["metadata"],
        **{key: value.copy() for key, value in vector.arrays.items()},
    )


def _write_raw_npz(
    path: Path,
    descriptor: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> None:
    np.savez_compressed(
        path,
        descriptor_json=np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        **arrays,
    )


@pytest.mark.parametrize("name", ["minimal", "zero_rank"])
def test_independent_verifier_accepts_normative_vectors(tmp_path: Path, name: str) -> None:
    vector = observation_contract_vector(name)
    path = tmp_path / f"{name}.npz"
    save_observation_belief_export(path, _artifact_from_vector(name))

    report = verify_observation_belief(path)

    assert report.artifact_id == vector.expected_artifact_id
    assert report.to_dict()["status"] == "valid"
    assert report.to_dict()["verifier_implementation"] == VERIFIER_IMPLEMENTATION
    assert report.to_dict()["report_id"] == report.report_id
    assert len(report.arrays) == 15


def test_independent_verifier_report_is_atomic_and_content_addressed(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "observation.npz"
    save_observation_belief_export(artifact_path, _artifact_from_vector("minimal"))
    report = verify_observation_belief(artifact_path)
    report_path = tmp_path / "report.json"

    write_verification_report(report_path, report)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload == report.to_dict()
    assert payload["report_id"] == report.report_id
    with pytest.raises(FileExistsError):
        write_verification_report(report_path, report)


def test_independent_verifier_rejects_the_complete_invalid_corpus(
    tmp_path: Path,
) -> None:
    for case in observation_contract_invalid_cases():
        invalid = invalid_observation_contract_vector(case["id"])
        descriptor = dict(invalid.descriptor)
        arrays = {name: value.copy() for name, value in invalid.arrays.items()}
        if invalid.mode == "digest_mismatch":
            descriptor["artifact_id"] = invalid.original_artifact_id
        else:
            descriptor["artifact_id"] = observation_contract_artifact_id(
                descriptor, arrays
            )
        path = tmp_path / f"{invalid.case_id}.npz"
        _write_raw_npz(path, descriptor, arrays)
        with pytest.raises(ValueError):
            verify_observation_belief(path)


def test_independent_verifier_imports_no_prob4d_module() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    code = """
import sys
import prob4d_independent_verifier
assert not any(name == 'prob4d' or name.startswith('prob4d.') for name in sys.modules)
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "ok"
