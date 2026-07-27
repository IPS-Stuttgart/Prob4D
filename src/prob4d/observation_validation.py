"""Strict loading and command-line validation for observation-belief artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .observation_contract import (
    OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION,
    ObservationBeliefExportV1,
)
from .observation_contract_bundle import observation_contract_schema

_SCHEMA = observation_contract_schema()
_REQUIRED_ARRAYS = frozenset(_SCHEMA["arrays"]["fields"])
_REQUIRED_DESCRIPTOR_FIELDS = frozenset(_SCHEMA["descriptor"]["fields"])
_ARRAY_DTYPES = {
    name: np.dtype(str(specification["dtype"]))
    for name, specification in _SCHEMA["arrays"]["fields"].items()
}


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def load_observation_belief_export(
    path: str | Path,
) -> ObservationBeliefExportV1:
    """Load, reconstruct, and content-validate one portable observation belief."""

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            if "descriptor_json" not in archive:
                raise ValueError("observation artifact has no descriptor_json")
            try:
                descriptor = json.loads(str(archive["descriptor_json"]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "observation artifact descriptor is not valid JSON"
                ) from error
            arrays = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name != "descriptor_json"
            }
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("observation artifact"):
            raise
        raise ValueError("observation artifact is not a valid non-pickled NPZ") from error

    if not isinstance(descriptor, dict):
        raise ValueError("observation artifact descriptor must be a JSON object")
    descriptor_fields = set(descriptor)
    missing_descriptor = _REQUIRED_DESCRIPTOR_FIELDS - descriptor_fields
    extra_descriptor = descriptor_fields - _REQUIRED_DESCRIPTOR_FIELDS
    if missing_descriptor or extra_descriptor:
        raise ValueError(
            "observation artifact descriptor changed; "
            f"missing={sorted(missing_descriptor)}, extra={sorted(extra_descriptor)}"
        )
    if descriptor["schema_name"] != OBSERVATION_BELIEF_SCHEMA:
        raise ValueError("unsupported observation-belief schema")
    if int(descriptor["schema_version"]) != OBSERVATION_BELIEF_VERSION:
        raise ValueError("unsupported observation-belief version")

    missing_arrays = _REQUIRED_ARRAYS - arrays.keys()
    extra_arrays = arrays.keys() - _REQUIRED_ARRAYS
    if missing_arrays or extra_arrays:
        raise ValueError(
            "observation artifact arrays changed; "
            f"missing={sorted(missing_arrays)}, extra={sorted(extra_arrays)}"
        )
    for name, expected_dtype in _ARRAY_DTYPES.items():
        if arrays[name].dtype != expected_dtype:
            raise ValueError(
                f"observation artifact array {name!r} has dtype "
                f"{arrays[name].dtype}, expected {expected_dtype}"
            )

    expected_artifact_id = str(descriptor["artifact_id"])
    _require_sha256(expected_artifact_id, name="artifact_id")
    artifact = ObservationBeliefExportV1(
        case_id=str(descriptor["case_id"]),
        stream_id=str(descriptor["stream_id"]),
        causal_frame_stop=int(descriptor["causal_frame_stop"]),
        view_names=tuple(map(str, descriptor["view_names"])),
        window_names=tuple(map(str, descriptor["window_names"])),
        factor_names=tuple(map(str, descriptor["factor_names"])),
        source_repository=str(descriptor["source_repository"]),
        source_revision=str(descriptor["source_revision"]),
        source_artifact_sha256=str(descriptor["source_artifact_sha256"]),
        metadata=descriptor["metadata"],
        **arrays,
    )
    if artifact.artifact_id != expected_artifact_id:
        raise ValueError("observation artifact digest does not match its payload")
    return artifact


def validation_summary(artifact: ObservationBeliefExportV1) -> dict[str, Any]:
    """Return the standard contract summary with an explicit validation status."""

    return {"status": "valid", **artifact.summary()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a content-addressed Phys4D observation artifact."
    )
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args(argv)
    artifact = load_observation_belief_export(arguments.artifact)
    print(json.dumps(validation_summary(artifact), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_observation_belief_export",
    "main",
    "validation_summary",
]
