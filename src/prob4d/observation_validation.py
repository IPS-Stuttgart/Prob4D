"""Strict loading and command-line validation for observation-belief artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ._strict_json import (
    loads_json_object,
    require_exact_integer,
    require_exact_string,
    require_mapping,
    require_nonempty_string,
    require_sha256,
    require_string_sequence,
)
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


def _descriptor_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(
            "observation artifact descriptor_json must be one scalar UTF-8 string"
        )
    item = array.item()
    if isinstance(item, bytes):
        try:
            return item.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "observation artifact descriptor_json must contain UTF-8 text"
            ) from error
    if type(item) is not str:
        raise ValueError(
            "observation artifact descriptor_json must be one scalar UTF-8 string"
        )
    return item


def load_observation_belief_export(
    path: str | Path,
) -> ObservationBeliefExportV1:
    """Load, reconstruct, and content-validate one portable observation belief."""

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            if "descriptor_json" not in archive:
                raise ValueError("observation artifact has no descriptor_json")
            descriptor = loads_json_object(
                _descriptor_text(archive["descriptor_json"]),
                name="observation artifact descriptor",
            )
            arrays = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name != "descriptor_json"
            }
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "observation artifact"
        ):
            raise
        raise ValueError(
            "observation artifact is not a valid non-pickled NPZ"
        ) from error

    descriptor_fields = set(descriptor)
    missing_descriptor = _REQUIRED_DESCRIPTOR_FIELDS - descriptor_fields
    extra_descriptor = descriptor_fields - _REQUIRED_DESCRIPTOR_FIELDS
    if missing_descriptor or extra_descriptor:
        raise ValueError(
            "observation artifact descriptor changed; "
            f"missing={sorted(missing_descriptor)}, "
            f"extra={sorted(extra_descriptor)}"
        )

    schema_name = require_exact_string(
        descriptor["schema_name"],
        name="observation artifact schema_name",
    )
    if schema_name != OBSERVATION_BELIEF_SCHEMA:
        raise ValueError("unsupported observation-belief schema")
    schema_version = require_exact_integer(
        descriptor["schema_version"],
        name="observation artifact schema_version",
        minimum=1,
    )
    if schema_version != OBSERVATION_BELIEF_VERSION:
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

    expected_artifact_id = require_sha256(
        descriptor["artifact_id"],
        name="observation artifact artifact_id",
    )
    artifact = ObservationBeliefExportV1(
        case_id=require_nonempty_string(
            descriptor["case_id"],
            name="observation artifact case_id",
        ),
        stream_id=require_nonempty_string(
            descriptor["stream_id"],
            name="observation artifact stream_id",
        ),
        causal_frame_stop=require_exact_integer(
            descriptor["causal_frame_stop"],
            name="observation artifact causal_frame_stop",
            minimum=1,
        ),
        view_names=require_string_sequence(
            descriptor["view_names"],
            name="observation artifact view_names",
        ),
        window_names=require_string_sequence(
            descriptor["window_names"],
            name="observation artifact window_names",
        ),
        factor_names=require_string_sequence(
            descriptor["factor_names"],
            name="observation artifact factor_names",
            allow_empty=True,
        ),
        source_repository=require_nonempty_string(
            descriptor["source_repository"],
            name="observation artifact source_repository",
        ),
        source_revision=require_nonempty_string(
            descriptor["source_revision"],
            name="observation artifact source_revision",
        ),
        source_artifact_sha256=require_sha256(
            descriptor["source_artifact_sha256"],
            name="observation artifact source_artifact_sha256",
        ),
        metadata=require_mapping(
            descriptor["metadata"],
            name="observation artifact metadata",
        ),
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
