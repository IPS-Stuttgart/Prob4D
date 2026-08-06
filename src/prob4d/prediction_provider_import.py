"""Strict generic import into the canonical prediction-provider manifest.

External providers write canonical :class:`~prob4d.data.PredictionWindow` NPZ
payloads and a small source specification.  This module validates the exact
payload bytes, per-output causal lineage, provider/model identities, dependence
semantics, and optional fields before materializing the single canonical
``PredictionProviderManifestV1`` contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_revision,
    require_sha256,
    require_string_sequence,
)
from .data import PredictionWindow
from .prediction_provider_manifest import (
    PREDICTION_PROVIDER_MANIFEST_VERSION,
    SOURCE_DEPENDENCY_SEMANTICS,
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    _relative_member,
    _resolved_member,
    save_prediction_provider_manifest,
    verify_prediction_provider_manifest,
)

PREDICTION_PROVIDER_IMPORT_SPEC_SCHEMA: Final = (
    "prob4d.prediction-provider-import-spec"
)
PREDICTION_PROVIDER_IMPORT_SPEC_VERSION: Final = 1

_SPEC_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "sequence_id",
        "provider_family",
        "provider_repository",
        "provider_revision",
        "provider_run_id",
        "model_set_id",
        "loader_id",
        "coordinate_semantics",
        "point_semantics",
        "flow_semantics",
        "ray_semantics",
        "source_dependency_semantics",
        "payloads",
        "metadata",
    }
)
_SPEC_PAYLOAD_FIELDS: Final = frozenset(
    {
        "product_role",
        "window_id",
        "path",
        "view_id",
        "stochastic_member_id",
        "dependence_group_ids",
        "frame_lineage",
    }
)
_RESERVED_METADATA_FIELDS: Final = frozenset(
    {
        "source_adapter",
        "source_import_spec_sha256",
        "source_import_spec_schema_version",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read import member {path.name!r}") from error
    return digest.hexdigest()


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    try:
        information = path.stat()
    except OSError as error:
        raise ValueError(f"cannot stat import member {path.name!r}") from error
    return (
        int(information.st_dev),
        int(information.st_ino),
        int(information.st_size),
        int(information.st_mtime_ns),
    )


def _snapshot_window(
    path: Path,
    *,
    expected_window_id: str,
) -> tuple[PredictionWindow, str, int]:
    """Load one canonical payload while rejecting admission-time mutation."""

    if not path.is_file():
        raise ValueError(f"prediction payload {path.name!r} is missing")
    before = _file_signature(path)
    initial_sha256 = _sha256_file(path)
    try:
        window = PredictionWindow.from_npz(path)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(
            f"prediction payload {path.name!r} is not a canonical PredictionWindow"
        ) from error
    final_sha256 = _sha256_file(path)
    after = _file_signature(path)
    if before != after or initial_sha256 != final_sha256:
        raise ValueError("prediction payload changed during generic import")
    if window.window_id != expected_window_id:
        raise ValueError("import specification and payload window IDs differ")
    return window, final_sha256, after[2]


def _load_specification(
    path: Path,
) -> tuple[Mapping[str, Any], str, tuple[int, int, int, int]]:
    """Read one strict specification from a stable byte snapshot."""

    before = _file_signature(path)
    initial_sha256 = _sha256_file(path)
    record = load_json_object(path, name="prediction-provider import specification")
    final_sha256 = _sha256_file(path)
    after = _file_signature(path)
    if before != after or initial_sha256 != final_sha256:
        raise ValueError("prediction-provider import specification changed during import")
    require_exact_fields(record, _SPEC_FIELDS, name="provider import specification")
    if record["schema"] != PREDICTION_PROVIDER_IMPORT_SPEC_SCHEMA:
        raise ValueError("unsupported prediction-provider import specification schema")
    version = require_exact_integer(
        record["schema_version"],
        name="prediction-provider import specification version",
        minimum=1,
    )
    if version != PREDICTION_PROVIDER_IMPORT_SPEC_VERSION:
        raise ValueError("unsupported prediction-provider import specification version")
    return record, final_sha256, after


def _lineage_from_specification(
    value: object,
    *,
    name: str,
) -> tuple[PredictionFrameLineageV1, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty JSON array")
    return tuple(PredictionFrameLineageV1.from_record(item) for item in value)


def import_prediction_provider_specification(
    specification_path: str | Path,
    output_path: str | Path,
) -> PredictionProviderManifestV1:
    """Validate external canonical payloads and write one neutral manifest.

    Payload paths in the source specification are resolved relative to the
    specification file.  Persisted paths are rewritten relative to the output
    manifest, while exact payload bytes and all scientific/provenance semantics
    remain identity-bearing through ``PredictionPayloadDescriptorV1``.
    """

    specification = Path(specification_path).resolve()
    output = Path(output_path).resolve()
    record, specification_sha256, specification_signature = _load_specification(
        specification
    )

    raw_payloads = record["payloads"]
    if not isinstance(raw_payloads, list) or not raw_payloads:
        raise ValueError("prediction-provider import specification requires payloads")

    payloads: list[PredictionPayloadDescriptorV1] = []
    for index, raw_payload in enumerate(raw_payloads):
        payload_record = require_mapping(
            raw_payload,
            name=f"provider import payload {index}",
        )
        require_exact_fields(
            payload_record,
            _SPEC_PAYLOAD_FIELDS,
            name=f"provider import payload {index}",
        )
        window_id = require_exact_string(
            payload_record["window_id"],
            name=f"provider import payload {index} window_id",
        )
        source_member = _resolved_member(
            specification.parent,
            require_exact_string(
                payload_record["path"],
                name=f"provider import payload {index} path",
            ),
            name=f"provider import payload {index} path",
        )
        window, sha256, byte_count = _snapshot_window(
            source_member,
            expected_window_id=window_id,
        )
        lineage = _lineage_from_specification(
            payload_record["frame_lineage"],
            name=f"provider import payload {index} frame_lineage",
        )
        output_frame_ids = tuple(int(value) for value in window.frame_indices)
        if tuple(item.output_frame_id for item in lineage) != output_frame_ids:
            raise ValueError(
                "provider import frame lineage differs from payload frame identities"
            )
        dependence_groups = require_string_sequence(
            payload_record["dependence_group_ids"],
            name=f"provider import payload {index} dependence_group_ids",
        )
        payloads.append(
            PredictionPayloadDescriptorV1(
                product_role=payload_record["product_role"],
                window_id=window_id,
                path=_relative_member(
                    source_member,
                    root=output.parent,
                    name=f"provider import payload {index} output path",
                ),
                sha256=sha256,
                byte_count=byte_count,
                view_id=payload_record["view_id"],
                stochastic_member_id=payload_record["stochastic_member_id"],
                dependence_group_ids=dependence_groups,
                dense_storage_dtype=window.dense_storage_dtype,
                has_scene_flow=window.scene_flow is not None,
                has_ray_directions=window.ray_directions is not None,
                frame_lineage=lineage,
            )
        )

    payloads.sort(
        key=lambda item: (
            item.view_id,
            item.output_frame_ids[0],
            item.window_id,
            item.stochastic_member_id,
        )
    )
    metadata = dict(
        require_finite_json_mapping(
            record["metadata"],
            name="provider import metadata",
        )
    )
    conflicting = sorted(_RESERVED_METADATA_FIELDS.intersection(metadata))
    if conflicting:
        raise ValueError(
            "provider import metadata uses reserved fields: " + ", ".join(conflicting)
        )
    metadata.update(
        {
            "source_adapter": "prob4d-external-provider-import-spec-v1",
            "source_import_spec_sha256": specification_sha256,
            "source_import_spec_schema_version": (
                PREDICTION_PROVIDER_IMPORT_SPEC_VERSION
            ),
        }
    )

    source_semantics = require_exact_string(
        record["source_dependency_semantics"],
        name="source_dependency_semantics",
    )
    if source_semantics != SOURCE_DEPENDENCY_SEMANTICS:
        raise ValueError("unsupported source-dependency semantics")
    manifest = PredictionProviderManifestV1(
        sequence_id=record["sequence_id"],
        provider_family=record["provider_family"],
        provider_repository=record["provider_repository"],
        provider_revision=require_revision(
            record["provider_revision"],
            name="provider_revision",
        ),
        provider_run_id=require_sha256(
            record["provider_run_id"],
            name="provider_run_id",
        ),
        model_set_id=require_sha256(record["model_set_id"], name="model_set_id"),
        loader_id=require_sha256(record["loader_id"], name="loader_id"),
        coordinate_semantics=record["coordinate_semantics"],
        point_semantics=record["point_semantics"],
        flow_semantics=record["flow_semantics"],
        ray_semantics=record["ray_semantics"],
        source_dependency_semantics=source_semantics,
        payloads=tuple(payloads),
        metadata=metadata,
    )

    if _file_signature(specification) != specification_signature:
        raise ValueError("prediction-provider import specification changed during import")
    if _sha256_file(specification) != specification_sha256:
        raise ValueError("prediction-provider import specification changed during import")
    save_prediction_provider_manifest(output, manifest)
    verified, _ = verify_prediction_provider_manifest(output)
    return verified


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import external canonical predictions into the neutral manifest."
    )
    parser.add_argument("specification")
    parser.add_argument("output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    manifest = import_prediction_provider_specification(
        arguments.specification,
        arguments.output,
    )
    print(json.dumps(manifest.summary(), indent=2, sort_keys=True))
    return 0


__all__ = [
    "PREDICTION_PROVIDER_IMPORT_SPEC_SCHEMA",
    "PREDICTION_PROVIDER_IMPORT_SPEC_VERSION",
    "import_prediction_provider_specification",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
