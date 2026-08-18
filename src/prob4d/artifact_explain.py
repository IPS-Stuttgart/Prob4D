"""Human-readable, fail-closed explanations for Prob4D artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from ._strict_json import load_json_object, loads_json_object
from .gauge_tree_prior_artifact import (
    GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA,
    GAUGE_TREE_PRIOR_ARTIFACT_VERSION,
    load_gauge_tree_prior_artifact,
)
from .gauge_tree_prior_artifact import (
    artifact_summary as gauge_tree_prior_summary,
)
from .observation_contract import OBSERVATION_BELIEF_SCHEMA, OBSERVATION_BELIEF_VERSION
from .observation_validation import load_observation_belief_export, validation_summary
from .prediction_store import (
    PREDICTION_BUNDLE_STORE_SCHEMA,
    PREDICTION_BUNDLE_STORE_VERSION,
    PREDICTION_STORE_MANIFEST,
    prediction_bundle_store_summary,
)
from .tree_sparse_observation_artifact import (
    TREE_SPARSE_OBSERVATION_ARTIFACT_SCHEMA,
    TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION,
    load_tree_sparse_observation_artifact,
)

_IDENTITY_KEYS: Final = (
    "artifact_id",
    "manifest_id",
    "store_id",
    "portfolio_id",
    "report_id",
    "receipt_id",
    "decision_id",
    "authorization_id",
    "lock_id",
    "bundle_id",
    "prior_id",
    "calibration_id",
    "model_set_id",
)
_CONTEXT_KEYS: Final = (
    "sequence_id",
    "case_id",
    "stream_id",
    "provider_id",
    "provider_name",
    "provider_version",
    "provider_revision",
    "causal_frame_stop",
    "source_repository",
    "source_revision",
    "source_artifact_sha256",
    "claim_boundary",
    "storage_semantics",
)


def _descriptor_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError("NPZ descriptor_json must be one scalar UTF-8 string")
    item = array.item()
    if isinstance(item, bytes):
        try:
            return item.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("NPZ descriptor_json must contain UTF-8 text") from error
    if type(item) is not str:
        raise ValueError("NPZ descriptor_json must be one scalar UTF-8 string")
    return item


def _schema_name(value: Mapping[str, Any]) -> str | None:
    for key in ("schema_name", "schema"):
        candidate = value.get(key)
        if type(candidate) is str and candidate:
            return candidate
    return None


def _schema_version(value: Mapping[str, Any]) -> int | None:
    candidate = value.get("schema_version")
    return candidate if type(candidate) is int else None


def _selected_fields(
    value: Mapping[str, Any],
    keys: Sequence[str],
) -> dict[str, Any]:
    return {key: value[key] for key in keys if key in value}


def _known_explanation(
    *,
    path: Path,
    artifact_format: str,
    artifact_kind: str,
    schema: str,
    schema_version: int,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(summary)
    normalized.pop("status", None)
    normalized.pop("valid", None)
    identity = _selected_fields(normalized, _IDENTITY_KEYS)
    context = _selected_fields(normalized, _CONTEXT_KEYS)
    details = {
        key: value
        for key, value in normalized.items()
        if key not in identity
        and key not in context
        and key not in {"schema", "schema_name", "schema_version"}
    }
    return {
        "status": "valid",
        "path": str(path),
        "format": artifact_format,
        "artifact_kind": artifact_kind,
        "validation_scope": "strict-schema-and-content-address",
        "schema": schema,
        "schema_version": schema_version,
        "identity": identity,
        "context": context,
        "summary": details,
    }


def _structural_explanation(
    *,
    path: Path,
    artifact_format: str,
    artifact_kind: str,
    descriptor: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    explanation: dict[str, Any] = {
        "status": "structural-only",
        "path": str(path),
        "format": artifact_format,
        "artifact_kind": artifact_kind,
        "validation_scope": (
            "strict-JSON-object-syntax-only"
            if artifact_format == "json"
            else "non-pickled-NPZ-container-only"
        ),
        "schema": _schema_name(descriptor),
        "schema_version": _schema_version(descriptor),
        "identity": _selected_fields(descriptor, _IDENTITY_KEYS),
        "context": _selected_fields(descriptor, _CONTEXT_KEYS),
        "descriptor_fields": sorted(descriptor),
        "warnings": [
            "No registered strict loader matched this artifact; no schema semantics "
            "or content digest were verified."
        ],
    }
    if members:
        explanation["members"] = list(members)
    return explanation


def _tree_sparse_summary(path: Path) -> dict[str, Any]:
    loaded = load_tree_sparse_observation_artifact(path)
    manifest = loaded.manifest
    return {
        "artifact_id": manifest.artifact_id,
        "sequence_id": manifest.sequence_id,
        "case_id": manifest.case_id,
        "stream_id": manifest.stream_id,
        "causal_frame_stop": manifest.causal_frame_stop,
        "observation_count": manifest.observation_count,
        "gauge_count": len(manifest.gauge_ids),
        "view_count": len(manifest.view_id_table),
        "factor_count": len(manifest.factor_id_table),
        "correlation_group_count": len(manifest.correlation_group_id_table),
        "gauge_tree_prior_artifact_id": manifest.gauge_tree_prior_artifact_id,
        "source_repository": manifest.source_repository,
        "source_revision": manifest.source_revision,
        "claim_boundary": manifest.claim_boundary,
        "storage_semantics": manifest.storage_semantics,
    }


def _explain_json(path: Path, *, require_strict: bool) -> dict[str, Any]:
    descriptor = load_json_object(path, name="Prob4D artifact")
    schema = _schema_name(descriptor)
    if schema == GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA:
        summary = gauge_tree_prior_summary(load_gauge_tree_prior_artifact(path))
        return _known_explanation(
            path=path,
            artifact_format="json-with-npy-sidecars",
            artifact_kind="gauge-tree-prior-artifact-v1",
            schema=GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA,
            schema_version=GAUGE_TREE_PRIOR_ARTIFACT_VERSION,
            summary=summary,
        )
    if schema == TREE_SPARSE_OBSERVATION_ARTIFACT_SCHEMA:
        return _known_explanation(
            path=path,
            artifact_format="json-with-npy-sidecars",
            artifact_kind="tree-sparse-observation-artifact-v1",
            schema=TREE_SPARSE_OBSERVATION_ARTIFACT_SCHEMA,
            schema_version=TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION,
            summary=_tree_sparse_summary(path),
        )
    if schema == PREDICTION_BUNDLE_STORE_SCHEMA:
        return _known_explanation(
            path=path.parent,
            artifact_format="directory-with-json-and-npy-members",
            artifact_kind="prediction-bundle-store-v1",
            schema=PREDICTION_BUNDLE_STORE_SCHEMA,
            schema_version=PREDICTION_BUNDLE_STORE_VERSION,
            summary=prediction_bundle_store_summary(path.parent),
        )
    if require_strict:
        raise ValueError(
            "artifact has no registered strict loader; rerun without --require-strict "
            "for a structural explanation"
        )
    return _structural_explanation(
        path=path,
        artifact_format="json",
        artifact_kind="unrecognized-json-object",
        descriptor=descriptor,
    )


def _npz_members(
    archive: np.lib.npyio.NpzFile,
    *,
    include_arrays: bool,
) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for name in sorted(archive.files):
        member: dict[str, Any] = {"name": name}
        if include_arrays:
            array = np.asarray(archive[name])
            member.update(
                {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "nbytes": int(array.nbytes),
                }
            )
        members.append(member)
    return members


def _explain_npz(
    path: Path,
    *,
    require_strict: bool,
    include_arrays: bool,
) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            members = _npz_members(archive, include_arrays=include_arrays)
            if "descriptor_json" not in archive:
                descriptor: dict[str, Any] = {}
            else:
                descriptor = loads_json_object(
                    _descriptor_text(archive["descriptor_json"]),
                    name="NPZ descriptor_json",
                )
    except OSError as error:
        raise ValueError("artifact is not a valid non-pickled NPZ") from error
    except ValueError as error:
        if str(error).startswith("NPZ descriptor_json"):
            raise
        raise ValueError("artifact is not a valid non-pickled NPZ") from error

    if _schema_name(descriptor) == OBSERVATION_BELIEF_SCHEMA:
        artifact = load_observation_belief_export(path)
        explanation = _known_explanation(
            path=path,
            artifact_format="npz",
            artifact_kind="observation-belief-v1",
            schema=OBSERVATION_BELIEF_SCHEMA,
            schema_version=OBSERVATION_BELIEF_VERSION,
            summary=validation_summary(artifact),
        )
        if include_arrays:
            explanation["members"] = members
        return explanation
    if require_strict:
        raise ValueError(
            "NPZ artifact has no registered strict loader; rerun without "
            "--require-strict for a structural explanation"
        )
    return _structural_explanation(
        path=path,
        artifact_format="npz",
        artifact_kind="unrecognized-npz",
        descriptor=descriptor,
        members=members,
    )


def _explain_directory(path: Path, *, require_strict: bool) -> dict[str, Any]:
    manifest = path / PREDICTION_STORE_MANIFEST
    if manifest.is_symlink():
        raise ValueError("artifact directory manifest must not be a symbolic link")
    if not manifest.is_file():
        raise ValueError(
            f"artifact directory has no regular {PREDICTION_STORE_MANIFEST}"
        )
    return _explain_json(manifest, require_strict=require_strict)


def explain_artifact(
    artifact: str | Path,
    *,
    require_strict: bool = False,
    include_arrays: bool = False,
) -> dict[str, Any]:
    """Explain one artifact without overstating unrecognized structural checks."""

    path = Path(artifact)
    if path.is_symlink():
        raise ValueError("artifact path must not be a symbolic link")
    if path.is_dir():
        return _explain_directory(path, require_strict=require_strict)
    if not path.is_file():
        raise ValueError(f"artifact is not a regular file or directory: {path}")
    if path.suffix.lower() == ".npz":
        return _explain_npz(
            path,
            require_strict=require_strict,
            include_arrays=include_arrays,
        )
    return _explain_json(path, require_strict=require_strict)


def _format_value(value: Any) -> str:
    if isinstance(value, (Mapping, list, tuple)) or value is None or isinstance(value, bool):
        return json.dumps(value, sort_keys=True, allow_nan=False)
    return str(value)


def render_text(explanation: Mapping[str, Any]) -> str:
    """Render a deterministic human-readable explanation."""

    lines = [
        f"Artifact: {explanation['path']}",
        f"Kind: {explanation['artifact_kind']}",
        f"Format: {explanation['format']}",
        f"Status: {explanation['status']}",
        f"Validation: {explanation['validation_scope']}",
    ]
    schema = explanation.get("schema")
    if schema is not None:
        lines.append(f"Schema: {schema} v{explanation.get('schema_version')}")
    for section in ("identity", "context", "summary"):
        values = explanation.get(section)
        if not isinstance(values, Mapping) or not values:
            continue
        lines.extend(["", section.capitalize() + ":"])
        for key in sorted(values):
            lines.append(f"  {key}: {_format_value(values[key])}")
    members = explanation.get("members")
    if isinstance(members, list) and members:
        lines.extend(["", "Members:"])
        for member in members:
            if not isinstance(member, Mapping):
                continue
            name = member.get("name", "<unnamed>")
            details = ", ".join(
                f"{key}={_format_value(member[key])}"
                for key in ("dtype", "shape", "nbytes")
                if key in member
            )
            lines.append(f"  {name}" + (f" ({details})" if details else ""))
    warnings = explanation.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grouped ``prob4d artifact explain`` command."""

    parser = argparse.ArgumentParser(
        prog="prob4d artifact explain",
        description=(
            "Explain a Prob4D JSON, NPZ, or prediction-store artifact. Known "
            "schemas use their strict loader; unknown artifacts are clearly "
            "reported as structural-only."
        )
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--require-strict",
        action="store_true",
        help="reject artifacts without a registered strict schema loader",
    )
    parser.add_argument(
        "--arrays",
        action="store_true",
        help="include NPZ member dtype, shape, and uncompressed byte counts",
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    try:
        explanation = explain_artifact(
            arguments.artifact,
            require_strict=arguments.require_strict,
            include_arrays=arguments.arrays,
        )
    except (OSError, ValueError) as error:
        print(f"unable to explain artifact: {error}", file=sys.stderr)
        return 2
    if arguments.json_output:
        print(json.dumps(explanation, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(render_text(explanation), end="")
    return 0


__all__ = ["explain_artifact", "main", "render_text"]


if __name__ == "__main__":
    raise SystemExit(main())
