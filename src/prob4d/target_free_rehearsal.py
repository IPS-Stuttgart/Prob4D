"""Target-free rehearsal for the portable observation admission boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np

from prob4d_independent_verifier import (
    VERIFIER_IMPLEMENTATION,
    verify_observation_belief,
    write_verification_report,
)

from ._atomic_file import atomic_write_text, publish_temporary_file
from ._strict_json import load_json_object, require_exact_fields, require_sha256
from .causal_stream_contract import PROB4D_CAUSAL_STREAM_ID, PROB4D_SOURCE_REPOSITORY
from .observation_contract import (
    ObservationBeliefExportV1,
    file_sha256,
    save_observation_belief_export,
)
from .observation_contract_bundle import (
    observation_contract_artifact_id,
    observation_contract_bundle_manifest,
    observation_contract_vector,
)
from .observation_validation import load_observation_belief_export
from .project_identity import prob4d_project_identity
from .provider_v2 import prob4d_provider_manifest
from .provider_v2_loading import load_claim_bearing_observation_belief
from .public_api_manifest import build_public_api_manifest

TARGET_FREE_REHEARSAL_SCHEMA = "prob4d.target-free-rehearsal-receipt"
TARGET_FREE_REHEARSAL_VERSION = 1
TARGET_FREE_REHEARSAL_PROTOCOL_ID = "prob4d-target-free-observation-rehearsal-v1"
TARGET_FREE_REHEARSAL_CLAIM_BOUNDARY = (
    "This receipt proves that a target-free normative observation artifact passes the "
    "official loader and an independently implemented verifier, that an unattested "
    "artifact cannot enter the claim-bearing loader, and that registered adversarial "
    "controls fail closed. It opens no source suffix, target payload, target outcome, or "
    "scientific evidence and does not establish provider accuracy, calibration, physical "
    "benefit, deployment safety, or state of the art."
)
_NEGATIVE_CONTROL_IDS = (
    "future-observation-frame",
    "duplicate-observation-identity",
    "non-positive-definite-local-covariance",
    "wrong-integer-dtype",
    "tampered-payload-preserves-artifact-id",
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "protocol_id",
        "source_revision",
        "package_version",
        "project_identity",
        "public_api_manifest_id",
        "provider_manifest_id",
        "contract_bundle_sha256",
        "environment",
        "positive_control",
        "negative_controls",
        "target_access",
        "claim_boundary",
        "receipt_id",
    }
)


@dataclass(frozen=True, slots=True)
class RejectionResult:
    """Whether two independent validation paths rejected one control."""

    official_rejected: bool
    official_error_type: str
    independent_rejected: bool
    independent_error_type: str

    def __post_init__(self) -> None:
        if not self.official_rejected or not self.independent_rejected:
            raise ValueError("every rehearsal control must fail closed in both validators")
        if not self.official_error_type or not self.independent_error_type:
            raise ValueError("rejection evidence must retain both exception types")

    def to_dict(self) -> dict[str, object]:
        return {
            "official_rejected": self.official_rejected,
            "official_error_type": self.official_error_type,
            "independent_rejected": self.independent_rejected,
            "independent_error_type": self.independent_error_type,
        }


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _receipt_id(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_id", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _validated_revision(value: str) -> str:
    revision = str(value)
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("source_revision must be one exact lowercase 40-character commit")
    return revision


def _artifact_from_normative_vector(source_revision: str) -> ObservationBeliefExportV1:
    vector = observation_contract_vector("minimal")
    descriptor = dict(vector.descriptor)
    metadata = dict(cast(Mapping[str, Any], descriptor["metadata"]))
    metadata["target_free_rehearsal"] = {
        "protocol_id": TARGET_FREE_REHEARSAL_PROTOCOL_ID,
        "source_suffix_payloads_opened": 0,
        "target_payloads_opened": 0,
        "target_outcomes_opened": 0,
        "scientific_evidence": False,
    }
    return ObservationBeliefExportV1(
        case_id="target-free-observation-rehearsal",
        stream_id=PROB4D_CAUSAL_STREAM_ID,
        causal_frame_stop=int(descriptor["causal_frame_stop"]),
        view_names=tuple(cast(Sequence[str], descriptor["view_names"])),
        window_names=tuple(cast(Sequence[str], descriptor["window_names"])),
        factor_names=tuple(cast(Sequence[str], descriptor["factor_names"])),
        source_repository=PROB4D_SOURCE_REPOSITORY,
        source_revision=source_revision,
        source_artifact_sha256=cast(str, descriptor["source_artifact_sha256"]),
        metadata=metadata,
        **{name: values.copy() for name, values in vector.arrays.items()},
    )


def _publish_official_artifact(
    path: Path,
    artifact: ObservationBeliefExportV1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_observation_belief_export(temporary, artifact)
        publish_temporary_file(temporary, path, overwrite=False)
    finally:
        temporary.unlink(missing_ok=True)


def _write_raw_artifact(
    path: Path,
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor_fd, "wb") as stream:
            np.savez_compressed(  # type: ignore[arg-type]
                stream,
                descriptor_json=np.asarray(
                    json.dumps(
                        dict(descriptor),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                ),
                **arrays,
            )
            stream.flush()
            os.fsync(stream.fileno())
        publish_temporary_file(temporary, path, overwrite=False)
    finally:
        temporary.unlink(missing_ok=True)


def _mutated_payloads(
    artifact: ObservationBeliefExportV1,
) -> tuple[tuple[str, dict[str, Any], dict[str, np.ndarray]], ...]:
    base_descriptor = artifact.descriptor()
    base_arrays = {name: values.copy() for name, values in artifact.arrays().items()}
    base_artifact_id = observation_contract_artifact_id(base_descriptor, base_arrays)
    controls: list[tuple[str, dict[str, Any], dict[str, np.ndarray]]] = []

    def semantic_control(
        identifier: str,
        array_name: str,
        replacement: np.ndarray,
    ) -> None:
        descriptor = dict(base_descriptor)
        arrays = {name: values.copy() for name, values in base_arrays.items()}
        arrays[array_name] = replacement
        descriptor["artifact_id"] = observation_contract_artifact_id(descriptor, arrays)
        controls.append((identifier, descriptor, arrays))

    future_frames = base_arrays["frame_ids"].copy()
    future_frames[-1] = artifact.causal_frame_stop
    semantic_control("future-observation-frame", "frame_ids", future_frames)

    duplicate_entities = base_arrays["entity_ids"].copy()
    duplicate_entities[1] = duplicate_entities[0]
    semantic_control(
        "duplicate-observation-identity",
        "entity_ids",
        duplicate_entities,
    )

    indefinite_covariance = base_arrays["local_covariance_m2"].copy()
    indefinite_covariance[0, 2, 2] = -abs(indefinite_covariance[0, 2, 2])
    semantic_control(
        "non-positive-definite-local-covariance",
        "local_covariance_m2",
        indefinite_covariance,
    )

    wrong_dtype = base_arrays["frame_ids"].astype(np.int32)
    semantic_control("wrong-integer-dtype", "frame_ids", wrong_dtype)

    tampered_descriptor = dict(base_descriptor)
    tampered_arrays = {name: values.copy() for name, values in base_arrays.items()}
    tampered_arrays["mean_xyz_m"][0, 0] += 0.25
    tampered_descriptor["artifact_id"] = base_artifact_id
    controls.append(
        (
            "tampered-payload-preserves-artifact-id",
            tampered_descriptor,
            tampered_arrays,
        )
    )
    return tuple(controls)


def _capture_rejection(path: Path) -> RejectionResult:
    official_rejected = False
    official_error_type = ""
    try:
        load_observation_belief_export(path)
    except Exception as error:  # noqa: BLE001 - receipt records the fail-closed type.
        official_rejected = True
        official_error_type = type(error).__name__

    independent_rejected = False
    independent_error_type = ""
    try:
        verify_observation_belief(path)
    except Exception as error:  # noqa: BLE001 - receipt records the fail-closed type.
        independent_rejected = True
        independent_error_type = type(error).__name__
    return RejectionResult(
        official_rejected=official_rejected,
        official_error_type=official_error_type,
        independent_rejected=independent_rejected,
        independent_error_type=independent_error_type,
    )


def _claim_bearing_rejection(path: Path) -> dict[str, object]:
    try:
        load_claim_bearing_observation_belief(path)
    except Exception as error:  # noqa: BLE001 - expected boundary rejection.
        return {
            "status": "rejected-as-required",
            "error_type": type(error).__name__,
        }
    raise ValueError("unattested rehearsal artifact entered the claim-bearing loader")


def run_target_free_rehearsal(
    output_dir: str | Path,
    *,
    source_revision: str,
) -> dict[str, Any]:
    """Run and seal one target-free positive and adversarial contract rehearsal."""

    revision = _validated_revision(source_revision)
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"rehearsal output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    artifact = _artifact_from_normative_vector(revision)
    valid_path = root / "valid_observation_belief.npz"
    _publish_official_artifact(valid_path, artifact)
    official = load_observation_belief_export(valid_path)
    independent = verify_observation_belief(valid_path)
    if official.artifact_id != independent.artifact_id:
        raise ValueError("official and independent artifact identities disagree")

    report_path = root / "independent_verification.json"
    write_verification_report(report_path, independent)
    positive_control = {
        "artifact": valid_path.name,
        "artifact_id": official.artifact_id,
        "artifact_file_sha256": file_sha256(valid_path),
        "official_loader_status": "valid",
        "independent_verifier_status": "valid",
        "independent_verifier_implementation": VERIFIER_IMPLEMENTATION,
        "independent_report": report_path.name,
        "independent_report_id": independent.report_id,
        "claim_bearing_loader": _claim_bearing_rejection(valid_path),
    }

    negative_dir = root / "negative_controls"
    negative_controls: list[dict[str, object]] = []
    for identifier, descriptor, arrays in _mutated_payloads(artifact):
        path = negative_dir / f"{identifier}.npz"
        _write_raw_artifact(path, descriptor, arrays)
        rejection = _capture_rejection(path)
        negative_controls.append(
            {
                "control_id": identifier,
                "artifact": path.relative_to(root).as_posix(),
                "artifact_file_sha256": file_sha256(path),
                **rejection.to_dict(),
            }
        )
    if tuple(item["control_id"] for item in negative_controls) != _NEGATIVE_CONTROL_IDS:
        raise ValueError("rehearsal negative-control roster changed")

    public_api = build_public_api_manifest()
    provider_manifest = prob4d_provider_manifest(provider_revision=revision)
    contract_bundle = observation_contract_bundle_manifest()
    receipt: dict[str, Any] = {
        "schema_name": TARGET_FREE_REHEARSAL_SCHEMA,
        "schema_version": TARGET_FREE_REHEARSAL_VERSION,
        "protocol_id": TARGET_FREE_REHEARSAL_PROTOCOL_ID,
        "source_revision": revision,
        "package_version": version("prob4d"),
        "project_identity": prob4d_project_identity(),
        "public_api_manifest_id": public_api["manifest_id"],
        "provider_manifest_id": provider_manifest["manifest_id"],
        "contract_bundle_sha256": contract_bundle["bundle_sha256"],
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.system().lower(),
        },
        "positive_control": positive_control,
        "negative_controls": negative_controls,
        "target_access": {
            "source_suffix_payloads_opened": 0,
            "target_payloads_opened": 0,
            "target_outcomes_opened": 0,
            "scientific_evidence": False,
        },
        "claim_boundary": TARGET_FREE_REHEARSAL_CLAIM_BOUNDARY,
    }
    receipt["receipt_id"] = _receipt_id(receipt)
    atomic_write_text(
        root / "target_free_rehearsal_receipt.json",
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
        overwrite=False,
    )
    return receipt


def verify_target_free_rehearsal(path: str | Path) -> dict[str, Any]:
    """Replay every receipt assertion against its retained artifacts."""

    receipt_path = Path(path)
    receipt = load_json_object(receipt_path, name="target-free rehearsal receipt")
    require_exact_fields(receipt, _RECEIPT_FIELDS, name="target-free rehearsal receipt")
    if receipt["schema_name"] != TARGET_FREE_REHEARSAL_SCHEMA:
        raise ValueError("unsupported target-free rehearsal schema")
    if receipt["schema_version"] != TARGET_FREE_REHEARSAL_VERSION:
        raise ValueError("unsupported target-free rehearsal version")
    if receipt["protocol_id"] != TARGET_FREE_REHEARSAL_PROTOCOL_ID:
        raise ValueError("target-free rehearsal protocol identity changed")
    _validated_revision(cast(str, receipt["source_revision"]))
    require_sha256(receipt["receipt_id"], name="target-free rehearsal receipt_id")
    if receipt["receipt_id"] != _receipt_id(receipt):
        raise ValueError("target-free rehearsal receipt ID does not match its content")
    if receipt["claim_boundary"] != TARGET_FREE_REHEARSAL_CLAIM_BOUNDARY:
        raise ValueError("target-free rehearsal claim boundary changed")
    if receipt["target_access"] != {
        "source_suffix_payloads_opened": 0,
        "target_payloads_opened": 0,
        "target_outcomes_opened": 0,
        "scientific_evidence": False,
    }:
        raise ValueError("target-free rehearsal records prohibited data access")
    revision = cast(str, receipt["source_revision"])
    if receipt["package_version"] != version("prob4d"):
        raise ValueError("target-free rehearsal package version differs from this runtime")
    if receipt["project_identity"] != prob4d_project_identity():
        raise ValueError("target-free rehearsal project identity changed")
    if receipt["public_api_manifest_id"] != build_public_api_manifest()["manifest_id"]:
        raise ValueError("target-free rehearsal public-API manifest changed")
    if receipt["provider_manifest_id"] != prob4d_provider_manifest(
        provider_revision=revision
    )["manifest_id"]:
        raise ValueError("target-free rehearsal provider manifest changed")
    if receipt["contract_bundle_sha256"] != observation_contract_bundle_manifest()[
        "bundle_sha256"
    ]:
        raise ValueError("target-free rehearsal contract bundle changed")

    root = receipt_path.parent
    positive = cast(Mapping[str, Any], receipt["positive_control"])
    valid_path = root / cast(str, positive["artifact"])
    if file_sha256(valid_path) != positive["artifact_file_sha256"]:
        raise ValueError("positive-control artifact digest changed")
    official = load_observation_belief_export(valid_path)
    independent = verify_observation_belief(valid_path)
    if official.artifact_id != positive["artifact_id"] or (
        independent.artifact_id != positive["artifact_id"]
    ):
        raise ValueError("positive-control artifact identity changed")
    if independent.report_id != positive["independent_report_id"]:
        raise ValueError("independent verification report identity changed")
    report_path = root / cast(str, positive["independent_report"])
    if load_json_object(report_path, name="independent verification report") != (
        independent.to_dict()
    ):
        raise ValueError("retained independent verification report changed")
    _claim_bearing_rejection(valid_path)

    controls = receipt["negative_controls"]
    if type(controls) is not list or tuple(
        item.get("control_id") for item in controls if isinstance(item, Mapping)
    ) != _NEGATIVE_CONTROL_IDS:
        raise ValueError("target-free rehearsal negative-control roster changed")
    for item in controls:
        if not isinstance(item, Mapping):
            raise ValueError("target-free rehearsal negative control is not a mapping")
        control_path = root / cast(str, item["artifact"])
        if file_sha256(control_path) != item["artifact_file_sha256"]:
            raise ValueError("negative-control artifact digest changed")
        replay = _capture_rejection(control_path).to_dict()
        for key, value in replay.items():
            if item.get(key) != value:
                raise ValueError(
                    f"negative control {item.get('control_id')!r} replay changed"
                )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d diagnostic target-free-rehearsal",
        description=(
            "Run or verify the target-free observation-contract positive and "
            "adversarial rehearsal."
        ),
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="run and seal a new rehearsal")
    run_parser.add_argument("output_dir", type=Path)
    run_parser.add_argument("--source-revision", required=True)
    verify_parser = subparsers.add_parser("verify", help="replay a sealed receipt")
    verify_parser.add_argument("receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    if arguments.action == "run":
        receipt = run_target_free_rehearsal(
            arguments.output_dir,
            source_revision=arguments.source_revision,
        )
    else:
        receipt = verify_target_free_rehearsal(arguments.receipt)
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TARGET_FREE_REHEARSAL_CLAIM_BOUNDARY",
    "TARGET_FREE_REHEARSAL_PROTOCOL_ID",
    "TARGET_FREE_REHEARSAL_SCHEMA",
    "TARGET_FREE_REHEARSAL_VERSION",
    "RejectionResult",
    "main",
    "run_target_free_rehearsal",
    "verify_target_free_rehearsal",
]
