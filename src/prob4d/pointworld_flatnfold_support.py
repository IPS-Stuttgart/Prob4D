"""Outcome-blind PointWorld--Flat'n'Fold support qualification.

The inventory in this module is evaluated before PointWorld prediction payloads,
provider residuals, or target outcomes are opened.  It converts an explicit
three-camera garment/demo roster into the existing replayable
``ProviderSupportFeasibilityV1`` contract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .provider_support_feasibility import (
    ProviderSupportFeasibilityRequestV1,
    ProviderSupportStreamV1,
    evaluate_provider_support_feasibility,
    write_provider_support_feasibility,
    write_provider_support_feasibility_request,
)

POINTWORLD_FLATNFOLD_INVENTORY_SCHEMA = (
    "prob4d.pointworld-flatnfold-support-inventory"
)
POINTWORLD_FLATNFOLD_INVENTORY_VERSION = 1
POINTWORLD_REPOSITORY = "NVlabs/PointWorld"
FLATNFOLD_REPOSITORY = "lipeng-zhuang521/flat-n-fold"
PROB4D_REPOSITORY = "IPS-Stuttgart/Prob4D"

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "protocol_id",
        "source_revision",
        "pointworld_revision",
        "checkpoint_sha256",
        "model_set_id",
        "loader_id",
        "cohort_binding_id",
        "promotion_lock_id",
        "flatnfold_revision",
        "dataset_bytes_id",
        "coordinate_semantics",
        "required_camera_ids",
        "admission_rule",
        "minimum_supported_fraction",
        "permitted_technical_exclusion_codes",
        "maximum_technical_exclusions",
        "prediction_payloads_opened",
        "residuals_used",
        "target_outcomes_used",
        "streams",
        "metadata",
    }
)
_STREAM_FIELDS = frozenset(
    {
        "garment_id",
        "demonstration_id",
        "camera_id",
        "causal_frame_start",
        "causal_frame_stop_exclusive",
        "required_frame_ids",
        "available_frame_ids",
        "geometry_supported_frame_ids",
        "minimum_geometry_support_fraction",
        "intrinsics_id",
        "extrinsics_id",
        "metric_anchor_id",
        "action_sequence_id",
        "technical_failure_code",
        "metadata",
    }
)
_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: Path) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"inventory contains duplicate key {key!r}")
            result[key] = item
        return result

    def reject_constant(token: str) -> Any:
        raise ValueError(f"inventory contains non-finite number {token!r}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except OSError as error:
        raise ValueError("cannot read PointWorld--Flat'n'Fold inventory") from error
    except json.JSONDecodeError as error:
        raise ValueError("inventory must contain valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("inventory must contain one JSON object")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be one JSON object")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    fields: frozenset[str],
    *,
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(fields - actual)
    extra = sorted(actual - fields)
    if missing or extra:
        raise ValueError(f"{name} fields changed; missing={missing}, extra={extra}")


def _string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name=name)


def _revision(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if _REVISION.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase 40- or 64-hex revision")
    return result


def _digest(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _real(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be one finite real number")
    result = float(value)
    if not (result == result and abs(result) != float("inf")):
        raise ValueError(f"{name} must be one finite real number")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be Boolean")
    return value


def _list(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be one JSON array")
    return value


def _string_tuple(value: object, *, name: str, nonempty: bool) -> tuple[str, ...]:
    items = tuple(
        _string(item, name=f"{name}[{index}]")
        for index, item in enumerate(_list(value, name=name))
    )
    if nonempty and not items:
        raise ValueError(f"{name} must not be empty")
    if items != tuple(sorted(items)) or len(items) != len(set(items)):
        raise ValueError(f"{name} must be sorted and unique")
    return items


def _integer_tuple(value: object, *, name: str, nonempty: bool) -> tuple[int, ...]:
    items = tuple(
        _integer(item, name=f"{name}[{index}]")
        for index, item in enumerate(_list(value, name=name))
    )
    if nonempty and not items:
        raise ValueError(f"{name} must not be empty")
    if items != tuple(sorted(items)) or len(items) != len(set(items)):
        raise ValueError(f"{name} must be sorted and unique")
    return items


def _finite_json(value: object, *, name: str) -> Mapping[str, Any]:
    mapping = _mapping(value, name=name)
    try:
        json.dumps(mapping, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite JSON") from error
    return mapping


def build_pointworld_flatnfold_support_request(
    inventory_path: str | Path,
) -> ProviderSupportFeasibilityRequestV1:
    """Build a support request from a strict unopened Flat'n'Fold inventory."""

    inventory = _load_json(Path(inventory_path))
    _exact_fields(inventory, _TOP_LEVEL_FIELDS, name="support inventory")
    if inventory["schema_name"] != POINTWORLD_FLATNFOLD_INVENTORY_SCHEMA:
        raise ValueError("unsupported PointWorld--Flat'n'Fold inventory schema")
    if inventory["schema_version"] != POINTWORLD_FLATNFOLD_INVENTORY_VERSION:
        raise ValueError("unsupported PointWorld--Flat'n'Fold inventory version")

    prediction_payloads_opened = _boolean(
        inventory["prediction_payloads_opened"],
        name="prediction_payloads_opened",
    )
    residuals_used = _boolean(inventory["residuals_used"], name="residuals_used")
    target_outcomes_used = _boolean(
        inventory["target_outcomes_used"],
        name="target_outcomes_used",
    )
    if prediction_payloads_opened or residuals_used or target_outcomes_used:
        raise ValueError(
            "support inventory must be frozen before payload, residual, or target access"
        )

    required_cameras = _string_tuple(
        inventory["required_camera_ids"],
        name="required_camera_ids",
        nonempty=True,
    )
    if len(required_cameras) != 3:
        raise ValueError("Flat'n'Fold qualification requires exactly three cameras")
    camera_set = set(required_cameras)

    stream_records = _list(inventory["streams"], name="streams")
    if not stream_records:
        raise ValueError("support inventory requires at least one stream")

    support_streams: list[ProviderSupportStreamV1] = []
    demo_cameras: dict[tuple[str, str], set[str]] = defaultdict(set)
    demo_actions: dict[tuple[str, str], set[str]] = defaultdict(set)
    demo_intervals: dict[
        tuple[str, str],
        set[tuple[int, int, tuple[int, ...]]],
    ] = defaultdict(set)
    seen_streams: set[tuple[str, str, str]] = set()

    for index, raw_stream in enumerate(stream_records):
        stream = _mapping(raw_stream, name=f"stream {index}")
        _exact_fields(stream, _STREAM_FIELDS, name=f"stream {index}")
        garment_id = _string(stream["garment_id"], name=f"stream {index} garment_id")
        demonstration_id = _string(
            stream["demonstration_id"],
            name=f"stream {index} demonstration_id",
        )
        camera_id = _string(stream["camera_id"], name=f"stream {index} camera_id")
        if camera_id not in camera_set:
            raise ValueError(f"stream {index} uses an undeclared camera")
        stream_key = (garment_id, demonstration_id, camera_id)
        if stream_key in seen_streams:
            raise ValueError("support inventory contains a duplicate camera stream")
        seen_streams.add(stream_key)

        start = _integer(
            stream["causal_frame_start"],
            name=f"stream {index} causal_frame_start",
        )
        stop = _integer(
            stream["causal_frame_stop_exclusive"],
            name=f"stream {index} causal_frame_stop_exclusive",
            minimum=1,
        )
        if stop <= start:
            raise ValueError("causal frame stop must exceed start")
        required = _integer_tuple(
            stream["required_frame_ids"],
            name=f"stream {index} required_frame_ids",
            nonempty=True,
        )
        available = _integer_tuple(
            stream["available_frame_ids"],
            name=f"stream {index} available_frame_ids",
            nonempty=False,
        )
        geometry = _integer_tuple(
            stream["geometry_supported_frame_ids"],
            name=f"stream {index} geometry_supported_frame_ids",
            nonempty=False,
        )
        action_sequence_id = _digest(
            stream["action_sequence_id"],
            name=f"stream {index} action_sequence_id",
        )
        demo_key = (garment_id, demonstration_id)
        demo_cameras[demo_key].add(camera_id)
        demo_actions[demo_key].add(action_sequence_id)
        demo_intervals[demo_key].add((start, stop, required))

        support_streams.append(
            ProviderSupportStreamV1(
                group_id=garment_id,
                stream_id=f"{demonstration_id}:{camera_id}",
                causal_frame_start=start,
                causal_frame_stop_exclusive=stop,
                required_frame_ids=required,
                available_frame_ids=available,
                geometry_supported_frame_ids=geometry,
                minimum_geometry_support_fraction=_real(
                    stream["minimum_geometry_support_fraction"],
                    name=(
                        f"stream {index} minimum_geometry_support_fraction"
                    ),
                ),
                intrinsics_required=True,
                intrinsics_id=_digest(
                    stream["intrinsics_id"],
                    name=f"stream {index} intrinsics_id",
                ),
                extrinsics_required=True,
                extrinsics_id=_digest(
                    stream["extrinsics_id"],
                    name=f"stream {index} extrinsics_id",
                ),
                metric_anchor_required=True,
                metric_anchor_id=_digest(
                    stream["metric_anchor_id"],
                    name=f"stream {index} metric_anchor_id",
                ),
                technical_failure_code=_optional_string(
                    stream["technical_failure_code"],
                    name=f"stream {index} technical_failure_code",
                ),
                metadata={
                    "garment_id": garment_id,
                    "demonstration_id": demonstration_id,
                    "camera_id": camera_id,
                    "action_sequence_id": action_sequence_id,
                    "dataset_bytes_id": _digest(
                        inventory["dataset_bytes_id"],
                        name="dataset_bytes_id",
                    ),
                    "source_metadata": _finite_json(
                        stream["metadata"],
                        name=f"stream {index} metadata",
                    ),
                },
            )
        )

    for demo_key in sorted(demo_cameras):
        if demo_cameras[demo_key] != camera_set:
            raise ValueError(
                "every Flat'n'Fold demonstration must retain all required cameras"
            )
        if len(demo_actions[demo_key]) != 1:
            raise ValueError(
                "all cameras of one demonstration must bind one action sequence"
            )
        if len(demo_intervals[demo_key]) != 1:
            raise ValueError(
                "all cameras of one demonstration must use one causal frame schedule"
            )

    support_streams.sort(key=lambda item: item.key)
    admission_rule = _string(inventory["admission_rule"], name="admission_rule")
    if admission_rule not in {"all-streams", "minimum-stream-fraction"}:
        raise ValueError("unsupported admission_rule")
    minimum_supported = _real(
        inventory["minimum_supported_fraction"],
        name="minimum_supported_fraction",
    )
    if not 0.0 <= minimum_supported <= 1.0:
        raise ValueError("minimum_supported_fraction must lie in [0, 1]")

    dataset_bytes_id = _digest(
        inventory["dataset_bytes_id"],
        name="dataset_bytes_id",
    )
    checkpoint_sha256 = _digest(
        inventory["checkpoint_sha256"],
        name="checkpoint_sha256",
    )
    top_metadata = _finite_json(inventory["metadata"], name="metadata")
    return ProviderSupportFeasibilityRequestV1(
        protocol_id=_string(inventory["protocol_id"], name="protocol_id"),
        source_repository=PROB4D_REPOSITORY,
        source_revision=_revision(
            inventory["source_revision"],
            name="source_revision",
        ),
        provider_family="pointworld",
        provider_repository=POINTWORLD_REPOSITORY,
        provider_revision=_revision(
            inventory["pointworld_revision"],
            name="pointworld_revision",
        ),
        model_set_id=_digest(inventory["model_set_id"], name="model_set_id"),
        loader_id=_digest(inventory["loader_id"], name="loader_id"),
        cohort_binding_id=_digest(
            inventory["cohort_binding_id"],
            name="cohort_binding_id",
        ),
        promotion_lock_id=_digest(
            inventory["promotion_lock_id"],
            name="promotion_lock_id",
        ),
        coordinate_semantics=_string(
            inventory["coordinate_semantics"],
            name="coordinate_semantics",
        ),
        admission_rule=admission_rule,
        minimum_supported_fraction=minimum_supported,
        permitted_technical_exclusion_codes=_string_tuple(
            inventory["permitted_technical_exclusion_codes"],
            name="permitted_technical_exclusion_codes",
            nonempty=False,
        ),
        maximum_technical_exclusions=_integer(
            inventory["maximum_technical_exclusions"],
            name="maximum_technical_exclusions",
        ),
        prediction_payloads_opened=False,
        residuals_used=False,
        target_outcomes_used=False,
        streams=tuple(support_streams),
        metadata={
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_repository": FLATNFOLD_REPOSITORY,
            "dataset_revision": _revision(
                inventory["flatnfold_revision"],
                name="flatnfold_revision",
            ),
            "dataset_bytes_id": dataset_bytes_id,
            "required_camera_ids": list(required_cameras),
            "statistical_unit": "complete-physical-garment",
            "demonstration_count": len(demo_cameras),
            "garment_count": len({key[0] for key in demo_cameras}),
            "inventory_metadata": top_metadata,
        },
    )


def scaffold_pointworld_flatnfold_support_inventory(path: str | Path) -> None:
    """Write an intentionally incomplete no-clobber source inventory scaffold."""

    target = Path(path)
    if target.is_symlink():
        raise ValueError("support inventory output path is a symbolic link")
    target.parent.mkdir(parents=True, exist_ok=True)
    camera_ids = ["camera-0", "camera-1", "camera-2"]
    streams = []
    for index, camera_id in enumerate(camera_ids):
        streams.append(
            {
                "garment_id": "REPLACE_WITH_COMPLETE_GARMENT_ID",
                "demonstration_id": "REPLACE_WITH_DEMONSTRATION_ID",
                "camera_id": camera_id,
                "causal_frame_start": 0,
                "causal_frame_stop_exclusive": 10,
                "required_frame_ids": list(range(10)),
                "available_frame_ids": [],
                "geometry_supported_frame_ids": [],
                "minimum_geometry_support_fraction": 1.0,
                "intrinsics_id": f"REPLACE_WITH_CAMERA_{index}_INTRINSICS_SHA256",
                "extrinsics_id": f"REPLACE_WITH_CAMERA_{index}_EXTRINSICS_SHA256",
                "metric_anchor_id": "REPLACE_WITH_METRIC_ANCHOR_SHA256",
                "action_sequence_id": "REPLACE_WITH_ACTION_SEQUENCE_SHA256",
                "technical_failure_code": None,
                "metadata": {},
            }
        )
    record = {
        "schema_name": POINTWORLD_FLATNFOLD_INVENTORY_SCHEMA,
        "schema_version": POINTWORLD_FLATNFOLD_INVENTORY_VERSION,
        "protocol_id": "pointworld-flatnfold-source-qualification-v1",
        "source_revision": "REPLACE_WITH_PROB4D_REVISION",
        "pointworld_revision": "REPLACE_WITH_POINTWORLD_REVISION",
        "checkpoint_sha256": "REPLACE_WITH_POINTWORLD_CHECKPOINT_SHA256",
        "model_set_id": "REPLACE_WITH_MODEL_SET_SHA256",
        "loader_id": "REPLACE_WITH_LOADER_SHA256",
        "cohort_binding_id": "REPLACE_WITH_COHORT_BINDING_SHA256",
        "promotion_lock_id": "REPLACE_WITH_PROMOTION_LOCK_SHA256",
        "flatnfold_revision": "REPLACE_WITH_FLATNFOLD_REVISION",
        "dataset_bytes_id": "REPLACE_WITH_DATASET_BYTES_SHA256",
        "coordinate_semantics": "metric-baxter-base",
        "required_camera_ids": camera_ids,
        "admission_rule": "all-streams",
        "minimum_supported_fraction": 1.0,
        "permitted_technical_exclusion_codes": [],
        "maximum_technical_exclusions": 0,
        "prediction_payloads_opened": False,
        "residuals_used": False,
        "target_outcomes_used": False,
        "streams": streams,
        "metadata": {
            "ready_for_evaluation": False,
            "statistical_unit": "complete-physical-garment",
        },
    }
    try:
        with target.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="build PointWorld--Flat'n'Fold support feasibility artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    scaffold = subparsers.add_parser("scaffold")
    scaffold.add_argument("inventory")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("inventory")
    evaluate.add_argument("request")
    evaluate.add_argument("result")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    if arguments.command == "scaffold":
        scaffold_pointworld_flatnfold_support_inventory(arguments.inventory)
        print(
            json.dumps(
                {
                    "inventory": str(arguments.inventory),
                    "ready_for_evaluation": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    request = build_pointworld_flatnfold_support_request(arguments.inventory)
    write_provider_support_feasibility_request(arguments.request, request)
    result = evaluate_provider_support_feasibility(request)
    write_provider_support_feasibility(arguments.result, result)
    print(
        json.dumps(
            {
                "request_id": request.request_id,
                "provider_support_feasibility_id": (
                    result.provider_support_feasibility_id
                ),
                "support_feasible": result.support_feasible,
                "stream_count": result.stream_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.support_feasible else 2


__all__ = [
    "FLATNFOLD_REPOSITORY",
    "POINTWORLD_FLATNFOLD_INVENTORY_SCHEMA",
    "POINTWORLD_FLATNFOLD_INVENTORY_VERSION",
    "POINTWORLD_REPOSITORY",
    "PROB4D_REPOSITORY",
    "build_pointworld_flatnfold_support_request",
    "main",
    "scaffold_pointworld_flatnfold_support_inventory",
]


if __name__ == "__main__":
    raise SystemExit(main())
