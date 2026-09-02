"""Content-addressed protocol and result I/O for the Tracking Cloth study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA = "prob4d.tracking-cloth-approximate-orbit-tube.v1"
CALIBRATION_SCHEMA = "prob4d.tracking-cloth-approximate-orbit-calibration.v1"
RESULT_SCHEMA = "prob4d.tracking-cloth-approximate-orbit-result.v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("protocol must be one JSON object")
    unsigned = dict(value)
    supplied = unsigned.pop("protocol_id", None)
    if type(supplied) is not str or _sha256(unsigned) != supplied:
        raise ValueError("protocol identity changed")
    if value.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("protocol schema changed")
    dataset = value["dataset"]
    if dataset["expected_csv_files"] != 120 or dataset["expected_cohort_files"] != 27:
        raise ValueError("dataset roster changed")
    prior_ids = dataset["prior_header_only_group_ids"]
    if sorted(prior_ids) != sorted(set(prior_ids)):
        raise ValueError("prior header-only group IDs must be unique")
    if len(dataset["prior_header_only_group_ids"]) != 27:
        raise ValueError("expected exactly 27 prior header-only groups")
    split = value["split"]
    if split["calibration_per_material"] != 6 or split["target_per_material"] != 3:
        raise ValueError("within-material split changed")
    if split["expected_calibration_groups"] != 18 or split["expected_target_groups"] != 9:
        raise ValueError("split group counts changed")
    if split["target_trajectory_values_opened_before_protocol_freeze"] is not False:
        raise ValueError("target trajectory values were already opened")
    calibration = value["calibration"]
    if calibration["requested_miscoverage"] != 0.10:
        raise ValueError("miscoverage changed")
    if calibration["independent_unit"] != "complete-recording":
        raise ValueError("independent unit changed")
    if value["information_order"]["target_values_may_be_read_during_calibration"] is not False:
        raise ValueError("calibration may not read target values")
    if value["information_order"]["target_side_retuning_allowed"] is not False:
        raise ValueError("target-side retuning was enabled")
    return value


def _load_calibration(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict or value.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unexpected calibration schema")
    unsigned = dict(value)
    supplied = unsigned.pop("calibration_id", None)
    if type(supplied) is not str or _sha256(unsigned) != supplied:
        raise ValueError("calibration identity changed")
    if value.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("calibration protocol does not match")
    information_order = value.get("information_order", {})
    if information_order.get("target_trajectory_values_parsed") is not False:
        raise ValueError("calibration accessed target trajectory values")
    return value
