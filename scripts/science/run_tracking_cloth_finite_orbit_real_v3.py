#!/usr/bin/env python3
"""Run the header-qualified Tracking Cloth finite-orbit replication v3.

The v2 source seal selected marker labels 1/20/5 before target-header access.
Its bounded support stage found that all 15 Hitting/Tablecloth recordings
support those labels, while all 27 Self-collisions recordings use a different
identity namespace. No target trajectory value was parsed. This script freezes
and evaluates exactly those 15 support-positive recordings without changing the
controlled factor, inference rule, registered criteria, or source selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

V2_FILENAME = "run_tracking_cloth_finite_orbit_real_v2.py"
V2_GIT_BLOB_SHA1 = "921ff4dedb2759f85a5f3cbb821c699da7544baa"
PROTOCOL_ID = "9c0fb1a4191743a5038a2f26e521db1640fd5abfc3cac389e851485b7836a472"
PROTOCOL_SCHEMA = "prob4d.tracking-cloth-finite-orbit-real.v3"
RESULT_SCHEMA = "prob4d.tracking-cloth-finite-orbit-result.v3"
EXPECTED_SOURCE = 24
EXPECTED_TARGET = 15
SELECTED_TRIPLET = ("1", "20", "5")
QUALIFICATION_PROTOCOL_ID = "4652d72f9e8d4c80c69df86b7a48a6f4e307e4131ea3bea04e09deed10db5eb0"
QUALIFICATION_RESULT_ID = "5f130c8643c18346ad55e7d5997deefadc20481d14c8024e641ce811f23119e0"
QUALIFICATION_SOURCE_SEAL_ID = "1bb2236a087f5d309199fadd1cc2ebcc6a6242eaf6ab4f65a0d077ab9545ba52"
QUALIFICATION_SUPPORT_ID = "2d3b385f69bbeace010412771ad988bc51509cfe61e071cdc6ed0f7803938abb"
QUALIFICATION_ARTIFACT_DIGEST = (
    "sha256:c7c61a117c1e3dbcf33997c528fb1703bfae44cbdfb560d176a9880bc04b05e3"
)
EXPECTED_SELECTION = {
    "anchor_distance_q10_mm": 682.8316929700749,
    "probe_radius_median_mm": 343.3444426753403,
    "probe_radius_q10_mm": 326.8171781029235,
    "score": 670.801578339614,
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(value)}\0".encode() + value,
        usedforsecurity=False,
    ).hexdigest()


def _stable_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode()).hexdigest()[:16]


def _load_v2() -> ModuleType:
    source = Path(__file__).with_name(V2_FILENAME)
    if _git_blob_sha1(source.read_bytes()) != V2_GIT_BLOB_SHA1:
        raise RuntimeError("registered finite-orbit v2 implementation changed")
    module_name = "tracking_cloth_finite_orbit_v2_frozen"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load registered finite-orbit v2 implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _expected_qualification() -> dict[str, Any]:
    return {
        "protocol_path": "protocols/tracking-cloth-finite-orbit-real-v2.json",
        "protocol_git_blob_sha1": "641d05f1dcae00a2f36888a7e071be64eaf9cb45",
        "protocol_id": QUALIFICATION_PROTOCOL_ID,
        "run_id": 33532387635,
        "head_sha": "164a95c93cba679c911eeced22b48516a90720f2",
        "artifact_id": 9810268001,
        "artifact_name": "tracking-cloth-finite-orbit-real-v2-result-33532387635",
        "artifact_digest": QUALIFICATION_ARTIFACT_DIGEST,
        "result_id": QUALIFICATION_RESULT_ID,
        "source_seal_id": QUALIFICATION_SOURCE_SEAL_ID,
        "support_id": QUALIFICATION_SUPPORT_ID,
        "status": "target-marker-support-negative",
        "required_marker_labels": list(SELECTED_TRIPLET),
        "supported_target_group_count": EXPECTED_TARGET,
        "unsupported_target_group_count": 27,
        "target_trajectory_values_parsed": False,
        "unsupported_groups_replaced": False,
    }


def _load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("v3 protocol must be one JSON object")
    unsigned = dict(value)
    supplied = unsigned.pop("protocol_id", None)
    if supplied != PROTOCOL_ID or _sha256(unsigned) != supplied:
        raise ValueError("v3 protocol identity changed")
    if value.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("v3 protocol schema changed")
    dataset = value.get("dataset", {})
    if dataset.get("expected_source_files") != EXPECTED_SOURCE:
        raise ValueError("source roster size changed")
    if dataset.get("expected_target_files") != EXPECTED_TARGET:
        raise ValueError("target roster size changed")
    paths = dataset.get("target_relative_paths")
    group_ids = dataset.get("target_group_ids")
    if not isinstance(paths, list) or len(paths) != EXPECTED_TARGET:
        raise ValueError("target path roster changed")
    if not isinstance(group_ids, list):
        raise ValueError("target group IDs are missing")
    if group_ids != [_stable_id(relative_path) for relative_path in paths]:
        raise ValueError("target path/group identity binding changed")
    if value.get("support_qualification") != _expected_qualification():
        raise ValueError("support qualification binding changed")
    marker_support = value.get("marker_support", {})
    if marker_support.get("selected_marker_triplet") != list(SELECTED_TRIPLET):
        raise ValueError("source-selected marker triplet changed")
    settings = value.get("scientific_settings", {})
    if settings.get("source_frames_per_recording") != 12:
        raise ValueError("source sampling changed")
    if settings.get("target_frames_per_recording") != 128:
        raise ValueError("target sampling changed")
    if settings.get("minimum_target_cases") != 1000:
        raise ValueError("minimum target case count changed")
    order = value.get("information_order", {})
    if order.get("target_trajectory_values_used_for_support_qualification") is not False:
        raise ValueError("support qualification accessed target outcomes")
    if order.get("target_trajectory_values_opened_before_v3_freeze") is not False:
        raise ValueError("v3 was not frozen before target trajectory access")
    if order.get("target_side_retuning_allowed") is not False:
        raise ValueError("target-side retuning was enabled")
    if order.get("unsupported_target_replacement_allowed") is not False:
        raise ValueError("unsupported target replacement was enabled")
    return value


def _configure_v2(v2: ModuleType, protocol: dict[str, Any]) -> None:
    target_paths = set(protocol["dataset"]["target_relative_paths"])
    original_subset = v2._subset
    original_effective_parent = v2._effective_parent
    original_load_base = v2._load_base
    original_summary = v2._summary

    def subset(
        recordings: list[Any],
        *,
        source: bool,
    ) -> tuple[list[Any], list[Any]]:
        if source:
            return original_subset(recordings, source=True)
        eligible = [row for row in recordings if row.relative_path in target_paths]
        excluded = [row for row in recordings if row.relative_path not in target_paths]
        actual = {row.relative_path for row in eligible}
        if actual != target_paths or len(eligible) != EXPECTED_TARGET:
            raise ValueError("support-qualified target roster is incomplete or changed")
        return eligible, excluded

    def effective_parent(parent: dict[str, Any]) -> dict[str, Any]:
        effective = original_effective_parent(parent)
        effective["geometry"]["target_frames_per_recording"] = 128
        if effective["geometry"]["minimum_target_cases"] != 1000:
            raise ValueError("parent minimum target case criterion changed")
        return effective

    def load_base() -> ModuleType:
        base = original_load_base()
        original_select = base._select_marker_triplet

        def select_marker_triplet(
            *args: Any,
            **kwargs: Any,
        ) -> tuple[tuple[str, str, str], dict[str, float]]:
            triplet, details = original_select(*args, **kwargs)
            if triplet != SELECTED_TRIPLET:
                raise ValueError(f"source-selected triplet changed: {triplet}")
            for name, expected in EXPECTED_SELECTION.items():
                measured = float(details[name])
                if abs(measured - expected) > 1e-9 * max(1.0, abs(expected)):
                    raise ValueError(f"source selection statistic changed: {name}")
            return triplet, details

        base._select_marker_triplet = select_marker_triplet
        return base

    def summary(
        result: dict[str, Any],
        base: ModuleType,
        triplet: tuple[str, str, str],
    ) -> str:
        text = original_summary(result, base, triplet)
        text = text.replace(
            "# Tracking Cloth finite-orbit replication v2",
            "# Tracking Cloth support-qualified finite-orbit replication v3",
            1,
        )
        qualification = (
            "The exact 15-recording target roster was frozen from v2 marker-label "
            "headers only. No target trajectory value was used to select a recording, "
            "marker, threshold, or method. The 27 incompatible self-collision "
            "recordings remain a separately reported support-negative cohort.\n"
        )
        return text.rstrip() + "\n\n## Header-only support qualification\n\n" + qualification

    v2.PROTOCOL_ID = PROTOCOL_ID
    v2.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    v2.RESULT_SCHEMA = RESULT_SCHEMA
    v2.EXPECTED_SOURCE = EXPECTED_SOURCE
    v2.EXPECTED_TARGET = EXPECTED_TARGET
    v2._load_protocol = _load_protocol
    v2._subset = subset
    v2._effective_parent = effective_parent
    v2._load_base = load_base
    v2._summary = summary


def run(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol).resolve()
    protocol = _load_protocol(protocol_path)
    v2 = _load_v2()
    _configure_v2(v2, protocol)
    status = int(v2.run(args))
    output = Path(args.output_dir).resolve()
    result_path = output / "result.json"
    if not result_path.is_file():
        raise RuntimeError("v3 replication emitted no terminal result")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["schema"] = RESULT_SCHEMA
    result["support_qualification"] = protocol["support_qualification"]
    result["frozen_target_group_ids"] = protocol["dataset"]["target_group_ids"]
    result["scientific_settings"] = protocol["scientific_settings"]
    result.pop("result_id", None)
    result["result_id"] = _sha256(result)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
