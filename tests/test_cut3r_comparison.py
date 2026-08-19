from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from prob4d.cut3r_comparison import (
    CUT3R_COMPARISON_GROUP_UNIT,
    CUT3R_COMPARISON_SCHEMA,
    CUT3R_PROVIDER_ENDPOINTS,
    build_cut3r_comparison_lock,
    cut3r_comparison_summary,
    load_cut3r_comparison_lock,
    main,
    validate_cut3r_comparison_lock,
    write_cut3r_comparison_lock,
)


def _case(case_id: str, digest_character: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "input_video_sha256": digest_character * 64,
        "input_video_byte_count": 1000,
        "frame_start": 0,
        "frame_stop_exclusive": 100,
        "evaluation_frame_start": 20,
        "evaluation_frame_stop_exclusive": 90,
    }


def _specification() -> dict[str, Any]:
    return {
        "protocol_name": "cut3r-source-comparison-v1",
        "provider_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "prob4d_revision": "c" * 40,
        "prob4d_distribution_sha256": "d" * 64,
        "window_size": 25,
        "overlap": 8,
        "confidence_threshold": 1.5,
        "storage_dtype": "float32",
        "random_seeds": [7, 11],
        "groups": [
            {"group_id": "object-a", "cases": [_case("case-a", "1")]},
            {"group_id": "object-b", "cases": [_case("case-b", "2")]},
            {"group_id": "object-c", "cases": [_case("case-c", "3")]},
        ],
        "group_roles": {
            "development": ["object-a"],
            "calibration": ["object-b"],
            "source_evaluation": ["object-c"],
        },
        "include_revisit_diagnostic": True,
    }


def test_lock_is_deterministic_group_aware_and_source_only() -> None:
    first = build_cut3r_comparison_lock(_specification())
    second = build_cut3r_comparison_lock(_specification())

    assert first == second
    assert first["schema"] == CUT3R_COMPARISON_SCHEMA
    assert first["group_unit"] == CUT3R_COMPARISON_GROUP_UNIT
    assert first["source_access"] == "source-only"
    assert first["target_access"] == "forbidden"
    assert first["windowing"]["stride"] == 17
    assert first["provider_endpoints"] == list(CUT3R_PROVIDER_ENDPOINTS)
    assert [arm["arm_id"] for arm in first["arms"]] == [
        "native-continuous",
        "restarted-newest",
        "restarted-prob4d-fused",
        "revisit-diagnostic",
    ]
    assert first["arms"][-1]["causal"] is False
    assert first["arms"][-1]["claim_eligible"] is False
    assert validate_cut3r_comparison_lock(first) == first

    summary = cut3r_comparison_summary(first)
    assert summary["independent_group_count"] == 3
    assert summary["case_count"] == 3
    assert summary["group_role_counts"] == {
        "development": 1,
        "calibration": 1,
        "source_evaluation": 1,
    }
    assert summary["claim_eligible_contrasts"] == [
        "prob4d-fusion-value",
        "provider-recurrence-value",
    ]


def test_revisit_arm_can_be_disabled_but_never_promoted() -> None:
    specification = _specification()
    specification["include_revisit_diagnostic"] = False
    lock = build_cut3r_comparison_lock(specification)

    assert lock["arms"][-1]["enabled"] is False
    assert lock["registered_contrasts"][-1]["enabled"] is False

    tampered = deepcopy(lock)
    tampered["arms"][-1]["claim_eligible"] = True
    with pytest.raises(ValueError, match="arms changed"):
        validate_cut3r_comparison_lock(tampered)


def test_group_roles_must_be_disjoint_and_complete() -> None:
    specification = _specification()
    specification["group_roles"]["calibration"] = ["object-a"]

    with pytest.raises(ValueError, match="must be disjoint"):
        build_cut3r_comparison_lock(specification)

    specification = _specification()
    specification["group_roles"]["source_evaluation"] = ["unknown"]
    with pytest.raises(ValueError, match="partition every group"):
        build_cut3r_comparison_lock(specification)


def test_cases_cannot_cross_groups_or_escape_source_interval() -> None:
    specification = _specification()
    specification["groups"][1]["cases"][0]["case_id"] = "case-a"
    with pytest.raises(ValueError, match="appears in multiple groups"):
        build_cut3r_comparison_lock(specification)

    specification = _specification()
    specification["groups"][0]["cases"][0]["evaluation_frame_stop_exclusive"] = 101
    with pytest.raises(ValueError, match="must lie inside"):
        build_cut3r_comparison_lock(specification)


def test_lock_round_trip_is_no_clobber(tmp_path: Path) -> None:
    destination = tmp_path / "cut3r-comparison-lock.json"
    lock = build_cut3r_comparison_lock(_specification())

    assert write_cut3r_comparison_lock(destination, lock) == lock
    assert load_cut3r_comparison_lock(destination) == lock
    assert write_cut3r_comparison_lock(destination, lock) == lock

    changed_specification = _specification()
    changed_specification["protocol_name"] = "different-protocol"
    different = build_cut3r_comparison_lock(changed_specification)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_cut3r_comparison_lock(destination, different)


def test_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_cut3r_comparison_lock(path)


def test_cli_build_verify_and_summarize(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    specification = tmp_path / "specification.json"
    lock_path = tmp_path / "lock.json"
    specification.write_text(
        json.dumps(_specification(), sort_keys=True),
        encoding="utf-8",
    )

    assert main(["build", str(specification), "--output", str(lock_path)]) == 0
    lock_id = capsys.readouterr().out.strip()
    assert lock_id == load_cut3r_comparison_lock(lock_path)["lock_id"]

    assert main(["verify", str(lock_path)]) == 0
    assert capsys.readouterr().out.strip() == lock_id

    assert main(["summarize", str(lock_path), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["independent_group_count"] == 3
    assert summary["target_access"] == "forbidden"
