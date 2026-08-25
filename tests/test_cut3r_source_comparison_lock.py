from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from prob4d.cut3r_source_comparison_plan import validate_execution_plan

ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "protocols"
    / "execution_requests"
    / "cut3r_deform360_source_comparison_execution_v1.json"
)
PLAN_FILE_SHA256 = "3ea10036bdd3d0b516d06ef210204cb7644e8b06cc8c2a3391c4742f3940ef7f"
PLAN_ID = "0dbb6b3a46e2c895259fd5f4a4691c1d6d3c43b0e71774171bbfb3a20239953c"
IMPLEMENTATION_REVISION = "8d0310e93269c0489b53564fd8077763829c4371"
AMENDED_PLAN = (
    ROOT
    / "protocols"
    / "execution_requests"
    / "cut3r_deform360_source_comparison_execution_v1_1.json"
)
AMENDED_PLAN_FILE_SHA256 = "d4eceb6a44f154901227f1e2ac0e832874869179e55f74ce18f10d6a352d6b00"
AMENDED_PLAN_ID = "ab460acf8ba85d8e5470126e6e9e2fc445d16ad506612b10f1a926a614c60f98"
AMENDED_IMPLEMENTATION_REVISION = "83ce1c546d4c7d0ebca740334a8ad969666a1d0c"
AMENDED_SMOKE_CASE_SHA256 = "8ddd8b05edcd78515fc7c5647e2736060630e0932a59702495b2853a68f02fa7"
SMOKE_RESULT = ROOT / "evidence/cut3r-source-comparison-smoke-v1/summary.json"
SMOKE_RESULT_FILE_SHA256 = "550ce0c8858c4730548d08af50589c820ed3023cfbbae3050eea0369bcd7845f"
SMOKE_RESULT_ID = "ef4e5bf187570e918df1d7d14434b4ae55f983c347104b9c6f7ad52b42f7a7bf"


def _git_blob(revision: str, relative: str) -> bytes:
    return subprocess.run(
        ("git", "-C", str(ROOT), "show", f"{revision}:{relative}"),
        check=True,
        capture_output=True,
    ).stdout


def _is_shallow_checkout() -> bool:
    result = subprocess.run(
        ("git", "-C", str(ROOT), "rev-parse", "--is-shallow-repository"),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() == "true"


def _reviewed_blob(revision: str, relative: str, expected_sha256: str) -> bytes:
    try:
        return _git_blob(revision, relative)
    except subprocess.CalledProcessError:
        # The ordinary test matrix deliberately uses a depth-one pull-request
        # checkout, so the reviewed implementation commit may not be present as
        # a local Git object. In that specific case, retain content-level
        # verification against the checked-out merge candidate. A full-history
        # checkout must resolve the pinned revision and may not use this path.
        if not _is_shallow_checkout():
            raise
        current = (ROOT / relative).read_bytes()
        actual_sha256 = hashlib.sha256(current).hexdigest()
        if actual_sha256 != expected_sha256:
            pytest.skip("historical implementation is absent from the shallow checkout")
        return current


def test_checked_in_execution_plan_is_exact_and_target_closed() -> None:
    payload = PLAN.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PLAN_FILE_SHA256
    plan = validate_execution_plan(json.loads(payload))

    assert plan["plan_id"] == PLAN_ID
    assert plan["implementation"]["revision"] == IMPLEMENTATION_REVISION
    assert plan["execution"]["case_count"] == 40
    assert plan["execution"]["group_count"] == 10
    assert plan["method"]["random_seeds"] == [7, 11, 19]
    assert plan["method"]["window_schedule"] == [
        {"start": 0, "stop": 25, "window_id": "window-000000-000025"},
        {"start": 17, "stop": 42, "window_id": "window-000017-000042"},
        {"start": 33, "stop": 58, "window_id": "window-000033-000058"},
    ]
    boundary = plan["information_boundary"]
    assert boundary["source_rgb_decode_authorized"] is True
    assert boundary["source_outcomes_authorized"] is False
    assert boundary["target_payloads_opened"] is False
    assert boundary["target_outcomes_opened"] is False
    assert boundary["bayesian_phystwin_executed"] is False
    assert boundary["causal4d_executed"] is False


def test_execution_plan_binds_reviewed_implementation_blobs() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    source_hashes = plan["implementation"]["source_file_sha256"]

    for relative, expected in source_hashes.items():
        blob = _reviewed_blob(IMPLEMENTATION_REVISION, relative, expected)
        assert hashlib.sha256(blob).hexdigest() == expected


def test_smoke_result_is_exact_and_pre_science() -> None:
    payload = SMOKE_RESULT.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == SMOKE_RESULT_FILE_SHA256
    result = json.loads(payload)
    artifact_id = result.pop("artifact_id")
    canonical = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert artifact_id == SMOKE_RESULT_ID == hashlib.sha256(canonical).hexdigest()
    assert result["decision"] == "pre-science-technical-failure-no-retry"
    assert result["attempt_count"] == 1
    assert result["retry_performed"] is False
    assert result["output_file_count_after_failure"] == 0
    assert not any(result["information_boundary"].values())


def test_amended_plan_is_exact_distinct_and_method_preserving() -> None:
    payload = AMENDED_PLAN.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == AMENDED_PLAN_FILE_SHA256
    amended = validate_execution_plan(json.loads(payload))
    parent = validate_execution_plan(json.loads(PLAN.read_text(encoding="utf-8")))

    assert amended["plan_id"] == AMENDED_PLAN_ID
    assert amended["schema_version"] == 2
    assert amended["implementation"]["revision"] == AMENDED_IMPLEMENTATION_REVISION
    assert amended["method"] == parent["method"]
    assert amended["cases"] == parent["cases"]
    assert amended["provider"]["revision"] == parent["provider"]["revision"]
    assert amended["provider"]["checkpoint_sha256"] == parent["provider"][
        "checkpoint_sha256"
    ]
    policy = amended["execution"]["smoke_policy"]
    assert policy["registered_case_id_sha256"] == AMENDED_SMOKE_CASE_SHA256
    assert policy["registered_case_id_sha256"] != amended["amendment"][
        "prior_case_id_sha256"
    ]
    assert policy["attempt_limit"] == 1
    assert policy["source_shards_require_ordinary_success_custody"] is True


def test_amended_plan_binds_reviewed_implementation_blobs() -> None:
    amended = json.loads(AMENDED_PLAN.read_text(encoding="utf-8"))
    for relative, expected in amended["implementation"]["source_file_sha256"].items():
        blob = _reviewed_blob(AMENDED_IMPLEMENTATION_REVISION, relative, expected)
        assert hashlib.sha256(blob).hexdigest() == expected
