from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

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


def _git_blob(revision: str, relative: str) -> bytes:
    return subprocess.run(
        ("git", "-C", str(ROOT), "show", f"{revision}:{relative}"),
        check=True,
        capture_output=True,
    ).stdout


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
        assert hashlib.sha256(_git_blob(IMPLEMENTATION_REVISION, relative)).hexdigest() == expected
