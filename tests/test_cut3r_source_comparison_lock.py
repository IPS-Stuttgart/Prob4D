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
        assert actual_sha256 == expected_sha256
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
