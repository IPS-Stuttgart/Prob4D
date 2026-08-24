from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "cut3r-exact-retry-comment-dispatch.yml"
HISTORICAL_EXECUTION_SHA = "8b923e8cd67ca65f09312cffe305e36852f36fbb"
RETAINED_REQUEST_ID = "8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e"
COMMAND = (
    f"/prob4d-dispatch-cut3r-source-freeze-v2 {HISTORICAL_EXECUTION_SHA} {RETAINED_REQUEST_ID}"
)


def test_comment_dispatch_is_exact_actor_and_issue_bound() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "\n  issue_comment:\n    types: [created]" in text
    assert "pull_request_target:" not in text
    assert "github.event.issue.number == 49" in text
    assert "github.actor == 'FlorianPfaff'" in text
    assert "github.event.comment.user.login == 'FlorianPfaff'" in text
    assert f"github.event.comment.body == '{COMMAND}'" in text
    assert f"DISPATCH_COMMAND: {COMMAND}" in text
    assert "actions: write" in text
    assert "issues: write" in text
    assert "contents: read" in text


def test_comment_dispatch_uses_reviewed_default_branch_bytes() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "ref: ${{ github.sha }}" in text
    assert "fetch-depth: 0" in text
    assert "persist-credentials: false" in text
    assert 'test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"' in text
    assert 'test "$EVENT_COMMENT_BODY" = "$DISPATCH_COMMAND"' in text
    assert "authorize-retry" in text
    assert '--execution-revision "$HISTORICAL_EXECUTION_SHA"' in text
    assert '--expected-request-id "$RETAINED_REQUEST_ID"' in text
    assert HISTORICAL_EXECUTION_SHA in text
    assert RETAINED_REQUEST_ID in text
    assert (
        "runs-on: [self-hosted, Linux, X64, nvidia-smi, "
        "data-prob4d-deform360-source-v1, prob4d-cut3r]"
    ) in text


def test_comment_dispatch_discovers_before_dispatch_and_fails_closed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    listing = '"?event=workflow_dispatch&branch=main&per_page=50"'
    assert listing in text
    assert "existing = relevant_runs()" in text
    assert "No duplicate retry was dispatched" in text
    assert "superseded run is not a terminal zero-evidence failure" in text
    assert "superseded source-freeze job succeeded; duplicate forbidden" in text
    assert 'SUPERSEDED_FAILURE_ARTIFACT_ID: "9532584642"' in text
    assert ("SUPERSEDED_FAILURE_ARTIFACT_NAME: cut3r-source-freeze-v2-failed-32621813949-2") in text
    assert 'SUPERSEDED_FAILURE_ARTIFACT_SIZE: "3106"' in text
    assert 'artifacts.get("total_count") != 1' in text
    assert '"expired": False' in text
    assert "superseded failure-evidence artifact run binding mismatch" in text
    assert "before_ids =" in text
    assert "/dispatches" in text
    assert '"execution_sha": expected_head' in text
    assert '"request_id": os.environ["RETAINED_REQUEST_ID"]' in text
    assert "accepted workflow dispatch was not discoverable" in text
    assert "Exact retained CUT3R source-freeze retry dispatched" in text


def test_comment_dispatch_jobs_are_github_hosted() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    runs_on = [line.strip() for line in text.splitlines() if line.lstrip().startswith("runs-on:")]

    assert runs_on == ["runs-on: ubuntu-latest", "runs-on: ubuntu-latest"]
