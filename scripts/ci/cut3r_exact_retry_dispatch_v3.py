from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, NoReturn

EXPECTED_REPOSITORY = "IPS-Stuttgart/Prob4D"
ISSUE_NUMBER = 49
TARGET_WORKFLOW = "cut3r-source-freeze-auto-v2.yml"
TARGET_REF = "main"
HISTORICAL_EXECUTION_SHA = "8b923e8cd67ca65f09312cffe305e36852f36fbb"
RETAINED_REQUEST_ID = "8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e"
SUPERSEDED_RUN_ID = 32621813949
SUPERSEDED_EXECUTE_JOB_ID = 97376973894
RETRY_WINDOW_START_UTC = "2026-08-24T18:12:05Z"
FAILURE_ARTIFACT = {
    "id": 9532584642,
    "name": "cut3r-source-freeze-v2-failed-32621813949-2",
    "size_in_bytes": 3106,
    "digest": ("sha256:a7805e079ccb367d56634c62bb91a79fdac71babaa69c233e485135b9243a0a0"),
    "expired": False,
}

JsonObject = dict[str, Any]


class DispatchError(RuntimeError):
    """Fail-closed exact-retry dispatch error."""


def _fail(message: str) -> NoReturn:
    raise DispatchError(message)


def _require_object(value: Any, *, label: str) -> JsonObject:
    if not isinstance(value, dict):
        _fail(f"{label} is not a JSON object")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} is not a JSON list")
    return value


def _require_exact_fields(
    actual: JsonObject,
    expected: JsonObject,
    *,
    label: str,
) -> None:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            _fail(f"{label} {key} mismatch: {actual_value!r} != {expected_value!r}")


def validate_superseded_run(
    run_payload: Any,
    jobs_payload: Any,
    artifacts_payload: Any,
) -> JsonObject:
    """Validate the exact terminal failure and its diagnostic-only artifact."""

    run = _require_object(run_payload, label="superseded run")
    _require_exact_fields(
        run,
        {
            "id": SUPERSEDED_RUN_ID,
            "name": "Execute retained CUT3R source freeze automatically v2",
            "path": ".github/workflows/cut3r-source-freeze-auto-v2.yml",
            "head_sha": HISTORICAL_EXECUTION_SHA,
            "head_branch": "main",
            "event": "push",
            "status": "completed",
            "conclusion": "failure",
        },
        label="superseded run",
    )

    jobs = _require_object(jobs_payload, label="superseded jobs")
    job_rows = _require_list(jobs.get("jobs"), label="superseded jobs.jobs")
    execute_jobs = [
        row
        for row in job_rows
        if isinstance(row, dict)
        and row.get("name") == "Freeze retained source inputs from trusted merged main"
    ]
    if len(execute_jobs) != 1:
        _fail("expected exactly one superseded retained source-freeze job")
    execute_job = _require_object(execute_jobs[0], label="superseded execute job")
    _require_exact_fields(
        execute_job,
        {
            "id": SUPERSEDED_EXECUTE_JOB_ID,
            "run_id": SUPERSEDED_RUN_ID,
            "status": "completed",
            "conclusion": "failure",
        },
        label="superseded execute job",
    )

    step_rows = _require_list(
        execute_job.get("steps"),
        label="superseded execute job.steps",
    )
    steps_by_name = {
        str(row.get("name")): row
        for row in step_rows
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    expected_steps = {
        "Initialize isolated read-only execution workspace": "success",
        "Check out only the authorized merged revision": "success",
        "Verify exact clean checkout": "success",
        "Provision Python 3.12": "success",
        "Build and install the exact reviewed wheel": "success",
        "Run one target-closed source freeze": "failure",
        "Upload retained automatic source-freeze evidence": "success",
        "Remove isolated execution workspace and checkout residue": "success",
    }
    for step_name, expected_conclusion in expected_steps.items():
        step = _require_object(
            steps_by_name.get(step_name),
            label=f"superseded step {step_name!r}",
        )
        _require_exact_fields(
            step,
            {
                "status": "completed",
                "conclusion": expected_conclusion,
            },
            label=f"superseded step {step_name!r}",
        )

    artifacts = _require_object(
        artifacts_payload,
        label="superseded artifacts",
    )
    artifact_rows = _require_list(
        artifacts.get("artifacts"),
        label="superseded artifacts.artifacts",
    )
    total_count = artifacts.get("total_count", len(artifact_rows))
    if total_count != 1 or len(artifact_rows) != 1:
        _fail("superseded run has an unexpected artifact count")
    artifact = _require_object(
        artifact_rows[0],
        label="superseded failure artifact",
    )
    _require_exact_fields(
        artifact,
        FAILURE_ARTIFACT,
        label="superseded failure artifact",
    )
    workflow_run = _require_object(
        artifact.get("workflow_run"),
        label="superseded failure artifact.workflow_run",
    )
    _require_exact_fields(
        workflow_run,
        {
            "id": SUPERSEDED_RUN_ID,
            "repository_id": 1295794737,
            "head_repository_id": 1295794737,
            "head_branch": "main",
            "head_sha": HISTORICAL_EXECUTION_SHA,
        },
        label="superseded failure artifact.workflow_run",
    )
    return artifact


def select_relevant_runs(payload: Any) -> list[JsonObject]:
    response = _require_object(payload, label="target workflow runs")
    rows = _require_list(
        response.get("workflow_runs"),
        label="target workflow runs.workflow_runs",
    )
    window_start = datetime.fromisoformat(RETRY_WINDOW_START_UTC.replace("Z", "+00:00"))
    selected: list[JsonObject] = []
    for value in rows:
        if not isinstance(value, dict):
            continue
        created_raw = value.get("created_at")
        if not isinstance(created_raw, str):
            continue
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        if (
            value.get("event") == "workflow_dispatch"
            and value.get("head_branch") == TARGET_REF
            and created >= window_start
        ):
            selected.append(value)
    return sorted(
        selected,
        key=lambda row: (str(row.get("created_at")), int(row.get("id", 0))),
        reverse=True,
    )


class GitHubClient:
    def __init__(self, *, api_url: str, repository: str, token: str) -> None:
        if repository != EXPECTED_REPOSITORY:
            _fail(f"repository mismatch: {repository!r} != {EXPECTED_REPOSITORY!r}")
        if not token:
            _fail("GITHUB_TOKEN is empty")
        self.api_url = api_url.rstrip("/")
        self.repository = repository
        self.token = token

    def request_json(
        self,
        method: str,
        path: str,
        payload: JsonObject | None = None,
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_url}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "prob4d-cut3r-exact-retry-dispatch-v3",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            _fail(f"GitHub API {method} {path} failed with {error.code}: {detail}")
        return None if not body else json.loads(body)

    def post_comment(self, body: str) -> None:
        self.request_json(
            "POST",
            f"/repos/{self.repository}/issues/{ISSUE_NUMBER}/comments",
            {"body": body},
        )

    @property
    def encoded_target_workflow(self) -> str:
        return urllib.parse.quote(TARGET_WORKFLOW, safe="")

    @property
    def target_runs_path(self) -> str:
        return (
            f"/repos/{self.repository}/actions/workflows/"
            f"{self.encoded_target_workflow}/runs"
            f"?event=workflow_dispatch&branch={TARGET_REF}&per_page=50"
        )

    def relevant_runs(self) -> list[JsonObject]:
        return select_relevant_runs(self.request_json("GET", self.target_runs_path))


def _client_from_environment() -> GitHubClient:
    return GitHubClient(
        api_url=os.environ["API_URL"],
        repository=os.environ["REPOSITORY"],
        token=os.environ["GITHUB_TOKEN"],
    )


def publish_accepted_receipt() -> None:
    client = _client_from_environment()
    client.post_comment(
        "\n".join(
            (
                "## Exact CUT3R retry-v3 command accepted",
                "",
                f"- historical execution revision: `{HISTORICAL_EXECUTION_SHA}`",
                f"- retained request ID: `{RETAINED_REQUEST_ID}`",
                f"- command comment: {os.environ['COMMAND_COMMENT_URL']}",
                f"- helper workflow: {os.environ['HELPER_RUN_URL']}",
                "",
                "The hosted helper admits only the exact registered diagnostic "
                "failure artifact and first resolves any existing target retry. "
                "No retained path or scientific outcome is opened.",
            )
        )
    )


def resolve_or_dispatch() -> None:
    client = _client_from_environment()
    existing = client.relevant_runs()
    if existing:
        run = existing[0]
        client.post_comment(
            "\n".join(
                (
                    "## Existing CUT3R retry resolved by v3 dispatcher",
                    "",
                    f"- workflow run: {run.get('html_url')}",
                    f"- run ID: `{run.get('id')}`",
                    f"- status: `{run.get('status')}`",
                    f"- conclusion: `{run.get('conclusion')}`",
                    f"- created at: `{run.get('created_at')}`",
                    f"- command comment: {os.environ['COMMAND_COMMENT_URL']}",
                    f"- resolver workflow: {os.environ['HELPER_RUN_URL']}",
                    "",
                    "No duplicate target retry was dispatched and no outcome was opened.",
                )
            )
        )
        return

    run_path = f"/repos/{client.repository}/actions/runs/{SUPERSEDED_RUN_ID}"
    run_payload = client.request_json("GET", run_path)
    jobs_payload = client.request_json(
        "GET",
        f"{run_path}/jobs?per_page=100",
    )
    artifacts_payload = client.request_json(
        "GET",
        f"{run_path}/artifacts?per_page=100",
    )
    admitted_artifact = validate_superseded_run(
        run_payload,
        jobs_payload,
        artifacts_payload,
    )

    before_ids = {
        int(row["id"]) for row in client.relevant_runs() if isinstance(row.get("id"), int)
    }
    client.request_json(
        "POST",
        (
            f"/repos/{client.repository}/actions/workflows/"
            f"{client.encoded_target_workflow}/dispatches"
        ),
        {
            "ref": TARGET_REF,
            "inputs": {
                "execution_sha": HISTORICAL_EXECUTION_SHA,
                "request_id": RETAINED_REQUEST_ID,
            },
        },
    )

    discovered: JsonObject | None = None
    for _ in range(30):
        time.sleep(2)
        for row in client.relevant_runs():
            run_id = row.get("id")
            if isinstance(run_id, int) and run_id not in before_ids:
                discovered = row
                break
        if discovered is not None:
            break

    if discovered is None:
        client.post_comment(
            "\n".join(
                (
                    "## Exact CUT3R retry-v3 dispatch not yet discoverable",
                    "",
                    f"- historical execution revision: `{HISTORICAL_EXECUTION_SHA}`",
                    f"- retained request ID: `{RETAINED_REQUEST_ID}`",
                    f"- admitted diagnostic artifact ID: `{admitted_artifact['id']}`",
                    f"- helper workflow: {os.environ['HELPER_RUN_URL']}",
                    "",
                    "GitHub accepted the workflow dispatch, but the target run did "
                    "not appear during the bounded discovery interval. No outcome "
                    "was opened.",
                )
            )
        )
        _fail("accepted target workflow dispatch was not discoverable")

    client.post_comment(
        "\n".join(
            (
                "## Exact retained CUT3R source-freeze retry dispatched by v3",
                "",
                f"- workflow run: {discovered.get('html_url')}",
                f"- run ID: `{discovered.get('id')}`",
                f"- status: `{discovered.get('status')}`",
                f"- historical execution revision: `{HISTORICAL_EXECUTION_SHA}`",
                f"- retained request ID: `{RETAINED_REQUEST_ID}`",
                f"- admitted diagnostic artifact ID: `{admitted_artifact['id']}`",
                f"- admitted artifact digest: `{admitted_artifact['digest']}`",
                f"- command comment: {os.environ['COMMAND_COMMENT_URL']}",
                f"- dispatcher workflow: {os.environ['HELPER_RUN_URL']}",
                "",
                "The target workflow independently repeats exact authorization, "
                "checks retained-variable presence before queueing, uses the routed "
                "retained-data/CUT3R labels, and remains target-closed.",
            )
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch the exact retained CUT3R source-freeze retry v3."
    )
    parser.add_argument(
        "phase",
        choices=("accepted", "dispatch"),
        help="Publish the accepted receipt or resolve/dispatch the target run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.phase == "accepted":
            publish_accepted_receipt()
        else:
            resolve_or_dispatch()
    except DispatchError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
