from __future__ import annotations

import argparse
import copy
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

REPOSITORY = "IPS-Stuttgart/Prob4D"
ISSUE = 49
WORKFLOW = "cut3r-source-freeze-auto-v2.yml"
HISTORICAL_SHA = "8b923e8cd67ca65f09312cffe305e36852f36fbb"
REQUEST_ID = "8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e"
FAILED_RUN_ID = 32771242880
FAILED_HEAD_SHA = "30f85ecd6c2395b333f723db47a18f6677d7c58f"
FAILED_CREATED_AT = "2026-08-24T19:58:36Z"
FAILED_ATTEMPT = 3
FAILED_EXECUTE_JOB_ID = 97709306705
EXPECTED_JOBS = {
    "Authorize exact merged-main v2 request": (97709306242, "success"),
    "Freeze retained source inputs from trusted merged main": (
        FAILED_EXECUTE_JOB_ID,
        "failure",
    ),
    "Check retained execution configuration": (97709306713, "success"),
    "Publish queued run pointer": (97709307010, "success"),
    "Bound self-hosted runner acceptance wait": (97709307652, "success"),
    "Hosted automatic-execution contract": (97709332603, "success"),
    "Publish terminal v2 source-freeze receipt": (97709374957, "success"),
}
EXPECTED_STEPS = {
    "Set up job": "success",
    "Bind the sole retained Deform360 runner": "success",
    "Initialize isolated read-only execution workspace": "success",
    "Check out only the authorized merged revision": "success",
    "Verify exact clean checkout": "success",
    "Provision Python 3.12": "success",
    "Build and install the exact reviewed wheel": "success",
    "Run one target-closed source freeze": "failure",
    "Record exact automatic-execution environment": "success",
    "Upload retained automatic source-freeze evidence": "success",
    "Remove isolated execution workspace and checkout residue": "success",
    "Post Provision Python 3.12": "skipped",
    "Post Check out only the authorized merged revision": "success",
    "Complete job": "success",
}
EXPECTED_ARTIFACTS = {
    9551988484: {
        "id": 9551988484,
        "name": "cut3r-source-freeze-v2-failed-32771242880-3",
        "size_in_bytes": 4326,
        "digest": (
            "sha256:68f22308ab86190d68c196b747ffcf6f217e670a07da15377da35b7bbb61b57e"
        ),
        "expired": False,
    },
    9551181122: {
        "id": 9551181122,
        "name": "cut3r-source-freeze-v2-failed-32771242880-2",
        "size_in_bytes": 4384,
        "digest": (
            "sha256:d1eff3af637eb297e72334693b1c51723f4eb9487a6cf6a8957d130bc34b9721"
        ),
        "expired": False,
    },
}
LATEST_ARTIFACT_ID = 9551988484

Json = dict[str, Any]


class DispatchError(RuntimeError):
    """Fail-closed exact replacement error."""


def fail(message: str) -> NoReturn:
    raise DispatchError(message)


def obj(value: Any, label: str) -> Json:
    if not isinstance(value, dict):
        fail(f"{label} is not an object")
    return value


def rows(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{label} is not a list")
    return value


def exact(actual: Json, expected: Json, label: str) -> None:
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            fail(
                f"{label} {key} mismatch: "
                f"{actual.get(key)!r} != {expected_value!r}"
            )


def validate_workflow(text: str) -> None:
    required = (
        "runs-on: [self-hosted, host-workstation2]",
        'test "$RUNNER_NAME" = "workstation2"',
        'test "$RUNNER_OS" = "Linux"',
        'test "$RUNNER_ARCH" = "X64"',
        "command -v nvidia-smi >/dev/null 2>&1",
        'cut3r_checkout=$(/usr/bin/realpath -e -- "$CUT3R_CHECKOUT")',
        "GIT_CONFIG_COUNT=1",
        "GIT_CONFIG_KEY_0=safe.directory",
        'GIT_CONFIG_VALUE_0="$cut3r_checkout"',
        '--cut3r-checkout "$cut3r_checkout"',
    )
    missing = [token for token in required if token not in text]
    if missing:
        fail(f"target workflow lacks scoped-trust tokens: {missing!r}")
    forbidden = ("safe.directory=*", "git config --global", "sudo ")
    present = [token for token in forbidden if token in text]
    if present:
        fail(f"target workflow contains broad or privileged trust: {present!r}")


def validate_failure(run_value: Any, jobs_value: Any, artifacts_value: Any) -> Json:
    run = obj(run_value, "failure run")
    exact(
        run,
        {
            "id": FAILED_RUN_ID,
            "name": "Execute retained CUT3R source freeze automatically v2",
            "path": ".github/workflows/cut3r-source-freeze-auto-v2.yml",
            "head_sha": FAILED_HEAD_SHA,
            "head_branch": "main",
            "event": "workflow_dispatch",
            "status": "completed",
            "conclusion": "failure",
            "run_attempt": FAILED_ATTEMPT,
            "created_at": FAILED_CREATED_AT,
        },
        "failure run",
    )

    jobs = {
        row.get("name"): row
        for row in rows(obj(jobs_value, "jobs").get("jobs"), "jobs.jobs")
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    if set(jobs) != set(EXPECTED_JOBS):
        fail("failure job roster mismatch")
    for name, (job_id, conclusion) in EXPECTED_JOBS.items():
        exact(
            obj(jobs[name], f"job {name!r}"),
            {
                "id": job_id,
                "run_id": FAILED_RUN_ID,
                "status": "completed",
                "conclusion": conclusion,
            },
            f"job {name!r}",
        )

    execute = obj(
        jobs["Freeze retained source inputs from trusted merged main"],
        "execute job",
    )
    steps = {
        row.get("name"): row
        for row in rows(execute.get("steps"), "execute steps")
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    if set(steps) != set(EXPECTED_STEPS):
        fail("failure execute-step roster mismatch")
    for name, conclusion in EXPECTED_STEPS.items():
        exact(
            obj(steps[name], f"step {name!r}"),
            {"status": "completed", "conclusion": conclusion},
            f"step {name!r}",
        )

    artifact_payload = obj(artifacts_value, "artifacts")
    artifact_rows = rows(artifact_payload.get("artifacts"), "artifacts.artifacts")
    if artifact_payload.get("total_count") != 2 or len(artifact_rows) != 2:
        fail("failure artifact count mismatch")
    artifacts = {
        row.get("id"): row
        for row in artifact_rows
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    }
    if set(artifacts) != set(EXPECTED_ARTIFACTS):
        fail("failure artifact identity roster mismatch")
    for artifact_id, expected in EXPECTED_ARTIFACTS.items():
        artifact = obj(artifacts[artifact_id], f"artifact {artifact_id}")
        exact(artifact, expected, f"artifact {artifact_id}")
        exact(
            obj(artifact.get("workflow_run"), f"artifact {artifact_id}.workflow_run"),
            {
                "id": FAILED_RUN_ID,
                "repository_id": 1295794737,
                "head_repository_id": 1295794737,
                "head_branch": "main",
                "head_sha": FAILED_HEAD_SHA,
            },
            f"artifact {artifact_id}.workflow_run",
        )
    return obj(artifacts[LATEST_ARTIFACT_ID], "latest artifact")


def newer_retries(payload: Any) -> list[Json]:
    threshold = datetime.fromisoformat(FAILED_CREATED_AT.replace("Z", "+00:00"))
    selected = []
    workflow_rows = rows(
        obj(payload, "workflow runs").get("workflow_runs"),
        "workflow runs",
    )
    for row in workflow_rows:
        if not isinstance(row, dict):
            continue
        created_raw = row.get("created_at")
        if not isinstance(created_raw, str):
            continue
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        if (
            row.get("event") == "workflow_dispatch"
            and row.get("head_branch") == "main"
            and row.get("id") != FAILED_RUN_ID
            and created > threshold
        ):
            selected.append(row)
    return sorted(
        selected,
        key=lambda row: (str(row.get("created_at")), int(row.get("id", 0))),
        reverse=True,
    )


def request_json(method: str, path: str, payload: Json | None = None) -> Any:
    api_url = os.environ["API_URL"].rstrip("/")
    repository = os.environ["REPOSITORY"]
    if repository != REPOSITORY:
        fail(f"repository mismatch: {repository!r}")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{api_url}{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Content-Type": "application/json",
            "User-Agent": "prob4d-cut3r-safe-directory-retry-v4",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        fail(f"GitHub API {method} {path} failed with {error.code}: {detail}")
    return None if not body else json.loads(body)


def comment(body: str) -> None:
    request_json("POST", f"/repos/{REPOSITORY}/issues/{ISSUE}/comments", {"body": body})


def target_runs_path() -> str:
    encoded = urllib.parse.quote(WORKFLOW, safe="")
    return (
        f"/repos/{REPOSITORY}/actions/workflows/{encoded}/runs"
        "?event=workflow_dispatch&branch=main&per_page=50"
    )


def dispatch() -> None:
    validate_workflow(Path(os.environ["TARGET_WORKFLOW_PATH"]).read_text(encoding="utf-8"))
    run_path = f"/repos/{REPOSITORY}/actions/runs/{FAILED_RUN_ID}"
    artifact = validate_failure(
        request_json("GET", run_path),
        request_json("GET", f"{run_path}/jobs?filter=latest&per_page=100"),
        request_json("GET", f"{run_path}/artifacts?per_page=100"),
    )
    runs_path = target_runs_path()
    existing = newer_retries(request_json("GET", runs_path))
    if existing:
        run = existing[0]
        comment(
            "\n".join(
                (
                    "## Existing CUT3R safe-directory replacement resolved",
                    "",
                    f"- workflow run: {run.get('html_url')}",
                    f"- run ID: `{run.get('id')}`",
                    f"- status: `{run.get('status')}`",
                    f"- conclusion: `{run.get('conclusion')}`",
                    f"- command comment: {os.environ['COMMAND_COMMENT_URL']}",
                    "",
                    "No duplicate replacement was dispatched.",
                )
            )
        )
        return

    before = {
        row.get("id")
        for row in rows(
            obj(request_json("GET", runs_path), "workflow runs").get("workflow_runs"),
            "workflow runs",
        )
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    }
    comment(
        "\n".join(
            (
                "## Exact CUT3R attempt-3 failure admitted for replacement",
                "",
                f"- failed run ID: `{FAILED_RUN_ID}`",
                f"- execute job ID: `{FAILED_EXECUTE_JOB_ID}`",
                f"- diagnostic artifact ID: `{artifact['id']}`",
                f"- diagnostic digest: `{artifact['digest']}`",
                f"- command comment: {os.environ['COMMAND_COMMENT_URL']}",
                "",
                "The failure occurred during Git revision resolution before a "
                "source-freeze decision. One current-main replacement is admitted.",
            )
        )
    )
    encoded = urllib.parse.quote(WORKFLOW, safe="")
    request_json(
        "POST",
        f"/repos/{REPOSITORY}/actions/workflows/{encoded}/dispatches",
        {
            "ref": "main",
            "inputs": {"execution_sha": HISTORICAL_SHA, "request_id": REQUEST_ID},
        },
    )

    discovered: Json | None = None
    for _ in range(30):
        time.sleep(2)
        payload = obj(request_json("GET", runs_path), "workflow runs")
        for row in rows(payload.get("workflow_runs"), "workflow runs"):
            if isinstance(row, dict) and row.get("id") not in before:
                discovered = row
                break
        if discovered is not None:
            break
    if discovered is None:
        fail("accepted replacement dispatch was not discoverable")
    comment(
        "\n".join(
            (
                "## Exact retained CUT3R replacement dispatched",
                "",
                f"- workflow run: {discovered.get('html_url')}",
                f"- run ID: `{discovered.get('id')}`",
                f"- historical revision: `{HISTORICAL_SHA}`",
                f"- retained request ID: `{REQUEST_ID}`",
                f"- dispatcher: {os.environ['HELPER_RUN_URL']}",
                "",
                "The target workflow remains source-only and target-closed.",
            )
        )
    )


def self_test(workflow_path: Path) -> None:
    validate_workflow(workflow_path.read_text(encoding="utf-8"))
    workflow_run = {
        "id": FAILED_RUN_ID,
        "repository_id": 1295794737,
        "head_repository_id": 1295794737,
        "head_branch": "main",
        "head_sha": FAILED_HEAD_SHA,
    }
    run = {
        "id": FAILED_RUN_ID,
        "name": "Execute retained CUT3R source freeze automatically v2",
        "path": ".github/workflows/cut3r-source-freeze-auto-v2.yml",
        "head_sha": FAILED_HEAD_SHA,
        "head_branch": "main",
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "failure",
        "run_attempt": FAILED_ATTEMPT,
        "created_at": FAILED_CREATED_AT,
    }
    jobs = {
        "jobs": [
            {
                "id": job_id,
                "name": name,
                "run_id": FAILED_RUN_ID,
                "status": "completed",
                "conclusion": conclusion,
                "steps": (
                    [
                        {
                            "name": step_name,
                            "status": "completed",
                            "conclusion": step_conclusion,
                        }
                        for step_name, step_conclusion in EXPECTED_STEPS.items()
                    ]
                    if job_id == FAILED_EXECUTE_JOB_ID
                    else []
                ),
            }
            for name, (job_id, conclusion) in EXPECTED_JOBS.items()
        ]
    }
    artifacts = {
        "total_count": 2,
        "artifacts": [
            {**expected, "workflow_run": copy.deepcopy(workflow_run)}
            for expected in EXPECTED_ARTIFACTS.values()
        ],
    }
    admitted = validate_failure(run, jobs, artifacts)
    if admitted["id"] != LATEST_ARTIFACT_ID:
        fail("self-test returned the wrong diagnostic artifact")
    drift = copy.deepcopy(artifacts)
    drift["artifacts"][0]["digest"] = "sha256:drift"
    try:
        validate_failure(run, jobs, drift)
    except DispatchError:
        pass
    else:
        fail("self-test accepted diagnostic digest drift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("dispatch", "self-test"))
    parser.add_argument("--workflow", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.phase == "dispatch":
            dispatch()
        else:
            if args.workflow is None:
                fail("--workflow is required for self-test")
            self_test(args.workflow)
    except (DispatchError, OSError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
