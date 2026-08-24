from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "cut3r_exact_retry_dispatch_v3.py"
WORKFLOW = ROOT / ".github" / "workflows" / "cut3r-exact-retry-v3-comment-dispatch.yml"
HISTORICAL_EXECUTION_SHA = "8b923e8cd67ca65f09312cffe305e36852f36fbb"
RETAINED_REQUEST_ID = "8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e"
COMMAND = (
    f"/prob4d-dispatch-cut3r-source-freeze-v3 {HISTORICAL_EXECUTION_SHA} {RETAINED_REQUEST_ID}"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cut3r_exact_retry_dispatch_v3",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_run() -> dict[str, object]:
    return {
        "id": 32621813949,
        "name": "Execute retained CUT3R source freeze automatically v2",
        "path": ".github/workflows/cut3r-source-freeze-auto-v2.yml",
        "head_sha": HISTORICAL_EXECUTION_SHA,
        "head_branch": "main",
        "event": "push",
        "status": "completed",
        "conclusion": "failure",
    }


def _valid_jobs() -> dict[str, object]:
    step_conclusions = {
        "Initialize isolated read-only execution workspace": "success",
        "Check out only the authorized merged revision": "success",
        "Verify exact clean checkout": "success",
        "Provision Python 3.12": "success",
        "Build and install the exact reviewed wheel": "success",
        "Run one target-closed source freeze": "failure",
        "Upload retained automatic source-freeze evidence": "success",
        "Remove isolated execution workspace and checkout residue": "success",
    }
    return {
        "jobs": [
            {
                "id": 97376973894,
                "name": "Freeze retained source inputs from trusted merged main",
                "run_id": 32621813949,
                "status": "completed",
                "conclusion": "failure",
                "steps": [
                    {
                        "name": name,
                        "status": "completed",
                        "conclusion": conclusion,
                    }
                    for name, conclusion in step_conclusions.items()
                ],
            }
        ]
    }


def _valid_artifacts() -> dict[str, object]:
    return {
        "total_count": 1,
        "artifacts": [
            {
                "id": 9532584642,
                "name": "cut3r-source-freeze-v2-failed-32621813949-2",
                "size_in_bytes": 3106,
                "digest": (
                    "sha256:a7805e079ccb367d56634c62bb91a79fdac71babaa69c233e485135b9243a0a0"
                ),
                "expired": False,
                "workflow_run": {
                    "id": 32621813949,
                    "repository_id": 1295794737,
                    "head_repository_id": 1295794737,
                    "head_branch": "main",
                    "head_sha": HISTORICAL_EXECUTION_SHA,
                },
            }
        ],
    }


def _valid_registered_timeout_run() -> dict[str, object]:
    return {
        "id": 32764290533,
        "name": "Execute retained CUT3R source freeze automatically v2",
        "path": ".github/workflows/cut3r-source-freeze-auto-v2.yml",
        "head_sha": "78a209c2b217c264ab8b7bebfcc42fe7cd7d2ebf",
        "head_branch": "main",
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "cancelled",
        "run_attempt": 1,
        "created_at": "2026-08-24T18:46:15Z",
    }


def _valid_registered_timeout_jobs() -> dict[str, object]:
    rows = (
        (97550105116, "Authorize exact merged-main v2 request", "success"),
        (97550105469, "Hosted automatic-execution contract", "success"),
        (97550286850, "Check retained execution configuration", "success"),
        (97550323593, "Publish queued run pointer", "success"),
        (
            97550358844,
            "Freeze retained source inputs from trusted merged main",
            "cancelled",
        ),
        (
            97550358906,
            "Bound self-hosted runner acceptance wait",
            "cancelled",
        ),
        (97556521290, "Publish terminal v2 source-freeze receipt", "success"),
    )
    return {
        "jobs": [
            {
                "id": job_id,
                "name": name,
                "run_id": 32764290533,
                "status": "completed",
                "conclusion": conclusion,
                "steps": None if job_id == 97550358844 else [],
            }
            for job_id, name, conclusion in rows
        ]
    }


def _valid_registered_timeout_artifacts() -> dict[str, object]:
    return {"total_count": 0, "artifacts": []}


def test_exact_failure_diagnostic_is_admitted() -> None:
    module = _load_module()

    admitted = module.validate_superseded_run(
        _valid_run(),
        _valid_jobs(),
        _valid_artifacts(),
    )

    assert admitted["id"] == 9532584642
    assert admitted["size_in_bytes"] == 3106
    assert admitted["name"].startswith("cut3r-source-freeze-v2-failed-")


def test_result_or_additional_artifact_is_rejected() -> None:
    module = _load_module()
    artifacts = _valid_artifacts()
    rows = artifacts["artifacts"]
    assert isinstance(rows, list)
    rows.append(
        {
            "id": 999,
            "name": "cut3r-deform360-source-freeze-result",
            "size_in_bytes": 100,
            "digest": "sha256:unexpected",
            "expired": False,
        }
    )
    artifacts["total_count"] = 2

    with pytest.raises(
        module.DispatchError,
        match="unexpected artifact count",
    ):
        module.validate_superseded_run(
            _valid_run(),
            _valid_jobs(),
            artifacts,
        )


def test_failure_artifact_digest_drift_is_rejected() -> None:
    module = _load_module()
    artifacts = _valid_artifacts()
    rows = artifacts["artifacts"]
    assert isinstance(rows, list)
    artifact = rows[0]
    assert isinstance(artifact, dict)
    artifact["digest"] = "sha256:drift"

    with pytest.raises(module.DispatchError, match="digest mismatch"):
        module.validate_superseded_run(
            _valid_run(),
            _valid_jobs(),
            artifacts,
        )


def test_successful_scientific_job_is_rejected() -> None:
    module = _load_module()
    jobs = _valid_jobs()
    rows = jobs["jobs"]
    assert isinstance(rows, list)
    execute_job = rows[0]
    assert isinstance(execute_job, dict)
    execute_job["conclusion"] = "success"

    with pytest.raises(module.DispatchError, match="conclusion mismatch"):
        module.validate_superseded_run(
            _valid_run(),
            jobs,
            _valid_artifacts(),
        )


def test_exact_registered_runner_timeout_is_admitted() -> None:
    module = _load_module()

    admitted = module.validate_registered_runner_timeout(
        _valid_registered_timeout_run(),
        _valid_registered_timeout_jobs(),
        _valid_registered_timeout_artifacts(),
    )

    assert admitted["id"] == 32764290533


def test_registered_timeout_identity_and_job_outcome_drift_are_rejected() -> None:
    module = _load_module()
    run = _valid_registered_timeout_run()
    run["head_sha"] = "drift"

    with pytest.raises(module.DispatchError, match="head_sha mismatch"):
        module.validate_registered_runner_timeout(
            run,
            _valid_registered_timeout_jobs(),
            _valid_registered_timeout_artifacts(),
        )

    jobs = _valid_registered_timeout_jobs()
    rows = jobs["jobs"]
    assert isinstance(rows, list)
    authorize_job = rows[0]
    assert isinstance(authorize_job, dict)
    authorize_job["conclusion"] = "failure"

    with pytest.raises(module.DispatchError, match="conclusion mismatch"):
        module.validate_registered_runner_timeout(
            _valid_registered_timeout_run(),
            jobs,
            _valid_registered_timeout_artifacts(),
        )


def test_registered_timeout_with_retained_steps_is_rejected() -> None:
    module = _load_module()
    jobs = _valid_registered_timeout_jobs()
    rows = jobs["jobs"]
    assert isinstance(rows, list)
    execute_job = rows[4]
    assert isinstance(execute_job, dict)
    execute_job["steps"] = [
        {
            "name": "Set up job",
            "status": "completed",
            "conclusion": "success",
        }
    ]

    with pytest.raises(module.DispatchError, match="unexpectedly has steps"):
        module.validate_registered_runner_timeout(
            _valid_registered_timeout_run(),
            jobs,
            _valid_registered_timeout_artifacts(),
        )


def test_registered_timeout_with_any_artifact_is_rejected() -> None:
    module = _load_module()
    artifacts = {
        "total_count": 1,
        "artifacts": [{"id": 1, "name": "unexpected"}],
    }

    with pytest.raises(module.DispatchError, match="unexpectedly has artifacts"):
        module.validate_registered_runner_timeout(
            _valid_registered_timeout_run(),
            _valid_registered_timeout_jobs(),
            artifacts,
        )


def test_only_single_registered_timeout_bypasses_deduplication() -> None:
    module = _load_module()
    registered = {"id": 32764290533}

    assert module.is_only_registered_timeout([registered])
    assert not module.is_only_registered_timeout([])
    assert not module.is_only_registered_timeout([{"id": 32764290534}])
    assert not module.is_only_registered_timeout(
        [registered, {"id": 32764290534}]
    )


def test_relevant_run_selection_is_target_workflow_dispatch_only() -> None:
    module = _load_module()
    payload = {
        "workflow_runs": [
            {
                "id": 3,
                "event": "workflow_dispatch",
                "head_branch": "main",
                "created_at": "2026-08-24T18:30:00Z",
            },
            {
                "id": 2,
                "event": "push",
                "head_branch": "main",
                "created_at": "2026-08-24T18:31:00Z",
            },
            {
                "id": 1,
                "event": "workflow_dispatch",
                "head_branch": "other",
                "created_at": "2026-08-24T18:32:00Z",
            },
        ]
    }

    selected = module.select_relevant_runs(payload)

    assert [row["id"] for row in selected] == [3]


def test_workflow_is_exact_actor_issue_and_default_branch_bound() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    runs_on = [line.strip() for line in text.splitlines() if line.lstrip().startswith("runs-on:")]

    assert "\n  issue_comment:\n    types: [created]" in text
    assert "pull_request_target:" not in text
    assert "github.event.pull_request.head.sha" not in text
    assert "allow-unsafe-pr-checkout" not in text
    assert "Check out reviewed merge candidate" in text
    assert "github.event.issue.number == 49" in text
    assert "github.actor == 'FlorianPfaff'" in text
    assert "github.event.comment.user.login == 'FlorianPfaff'" in text
    assert f"github.event.comment.body == '{COMMAND}'" in text
    assert f"DISPATCH_COMMAND: {COMMAND}" in text
    assert text.count("ref: ${{ github.sha }}") == 1
    assert "fetch-depth: 0" in text
    assert "persist-credentials: false" in text
    assert runs_on == ["runs-on: ubuntu-latest", "runs-on: ubuntu-latest"]


def test_workflow_reauthorizes_and_uses_routed_target() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "authorize-retry" in text
    assert '--execution-revision "$HISTORICAL_EXECUTION_SHA"' in text
    assert '--expected-request-id "$RETAINED_REQUEST_ID"' in text
    assert "runs-on: [self-hosted, host-workstation2]" in text
    assert 'test "$RUNNER_NAME" = "workstation2"' in text
    assert 'test "$RUNNER_OS" = "Linux"' in text
    assert 'test "$RUNNER_ARCH" = "X64"' in text
    assert "command -v nvidia-smi >/dev/null 2>&1" in text
    assert "cut3r_exact_retry_dispatch_v3.py accepted" in text
    assert "cut3r_exact_retry_dispatch_v3.py dispatch" in text
    assert "actions: write" in text
    assert "issues: write" in text
    assert "contents: read" in text


def test_validation_does_not_mutate_inputs() -> None:
    module = _load_module()
    run = _valid_run()
    jobs = _valid_jobs()
    artifacts = _valid_artifacts()
    before = deepcopy((run, jobs, artifacts))

    module.validate_superseded_run(run, jobs, artifacts)

    assert (run, jobs, artifacts) == before


def test_registered_timeout_validation_does_not_mutate_inputs() -> None:
    module = _load_module()
    run = _valid_registered_timeout_run()
    jobs = _valid_registered_timeout_jobs()
    artifacts = _valid_registered_timeout_artifacts()
    before = deepcopy((run, jobs, artifacts))

    module.validate_registered_runner_timeout(run, jobs, artifacts)

    assert (run, jobs, artifacts) == before

