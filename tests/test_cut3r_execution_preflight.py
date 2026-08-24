from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "cut3r_execution_preflight.py"
REQUEST_RELATIVE = Path("protocols/execution_requests/cut3r_deform360_source_freeze_v2.json")
PROTOCOL_RELATIVE = Path("protocols/cut3r_deform360_source_v1.json")
DRIVER_RELATIVE = Path("scripts/science/run_cut3r_source_freeze_execution.py")


def _module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "cut3r_execution_preflight",
        SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _run(*arguments: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _canonical_id(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _request(protocol_blob: str) -> dict[str, object]:
    request: dict[str, object] = {
        "authorization_mode": "merged-main-read-only-self-hosted-v1",
        "claim_boundary": "fixture target-closed request",
        "comparison_execution_authorized": False,
        "environment_approval_required": False,
        "forbidden_target_group_count": 12,
        "issue_number": 49,
        "profile": "cut3r-deform360-source-freeze",
        "repository_write_token_on_self_hosted": False,
        "schema": "prob4d.cut3r-deform360-source-freeze-execution-request",
        "schema_version": 2,
        "source_group_count": 10,
        "source_prediction_payloads_opened": False,
        "source_protocol_git_blob_sha": protocol_blob,
        "source_protocol_path": PROTOCOL_RELATIVE.as_posix(),
        "source_residuals_or_truth_opened": False,
        "source_rgb_frames_decoded": False,
        "supersedes_request_id": "0" * 64,
        "target_outcomes_opened": False,
        "target_payloads_opened": False,
    }
    request["request_id"] = _canonical_id(request)
    return request


def _repository(tmp_path: Path) -> tuple[Path, str, str, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _run("init", "-q", "-b", "main", cwd=repository)
    _run("config", "user.name", "Prob4D tests", cwd=repository)
    _run("config", "user.email", "tests@example.invalid", cwd=repository)

    protocol = repository / PROTOCOL_RELATIVE
    protocol.parent.mkdir(parents=True)
    protocol.write_text('{"schema":"fixture"}\n', encoding="utf-8")
    driver = repository / DRIVER_RELATIVE
    driver.parent.mkdir(parents=True)
    driver.write_text("raise SystemExit(0)\n", encoding="utf-8")
    protocol_blob = _run("hash-object", PROTOCOL_RELATIVE.as_posix(), cwd=repository)

    request = _request(protocol_blob)
    request_path = repository / REQUEST_RELATIVE
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _run("add", ".", cwd=repository)
    _run("commit", "-qm", "historical request", cwd=repository)
    historical = _run("rev-parse", "HEAD", cwd=repository)

    (repository / "unrelated.txt").write_text("later main change\n", encoding="utf-8")
    _run("add", "unrelated.txt", cwd=repository)
    _run("commit", "-qm", "later main change", cwd=repository)
    current = _run("rev-parse", "HEAD", cwd=repository)
    return repository, historical, current, str(request["request_id"])


def test_variable_readiness_reports_names_only() -> None:
    module = _module()
    secret_value = "/retained/private/path"

    report = module.variable_readiness(
        ["CUT3R_CHECKPOINT", "BPT_CHECKOUT", "CUT3R_CHECKPOINT"],
        environment={
            "BPT_CHECKOUT": secret_value,
            "CUT3R_CHECKPOINT": "   ",
        },
    )

    assert report["required_variables"] == ["BPT_CHECKOUT", "CUT3R_CHECKPOINT"]
    assert report["configured_variable_count"] == 1
    assert report["missing_variables"] == ["CUT3R_CHECKPOINT"]
    assert report["ready"] is False
    assert secret_value not in json.dumps(report)


def test_variable_readiness_rejects_invalid_names() -> None:
    module = _module()

    with pytest.raises(ValueError, match="required variable names"):
        module.variable_readiness(["bad-name"], environment={})
    with pytest.raises(ValueError, match="at least one"):
        module.variable_readiness([], environment={})


def test_exact_historical_retry_is_authorized(tmp_path: Path) -> None:
    module = _module()
    repository, historical, current, request_id = _repository(tmp_path)

    result = module.authorize_retry(
        repository=repository,
        request_path=repository / REQUEST_RELATIVE,
        current_revision=current,
        execution_revision=historical,
        expected_request_id=request_id,
    )

    assert result["head_sha"] == historical
    assert result["current_main_sha"] == current
    assert result["request_id"] == request_id
    assert result["trigger_mode"] == "workflow_dispatch_exact_retry"
    assert result["profile"] == "cut3r-deform360-source-freeze"


def test_retry_rejects_request_byte_drift(tmp_path: Path) -> None:
    module = _module()
    repository, historical, _, request_id = _repository(tmp_path)
    request_path = repository / REQUEST_RELATIVE
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["claim_boundary"] = "changed current request"
    identity = dict(request)
    identity.pop("request_id")
    request["request_id"] = _canonical_id(identity)
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _run("add", REQUEST_RELATIVE.as_posix(), cwd=repository)
    _run("commit", "-qm", "change request", cwd=repository)
    current = _run("rev-parse", "HEAD", cwd=repository)

    with pytest.raises(ValueError, match="request bytes differ"):
        module.authorize_retry(
            repository=repository,
            request_path=request_path,
            current_revision=current,
            execution_revision=historical,
            expected_request_id=request_id,
        )


def test_retry_rejects_request_field_drift(tmp_path: Path) -> None:
    module = _module()
    repository, historical, _, request_id = _repository(tmp_path)
    request_path = repository / REQUEST_RELATIVE
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["unexpected"] = "field"
    identity = dict(request)
    identity.pop("request_id")
    request["request_id"] = _canonical_id(identity)
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _run("add", REQUEST_RELATIVE.as_posix(), cwd=repository)
    _run("commit", "-qm", "add unexpected request field", cwd=repository)
    current = _run("rev-parse", "HEAD", cwd=repository)

    with pytest.raises(ValueError, match="request bytes differ"):
        module.authorize_retry(
            repository=repository,
            request_path=request_path,
            current_revision=current,
            execution_revision=historical,
            expected_request_id=request_id,
        )


def test_retry_rejects_nonancestor_revision(tmp_path: Path) -> None:
    module = _module()
    repository, _, current, request_id = _repository(tmp_path)
    _run("checkout", "--orphan", "unrelated-history", cwd=repository)
    _run("rm", "-rf", ".", cwd=repository)
    (repository / "orphan.txt").write_text("unrelated\n", encoding="utf-8")
    _run("add", "orphan.txt", cwd=repository)
    _run("commit", "-qm", "unrelated history", cwd=repository)
    unrelated = _run("rev-parse", "HEAD", cwd=repository)
    _run("checkout", "main", cwd=repository)

    with pytest.raises(ValueError, match="not an ancestor"):
        module.authorize_retry(
            repository=repository,
            request_path=repository / REQUEST_RELATIVE,
            current_revision=current,
            execution_revision=unrelated,
            expected_request_id=request_id,
        )


def test_check_variables_cli_returns_registered_not_ready_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    report_path = tmp_path / "readiness.json"
    output_path = tmp_path / "github-output.txt"
    monkeypatch.setenv("BPT_CHECKOUT", "/configured")
    monkeypatch.delenv("CUT3R_CHECKPOINT", raising=False)

    status = module.main(
        [
            "check-variables",
            "--required-variable",
            "BPT_CHECKOUT",
            "--required-variable",
            "CUT3R_CHECKPOINT",
            "--report",
            os.fspath(report_path),
            "--github-output",
            os.fspath(output_path),
            "--require-ready",
        ]
    )

    assert status == 3
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["missing_variables"] == ["CUT3R_CHECKPOINT"]
    outputs = output_path.read_text(encoding="utf-8")
    assert "ready=false" in outputs
    assert 'missing_variables_json=["CUT3R_CHECKPOINT"]' in outputs
