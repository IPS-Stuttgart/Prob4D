from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NoReturn

REPOSITORY = "IPS-Stuttgart/Prob4D"
ISSUE_NUMBER = 49
PREDECESSOR_SMOKE_PATH = Path("evidence/cut3r-source-comparison-smoke-v1/summary.json")
PREDECESSOR_SMOKE_FILE_SHA256 = "550ce0c8858c4730548d08af50589c820ed3023cfbbae3050eea0369bcd7845f"
PREDECESSOR_SMOKE_ID = "ef4e5bf187570e918df1d7d14434b4ae55f983c347104b9c6f7ad52b42f7a7bf"
PREDECESSOR_PLAN_PATH = Path(
    "protocols/execution_requests/cut3r_deform360_source_comparison_execution_v1.json"
)
PREDECESSOR_PLAN_FILE_SHA256 = "3ea10036bdd3d0b516d06ef210204cb7644e8b06cc8c2a3391c4742f3940ef7f"
PREDECESSOR_PLAN_ID = "0dbb6b3a46e2c895259fd5f4a4691c1d6d3c43b0e71774171bbfb3a20239953c"
SOURCE_FREEZE_ID = "5e739b92c2628c61fa99ae68da61d5814ca94d4b6de5720b75c4552de82d1b2c"
SMOKE_CASE_ID = "031-cotton-cloth-episode-0000-brics-odroid-010_cam0"
SMOKE_CASE_ID_SHA256 = "f306dc541c5c48e069c2ddc9942d75cdf537e81f65b028cbbcf35aaf410a9654"
DISPATCH_COMMAND = (
    "/prob4d-run-cut3r-source-comparison-v2 "
    f"{PREDECESSOR_SMOKE_ID} {PREDECESSOR_PLAN_ID} {SOURCE_FREEZE_ID}"
)
ADMISSION_MARKER = "<!-- prob4d-cut3r-source-comparison-v2-admitted -->"
COMPLETION_MARKER = "<!-- prob4d-cut3r-source-comparison-v2-complete -->"

Json = dict[str, Any]


class ContractError(RuntimeError):
    """Fail-closed prospective-execution contract error."""


def fail(message: str) -> NoReturn:
    raise ContractError(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        fail(f"expected a regular non-symbolic file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> Json:
    def reject_constant(value: str) -> None:
        fail(f"non-finite JSON constant in {label}: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> Json:
        result: Json = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"failed to load {label}: {error}")
    if type(value) is not dict:
        fail(f"{label} must be a JSON object")
    return value


def _require_fields(value: Json, expected: Json, *, label: str) -> None:
    for key, expected_value in expected.items():
        measured = value.get(key)
        if measured != expected_value:
            fail(f"{label} {key} mismatch: {measured!r} != {expected_value!r}")


def validate_predecessor_smoke(value: Json) -> None:
    _require_fields(
        value,
        {
            "schema": "prob4d.cut3r-source-comparison-smoke-result",
            "schema_version": 1,
            "artifact_id": PREDECESSOR_SMOKE_ID,
            "decision": "pre-science-technical-failure-no-retry",
            "attempt_count": 1,
            "execution_plan_id": PREDECESSOR_PLAN_ID,
            "source_freeze_id": SOURCE_FREEZE_ID,
            "case_id_sha256": SMOKE_CASE_ID_SHA256,
            "ordinary_success_count": 0,
            "retained_technical_failure_count": 1,
            "output_file_count_after_failure": 0,
            "retry_authorized": False,
            "retry_performed": False,
        },
        label="predecessor smoke",
    )
    failure = value.get("failure")
    if type(failure) is not dict:
        fail("predecessor smoke failure must be an object")
    _require_fields(
        failure,
        {
            "class": "ModuleNotFoundError",
            "message": "No module named 'dust3r'",
            "terminal_stage": "initialize-cut3r-runtime",
        },
        label="predecessor smoke failure",
    )
    boundary = value.get("information_boundary")
    if type(boundary) is not dict:
        fail("predecessor information_boundary must be an object")
    expected_false = {
        "source_rgb_frames_decoded",
        "cut3r_inference_executed",
        "source_predictions_written",
        "source_residuals_or_truth_opened",
        "candidate_reference_file_contents_opened",
        "target_payloads_opened",
        "target_outcomes_opened",
        "bayesian_phystwin_executed",
        "causal4d_executed",
    }
    if set(boundary) != expected_false:
        fail("predecessor information-boundary roster changed")
    if any(boundary[name] is not False for name in expected_false):
        fail("predecessor smoke exceeded the zero-information boundary")
    unsigned = dict(value)
    unsigned.pop("artifact_id")
    if _content_id(unsigned) != PREDECESSOR_SMOKE_ID:
        fail("predecessor smoke content identity is invalid")


def validate_predecessor_plan(value: Json) -> None:
    _require_fields(
        value,
        {
            "schema": "prob4d.cut3r-source-comparison-execution-plan",
            "schema_version": 1,
            "decision": "source-comparison-execution-authorized",
            "plan_id": PREDECESSOR_PLAN_ID,
            "source_freeze_id": SOURCE_FREEZE_ID,
        },
        label="predecessor execution plan",
    )
    unsigned = dict(value)
    unsigned.pop("plan_id")
    if _content_id(unsigned) != PREDECESSOR_PLAN_ID:
        fail("predecessor execution-plan content identity is invalid")
    execution = value.get("execution")
    if type(execution) is not dict:
        fail("predecessor execution section must be an object")
    _require_fields(
        execution,
        {
            "case_count": 40,
            "group_count": 10,
            "shard_count": 2,
            "failure_policy": "retain-once-no-replacement-no-retry-v1",
        },
        label="predecessor execution",
    )
    cases = value.get("cases")
    if type(cases) is not list or len(cases) != 40:
        fail("predecessor plan no longer contains 40 cases")
    matches = [row for row in cases if type(row) is dict and row.get("case_id") == SMOKE_CASE_ID]
    if len(matches) != 1 or matches[0].get("role") != "development":
        fail("the frozen v2 smoke case is not the predecessor development case")


def validate_runner_repair(text: str) -> None:
    required = (
        'for candidate in (checkout, checkout / "src"):',
        "from dust3r.inference import inference",
        "from dust3r.model import ARCroco3DStereo",
        "from dust3r.post_process import estimate_focal_knowing_depth",
        "from dust3r.utils.camera import pose_encoding_to_camera",
        "_retain_runtime_failure(",
    )
    missing = [token for token in required if token not in text]
    if missing:
        fail(f"localized CUT3R import repair is incomplete: {missing!r}")
    forbidden = (
        "from src.dust3r.inference",
        "from src.dust3r.model",
        "from src.dust3r.post_process",
        "from src.dust3r.utils.camera",
    )
    present = [token for token in forbidden if token in text]
    if present:
        fail(f"obsolete CUT3R import surface remains: {present!r}")


def validate_repository(repository: Path) -> Json:
    root = repository.resolve(strict=True)
    smoke_path = root / PREDECESSOR_SMOKE_PATH
    plan_path = root / PREDECESSOR_PLAN_PATH
    if _file_sha256(smoke_path) != PREDECESSOR_SMOKE_FILE_SHA256:
        fail("predecessor smoke file bytes changed")
    if _file_sha256(plan_path) != PREDECESSOR_PLAN_FILE_SHA256:
        fail("predecessor execution-plan file bytes changed")
    smoke = _load_json(smoke_path, label="predecessor smoke")
    plan = _load_json(plan_path, label="predecessor execution plan")
    validate_predecessor_smoke(smoke)
    validate_predecessor_plan(plan)
    runner = root / "scripts/science/run_cut3r_source_comparison.py"
    validate_runner_repair(runner.read_text(encoding="utf-8"))
    if hashlib.sha256(SMOKE_CASE_ID.encode("utf-8")).hexdigest() != SMOKE_CASE_ID_SHA256:
        fail("frozen smoke case identity changed")
    return {
        "predecessor_smoke_id": PREDECESSOR_SMOKE_ID,
        "predecessor_plan_id": PREDECESSOR_PLAN_ID,
        "source_freeze_id": SOURCE_FREEZE_ID,
        "smoke_case_id_sha256": SMOKE_CASE_ID_SHA256,
        "zero_information_predecessor": True,
        "localized_repair_present": True,
    }


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", os.fspath(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _request_json(method: str, path: str, payload: Json | None = None) -> Any:
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
            "User-Agent": "prob4d-cut3r-source-comparison-v2",
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


def _all_issue_comments() -> list[Json]:
    comments: list[Json] = []
    page = 1
    while True:
        payload = _request_json(
            "GET",
            f"/repos/{REPOSITORY}/issues/{ISSUE_NUMBER}/comments?per_page=100&page={page}",
        )
        if type(payload) is not list:
            fail("issue-comment response is not a list")
        rows = [row for row in payload if type(row) is dict]
        comments.extend(rows)
        if len(payload) < 100:
            return comments
        page += 1


def _comment(body: str) -> None:
    _request_json(
        "POST",
        f"/repos/{REPOSITORY}/issues/{ISSUE_NUMBER}/comments",
        {"body": body},
    )


def _write_output(name: str, value: str) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write(f"{name}={value}\n")


def admit(repository: Path) -> None:
    if os.environ.get("EVENT_COMMENT_BODY") != DISPATCH_COMMAND:
        fail("issue-comment command mismatch")
    if os.environ.get("EVENT_ACTOR") != "FlorianPfaff":
        fail("only FlorianPfaff may admit the v2 execution")
    expected_sha = os.environ["EXPECTED_SHA"]
    if re.fullmatch(r"[0-9a-f]{40}", expected_sha) is None:
        fail("expected revision is not an exact Git SHA")
    root = repository.resolve(strict=True)
    if _git(root, "rev-parse", "HEAD") != expected_sha:
        fail("checked-out default-branch revision mismatch")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        fail("authorization checkout is not clean")
    details = validate_repository(root)
    if any(ADMISSION_MARKER in str(row.get("body", "")) for row in _all_issue_comments()):
        fail("the prospective v2 execution was already admitted")
    _comment(
        "\n".join(
            (
                ADMISSION_MARKER,
                "## Prospective CUT3R source-comparison v2 admitted",
                "",
                f"- exact control-plane revision: `{expected_sha}`",
                f"- predecessor smoke ID: `{PREDECESSOR_SMOKE_ID}`",
                f"- predecessor plan ID: `{PREDECESSOR_PLAN_ID}`",
                f"- source-freeze ID: `{SOURCE_FREEZE_ID}`",
                f"- development-case identity: `{SMOKE_CASE_ID_SHA256}`",
                f"- command comment: {os.environ['COMMAND_COMMENT_URL']}",
                f"- workflow run: {os.environ['RUN_URL']}",
                "",
                "The predecessor stopped during Python import before video verification, "
                "RGB decoding, provider inference, prediction writing, source truth, or "
                "target access. Version 2 prospectively freezes the localized import "
                "repair. It will generate a new content-addressed plan before data access, "
                "run one development smoke, require independent custody, and only then "
                "run the two frozen source shards. No source truth, target payload, "
                "BayesianPhysTwin result, or Causal4D result is authorized.",
            )
        )
    )
    _write_output("authorized_sha", expected_sha)
    for key, value in details.items():
        _write_output(key, str(value).lower() if type(value) is bool else str(value))


def report() -> None:
    result = os.environ.get("EXECUTION_RESULT", "unknown")
    decision = os.environ.get("EXECUTION_DECISION", "not-produced")
    plan_id = os.environ.get("PLAN_ID", "not-produced")
    smoke_receipt = os.environ.get("SMOKE_RECEIPT_ID", "not-produced")
    shard_zero = os.environ.get("SHARD_ZERO_RECEIPT_ID", "not-produced")
    shard_one = os.environ.get("SHARD_ONE_RECEIPT_ID", "not-produced")
    success = result == "success" and decision == "source-predictions-custody-complete"
    boundary = (
        "The 40-case source predictions are sealed under independent custody. "
        "Source truth, physical residuals, candidate-reference contents, target "
        "payloads/outcomes, BayesianPhysTwin, and Causal4D remain unopened."
        if success
        else "No scientific performance claim is made. Source truth, target data, "
        "BayesianPhysTwin, and Causal4D remain unopened. A later stage may proceed "
        "only if the retained receipt explicitly satisfies its frozen gate."
    )
    _comment(
        "\n".join(
            (
                COMPLETION_MARKER,
                "## CUT3R source-comparison v2 terminal receipt",
                "",
                f"- execution result: `{result}`",
                f"- retained decision: `{decision}`",
                f"- generated plan ID: `{plan_id}`",
                f"- smoke custody receipt: `{smoke_receipt}`",
                f"- shard 0 custody receipt: `{shard_zero}`",
                f"- shard 1 custody receipt: `{shard_one}`",
                f"- workflow run: {os.environ['RUN_URL']}",
                "",
                boundary,
            )
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("self-test", "admit", "report"))
    parser.add_argument("--repository", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.phase == "self-test":
            print(json.dumps(validate_repository(args.repository), sort_keys=True))
        elif args.phase == "admit":
            admit(args.repository)
        else:
            report()
    except (ContractError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
