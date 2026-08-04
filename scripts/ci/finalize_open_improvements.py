from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPOSITORY_ROOT = Path.cwd()
PYTHON = sys.executable


def run(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(arguments), flush=True)
    return subprocess.run(
        arguments,
        cwd=REPOSITORY_ROOT,
        check=check,
        text=True,
    )


def read(path: str) -> str:
    return (REPOSITORY_ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    (REPOSITORY_ROOT / path).write_text(content, encoding="utf-8")


def checkout(branch: str) -> None:
    run("git", "reset", "--hard")
    run("git", "clean", "-fdx")
    run(
        "git",
        "fetch",
        "origin",
        "+refs/heads/main:refs/remotes/origin/main",
        f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
    )
    run("git", "checkout", "-B", branch, f"origin/{branch}")
    run("git", "merge", "--no-edit", "origin/main")


def install_development_environment() -> None:
    run(PYTHON, "-m", "pip", "install", "--upgrade", "pip")
    run(PYTHON, "-m", "pip", "install", "-e", ".[dev]")
    run(PYTHON, "-m", "pip", "check")


def clean_generated_files() -> None:
    for path in (
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
        "src/prob4d.egg-info",
    ):
        candidate = REPOSITORY_ROOT / path
        if candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.exists():
            candidate.unlink()
    run("git", "clean", "-fdX")


def commit_and_push(branch: str, message: str) -> None:
    clean_generated_files()
    run("git", "diff", "--check")
    run("git", "add", "-u")
    staged = subprocess.run(
        ("git", "diff", "--cached", "--quiet"),
        cwd=REPOSITORY_ROOT,
        check=False,
    ).returncode
    if staged:
        run("git", "commit", "-m", message)
    run("git", "push", "origin", f"HEAD:{branch}")


def extract_python_payload(
    workflow_path: str,
    *,
    start_marker: str,
    end_marker: str,
) -> str:
    workflow = read(workflow_path)
    start = workflow.index(start_marker)
    end = workflow.index(end_marker, start)
    return textwrap.dedent(workflow[start:end])


def finalize_joint_gauge() -> None:
    branch = "agent/joint-gauge-tracklet-evidence"
    checkout(branch)
    run(PYTHON, ".github/scripts/harden_joint_gauge_tracklet_evidence.py")
    run(PYTHON, ".github/scripts/harden_base_tracklet_ranking.py")

    path = "src/prob4d/cross_window_tracklets.py"
    text = read(path)
    if "from typing import Any, Literal, TypeAlias\n" not in text:
        text = text.replace(
            "from typing import Any, Literal\n",
            "from typing import Any, Literal, TypeAlias\n",
            1,
        )
    aliases = "FloatArray = NDArray[np.floating]\nIntArray = NDArray[np.integer]\n"
    typed_aliases = (
        "FloatArray: TypeAlias = NDArray[np.floating[Any]]\n"
        "IntArray: TypeAlias = NDArray[np.integer[Any]]\n"
    )
    if aliases in text:
        text = text.replace(aliases, typed_aliases, 1)
    if typed_aliases not in text:
        raise RuntimeError("joint-gauge TypeAlias hardening was not applied")
    write(path, text)

    run(
        PYTHON,
        "-m",
        "ruff",
        "check",
        "--fix",
        "src/prob4d/cross_window_tracklets.py",
        "src/prob4d/cross_window_tracklet_evidence.py",
        "tests/test_cross_window_tracklets.py",
        "tests/test_cross_window_tracklet_evidence.py",
    )
    run(
        PYTHON,
        "-m",
        "ruff",
        "format",
        "src/prob4d/cross_window_tracklets.py",
        "src/prob4d/cross_window_tracklet_evidence.py",
        "tests/test_cross_window_tracklets.py",
        "tests/test_cross_window_tracklet_evidence.py",
    )
    run(
        PYTHON,
        "-m",
        "ruff",
        "check",
        "src/prob4d/cross_window_tracklets.py",
        "src/prob4d/cross_window_tracklet_evidence.py",
        "tests/test_cross_window_tracklets.py",
        "tests/test_cross_window_tracklet_evidence.py",
    )
    run(
        PYTHON,
        "-m",
        "mypy",
        "src/prob4d/cross_window_tracklets.py",
        "src/prob4d/cross_window_tracklet_evidence.py",
    )
    run(
        PYTHON,
        "-m",
        "pytest",
        "-q",
        "tests/test_cross_window_tracklet_evidence.py",
        "tests/test_cross_window_tracklets.py",
        "tests/test_causal_tracklets.py",
    )

    for temporary in (
        ".github/scripts/harden_joint_gauge_tracklet_evidence.py",
        ".github/scripts/harden_base_tracklet_ranking.py",
        ".github/workflows/hosted-publish-pr92.yml",
    ):
        candidate = REPOSITORY_ROOT / temporary
        if candidate.exists():
            candidate.unlink()
    commit_and_push(branch, "Publish covariance-safe joint-gauge tracklet evidence")


def finalize_tracklet_contracts() -> None:
    branch = "agent/harden-tracklet-association-contracts"
    checkout(branch)
    workflow_path = ".github/workflows/hosted-repair-pr94.yml"
    payload = extract_python_payload(
        workflow_path,
        start_marker="          from pathlib import Path\n",
        end_marker="          PY\n      - name: Validate focused contracts",
    )
    exec(compile(payload, workflow_path, "exec"), {"__name__": "__main__"})

    run(
        PYTHON,
        "-m",
        "ruff",
        "check",
        "src/prob4d/causal_tracklets.py",
        "src/prob4d/cross_window_tracklets.py",
        "tests/test_causal_tracklets.py",
        "tests/test_cross_window_tracklets.py",
    )
    run(
        PYTHON,
        "-m",
        "mypy",
        "src/prob4d/causal_tracklets.py",
        "src/prob4d/cross_window_tracklets.py",
    )
    run(
        PYTHON,
        "-m",
        "pytest",
        "-q",
        "tests/test_causal_tracklets.py",
        "tests/test_cross_window_tracklets.py",
    )
    commit_and_push(branch, "Harden tracklet seeds and length-neutral support")


def finalize_material_identity() -> None:
    branch = "agent/material-identity-hypothesis-stream"
    checkout(branch)
    workflow_path = ".github/workflows/hosted-repair-pr95.yml"
    payload = extract_python_payload(
        workflow_path,
        start_marker="          from pathlib import Path\n",
        end_marker="          PY\n      - name: Validate focused contract",
    )
    exec(compile(payload, workflow_path, "exec"), {"__name__": "__main__"})

    run(
        PYTHON,
        "-m",
        "ruff",
        "check",
        "src/prob4d/material_identity_stream.py",
        "tests/test_material_identity_stream.py",
    )
    run(
        PYTHON,
        "-m",
        "mypy",
        "src/prob4d/material_identity_stream.py",
    )
    run(
        PYTHON,
        "-m",
        "pytest",
        "-q",
        "tests/test_material_identity_stream.py",
        "tests/test_causal_tracklets.py",
        "tests/test_cross_window_tracklets.py",
    )
    commit_and_push(branch, "Harden material-identity provenance and validation")


def finalize_selection_evidence() -> None:
    branch = "agent/replayable-selection-evidence"
    checkout(branch)

    run(
        PYTHON,
        "-m",
        "ruff",
        "check",
        "--fix",
        "src/prob4d/_selection_evidence_common.py",
        "src/prob4d/_selection_evidence_records.py",
        "src/prob4d/_selection_evidence_replay.py",
        "src/prob4d/selection_evidence.py",
        "tests/test_selection_evidence.py",
        "tests/test_selection_evidence_claim_boundary.py",
    )
    run(
        PYTHON,
        "-m",
        "ruff",
        "format",
        "src/prob4d/_selection_evidence_common.py",
        "src/prob4d/_selection_evidence_records.py",
        "src/prob4d/_selection_evidence_replay.py",
        "src/prob4d/selection_evidence.py",
        "tests/test_selection_evidence.py",
        "tests/test_selection_evidence_claim_boundary.py",
    )
    run(
        PYTHON,
        "-m",
        "ruff",
        "check",
        "src/prob4d/_selection_evidence_common.py",
        "src/prob4d/_selection_evidence_records.py",
        "src/prob4d/_selection_evidence_replay.py",
        "src/prob4d/selection_evidence.py",
        "tests/test_selection_evidence.py",
        "tests/test_selection_evidence_claim_boundary.py",
    )
    run(
        PYTHON,
        "-m",
        "mypy",
        "src/prob4d/_selection_evidence_common.py",
        "src/prob4d/_selection_evidence_records.py",
        "src/prob4d/_selection_evidence_replay.py",
        "src/prob4d/selection_evidence.py",
    )
    run(
        PYTHON,
        "-m",
        "pytest",
        "-q",
        "tests/test_selection_evidence.py",
        "tests/test_selection_evidence_claim_boundary.py",
    )
    commit_and_push(branch, "Refresh replayable selection evidence on current main")


def main() -> int:
    run("git", "config", "user.name", "github-actions[bot]")
    run(
        "git",
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
    )
    install_development_environment()
    finalize_joint_gauge()
    finalize_tracklet_contracts()
    finalize_material_identity()
    finalize_selection_evidence()
    print("All focused improvement branches validated and published.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
