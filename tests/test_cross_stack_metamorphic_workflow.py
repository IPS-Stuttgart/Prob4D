from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "cross-stack-metamorphic.yml"
BAYESIAN_PHYSTWIN_PIN = "c41974ad5583e2d426d12cf0afd3274b6a47d9b6"
CAUSAL4D_PIN = "d5260f65fb3c7660a576ac4c8742190406247103"


def test_cross_stack_workflow_is_hosted_read_only_and_exactly_pinned() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "permissions:\n  contents: read" in text
    assert "secrets." not in text
    assert text.count("persist-credentials: false") == 3
    assert f"BAYESIAN_PHYSTWIN_REF: {BAYESIAN_PHYSTWIN_PIN}" in text
    assert f"CAUSAL4D_REF: {CAUSAL4D_PIN}" in text
    assert "PROB4D_REQUIRE_CROSS_STACK_METAMORPHIC: \"1\"" in text
    assert "actions/checkout@v" not in text
    assert "actions/setup-python@v" not in text
    assert "actions/upload-artifact@v" not in text


def test_cross_stack_workflow_uses_only_installed_wheels() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count("python -m build --wheel") == 3
    assert "test \"$(find \"${RUNNER_TEMP}/wheelhouse\"" in text
    assert "\"${python_bin}\" -m pip install pytest \"${RUNNER_TEMP}\"/wheelhouse/*.whl" in text
    assert "PYTHONNOUSERSITE=1" in text
    assert "env -u PYTHONPATH" in text
    assert "\"${RUNNER_TEMP}/cross-stack-metamorphic/bin/python\" -I -m pytest" in text
    assert "test_three_repository_metamorphic_v1.py" in text


def test_cross_stack_workflow_retains_replayable_diagnostics() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "cross-stack-metamorphic.log" in text
    assert "cross-stack-metamorphic.xml" in text
    assert "wheel-sha256.txt" in text
    assert "if-no-files-found: error" in text
    assert "retention-days: 30" in text


def test_cross_stack_paths_trigger_pull_request_and_push_runs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    pull_request, remainder = text.split("  push:\n", maxsplit=1)
    push, _ = remainder.split("  workflow_dispatch:\n", maxsplit=1)

    for path_filter in (
        '      - ".github/workflows/cross-stack-metamorphic.yml"',
        '      - "integration_tests/test_three_repository_metamorphic_v1.py"',
        '      - "tests/test_cross_stack_metamorphic_workflow.py"',
    ):
        assert path_filter in pull_request
        assert path_filter in push
        assert text.count(path_filter) == 2
