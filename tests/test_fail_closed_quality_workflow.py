from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "fail-closed-quality.yml"
_BLOCK_RUN = re.compile(r"^(?P<indent>\s*)run:\s*(?:[|>][-+]?)\s*$")
_INLINE_RUN = re.compile(r"^\s*run:\s*(?P<command>.+)$")


def _piped_run_block_errors(text: str, *, source: str) -> list[str]:
    lines = text.splitlines()
    errors: list[str] = []
    for index, line in enumerate(lines):
        block_match = _BLOCK_RUN.match(line)
        if block_match is not None:
            indentation = len(block_match.group("indent"))
            block: list[str] = []
            for candidate in lines[index + 1 :]:
                if not candidate.strip():
                    block.append(candidate)
                    continue
                stripped = candidate.lstrip()
                candidate_indentation = len(candidate) - len(stripped)
                if candidate_indentation <= indentation:
                    break
                block.append(candidate)
            command = "\n".join(block)
        else:
            inline_match = _INLINE_RUN.match(line)
            if inline_match is None:
                continue
            command = inline_match.group("command")
        if "| tee" not in command:
            continue
        if "set -o pipefail" in command or "set -euo pipefail" in command:
            continue
        errors.append(
            f"{source}:{index + 1}: a command piped through tee must "
            "enable pipefail"
        )
    return errors


def test_authoritative_quality_workflow_is_read_only_and_fail_closed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "name: Fail-closed quality" in text
    assert "permissions:\n  contents: read" in text
    assert "continue-on-error" not in text
    assert _piped_run_block_errors(
        text,
        source=WORKFLOW.relative_to(ROOT).as_posix(),
    ) == []


def test_authoritative_quality_workflow_covers_current_stable_surfaces() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    required = (
        "src/prob4d/_version.py",
        "src/prob4d/_provider_export_core.py",
        "src/prob4d/api/v2.py",
        "src/prob4d/provider_v1.py",
        "src/prob4d/provider_v2_factor_bundle.py",
        "src/prob4d/provider_v2_factors.py",
        "src/prob4d/public_api_manifest.py",
        "src/prob4d/sparse_observation_factors.py",
        "src/prob4d/source_diagnostics.py",
    )
    for path in required:
        assert path in text
    assert "src/prob4d/api/v1.py" not in text
    assert "python -m pip install -r requirements/ci/quality.txt" in text
    assert "python -m pip install --no-deps -e ." in text
    assert 'python -m pip install "numpy>=1.24,<2.3"' not in text


def test_pipefail_policy_rejects_masked_diagnostic_pipelines() -> None:
    unsafe_inline = "run: command 2>&1 | tee diagnostics.txt\n"
    safe_inline = "run: set -o pipefail; command 2>&1 | tee diagnostics.txt\n"
    unsafe_block = "run: |\n  command 2>&1 | tee diagnostics.txt\n"
    safe_block = (
        "run: |\n"
        "  set -o pipefail\n"
        "  command 2>&1 | tee diagnostics.txt\n"
    )

    assert len(_piped_run_block_errors(unsafe_inline, source="fixture.yml")) == 1
    assert _piped_run_block_errors(safe_inline, source="fixture.yml") == []
    assert len(_piped_run_block_errors(unsafe_block, source="fixture.yml")) == 1
    assert _piped_run_block_errors(safe_block, source="fixture.yml") == []
