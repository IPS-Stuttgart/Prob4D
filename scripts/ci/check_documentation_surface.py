#!/usr/bin/env python3
"""Reject active documentation that drifts from the Prob4D 0.5 public surface."""

from __future__ import annotations

import argparse
import ast
import re
import shlex
from collections.abc import Iterable, Sequence
from pathlib import Path

from prob4d.command_registry import COMMANDS
from prob4d.public_api_manifest import build_public_api_manifest

_FENCE = re.compile(r"^\s*```(?P<language>[A-Za-z0-9_+-]*)\s*$")
_SHELL_LANGUAGES = frozenset({"bash", "console", "sh", "shell", "text"})
_PYTHON_LANGUAGES = frozenset({"py", "python", "python3"})
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")

# Prob4D 0.5 deliberately installs only the grouped ``prob4d`` executable.
_REMOVED_EXECUTABLES = frozenset(
    {
        "prob4d-ablate",
        "prob4d-ablate-provider-v2-gauge",
        "prob4d-benchmark",
        "prob4d-evaluate-provider",
        "prob4d-export-calibrated-observation-belief",
        "prob4d-export-exploratory-observation-belief",
        "prob4d-export-observation-belief",
        "prob4d-finite-sample-preflight",
        "prob4d-motioncrafter",
        "prob4d-phystwin",
        "prob4d-phystwin-state",
        "prob4d-phystwin-uncertainty",
        "prob4d-provider-manifest",
        "prob4d-sintel-uncertainty",
        "prob4d-target-admit",
        "prob4d-target-verify",
        "prob4d-validate-observation",
        "prob4d-vggt-baseline",
        "prob4d-visual-bias-calibration",
    }
)


def _active_markdown_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    readme = root / "README.md"
    if readme.is_file():
        files.append(readme)
    docs = root / "docs"
    if docs.is_dir():
        for path in docs.rglob("*.md"):
            relative = path.relative_to(docs)
            if relative.parts and relative.parts[0] == "releases":
                continue
            files.append(path)
    return tuple(sorted(files))


def _fenced_blocks(text: str) -> Iterable[tuple[str, int, str]]:
    language: str | None = None
    start_line = 0
    lines: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _FENCE.match(line)
        if match is None:
            if language is not None:
                lines.append(line)
            continue
        if language is None:
            language = match.group("language").lower()
            start_line = line_number + 1
            lines = []
        else:
            yield language, start_line, "\n".join(lines)
            language = None
            start_line = 0
            lines = []


def _logical_shell_lines(block: str, *, start_line: int) -> Iterable[tuple[int, str]]:
    pending: list[str] = []
    pending_line = start_line
    for offset, raw_line in enumerate(block.splitlines(), start=0):
        line_number = start_line + offset
        stripped = raw_line.strip()
        if not pending:
            pending_line = line_number
        if stripped.endswith("\\"):
            pending.append(stripped[:-1].rstrip())
            continue
        pending.append(stripped)
        logical = " ".join(part for part in pending if part)
        if logical:
            yield pending_line, logical
        pending = []
    if pending:
        logical = " ".join(part for part in pending if part)
        if logical:
            yield pending_line, logical


def _command_tokens(line: str) -> list[str]:
    stripped = line.lstrip()
    if stripped.startswith("$ "):
        stripped = stripped[2:].lstrip()
    try:
        tokens = shlex.split(stripped, comments=True, posix=True)
    except ValueError:
        return []
    while tokens and (_ASSIGNMENT.match(tokens[0]) or tokens[0] == "env"):
        tokens.pop(0)
    while tokens and _ASSIGNMENT.match(tokens[0]):
        tokens.pop(0)
    return tokens


def _registered_routes() -> tuple[tuple[str, ...], ...]:
    return tuple(spec.route for spec in COMMANDS)


def _check_shell_block(
    path: Path,
    block: str,
    *,
    start_line: int,
) -> list[str]:
    issues: list[str] = []
    routes = _registered_routes()
    for line_number, line in _logical_shell_lines(block, start_line=start_line):
        tokens = _command_tokens(line)
        if not tokens:
            continue
        executable = tokens[0]
        if executable in _REMOVED_EXECUTABLES:
            issues.append(
                f"{path}:{line_number}: removed Prob4D executable {executable!r}; "
                "use the grouped 'prob4d' command"
            )
            continue
        if executable != "prob4d":
            continue
        arguments = tokens[1:]
        if not arguments or arguments[0].startswith("-"):
            continue
        if not any(
            len(arguments) >= len(route) and tuple(arguments[: len(route)]) == route
            for route in routes
        ):
            issues.append(
                f"{path}:{line_number}: undocumented or unregistered grouped command "
                f"{' '.join(tokens)!r}"
            )
    return issues


def _api_v2_exports() -> frozenset[str]:
    manifest = build_public_api_manifest()
    surfaces = manifest.get("surfaces")
    if not isinstance(surfaces, dict):
        raise RuntimeError("public API manifest has no surfaces mapping")
    api = surfaces.get("api_v2")
    if not isinstance(api, dict):
        raise RuntimeError("public API manifest has no api_v2 surface")
    exports = api.get("exports")
    if not isinstance(exports, list) or not all(isinstance(item, str) for item in exports):
        raise RuntimeError("public API manifest api_v2 exports are malformed")
    return frozenset(exports)


def _check_python_block(
    path: Path,
    block: str,
    *,
    start_line: int,
) -> list[str]:
    issues: list[str] = []
    try:
        module = ast.parse(block)
    except SyntaxError:
        # Many research snippets intentionally contain symbolic placeholders.
        # Import-surface enforcement applies to syntactically complete examples.
        return issues
    api_v2_exports = _api_v2_exports()
    for node in ast.walk(module):
        if not isinstance(node, ast.ImportFrom):
            continue
        line_number = start_line + node.lineno - 1
        if node.module == "prob4d":
            imported = {alias.name for alias in node.names}
            unsupported = sorted(imported - {"__version__"})
            if unsupported:
                issues.append(
                    f"{path}:{line_number}: package-root import exposes removed names: "
                    + ", ".join(unsupported)
                )
        elif node.module == "prob4d.api.v2":
            imported = {alias.name for alias in node.names}
            if "*" in imported:
                issues.append(
                    f"{path}:{line_number}: wildcard imports from prob4d.api.v2 are forbidden"
                )
                continue
            unsupported = sorted(imported - api_v2_exports)
            if unsupported:
                issues.append(
                    f"{path}:{line_number}: names absent from prob4d.api.v2: "
                    + ", ".join(unsupported)
                )
    return issues


def check_documentation_surface(root: str | Path) -> tuple[str, ...]:
    repository_root = Path(root).resolve()
    issues: list[str] = []
    for path in _active_markdown_files(repository_root):
        relative = path.relative_to(repository_root)
        text = path.read_text(encoding="utf-8")
        for language, start_line, block in _fenced_blocks(text):
            if language in _SHELL_LANGUAGES:
                issues.extend(
                    _check_shell_block(relative, block, start_line=start_line)
                )
            elif language in _PYTHON_LANGUAGES:
                issues.extend(
                    _check_python_block(relative, block, start_line=start_line)
                )
    return tuple(issues)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "validate active Markdown examples against the Prob4D grouped CLI "
            "and generated Python public API manifest"
        )
    )
    parser.add_argument(
        "repository_root",
        nargs="?",
        default=Path(__file__).resolve().parents[2],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    issues = check_documentation_surface(arguments.repository_root)
    if issues:
        print("Documentation public-surface drift detected:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("Documentation public surface matches the Prob4D 0.5 registry and API manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
