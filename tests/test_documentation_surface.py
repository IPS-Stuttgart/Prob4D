from __future__ import annotations

from pathlib import Path

from scripts.ci.check_documentation_surface import check_documentation_surface


def _write_doc(root: Path, content: str, *, path: str = "docs/example.md") -> None:
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def test_removed_standalone_executable_is_rejected(tmp_path: Path) -> None:
    _write_doc(
        tmp_path,
        """# Example

```bash
prob4d-evaluate-provider protocol.json
```
""",
    )

    issues = check_documentation_surface(tmp_path)

    assert len(issues) == 1
    assert "removed Prob4D executable" in issues[0]


def test_unregistered_grouped_command_is_rejected(tmp_path: Path) -> None:
    _write_doc(
        tmp_path,
        """# Example

```bash
prob4d nonexistent command
```
""",
    )

    issues = check_documentation_surface(tmp_path)

    assert len(issues) == 1
    assert "unregistered grouped command" in issues[0]


def test_registered_grouped_command_is_accepted(tmp_path: Path) -> None:
    _write_doc(
        tmp_path,
        """# Example

```bash
prob4d diagnostic finite-sample-preflight \\
  promotion-lock.json
```
""",
    )

    assert check_documentation_surface(tmp_path) == ()


def test_removed_package_root_import_is_rejected(tmp_path: Path) -> None:
    _write_doc(
        tmp_path,
        """# Example

```python
from prob4d import Sim3
```
""",
    )

    issues = check_documentation_surface(tmp_path)

    assert len(issues) == 1
    assert "package-root import exposes removed names" in issues[0]


def test_current_api_v2_import_is_accepted(tmp_path: Path) -> None:
    _write_doc(
        tmp_path,
        """# Example

```python
from prob4d.api.v2 import Sim3
```
""",
    )

    assert check_documentation_surface(tmp_path) == ()


def test_unknown_api_v2_import_is_rejected(tmp_path: Path) -> None:
    _write_doc(
        tmp_path,
        """# Example

```python
from prob4d.api.v2 import DefinitelyNotPublic
```
""",
    )

    issues = check_documentation_surface(tmp_path)

    assert len(issues) == 1
    assert "names absent from prob4d.api.v2" in issues[0]


def test_explicit_implementation_module_import_is_accepted(tmp_path: Path) -> None:
    _write_doc(
        tmp_path,
        """# Example

```python
from prob4d.source_reliability import build_source_reliability_features
```
""",
    )

    assert check_documentation_surface(tmp_path) == ()


def test_historical_release_notes_are_excluded(tmp_path: Path) -> None:
    _write_doc(
        tmp_path,
        """# Historical release

```bash
prob4d-evaluate-provider protocol.json
```
""",
        path="docs/releases/0.4.1.md",
    )

    assert check_documentation_surface(tmp_path) == ()


def test_repository_documentation_matches_current_surface() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    assert check_documentation_surface(repository_root) == ()
