from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN_EXCEPTION = Path("scripts/science/build_cut3r_deform360_source_freeze.py")


def _literal_false(value: ast.expr) -> bool:
    return isinstance(value, ast.Constant) and value.value is False


def _numpy_load_can_unpickle(call: ast.Call) -> bool:
    for keyword in call.keywords:
        if keyword.arg == "allow_pickle":
            return not _literal_false(keyword.value)
    if len(call.args) >= 3:
        return not _literal_false(call.args[2])
    return False


def _unsafe_calls(path: Path) -> Counter[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    pickle_modules: set[str] = set()
    numpy_modules: set[str] = set()
    pickle_functions: set[str] = set()
    numpy_load_functions: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pickle":
                    pickle_modules.add(alias.asname or alias.name)
                elif alias.name == "numpy":
                    numpy_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "pickle":
                pickle_functions.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name in {"load", "loads"}
                )
            elif node.module == "numpy":
                numpy_load_functions.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "load"
                )

    result: Counter[str] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in pickle_modules
            and function.attr in {"load", "loads"}
        ) or (isinstance(function, ast.Name) and function.id in pickle_functions):
            result["unrestricted-pickle-load"] += 1
            continue
        numpy_load = (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in numpy_modules
            and function.attr == "load"
        ) or (isinstance(function, ast.Name) and function.id in numpy_load_functions)
        if numpy_load and _numpy_load_can_unpickle(node):
            result["numpy-load-allows-pickle"] += 1
    return result


def test_unsafe_deserialization_is_confined_to_the_frozen_source_builder() -> None:
    actual: dict[Path, Counter[str]] = {}
    for source_root in (ROOT / "src" / "prob4d", ROOT / "scripts"):
        for path in sorted(source_root.rglob("*.py")):
            findings = _unsafe_calls(path)
            if findings:
                actual[path.relative_to(ROOT)] = findings

    assert actual == {
        FROZEN_EXCEPTION: Counter({"numpy-load-allows-pickle": 1}),
    }
