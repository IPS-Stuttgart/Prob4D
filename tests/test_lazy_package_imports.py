from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import prob4d


def _run_probe(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_root_import_is_minimal_and_does_not_load_implementations() -> None:
    _run_probe(
        """
        import sys
        import prob4d

        forbidden = {
            "prob4d.alignment_cycles",
            "prob4d.calibration",
            "prob4d.causal_gauge_graph",
            "prob4d.fusion",
            "prob4d.observation_export",
            "prob4d.prediction_store",
            "prob4d.sim3",
            "prob4d.source_reliability",
        }
        loaded = forbidden.intersection(sys.modules)
        if loaded:
            raise SystemExit(f"eager Prob4D imports: {sorted(loaded)}")
        if prob4d.__all__ != ["__version__"]:
            raise SystemExit(f"unexpected root exports: {prob4d.__all__!r}")
        if "Sim3" in prob4d.__dict__ or "Sim3" in dir(prob4d):
            raise SystemExit("implementation symbol leaked into the package root")
        if hasattr(prob4d, "_LAZY_EXPORTS"):
            raise SystemExit("historical lazy-root machinery is still installed")
        """
    )


def test_current_api_is_explicit_and_does_not_mutate_package_root() -> None:
    _run_probe(
        """
        import prob4d
        from prob4d.api.v2 import Sim3

        if Sim3.__module__ != "prob4d.sim3":
            raise SystemExit("api.v2 exported an unexpected Sim3 implementation")
        if "Sim3" in prob4d.__dict__:
            raise SystemExit("api.v2 import populated the package root")
        """
    )


def test_package_root_typing_stub_contains_only_version() -> None:
    stub_path = Path(prob4d.__file__).with_suffix(".pyi")
    assert stub_path.is_file()
    tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    annotations = [
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert imports == []
    assert annotations == ["__version__"]


def test_removed_root_attribute_raises_standard_attribute_error() -> None:
    with pytest.raises(
        AttributeError,
        match="module 'prob4d' has no attribute 'Sim3'",
    ):
        _ = prob4d.Sim3
