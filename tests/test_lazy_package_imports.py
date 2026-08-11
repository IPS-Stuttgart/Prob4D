from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from importlib import import_module
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


def test_root_import_does_not_eagerly_load_public_implementation_modules() -> None:
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
        if "Sim3" in prob4d.__dict__:
            raise SystemExit("lazy export was populated before first access")
        if "Sim3" not in prob4d.__all__:
            raise SystemExit("Sim3 is absent from the historical public inventory")
        if "Sim3" not in dir(prob4d):
            raise SystemExit("lazy export is absent from module introspection")
        """
    )


def test_lazy_root_export_is_loaded_once_and_cached() -> None:
    _run_probe(
        """
        import prob4d
        from prob4d.sim3 import Sim3

        if "Sim3" in prob4d.__dict__:
            raise SystemExit("root export was populated by the submodule import")
        if prob4d.Sim3 is not Sim3:
            raise SystemExit("lazy export differs from the owning module object")
        if prob4d.__dict__["Sim3"] is not Sim3:
            raise SystemExit("lazy export was not cached in the package root")
        if prob4d.Sim3 is not Sim3:
            raise SystemExit("cached export identity changed on repeated access")
        """
    )


def test_lazy_export_inventory_is_complete_unique_and_resolvable() -> None:
    assert len(prob4d.__all__) == len(set(prob4d.__all__))
    assert set(prob4d.__all__) == set(prob4d._LAZY_EXPORTS) | {"__version__"}

    for name, module_name in prob4d._LAZY_EXPORTS.items():
        owner = import_module(module_name)
        assert getattr(prob4d, name) is getattr(owner, name)


def test_package_root_typing_stub_covers_every_lazy_export() -> None:
    stub_path = Path(prob4d.__file__).with_suffix(".pyi")
    assert stub_path.is_file()
    tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    stub_exports: set[str] = set()
    has_version_annotation = False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.asname == alias.name
                stub_exports.add(alias.name)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__version__"
        ):
            has_version_annotation = True
    assert stub_exports == set(prob4d.__all__) - {"__version__"}
    assert has_version_annotation


def test_unknown_root_attribute_raises_standard_attribute_error() -> None:
    with pytest.raises(
        AttributeError,
        match="module 'prob4d' has no attribute 'definitely_not_an_export'",
    ):
        prob4d.definitely_not_an_export
