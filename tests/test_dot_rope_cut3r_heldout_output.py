from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_dot_rope_cut3r_heldout_confirmation.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("dot_heldout_confirmation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precreated_empty_output_is_removed_for_exclusive_provider_creation(
    tmp_path: Path,
) -> None:
    module = _load_script()
    output = tmp_path / "provider"
    output.mkdir()

    assert module._prepare_exclusive_output(output) == output
    assert not output.exists()


def test_nonempty_or_symbolic_output_fails_closed(tmp_path: Path) -> None:
    module = _load_script()
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "sentinel").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        module._prepare_exclusive_output(nonempty)
    assert (nonempty / "sentinel").read_text(encoding="utf-8") == "preserve"

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="symbolic link"):
        module._prepare_exclusive_output(link)
