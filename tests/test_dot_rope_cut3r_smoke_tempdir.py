from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_dot_rope_cut3r_native_provider.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("dot_rope_cut3r_provider", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_smoke_uses_an_uncreated_child_directory() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert '_make_synthetic_frames(\n                Path(temporary) / "frames", count=3' in text
    assert "_make_synthetic_frames(Path(temporary), count=3)" not in text


def test_synthetic_frames_are_created_inside_fresh_child_directory(tmp_path: Path) -> None:
    module = _load_script()
    with tempfile.TemporaryDirectory(dir=tmp_path, prefix="dot-cut3r-smoke-") as temporary:
        root = Path(temporary)
        destination = root / "frames"
        assert root.is_dir()
        assert not destination.exists()
        paths = module._make_synthetic_frames(destination, count=3)

        assert destination.is_dir()
        assert [path.name for path in paths] == [
            "synthetic-00.png",
            "synthetic-01.png",
            "synthetic-02.png",
        ]
        assert all(path.is_file() and path.parent == destination for path in paths)
