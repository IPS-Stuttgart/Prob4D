from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_dot_rope_cut3r_native_provider.py"


class _FakeFrame:
    def save(self, path: Path) -> None:
        path.write_bytes(b"synthetic-frame")


class _FakeDrawer:
    def line(self, *_args: object, **_kwargs: object) -> None:
        return None

    def ellipse(self, *_args: object, **_kwargs: object) -> None:
        return None


class _FakeImage:
    @staticmethod
    def fromarray(_array: object, *, mode: str) -> _FakeFrame:
        assert mode == "RGB"
        return _FakeFrame()


class _FakeImageDraw:
    @staticmethod
    def Draw(_frame: _FakeFrame) -> _FakeDrawer:  # noqa: N802 - mirrors Pillow API
        return _FakeDrawer()


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dot_rope_cut3r_provider", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_smoke_uses_an_uncreated_child_directory() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert '_make_synthetic_frames(Path(temporary) / "frames", count=3)' in text
    assert "_make_synthetic_frames(Path(temporary), count=3)" not in text


def test_synthetic_frames_are_created_inside_fresh_child_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pil = ModuleType("PIL")
    setattr(fake_pil, "Image", _FakeImage)
    setattr(fake_pil, "ImageDraw", _FakeImageDraw)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

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
