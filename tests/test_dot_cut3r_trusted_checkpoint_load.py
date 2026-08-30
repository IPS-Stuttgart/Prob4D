from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_dot_rope_cut3r_native_provider.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dot_cut3r_provider", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_checkpoint_loading_is_sha_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"not-the-trusted-checkpoint")
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)

    with pytest.raises(ValueError, match="trusted CUT3R checkpoint"):
        module._authorize_trusted_legacy_checkpoint_load(checkpoint)
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ

    monkeypatch.setattr(
        module,
        "_sha256",
        lambda _path: module.TRUSTED_CUT3R_CHECKPOINT_SHA256,
    )
    module._authorize_trusted_legacy_checkpoint_load(checkpoint)
    assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_model_runtime_calls_the_sha_bound_authorizer() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    constructor = text[text.index("class NativeRopeCut3RRuntime") : text.index("    def _reset_seed")]

    assert "_authorize_trusted_legacy_checkpoint_load(self.checkpoint)" in constructor
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in constructor
    assert "weights_only=False" not in constructor
