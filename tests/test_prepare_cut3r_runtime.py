from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/prepare_cut3r_runtime.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_cut3r_runtime", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_fixture(monkeypatch, module, tmp_path: Path) -> tuple[Path, Path, bytes]:
    checkout = tmp_path / "CUT3R"
    model_path = checkout / "src/dust3r/model.py"
    model_path.parent.mkdir(parents=True)
    source = (
        b"import torch\n\n"
        b"def load_model(model_path):\n"
        b'    ckpt = torch.load(model_path, map_location="cpu")\n'
        b"    return ckpt\n"
    )
    model_path.write_bytes(source)

    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"exact trusted checkpoint fixture")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    monkeypatch.setattr(
        module,
        "_TRUSTED_MODEL_GIT_BLOB_SHA1",
        module._git_blob_sha1(source),
    )
    monkeypatch.setattr(module, "_TRUSTED_CHECKPOINT_SHA256", checkpoint_sha256)
    monkeypatch.setenv("REQUEST_PATH", module._TRUSTED_REQUEST_PATH)
    monkeypatch.setenv("CUT3R_REVISION", module._TRUSTED_CUT3R_REVISION)
    monkeypatch.setenv("CUT3R_CHECKPOINT_SHA256", checkpoint_sha256)
    monkeypatch.setenv("CUT3R_RUNTIME_CHECKPOINT", str(checkpoint))
    return checkout, model_path, source


def _configure_smoke_fixture(
    monkeypatch,
    module,
    tmp_path: Path,
) -> tuple[Path, Path, bytes]:
    repository_root = tmp_path / "repository"
    source_path = repository_root / module._TRUSTED_PROVIDER_SCRIPT_RELATIVE_PATH
    source_path.parent.mkdir(parents=True)
    source = (
        b"with tempfile.TemporaryDirectory(prefix=\"dot-cut3r-smoke-\") as temporary:\n"
        b"            frame_paths = _make_synthetic_frames(Path(temporary), count=3)\n"
        b"            prediction = runtime.infer(frame_paths, image_size=512)\n"
    )
    source_path.write_bytes(source)
    monkeypatch.setattr(
        module,
        "_TRUSTED_PROVIDER_SCRIPT_GIT_BLOB_SHA1",
        module._git_blob_sha1(source),
    )
    monkeypatch.setenv("REQUEST_PATH", module._TRUSTED_REQUEST_PATH)
    monkeypatch.setenv("CUT3R_REVISION", module._TRUSTED_CUT3R_REVISION)
    monkeypatch.setenv(
        "CUT3R_CHECKPOINT_SHA256",
        module._TRUSTED_CHECKPOINT_SHA256,
    )
    return repository_root, source_path, source


def test_trusted_checkpoint_compatibility_is_hash_bound(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script()
    checkout, model_path, source = _configure_fixture(monkeypatch, module, tmp_path)

    record = module._prepare_trusted_checkpoint_compatibility(checkout.resolve())

    assert record is not None
    assert record["status"] == "trusted-legacy-checkpoint-loader-enabled"
    assert record["source_git_blob_sha1"] == module._git_blob_sha1(source)
    assert record["checkpoint_sha256"] == module._TRUSTED_CHECKPOINT_SHA256
    patched = model_path.read_text(encoding="utf-8")
    assert patched.count("weights_only=False") == 1
    unsigned = dict(record)
    artifact_id = unsigned.pop("artifact_id")
    assert artifact_id == module._content_id(unsigned)


def test_trusted_checkpoint_compatibility_rejects_source_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script()
    checkout, model_path, _ = _configure_fixture(monkeypatch, module, tmp_path)
    model_path.write_text("unexpected source\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source bytes changed"):
        module._prepare_trusted_checkpoint_compatibility(checkout.resolve())


def test_trusted_smoke_workspace_compatibility_is_hash_bound(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script()
    repository_root, source_path, source = _configure_smoke_fixture(
        monkeypatch,
        module,
        tmp_path,
    )

    record = module._prepare_trusted_smoke_workspace_compatibility(repository_root)

    assert record is not None
    assert record["status"] == "trusted-smoke-child-workspace-enabled"
    assert record["source_git_blob_sha1"] == module._git_blob_sha1(source)
    patched = source_path.read_text(encoding="utf-8")
    assert '_make_synthetic_frames(Path(temporary) / "frames", count=3)' in patched
    assert "_make_synthetic_frames(Path(temporary), count=3)" not in patched
    unsigned = dict(record)
    artifact_id = unsigned.pop("artifact_id")
    assert artifact_id == module._content_id(unsigned)


def test_trusted_smoke_workspace_compatibility_rejects_source_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script()
    repository_root, source_path, _ = _configure_smoke_fixture(
        monkeypatch,
        module,
        tmp_path,
    )
    source_path.write_text("unexpected source\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="provider source bytes changed"):
        module._prepare_trusted_smoke_workspace_compatibility(repository_root)


def test_unrelated_runtime_does_not_patch(monkeypatch, tmp_path: Path) -> None:
    module = _load_script()
    monkeypatch.delenv("REQUEST_PATH", raising=False)

    assert module._prepare_trusted_checkpoint_compatibility(tmp_path.resolve()) is None
    assert module._prepare_trusted_smoke_workspace_compatibility(tmp_path.resolve()) is None
