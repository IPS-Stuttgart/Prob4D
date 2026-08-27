from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prob4d import cut3r_runtime_contract as contract

_SOURCE_MEMBERS = (
    "src/croco/models/curope/__init__.py",
    "src/croco/models/curope/curope2d.py",
    "src/croco/models/curope/curope.cpp",
    "src/croco/models/curope/kernels.cu",
    "src/croco/models/curope/setup.py",
    "src/croco/models/pos_embed.py",
)


def _checkout(tmp_path: Path) -> tuple[Path, Path]:
    checkout = tmp_path / "CUT3R"
    for relative in _SOURCE_MEMBERS:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative}\n", encoding="utf-8")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    extension = checkout / "src/croco/models/curope" / f"curope{suffix}"
    extension.write_bytes(b"compiled-curope")
    return checkout, extension


def _install_fake_import(monkeypatch: pytest.MonkeyPatch, extension: Path) -> None:
    kernels = SimpleNamespace(__file__=str(extension), rope_2d=lambda *_args: None)
    curope2d = SimpleNamespace(_kernels=kernels)
    monkeypatch.setattr(importlib, "import_module", lambda _name: curope2d)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_native_runtime_receipt_is_content_bound_and_path_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, extension = _checkout(tmp_path)
    _install_fake_import(monkeypatch, extension)

    first = contract.require_compiled_cut3r_rope(checkout)
    second = contract.require_compiled_cut3r_rope(checkout)

    assert first == second
    assert first["status"] == "native-rope-verified"
    assert first["extension"]["relative_path"].startswith(
        "src/croco/models/curope/curope"
    )
    assert str(checkout) not in json.dumps(first, sort_keys=True)
    unsigned = dict(first)
    artifact_id = unsigned.pop("artifact_id")
    assert artifact_id == hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    contract.validate_cut3r_runtime_receipt(first)


def test_python_fallback_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _checkout(tmp_path)
    fallback = checkout / "src/croco/models/curope/curope.py"
    fallback.write_text("def rope_2d(*args): pass\n", encoding="utf-8")
    _install_fake_import(monkeypatch, fallback)

    with pytest.raises(
        contract.Cut3RRuntimeContractError,
        match="Python RoPE fallback",
    ):
        contract.require_compiled_cut3r_rope(checkout)


def test_out_of_tree_native_extension_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _checkout(tmp_path)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    external = tmp_path / f"curope{suffix}"
    external.write_bytes(b"external-extension")
    _install_fake_import(monkeypatch, external)

    with pytest.raises(
        contract.Cut3RRuntimeContractError,
        match="outside the pinned curope source tree",
    ):
        contract.require_compiled_cut3r_rope(checkout)


def test_missing_bound_source_member_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, extension = _checkout(tmp_path)
    (checkout / "src/croco/models/curope/kernels.cu").unlink()
    _install_fake_import(monkeypatch, extension)

    with pytest.raises(
        contract.Cut3RRuntimeContractError,
        match="required CUT3R RoPE source member is missing",
    ):
        contract.require_compiled_cut3r_rope(checkout)


def test_tampered_receipt_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, extension = _checkout(tmp_path)
    _install_fake_import(monkeypatch, extension)
    receipt = contract.require_compiled_cut3r_rope(checkout)
    receipt["extension"]["byte_count"] += 1

    with pytest.raises(
        contract.Cut3RRuntimeContractError,
        match="artifact ID is invalid",
    ):
        contract.validate_cut3r_runtime_receipt(receipt)
