"""Fail-closed attestation for CUT3R's native CUDA RoPE implementation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CUT3R_RUNTIME_CONTRACT_SCHEMA = "prob4d.cut3r-runtime-contract"
CUT3R_RUNTIME_CONTRACT_VERSION = 1

_CROCO_ROOT = Path("src/croco")
_CUROPE_ROOT = _CROCO_ROOT / "models/curope"
_REQUIRED_SOURCE_MEMBERS = (
    _CUROPE_ROOT / "__init__.py",
    _CUROPE_ROOT / "curope2d.py",
    _CUROPE_ROOT / "curope.cpp",
    _CUROPE_ROOT / "kernels.cu",
    _CUROPE_ROOT / "setup.py",
    _CROCO_ROOT / "models/pos_embed.py",
)


class Cut3RRuntimeContractError(RuntimeError):
    """Raised when CUT3R cannot prove use of its native RoPE implementation."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkout_member(checkout: Path, relative: Path, *, label: str) -> Path:
    candidate = checkout / relative
    cursor = checkout
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Cut3RRuntimeContractError(
                f"{label} traverses a symbolic link: {relative.as_posix()}"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise Cut3RRuntimeContractError(
            f"required {label} is missing: {relative.as_posix()}"
        ) from error
    try:
        resolved.relative_to(checkout)
    except ValueError as error:
        raise Cut3RRuntimeContractError(
            f"{label} escapes the pinned CUT3R checkout: {relative.as_posix()}"
        ) from error
    if not resolved.is_file():
        raise Cut3RRuntimeContractError(
            f"required {label} is not a regular file: {relative.as_posix()}"
        )
    return resolved


def _native_extension_path(module: object, *, checkout: Path, curope_root: Path) -> Path:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise Cut3RRuntimeContractError(
            "CUT3R RoPE kernel module does not expose a filesystem identity"
        )
    unresolved = Path(raw_path)
    if unresolved.is_symlink():
        raise Cut3RRuntimeContractError("CUT3R RoPE extension must not be a symbolic link")
    try:
        extension = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise Cut3RRuntimeContractError(
            "CUT3R RoPE kernel module points to a missing extension file"
        ) from error
    if not extension.is_file():
        raise Cut3RRuntimeContractError(
            "CUT3R RoPE kernel module is not backed by a regular file"
        )
    if not any(
        extension.name.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise Cut3RRuntimeContractError(
            "CUT3R resolved the Python RoPE fallback instead of a native extension"
        )
    try:
        extension.relative_to(curope_root)
    except ValueError as error:
        try:
            reported = extension.relative_to(checkout).as_posix()
        except ValueError:
            reported = "<outside-cut3r-checkout>"
        raise Cut3RRuntimeContractError(
            "CUT3R RoPE extension is outside the pinned curope source tree: "
            f"{reported}"
        ) from error
    return extension


def _source_records(checkout: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative in _REQUIRED_SOURCE_MEMBERS:
        source = _checkout_member(checkout, relative, label="CUT3R RoPE source member")
        records.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": _sha256(source),
                "byte_count": int(source.stat().st_size),
            }
        )
    return records


def require_compiled_cut3r_rope(checkout: Path) -> dict[str, Any]:
    """Require CUT3R's compiled RoPE kernel and return a content-bound receipt.

    This function is intentionally called before importing ``dust3r.model``. CUT3R
    otherwise silently permits a Python RoPE fallback, which is not an admissible
    runtime for the provider comparison.
    """

    try:
        resolved_checkout = checkout.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise Cut3RRuntimeContractError("CUT3R checkout does not exist") from error
    if not resolved_checkout.is_dir():
        raise Cut3RRuntimeContractError("CUT3R checkout is not a directory")

    croco_root = resolved_checkout / _CROCO_ROOT
    curope_root = resolved_checkout / _CUROPE_ROOT
    if not croco_root.is_dir() or not curope_root.is_dir():
        raise Cut3RRuntimeContractError(
            "CUT3R checkout is missing src/croco/models/curope"
        )

    croco_path = str(croco_root)
    while croco_path in sys.path:
        sys.path.remove(croco_path)
    sys.path.insert(0, croco_path)
    importlib.invalidate_caches()

    try:
        curope2d = importlib.import_module("models.curope.curope2d")
    except Exception as error:
        raise Cut3RRuntimeContractError(
            "CUT3R native RoPE is unavailable. Build the pinned checkout from "
            "src/croco/models/curope with 'python setup.py build_ext --inplace', "
            "then start a fresh Python process. The Python fallback is forbidden "
            "for provider qualification."
        ) from error

    kernels = getattr(curope2d, "_kernels", None)
    if kernels is None:
        raise Cut3RRuntimeContractError(
            "CUT3R curope2d did not expose its native kernel module"
        )
    extension = _native_extension_path(
        kernels,
        checkout=resolved_checkout,
        curope_root=curope_root,
    )
    if not callable(getattr(kernels, "rope_2d", None)):
        raise Cut3RRuntimeContractError(
            "CUT3R RoPE extension does not expose the required rope_2d symbol"
        )

    receipt: dict[str, Any] = {
        "schema": CUT3R_RUNTIME_CONTRACT_SCHEMA,
        "schema_version": CUT3R_RUNTIME_CONTRACT_VERSION,
        "status": "native-rope-verified",
        "import_target": "models.curope.curope2d._kernels",
        "extension": {
            "relative_path": extension.relative_to(resolved_checkout).as_posix(),
            "sha256": _sha256(extension),
            "byte_count": int(extension.stat().st_size),
        },
        "sources": _source_records(resolved_checkout),
    }
    receipt["artifact_id"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    return receipt


def validate_cut3r_runtime_receipt(receipt: Mapping[str, object]) -> None:
    """Validate the self-authenticating structure of a runtime receipt."""

    unsigned = dict(receipt)
    artifact_id = unsigned.pop("artifact_id", None)
    if not isinstance(artifact_id, str) or len(artifact_id) != 64:
        raise Cut3RRuntimeContractError("runtime receipt has no canonical artifact ID")
    if receipt.get("schema") != CUT3R_RUNTIME_CONTRACT_SCHEMA:
        raise Cut3RRuntimeContractError("runtime receipt schema is unsupported")
    if receipt.get("schema_version") != CUT3R_RUNTIME_CONTRACT_VERSION:
        raise Cut3RRuntimeContractError("runtime receipt version is unsupported")
    if receipt.get("status") != "native-rope-verified":
        raise Cut3RRuntimeContractError("runtime receipt does not attest native RoPE")
    expected = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if artifact_id != expected:
        raise Cut3RRuntimeContractError("runtime receipt artifact ID is invalid")
