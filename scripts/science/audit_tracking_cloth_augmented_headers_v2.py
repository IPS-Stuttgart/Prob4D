#!/usr/bin/env python3
"""Metadata-classification recovery for the frozen augmented-header audit.

The v1 audit failed before producing evidence because the public dataset stores
shake/twist recordings below ``Free-hanging`` while encoding the motion family
in the filename. This wrapper changes only that public-path classifier. It
loads the exact reviewed v1 implementation, replaces its metadata classifier,
and retains the v1 protocol, header-only parser, output schema, and explicit
prohibition on trajectory-value parsing and hashing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

BASE_FILENAME = "audit_tracking_cloth_augmented_headers_v1.py"
BASE_GIT_BLOB_SHA1 = "c42f1120fd7cd0e58f45a395efd4eb49ced6f4a5"

_MATERIALS = ("cotton", "denim", "polyester", "wool")
_SIZES = ("a2", "a3")


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload,
        usedforsecurity=False,
    ).hexdigest()


def _load_base() -> ModuleType:
    path = Path(__file__).with_name(BASE_FILENAME)
    payload = path.read_bytes()
    if _git_blob_sha1(payload) != BASE_GIT_BLOB_SHA1:
        raise RuntimeError("registered augmented-header audit v1 changed")
    name = "tracking_cloth_augmented_header_audit_v1_frozen"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load registered augmented-header audit v1")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tokens(value: str) -> set[str]:
    return {piece for piece in re.split(r"[^a-z0-9]+", value.casefold()) if piece}


def _material_size_category(relative_path: str) -> tuple[str, str, str]:
    """Classify only public path metadata, including Free-hanging filenames."""
    tokens = _tokens(relative_path)
    materials = [value for value in _MATERIALS if value in tokens]
    sizes = [value.upper() for value in _SIZES if value in tokens]
    if len(materials) != 1 or len(sizes) != 1:
        raise ValueError(f"ambiguous material/size metadata: {relative_path}")

    categories: list[str] = []
    for value in ("shake", "twist", "hitting", "tablecloth"):
        if value in tokens:
            categories.append(value)
    if "self" in tokens and ({"collision", "collisions"} & tokens):
        categories.append("self-collision")
    if len(categories) != 1:
        raise ValueError(f"ambiguous motion category metadata: {relative_path}")
    return materials[0], sizes[0], categories[0]


def main() -> int:
    base = _load_base()
    base._material_size_category = _material_size_category
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
