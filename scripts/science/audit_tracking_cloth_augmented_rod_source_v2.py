#!/usr/bin/env python3
"""Add the physically longest-pair rule to the frozen labelled source audit.

The target-facing hypothesis is simple: the two rod endpoints are the jointly
visible marker pair with the largest median Euclidean separation. This wrapper
loads the exact source-only v1 implementation, adds the negative median length
as an ascending ranking metric, and keeps all Self-collision trajectories
closed. The source protocol can therefore accept or reject this geometry-only
identifier before any unlabelled target trajectory is opened.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

BASE_FILENAME = "audit_tracking_cloth_augmented_rod_source_v1.py"
BASE_GIT_BLOB_SHA1 = "e9c5eb809f82fed679144e5b0dff70ba693175d7"


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload,
        usedforsecurity=False,
    ).hexdigest()


def _load_base() -> ModuleType:
    path = Path(__file__).with_name(BASE_FILENAME)
    payload = path.read_bytes()
    if _git_blob_sha1(payload) != BASE_GIT_BLOB_SHA1:
        raise RuntimeError("registered labelled rod-pair source audit v1 changed")
    name = "tracking_cloth_augmented_rod_source_audit_v1_frozen"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load labelled rod-pair source audit v1")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = _load_base()
    original = base._pair_rows

    def pair_rows(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        rows = original(*args, **kwargs)
        for row in rows:
            row["negative_median_distance_mm"] = -float(row["median_distance_mm"])
        return rows

    base._pair_rows = pair_rows
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
