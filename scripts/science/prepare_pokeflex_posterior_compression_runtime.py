#!/usr/bin/env python3
"""Apply one hash-bound, pre-data PokeFlex diagnostic shape repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_EXPECTED_SOURCE_GIT_BLOB_SHA1 = "0a4169f95149644fb9d00fca877a67b2672da36e"
_ORIGINAL = """        columns.append(
            np.concatenate(
                [coefficient * mode_xyz for coefficient in coefficients]
            )
        )
"""
_PATCHED = """        columns.append(
            np.concatenate(
                [coefficient * mode_xyz for coefficient in coefficients]
            ).reshape(-1)
        )
"""


def _git_blob_sha1(payload: bytes) -> str:
    framed = f"blob {len(payload)}\0".encode("ascii") + payload
    return hashlib.sha1(framed, usedforsecurity=False).hexdigest()


def _content_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_no_clobber(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(encoded, encoding="utf-8", newline="\n")
    except TypeError:
        path.write_text(encoded, encoding="utf-8")


def apply_repair(source_path: Path) -> dict[str, object]:
    if source_path.is_symlink():
        raise RuntimeError("PokeFlex diagnostic source must not be a symbolic link")
    source = source_path.resolve(strict=True)
    if not source.is_file():
        raise RuntimeError("PokeFlex diagnostic source must be a regular file")
    raw = source.read_bytes()
    source_blob = _git_blob_sha1(raw)
    if source_blob != _EXPECTED_SOURCE_GIT_BLOB_SHA1:
        raise RuntimeError("PokeFlex diagnostic source bytes changed")
    text = raw.decode("utf-8")
    if text.count(_ORIGINAL) != 1 or _PATCHED in text:
        raise RuntimeError("PokeFlex diagnostic shape-repair preimage changed")
    patched = text.replace(_ORIGINAL, _PATCHED, 1).encode("utf-8")
    source.write_bytes(patched)
    record: dict[str, object] = {
        "schema": "prob4d.pokeflex-posterior-compression-shape-repair.v1",
        "schema_version": 1,
        "status": "applied-before-real-data-access",
        "source_member": (
            "scripts/science/run_pokeflex_posterior_compression_real_geometry.py"
        ),
        "source_git_blob_sha1": source_blob,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "repair": (
            "Flatten each learned spatial-mode trajectory column from "
            "window-by-point-by-coordinate form to the same 3N observation "
            "coordinate vector used by the translation columns."
        ),
        "information_boundary": (
            "The repair is applied and tested before the self-hosted job reads "
            "any PokeFlex ZIP central directory or member payload."
        ),
    }
    record["artifact_id"] = _content_id(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = apply_repair(args.source)
    _write_no_clobber(args.output, record)
    print(record["artifact_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
