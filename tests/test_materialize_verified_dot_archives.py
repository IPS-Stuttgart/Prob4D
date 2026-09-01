from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest

from scripts.science.materialize_verified_dot_archives import (
    _materialize,
    _parse_entry,
    _resolve,
)


def _md5(payload: bytes) -> str:
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def test_parse_entry_accepts_plain_filename_and_lowercase_md5() -> None:
    digest = "0123456789abcdef0123456789abcdef"
    assert _parse_entry(f"R11-20.zip={digest}") == ("R11-20.zip", digest)


@pytest.mark.parametrize(
    "raw",
    [
        "missing-separator",
        "../R11-20.zip=0123456789abcdef0123456789abcdef",
        "folder/R11-20.zip=0123456789abcdef0123456789abcdef",
        "R11-20.zip=not-an-md5",
    ],
)
def test_parse_entry_rejects_ambiguous_or_unsafe_values(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_entry(raw)


def test_resolve_requires_exact_publisher_checksum() -> None:
    digest = "0123456789abcdef0123456789abcdef"
    metadata = {
        "data": {
            "latestVersion": {
                "files": [
                    {
                        "dataFile": {
                            "filename": "R11-20.zip",
                            "id": 123,
                            "filesize": 456,
                            "checksum": {"type": "MD5", "value": digest},
                        }
                    }
                ]
            }
        }
    }
    assert _resolve(metadata, [("R11-20.zip", digest)]) == [
        {
            "name": "R11-20.zip",
            "md5": digest,
            "bytes": 456,
            "datafile_id": 123,
        }
    ]
    with pytest.raises(RuntimeError, match="checksum changed"):
        _resolve(metadata, [("R11-20.zip", "f" * 32)])


def test_materialize_reuses_only_size_and_hash_verified_archive(
    tmp_path: Path,
) -> None:
    payload = b"compressed archive bytes are opaque to this helper"
    digest = _md5(payload)
    archive = tmp_path / "R11-20.zip"
    archive.write_bytes(payload)
    value = {
        "name": archive.name,
        "md5": digest,
        "bytes": len(payload),
        "datafile_id": 123,
    }
    result = _materialize(
        output_dir=tmp_path,
        datafile_api_root="https://example.invalid/api/access/datafile",
        value=value,
    )
    assert result == {
        **value,
        "path": str(archive),
        "measured_md5": digest,
    }
    assert archive.read_bytes() == payload


def test_resolve_rejects_duplicate_publisher_filename() -> None:
    digest = "0123456789abcdef0123456789abcdef"
    data_file = {
        "filename": "R11-20.zip",
        "id": 123,
        "filesize": 456,
        "checksum": {"type": "MD5", "value": digest},
    }
    metadata = {
        "data": {
            "latestVersion": {"files": [{"dataFile": data_file}, {"dataFile": dict(data_file)}]}
        }
    }
    with pytest.raises(RuntimeError, match="ambiguous"):
        _resolve(metadata, [("R11-20.zip", digest)])
