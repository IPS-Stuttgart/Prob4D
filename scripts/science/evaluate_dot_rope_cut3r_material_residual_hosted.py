#!/usr/bin/env python3
"""Hosted adapter for the frozen DOT CUT3R material-residual evaluator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

BASE_FILENAME = "evaluate_dot_rope_cut3r_material_residual.py"
BASE_GIT_BLOB_SHA1 = "45e86f943d1f1afdd706d9db11e07fa4faa89678"
ARCHIVE_NAME = "R01-10.zip"
ARCHIVE_BYTES = 1705947395
ARCHIVE_MD5 = "ca546ff5f22c0279123ccb18509858ee"


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def _load_base() -> ModuleType:
    source = Path(__file__).with_name(BASE_FILENAME)
    source_bytes = source.read_bytes()
    if _git_blob_sha1(source_bytes) != BASE_GIT_BLOB_SHA1:
        raise RuntimeError("frozen material-residual evaluator source changed")
    module_name = "dot_rope_cut3r_material_residual_frozen"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen material-residual evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _official_dataset(
    root: Path,
    protocol: Mapping[str, Any],
) -> Path:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("hosted dataset root must be a directory")
    if protocol.get("archive") != ARCHIVE_NAME:
        raise ValueError("protocol archive changed")
    archive = resolved / ARCHIVE_NAME
    if (
        not archive.is_file()
        or archive.is_symlink()
        or archive.stat().st_size != ARCHIVE_BYTES
    ):
        raise ValueError("official hosted DOT archive identity changed")
    if _md5(archive) != ARCHIVE_MD5:
        raise ValueError("official hosted DOT archive checksum changed")
    return archive


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-protocol")
    validate.add_argument("--protocol", type=Path, required=True)
    commands.add_parser("self-test")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--provider-bundle", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--repository-revision", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    base = _load_base()
    if args.command == "validate-protocol":
        value = base._protocol(args.protocol)
        print(f'{{"protocol_id": "{value["protocol_id"]}"}}')
        return 0
    if args.command == "self-test":
        return int(base._self_test())
    if args.command == "evaluate":
        base._dataset = _official_dataset
        return int(base._evaluate(args))
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
