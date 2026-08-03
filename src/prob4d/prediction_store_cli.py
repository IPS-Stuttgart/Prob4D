"""CLI for content-addressed memory-mapped prediction execution stores."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .data import DENSE_STORAGE_DTYPES
from .prediction_store import (
    materialize_prediction_bundle_store,
    prediction_bundle_store_summary,
)


def main_materialize(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d storage materialize",
        description=(
            "Convert a verified NPZ prediction bundle into an atomic, "
            "content-addressed NPY execution store."
        ),
    )
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--dense-storage-dtype",
        choices=DENSE_STORAGE_DTYPES,
        default="float32",
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    try:
        materialize_prediction_bundle_store(
            arguments.source_manifest,
            arguments.destination,
            dense_storage_dtype=arguments.dense_storage_dtype,
        )
        summary = prediction_bundle_store_summary(arguments.destination)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main_validate(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d storage validate",
        description="Hash-verify and summarize a memory-mapped prediction store.",
    )
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    try:
        summary = prediction_bundle_store_summary(arguments.directory)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = ["main_materialize", "main_validate"]
