"""Profile eager NPZ versus read-only memory-mapped prediction loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data import DENSE_STORAGE_DTYPES
from .io import load_prediction_bundle
from .prediction_store import load_prediction_bundle_store


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else 1024 * value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if arguments.backend == "eager_npz":
        bundle = load_prediction_bundle(
            arguments.input,
            dense_storage_dtype=arguments.dense_storage_dtype,
        )
        manifest_path = Path(arguments.input)
        identity = {
            "source_manifest_sha256": _sha256_file(manifest_path),
            "store_id": None,
        }
    else:
        bundle = load_prediction_bundle_store(arguments.input, verify_hashes=True)
        manifest_path = bundle.manifest_path
        store = bundle.metadata["prediction_execution_store"]
        identity = {
            "source_manifest_sha256": store["source_manifest_sha256"],
            "store_id": store["store_id"],
        }
    loading_seconds = time.perf_counter() - started
    summary = bundle.dense_storage_summary()
    return {
        "schema": "prob4d.prediction-store-memory-benchmark",
        "version": 1,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "configuration": {
            "backend": arguments.backend,
            "input": str(arguments.input),
            "dense_storage_dtype": arguments.dense_storage_dtype,
            "full_member_hash_verification": True,
        },
        "identity": identity,
        "timing_seconds": {"loading_and_validation": loading_seconds},
        "memory_bytes": {
            "peak_process_rss": _peak_rss_bytes(),
            "retained_dense_vectors": summary["retained_bytes"],
            "float64_equivalent": summary["float64_equivalent_bytes"],
        },
        "bundle": {
            "window_count": summary["window_count"],
            "dense_vector_field_count": summary["dense_vector_field_count"],
            "storage_dtypes": summary["storage_dtypes"],
            "manifest_path": str(manifest_path),
        },
        "claim_boundary": (
            "Process-level engineering profile only. Compare backends in separate "
            "fresh processes with the same host, revision, source bundle, and dtype."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--backend",
        choices=("eager_npz", "mmap_npy"),
        required=True,
    )
    parser.add_argument(
        "--dense-storage-dtype",
        choices=DENSE_STORAGE_DTYPES,
        default="float32",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = run_benchmark(arguments)
    except ValueError as error:
        parser.error(str(error))
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output_json is not None:
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
