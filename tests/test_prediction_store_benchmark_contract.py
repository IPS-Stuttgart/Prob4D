from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from prob4d import prediction_store_benchmark


class _FakeMappedBundle:
    manifest_path = Path("/tmp/fake-store/manifest.json")
    metadata = {
        "prediction_execution_store": {
            "source_manifest_sha256": "a" * 64,
            "store_id": "b" * 64,
        }
    }

    @staticmethod
    def dense_storage_summary() -> dict[str, object]:
        return {
            "retained_bytes": 128,
            "float64_equivalent_bytes": 256,
            "window_count": 3,
            "dense_vector_field_count": 6,
            "storage_dtypes": ["float32"],
        }


def test_mmap_benchmark_rejects_requested_dtype_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        prediction_store_benchmark,
        "load_prediction_bundle_store",
        lambda *_args, **_kwargs: _FakeMappedBundle(),
    )

    with pytest.raises(ValueError, match="storage dtype differs"):
        prediction_store_benchmark.run_benchmark(
            Namespace(
                backend="mmap_npy",
                input=Path("/tmp/fake-store"),
                dense_storage_dtype="float64",
            )
        )


def test_mmap_benchmark_records_actual_dtype_and_revision(monkeypatch) -> None:
    monkeypatch.setattr(
        prediction_store_benchmark,
        "load_prediction_bundle_store",
        lambda *_args, **_kwargs: _FakeMappedBundle(),
    )
    monkeypatch.setattr(
        prediction_store_benchmark,
        "_git_revision",
        lambda _root: "c" * 40,
    )

    result = prediction_store_benchmark.run_benchmark(
        Namespace(
            backend="mmap_npy",
            input=Path("/tmp/fake-store"),
            dense_storage_dtype="float32",
        )
    )

    assert result["repository_revision"] == "c" * 40
    assert result["configuration"]["dense_storage_dtype"] == "float32"
    assert result["bundle"]["storage_dtypes"] == ["float32"]
