from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from test_io import write_problem_bundle

from prob4d.benchmark import _write_fused_prediction, fuse_prediction_bundle_methods
from prob4d.benchmark_safe import (
    _sha256_file,
    _validate_existing_outputs,
    main as benchmark_main,
)
from prob4d.io import load_prediction_bundle
from prob4d.synthetic import make_synthetic_problem


def _existing_outputs(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    artifact_directory = tmp_path / "artifacts" / "sample"
    problem = make_synthetic_problem(
        seed=91,
        num_frames=40,
        height=4,
        width=6,
        overlap=15,
    )
    manifest_path, _ = write_problem_bundle(artifact_directory, problem)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["motioncrafter_commit"] = "b" * 40
    manifest["config"] = {
        "seed": 42,
        "seed_policy": "legacy-common",
        "model_source_set_sha256": "c" * 64,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    destinations = {
        "motioncrafter_disjoint": tmp_path / "disjoint" / "sample.npz",
        "motioncrafter_latent_linear": tmp_path / "latent" / "sample.npz",
        "prob4d_uniform": tmp_path / "uniform" / "sample.npz",
    }
    for source_name, destination_key in (
        ("baseline_disjoint.npz", "motioncrafter_disjoint"),
        ("baseline_latent_linear.npz", "motioncrafter_latent_linear"),
    ):
        destination = destinations[destination_key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_directory / source_name, destination)

    bundle = load_prediction_bundle(
        manifest_path,
        dense_storage_dtype="float32",
    )
    sequence = fuse_prediction_bundle_methods(
        bundle,
        method_names={"prob4d_uniform"},
    )["prob4d_uniform"]
    _write_fused_prediction(
        destinations["prob4d_uniform"],
        sequence,
        method_id="prob4d_uniform",
        include_covariance=True,
        metadata={
            "prob4d_revision": "a" * 40,
            "motioncrafter_revision": "b" * 40,
            "motioncrafter_seed_policy": "legacy-common",
            "motioncrafter_model_set_sha256": "c" * 64,
            "prediction_manifest_sha256": _sha256_file(manifest_path),
            "includes_covariance": True,
            "dense_storage_dtype": "float32",
        },
    )
    return artifact_directory, destinations


def test_existing_benchmark_outputs_are_validated_before_skip(tmp_path: Path) -> None:
    artifact_directory, destinations = _existing_outputs(tmp_path)

    manifest_path, frame_count, storage_summary = _validate_existing_outputs(
        artifact_directory=artifact_directory,
        destinations=destinations,
        fusion_methods=("prob4d_uniform",),
        prob4d_commit="a" * 40,
        motioncrafter_commit="b" * 40,
        seed_policy="legacy-common",
        model_set_sha256="c" * 64,
        include_covariance=True,
        dense_storage_dtype="float32",
    )

    assert manifest_path == artifact_directory / "predictions.json"
    assert frame_count == 40

    baseline = destinations["motioncrafter_disjoint"]
    baseline.write_bytes(baseline.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="differs from its bound source"):
        _validate_existing_outputs(
            artifact_directory=artifact_directory,
            destinations=destinations,
            fusion_methods=("prob4d_uniform",),
            prob4d_commit="a" * 40,
            motioncrafter_commit="b" * 40,
            seed_policy="legacy-common",
            model_set_sha256="c" * 64,
            include_covariance=True,
            dense_storage_dtype="float32",
        )


def test_pinned_benchmark_cli_rejects_mutable_model_defaults(
    tmp_path: Path,
    capsys: Any,
) -> None:
    with pytest.raises(SystemExit) as error:
        benchmark_main(
            [
                "--dataset-dir",
                str(tmp_path / "dataset"),
                "--output-dir",
                str(tmp_path / "output"),
                "--upstream-root",
                str(tmp_path / "MotionCrafter"),
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )

    assert error.value.code == 2
    assert "exact remote revision" in capsys.readouterr().err
