#!/usr/bin/env python3
"""Freeze/run a CPU-only exploratory Bayesian CUT3R prefix comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.cut3r_bayesian_prefix_dev import ARMS, Rows, predict_arms, score_arms
from prob4d.dot_rope_cut3r_study import bilinear_sample, content_id

ROOT = Path(__file__).resolve().parents[2]
SEQUENCES = ("R01", "R02", "R03")
PROVIDER_ID = "952421d140731b2a6eb99df3cbd348653e04863fa457aaa490be31fe0b4c06a7"
ARCHIVE_SHA = "07b2222808e7fef1e4a7fd86ee0329da7c50e31cac2992d4773db0e8ffc105d7"
BOUND_FILES = (
    "src/prob4d/cut3r_bayesian_prefix_dev.py",
    "scripts/science/run_cut3r_bayesian_prefix_dev.py",
    "src/prob4d/query_posterior.py",
    "src/prob4d/dot_rope_cut3r_study.py",
    "scripts/science/evaluate_dot_rope_cut3r_pooled.py",
    "scripts/science/run_dot_rope_cut3r_native_provider.py",
    "tests/test_cut3r_bayesian_prefix_dev.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["artifact_id"] = content_id(record)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def protocol() -> dict[str, Any]:
    return {
        "schema": "prob4d.cut3r-bayesian-prefix-development-v1",
        "sequences": list(SEQUENCES),
        "provider_bundle_id": PROVIDER_ID,
        "archive_sha256": ARCHIVE_SHA,
        "alignment_frames": [1, 2],
        "residual_update_frames": [3, 4, 5],
        "score_frames": [6, 7],
        "camera": "cam001",
        "arms": list(ARMS),
        "prior_std_prefix_span": 0.1,
        "observation_std_prefix_span": 0.02,
        "rbf_length_prefix_span": 0.25,
        "prior_global_fraction": 0.5,
        "shared_observation_correlation": 0.8,
        "normalization": "bbox_diagonal_of_finite_frame_1_2_marker_rows",
        "aggregation": "equal_supported_score_frame_then_equal_sequence",
        "source_files_sha256": {name: sha256(ROOT / name) for name in BOUND_FILES},
        "claim": "exploratory sparse-prefix 3D-supervised observed-frame reconstruction only",
        "prospective_new_object_confirmation": False,
        "provider_inference_rerun": False,
        "new_rgb_decoding": False,
        "future_3d_permitted_only_after_all_predictions_sealed": True,
        "future_rgb_is_in_existing_causal_provider_outputs": True,
        "future_2d_marker_locations_are_common_query_inputs": True,
        "metric_units": "prefix_span_not_asserted_metres",
        "baseline_uncertainty": "same fixed Gaussian wrapper, not native CUT3R covariance",
        "calibration_guarantee": False,
        "autopromotion": False,
        "hyperparameter_search": False,
        "closed_studies_reopened": False,
        "protected_targets_accessed": False,
    }


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(run: dict[str, Any], frame: int, coordinates: np.ndarray, pooled: Any) -> Rows:
    selected = np.flatnonzero(run["frames"] == frame)
    if len(selected) != 1:
        raise ValueError("requested frame absent or repeated")
    index = int(selected[0])
    points = np.asarray(run["points"][index], dtype=np.float64)
    confidence = np.asarray(run["confidence"][index], dtype=np.float64)
    width, height = run["original_sizes"][index]
    pixels, in_image, _ = pooled.cut3r_output_coordinates(
        coordinates,
        original_width=int(width),
        original_height=int(height),
        output_width=int(points.shape[1]),
        output_height=int(points.shape[0]),
    )
    xyz, valid = bilinear_sample(points, pixels)
    quality, quality_valid = bilinear_sample(confidence[..., None], pixels)
    valid &= in_image & quality_valid & np.isfinite(xyz).all(axis=1)
    valid &= np.isfinite(quality[:, 0]) & (quality[:, 0] > 0)
    identities = np.flatnonzero(valid)
    return Rows(np.full(len(identities), frame), identities, xyz[valid])


def _concat(parts: list[Rows]) -> Rows:
    return Rows(
        np.concatenate([part.frame for part in parts]),
        np.concatenate([part.identity for part in parts]),
        np.concatenate([part.points for part in parts]),
    )


def run_study(archive_path: Path, bundle: Path, output: Path, lock: Path) -> dict[str, Any]:
    expected = protocol()
    actual = json.loads(lock.read_text())
    artifact_id = actual.pop("artifact_id")
    if actual != expected or artifact_id != content_id(expected):
        raise ValueError("protocol or source bytes changed")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("tracked source worktree must be clean")
    output.mkdir(parents=False, exist_ok=False)
    if sha256(archive_path) != ARCHIVE_SHA:
        raise ValueError("official source archive bytes changed")
    base = _load_module(ROOT / BOUND_FILES[5], "cut3r_prefix_base")
    pooled = _load_module(ROOT / BOUND_FILES[4], "cut3r_prefix_pooled")
    pooled._ACTIVE_COORDINATE_COLUMNS = (0, 1)
    pooled._ACTIVE_COORDINATE_MODE = "pixel-zero-based"
    original_protocol = json.loads(
        (ROOT / "protocols/dot-rope-cut3r-native-provider-v1.json").read_text()
    )
    manifest = base._verify_provider_bundle(bundle, original_protocol)
    if manifest["provider_bundle_id"] != PROVIDER_ID:
        raise ValueError("wrong sealed provider")
    records = {row["sequence"]: row for row in manifest["outputs"] if row["run"] == "continuous"}
    if set(records) != set(SEQUENCES):
        raise ValueError("source provider roster changed")

    opened: list[dict[str, Any]] = []
    sealed: dict[str, dict[str, Any]] = {}
    cached: dict[str, tuple[dict[str, Any], Rows]] = {}
    stage = "prediction"
    barrier_written = False
    with zipfile.ZipFile(archive_path) as archive:

        def read_coordinates(sequence: str, dimension: int, frame: int) -> np.ndarray:
            if sequence not in SEQUENCES or frame not in range(1, 8) or dimension not in (2, 3):
                raise ValueError("source member outside allowlist")
            if dimension == 3 and frame > 5 and not barrier_written:
                raise ValueError("score truth requested before prediction barrier")
            member = base._coordinate_member(sequence, dimension, frame, "cam001")
            raw = archive.read(member)
            opened.append(
                {"member": member, "sha256": hashlib.sha256(raw).hexdigest(), "stage": stage}
            )
            return np.asarray(
                pooled._parse_coordinate_text(raw.decode("utf-8"), dimension), dtype=np.float64
            )

        for sequence in SEQUENCES:
            try:
                source_run = base._load_run(bundle, records[sequence])
                prefix_parts: list[Rows] = []
                prefix_truth: list[np.ndarray] = []
                query_parts: list[Rows] = []
                for frame in range(1, 8):
                    coordinates = read_coordinates(sequence, 2, frame)
                    rows = _sample(source_run, frame, coordinates, pooled)
                    if frame <= 5:
                        truth = read_coordinates(sequence, 3, frame)
                        if len(truth) != len(coordinates):
                            raise ValueError("prefix 2D/3D row identity count mismatch")
                        valid = np.isfinite(truth[rows.identity]).all(axis=1)
                        prefix_parts.append(
                            Rows(rows.frame[valid], rows.identity[valid], rows.points[valid])
                        )
                        prefix_truth.append(truth[rows.identity[valid]])
                    else:
                        query_parts.append(rows)
                query = _concat(query_parts)
                prediction = predict_arms(
                    _concat(prefix_parts), np.concatenate(prefix_truth), query
                )
                arrays = {
                    **{f"mean_{name}": prediction["means"][name] for name in ARMS},
                    **{f"covariance_{name}": prediction["covariances"][name] for name in ARMS},
                    "query_frame": query.frame,
                    "query_identity": query.identity,
                    "query_provider_points": query.points,
                    "normalization_center": prediction["normalization_center"],
                    "normalization_span": np.asarray(prediction["normalization_span"]),
                }
                path = output / f"{sequence}-predictions.npz"
                np.savez_compressed(path, **arrays)
                cached[sequence] = (prediction, query)
                sealed[sequence] = {
                    "status": "exact_fallback"
                    if prediction["bayesian_fallback"]
                    else "ordinary_success",
                    "file": path.name,
                    "sha256": sha256(path),
                    "prefix_count": prediction["prefix_count"],
                    "update_count": prediction["update_count"],
                    "query_count": len(query.frame),
                }
            except (ValueError, RuntimeError, KeyError, np.linalg.LinAlgError) as error:
                sealed[sequence] = {
                    "status": "unsealable",
                    "error": f"{type(error).__name__}: {error}",
                }
        barrier = seal(
            output / "prediction-barrier.json",
            {
                "schema": "prob4d.cut3r-bayesian-prefix-prediction-barrier-v1",
                "protocol_id": artifact_id,
                "implementation_revision": revision,
                "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                "provider_bundle_id": PROVIDER_ID,
                "cases": sealed,
                "future_3d_opened": False,
            },
        )
        if set(barrier["cases"]) != set(SEQUENCES):
            raise ValueError("incomplete prediction denominator")
        for sequence, record in sealed.items():
            if sequence in cached and sha256(output / record["file"]) != record["sha256"]:
                raise ValueError("sealed prediction changed")
        barrier_written = True
        stage = "scoring"
        scored: dict[str, Any] = {}
        for sequence, (prediction, query) in cached.items():
            try:
                later = {frame: read_coordinates(sequence, 3, frame) for frame in (6, 7)}
                truth = np.full(query.points.shape, np.nan)
                for index, (frame, identity) in enumerate(
                    zip(query.frame, query.identity, strict=True)
                ):
                    if identity < len(later[int(frame)]):
                        truth[index] = later[int(frame)][identity]
                scored[sequence] = {
                    "status": "scored",
                    "metrics": score_arms(prediction, query, truth),
                }
            except (ValueError, RuntimeError, KeyError, np.linalg.LinAlgError) as error:
                scored[sequence] = {
                    "status": "score_failure",
                    "error": f"{type(error).__name__}: {error}",
                }
    complete = [record["metrics"] for record in scored.values() if record["status"] == "scored"]
    aggregate = (
        {
            name: {
                key: float(np.mean([row[name][key] for row in complete]))
                for key in complete[0][name]
            }
            for name in ARMS
        }
        if complete
        else {}
    )
    return seal(
        output / "result.json",
        {
            "schema": "prob4d.cut3r-bayesian-prefix-development-result-v1",
            "protocol_id": artifact_id,
            "implementation_revision": revision,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "prediction_barrier_id": barrier["artifact_id"],
            "locked_sequence_count": 3,
            "prediction_accounting": sealed,
            "scored_sequence_count": len(complete),
            "complete_denominator": len(complete) == 3,
            "per_sequence": scored,
            "aggregate": aggregate,
            "aggregate_scope": "all_three"
            if len(complete) == 3
            else "descriptive_scorable_subset_only",
            "opened_coordinate_members": opened,
            "claim": expected["claim"],
            "prior_source_outcomes_already_opened": True,
            "provider_inference_rerun": False,
            "new_rgb_decoding": False,
            "protected_targets_accessed": False,
            "new_future_3d_access_only_after_prediction_barrier": True,
            "calibration_guarantee": False,
            "autopromotion": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--archive", type=Path, required=True)
    execute.add_argument("--provider-bundle", type=Path, required=True)
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        record = seal(args.output, protocol())
    else:
        record = run_study(args.archive, args.provider_bundle, args.output, args.protocol)
    print(
        json.dumps(
            {"artifact_id": record["artifact_id"], "output": str(args.output)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
