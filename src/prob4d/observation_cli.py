"""Command-line export and validation for Prob4D observation artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from ._observation_validation import _integer_scalar
from .observation import FUSION_METHOD_NAMES, ObservationArtifact, SourceWindowProvenance
from .observation_io import _sha256, load_observation_artifact, save_observation_artifact


def _git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def export_observation_artifact(
    prediction_manifest: str | Path,
    output_manifest: str | Path,
    *,
    method: str = "prob4d_uniform",
    causal_max_frame: int | None = None,
    correlation_group: str = "motioncrafter_shared_backbone",
    prob4d_revision: str | None = None,
) -> ObservationArtifact:
    """Re-fuse admissible windows and export a gauge-relative observation artifact."""

    if method not in FUSION_METHOD_NAMES:
        raise ValueError(f"unknown fusion method {method!r}")
    from .benchmark import fuse_prediction_bundle_methods
    from .io import load_prediction_bundle

    manifest_path = Path(prediction_manifest).resolve()
    bundle = load_prediction_bundle(manifest_path)
    windows = bundle.overlap_windows
    causal_limit = (
        None
        if causal_max_frame is None
        else _integer_scalar(causal_max_frame, name="causal_max_frame")
    )
    if causal_limit is not None:
        windows = [
            window
            for window in windows
            if int(np.max(window.frame_indices)) <= causal_limit
        ]
        if not windows:
            raise ValueError("no complete prediction window satisfies causal_max_frame")
        bundle = replace(bundle, overlap_windows=windows)

    sequence = fuse_prediction_bundle_methods(bundle, method_names={method})[method]
    source_windows = tuple(
        SourceWindowProvenance.from_prediction_window(
            window,
            correlation_group=correlation_group,
        )
        for window in windows
    )
    revision = prob4d_revision or _git_revision(Path(__file__).resolve().parents[2])
    motioncrafter_revision = bundle.metadata.get("motioncrafter_commit")
    if not isinstance(motioncrafter_revision, str) or not motioncrafter_revision:
        raise ValueError("prediction manifest does not identify the MotionCrafter revision")
    provenance = {
        "producer": "Prob4D",
        "producer_revision": revision,
        "source_model": "MotionCrafter",
        "source_model_revision": motioncrafter_revision,
        "source_manifest_sha256": _sha256(manifest_path),
        "method": method,
        "prediction_manifest": str(manifest_path),
    }
    artifact = ObservationArtifact.from_fused_sequence(
        sequence,
        source_windows,
        coordinate_status="gauge_relative",
        gauge_status="unresolved",
        covariance_units="gauge_unit^2",
        gauge_reference=source_windows[0].window_id,
        provenance=provenance,
        causal_max_frame=causal_limit,
        global_estimator_source_frame_limit=max(
            window.maximum_source_frame for window in source_windows
        ),
    )
    save_observation_artifact(output_manifest, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a saved observation artifact")
    validate.add_argument("manifest", type=Path)

    export = subparsers.add_parser(
        "export",
        help="re-fuse a prediction bundle and export a causally bounded observation artifact",
    )
    export.add_argument("prediction_manifest", type=Path)
    export.add_argument("output_manifest", type=Path)
    export.add_argument("--method", choices=FUSION_METHOD_NAMES, default="prob4d_uniform")
    export.add_argument("--causal-max-frame", type=int)
    export.add_argument(
        "--correlation-group",
        default="motioncrafter_shared_backbone",
    )
    export.add_argument("--prob4d-revision")

    arguments = parser.parse_args(argv)
    if arguments.command == "validate":
        artifact = load_observation_artifact(arguments.manifest)
        print(json.dumps(artifact.summary(), indent=2, sort_keys=True))
        return 0
    if arguments.command == "export":
        artifact = export_observation_artifact(
            arguments.prediction_manifest,
            arguments.output_manifest,
            method=arguments.method,
            causal_max_frame=arguments.causal_max_frame,
            correlation_group=arguments.correlation_group,
            prob4d_revision=arguments.prob4d_revision,
        )
        print(json.dumps(artifact.summary(), indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
