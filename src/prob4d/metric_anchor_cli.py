"""Create a content-addressed metric anchor for portable Prob4D observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ._metric_gauge_anchor import (
    MetricGaugeAnchor,
    prediction_window_sha256,
    save_metric_gauge_anchor,
)
from .observation_contract import file_sha256
from .sim3 import Sim3


def _load_covariance(path: Path | None) -> np.ndarray:
    if path is None:
        return np.zeros((7, 7), dtype=np.float64)
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"covariance"}:
                raise ValueError(
                    "metric-anchor covariance NPZ must contain only 'covariance'"
                )
            values = np.asarray(archive["covariance"])
    covariance = np.asarray(values, dtype=np.float64)
    if covariance.shape != (7, 7):
        raise ValueError("metric-anchor covariance must have shape (7, 7)")
    return covariance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_manifest", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--world-frame-id", required=True)
    parser.add_argument("--reference-window-id", required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument("--source-kind", default="prefix_registration")
    parser.add_argument(
        "--sim3-vector",
        type=float,
        nargs=7,
        metavar=("LOG_SCALE", "RX", "RY", "RZ", "TX", "TY", "TZ"),
        required=True,
    )
    parser.add_argument(
        "--covariance",
        type=Path,
        help=(
            "optional .npy matrix or .npz containing only 'covariance'; "
            "omitting it declares a fixed external calibration"
        ),
    )
    args = parser.parse_args(argv)

    anchor = MetricGaugeAnchor(
        window_id=args.reference_window_id,
        global_from_local=Sim3.from_vector(np.asarray(args.sim3_vector)),
        covariance=_load_covariance(args.covariance),
        coordinate_frame=args.world_frame_id,
        source_kind=args.source_kind,
        source_artifact_sha256=prediction_window_sha256(
            args.predictions_manifest,
            args.reference_window_id,
        ),
        calibration_artifact_sha256=file_sha256(args.calibration_artifact),
        case_id=args.case_id,
        metadata={
            "calibration_artifact_name": args.calibration_artifact.name,
        },
    )
    save_metric_gauge_anchor(args.output_json, anchor)
    print(
        json.dumps(
            {
                "artifact_id": anchor.artifact_id,
                "case_id": anchor.case_id,
                "window_id": anchor.window_id,
                "world_frame_id": anchor.world_frame_id,
                "source_artifact_sha256": anchor.source_artifact_sha256,
                "calibration_artifact_sha256": (
                    anchor.calibration_artifact_sha256
                ),
                "covariance_treatment": anchor.covariance_treatment,
                "output": str(args.output_json.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
