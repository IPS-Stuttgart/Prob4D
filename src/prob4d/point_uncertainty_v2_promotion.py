"""Evaluate and replay disjoint-group promotion evidence for point uncertainty v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._point_uncertainty_v2_promotion_eval import evaluate_point_uncertainty_v2_promotion
from ._point_uncertainty_v2_promotion_report import (
    PointUncertaintyPromotionReportV1,
    load_point_uncertainty_promotion_report_v1,
    write_point_uncertainty_promotion_report_v1,
)
from ._point_uncertainty_v2_promotion_types import (
    POINT_UNCERTAINTY_PROMOTION_CLAIM_BOUNDARY,
    POINT_UNCERTAINTY_PROMOTION_SCHEMA,
    POINT_UNCERTAINTY_PROMOTION_VERSION,
    PointUncertaintyGroupMetricsV1,
    PointUncertaintyPromotionPolicyV1,
)
from ._strict_calibration import load_point_uncertainty_calibration
from ._strict_json import load_json_object
from .point_uncertainty_v2 import load_point_uncertainty_calibration_v2


def _load_validation(path: Path) -> tuple[dict[str, np.ndarray], str, str, str]:
    payload = path.read_bytes()
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            "residual_xyz",
            "ray_directions",
            "tangent_reference",
            "features",
            "group_ids",
            "feature_names",
            "depth_squared",
            "disagreement_parallel_mean",
            "disagreement_lateral_mean",
            "provider_manifest_id",
            "cohort_binding_id",
        }
        if set(archive.files) != expected:
            raise ValueError("validation NPZ fields changed")
        arrays = {
            name: np.asarray(archive[name])
            for name in expected
            if name not in {"provider_manifest_id", "cohort_binding_id"}
        }
        provider_manifest_id = str(np.asarray(archive["provider_manifest_id"]).item())
        cohort_binding_id = str(np.asarray(archive["cohort_binding_id"]).item())
    return arrays, provider_manifest_id, cohort_binding_id, hashlib.sha256(payload).hexdigest()


def _evaluate_from_files(
    calibration_path: Path,
    baseline_calibration_path: Path,
    validation_path: Path,
    policy_path: Path,
) -> PointUncertaintyPromotionReportV1:
    calibration = load_point_uncertainty_calibration_v2(calibration_path)
    baseline_calibration = load_point_uncertainty_calibration(baseline_calibration_path)
    arrays, provider_manifest_id, cohort_binding_id, validation_sha256 = _load_validation(
        validation_path
    )
    policy = PointUncertaintyPromotionPolicyV1.from_dict(
        load_json_object(policy_path, name="point uncertainty promotion policy")
    )
    return evaluate_point_uncertainty_v2_promotion(
        calibration,
        residual_xyz=arrays["residual_xyz"],
        ray_directions=arrays["ray_directions"],
        tangent_reference=arrays["tangent_reference"],
        features=arrays["features"],
        feature_names=tuple(str(item) for item in arrays["feature_names"].tolist()),
        group_ids=tuple(str(item) for item in arrays["group_ids"].tolist()),
        baseline_calibration=baseline_calibration,
        depth_squared=arrays["depth_squared"],
        disagreement_parallel_mean=arrays["disagreement_parallel_mean"],
        disagreement_lateral_mean=arrays["disagreement_lateral_mean"],
        provider_manifest_id=provider_manifest_id,
        cohort_binding_id=cohort_binding_id,
        validation_sha256=validation_sha256,
        policy=policy,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--calibration", type=Path, required=True)
    evaluate.add_argument("--baseline-calibration", type=Path, required=True)
    evaluate.add_argument("--validation", type=Path, required=True)
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--calibration", type=Path, required=True)
    verify.add_argument("--baseline-calibration", type=Path, required=True)
    verify.add_argument("--validation", type=Path, required=True)
    verify.add_argument("--policy", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    recomputed = _evaluate_from_files(
        arguments.calibration,
        arguments.baseline_calibration,
        arguments.validation,
        arguments.policy,
    )
    if arguments.command == "evaluate":
        write_point_uncertainty_promotion_report_v1(
            arguments.output,
            recomputed,
            overwrite=arguments.overwrite,
        )
        report = recomputed
    else:
        report = load_point_uncertainty_promotion_report_v1(arguments.artifact)
        if report.to_dict() != recomputed.to_dict():
            raise ValueError("promotion report does not replay from calibration/validation inputs")
    print(
        json.dumps(
            {
                "point_uncertainty_promotion_id": report.point_uncertainty_promotion_id,
                "promote_candidate": report.promote_candidate,
                "summary": report.summary,
                "criteria": report.criteria,
            },
            sort_keys=True,
        )
    )
    return 0 if report.promote_candidate else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "POINT_UNCERTAINTY_PROMOTION_CLAIM_BOUNDARY",
    "POINT_UNCERTAINTY_PROMOTION_SCHEMA",
    "POINT_UNCERTAINTY_PROMOTION_VERSION",
    "PointUncertaintyGroupMetricsV1",
    "PointUncertaintyPromotionPolicyV1",
    "PointUncertaintyPromotionReportV1",
    "evaluate_point_uncertainty_v2_promotion",
    "load_point_uncertainty_promotion_report_v1",
    "write_point_uncertainty_promotion_report_v1",
]
