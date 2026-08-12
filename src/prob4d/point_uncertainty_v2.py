"""Experimental source-only three-axis point-uncertainty calibration.

Version 2 models conditional point covariance in a local orthonormal basis:
the viewing ray, a prediction-only reference tangent, and the orthogonal tangent.
It can be fitted only after SourceCovarianceLocalizationV1 has explicitly
localized the remaining failure to conditional point covariance and a matching
GaugePropagationReadinessV1 has admitted the declared gauge-propagation path.

The shared Sim(3) gauge covariance remains outside this model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._point_uncertainty_v2_common import (
    PointUncertaintyCalibrationPolicyV2,
    local_point_basis,
)
from ._point_uncertainty_v2_fit import (
    fit_point_uncertainty_calibration_v2,
    validate_point_uncertainty_v2_eligibility,
)
from ._point_uncertainty_v2_model import (
    POINT_UNCERTAINTY_CALIBRATION_CLAIM_BOUNDARY,
    POINT_UNCERTAINTY_CALIBRATION_SCHEMA,
    POINT_UNCERTAINTY_CALIBRATION_VERSION,
    PointUncertaintyCalibrationV2,
    load_point_uncertainty_calibration_v2,
    write_point_uncertainty_calibration_v2,
)
from ._strict_json import load_json_object
from .gauge_propagation_readiness import load_gauge_propagation_readiness
from .source_covariance_localization import load_source_covariance_localization


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--localization", type=Path, required=True)
    fit.add_argument("--propagation", type=Path, required=True)
    fit.add_argument("--training", type=Path, required=True)
    fit.add_argument("--policy", type=Path)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "fit":
        localization = load_source_covariance_localization(arguments.localization)
        propagation = load_gauge_propagation_readiness(arguments.propagation)
        validate_point_uncertainty_v2_eligibility(localization, propagation)
        training_bytes = arguments.training.read_bytes()
        with np.load(arguments.training, allow_pickle=False) as archive:
            expected = {
                "residual_xyz",
                "ray_directions",
                "tangent_reference",
                "features",
                "group_ids",
                "feature_names",
            }
            if set(archive.files) != expected:
                raise ValueError("training NPZ fields changed")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        policy = (
            None
            if arguments.policy is None
            else PointUncertaintyCalibrationPolicyV2.from_dict(
                load_json_object(arguments.policy, name="point uncertainty v2 policy")
            )
        )
        calibration = fit_point_uncertainty_calibration_v2(
            localization,
            propagation,
            residual_xyz=arrays["residual_xyz"],
            ray_directions=arrays["ray_directions"],
            tangent_reference=arrays["tangent_reference"],
            features=arrays["features"],
            feature_names=tuple(str(item) for item in arrays["feature_names"].tolist()),
            group_ids=tuple(str(item) for item in arrays["group_ids"].tolist()),
            source_training_sha256=hashlib.sha256(training_bytes).hexdigest(),
            policy=policy,
        )
        write_point_uncertainty_calibration_v2(
            arguments.output,
            calibration,
            overwrite=arguments.overwrite,
        )
    else:
        calibration = load_point_uncertainty_calibration_v2(arguments.artifact)

    print(
        json.dumps(
            {
                "point_uncertainty_calibration_id": (
                    calibration.point_uncertainty_calibration_id
                ),
                "fit_converged": calibration.fit_converged,
                "training_normalized_energy": list(
                    calibration.training_normalized_energy
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if calibration.fit_converged else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "POINT_UNCERTAINTY_CALIBRATION_CLAIM_BOUNDARY",
    "POINT_UNCERTAINTY_CALIBRATION_SCHEMA",
    "POINT_UNCERTAINTY_CALIBRATION_VERSION",
    "PointUncertaintyCalibrationPolicyV2",
    "PointUncertaintyCalibrationV2",
    "fit_point_uncertainty_calibration_v2",
    "load_point_uncertainty_calibration_v2",
    "local_point_basis",
    "write_point_uncertainty_calibration_v2",
]
