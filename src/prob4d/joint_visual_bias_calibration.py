"""Prospective joint shared and camera-specific visual-bias calibration.

The module constructs one complete bias design across several cameras, then
reuses the existing source-group fitter and observation-sidecar artifact path.
The frozen single-scope V1 implementation is not modified.
"""

from ._joint_visual_bias_calibration import (
    build_joint_visual_bias_nuisance_from_calibration,
    fit_joint_visual_bias_calibration,
    joint_visual_bias_layout_from_calibration,
    joint_visual_bias_selection_summary,
)
from ._joint_visual_bias_common import (
    JOINT_VISUAL_BIAS_BASIS_ORDER,
    JOINT_VISUAL_BIAS_CLAIM_BOUNDARY,
    JOINT_VISUAL_BIAS_COVARIANCE_SEMANTICS,
    JOINT_VISUAL_BIAS_LAYOUT_SCHEMA,
    JOINT_VISUAL_BIAS_LAYOUT_VERSION,
    JOINT_VISUAL_BIAS_METADATA_KEY,
    JointVisualBiasLayoutV1,
    expand_joint_visual_bias_jacobian,
)
from ._joint_visual_bias_group import JointVisualBiasCalibrationGroupV1

__all__ = [
    "JOINT_VISUAL_BIAS_BASIS_ORDER",
    "JOINT_VISUAL_BIAS_CLAIM_BOUNDARY",
    "JOINT_VISUAL_BIAS_COVARIANCE_SEMANTICS",
    "JOINT_VISUAL_BIAS_LAYOUT_SCHEMA",
    "JOINT_VISUAL_BIAS_LAYOUT_VERSION",
    "JOINT_VISUAL_BIAS_METADATA_KEY",
    "JointVisualBiasCalibrationGroupV1",
    "JointVisualBiasLayoutV1",
    "build_joint_visual_bias_nuisance_from_calibration",
    "expand_joint_visual_bias_jacobian",
    "fit_joint_visual_bias_calibration",
    "joint_visual_bias_layout_from_calibration",
    "joint_visual_bias_selection_summary",
]
