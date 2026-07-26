"""Probabilistic long-horizon fusion for MotionCrafter predictions."""

from importlib.metadata import PackageNotFoundError, version

from ._metric_gauge_anchor import (
    MetricGaugeAnchor,
    load_metric_gauge_anchor,
    prediction_window_sha256,
    save_metric_gauge_anchor,
)
from .data import PredictionWindow
from .observation_contract import (
    OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION,
    ObservationBeliefExportV1,
)
from .observation_export import (
    JointGaugePosterior,
    deterministic_covariance_root,
    estimate_joint_gauge_tree,
)
from .observation_factors import (
    LinearizedObservationFactor,
    ObservationFactor,
    ObservationFactorBundle,
    StackedObservationFactors,
    load_observation_factor_bundle,
    stack_observation_factors,
    write_observation_factor_bundle,
)
from .observation_validation import load_observation_belief_export
from .portable_observation import (
    APPROXIMATE_FIXED_LAG_COVARIANCE_LAYOUT,
    JOINT_GAUGE_COVARIANCE_LAYOUT,
    PROB4D_OBSERVATION_CONTRACT_VERSION,
    build_prob4d_observation_belief,
    save_observation_belief_export,
)
from .sim3 import Sim3

try:
    __version__ = version("prob4d")
except PackageNotFoundError:  # Source tree without installed distribution metadata.
    __version__ = "0.2.1"

__all__ = [
    "APPROXIMATE_FIXED_LAG_COVARIANCE_LAYOUT",
    "JOINT_GAUGE_COVARIANCE_LAYOUT",
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "PROB4D_OBSERVATION_CONTRACT_VERSION",
    "JointGaugePosterior",
    "LinearizedObservationFactor",
    "MetricGaugeAnchor",
    "ObservationBeliefExportV1",
    "ObservationFactor",
    "ObservationFactorBundle",
    "PredictionWindow",
    "Sim3",
    "StackedObservationFactors",
    "build_prob4d_observation_belief",
    "deterministic_covariance_root",
    "estimate_joint_gauge_tree",
    "load_metric_gauge_anchor",
    "load_observation_belief_export",
    "load_observation_factor_bundle",
    "prediction_window_sha256",
    "save_metric_gauge_anchor",
    "save_observation_belief_export",
    "stack_observation_factors",
    "write_observation_factor_bundle",
]
