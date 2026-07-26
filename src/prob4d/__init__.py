"""Probabilistic long-horizon fusion for MotionCrafter predictions."""

from .data import PredictionWindow
from .observation_contract import (
    OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION,
    ObservationBeliefExportV1,
    save_observation_belief_export,
)
from .observation_export import (
    JointGaugePosterior,
    MetricGaugeAnchor,
    build_prob4d_observation_belief,
    deterministic_covariance_root,
    estimate_joint_gauge_tree,
    load_metric_gauge_anchor,
    save_metric_gauge_anchor,
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
from .sim3 import Sim3

__all__ = [
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
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
    "load_observation_factor_bundle",
    "save_metric_gauge_anchor",
    "save_observation_belief_export",
    "stack_observation_factors",
    "write_observation_factor_bundle",
]
__version__ = "0.2.0"
