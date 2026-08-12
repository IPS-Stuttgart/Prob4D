"""Narrow compatibility bridge for historical provider-v1 artifacts.

Prob4D 0.5 removes provider-v1 execution and export entry points.  This module
retains only the immutable schema records, serializers, validators, and manifest
metadata needed to inspect or round-trip already frozen provider-v1 artifacts and
to keep the three-repository contract corpus reproducible.  Pin Prob4D 0.4.1 for
full provider-v1 execution.
"""

from __future__ import annotations

import hashlib
import json

from ._metric_gauge_anchor import (
    METRIC_GAUGE_ANCHOR_SCHEMA,
    METRIC_GAUGE_ANCHOR_VERSION,
    MetricGaugeAnchor,
    load_metric_gauge_anchor,
    save_metric_gauge_anchor,
)
from ._observation_factor_io import (
    load_observation_factor_bundle_v3,
    write_observation_factor_bundle_v3,
)
from .calibration import (
    GAUGE_COVARIANCE_CALIBRATION_SCHEMA,
    GAUGE_COVARIANCE_CALIBRATION_VERSION,
    POINT_UNCERTAINTY_CALIBRATION_SCHEMA,
    POINT_UNCERTAINTY_CALIBRATION_VERSION,
    GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1,
    load_gauge_covariance_calibration,
    load_point_uncertainty_calibration,
    save_gauge_covariance_calibration,
    save_point_uncertainty_calibration,
)
from .causal_stream_contract import (
    PROB4D_CAUSAL_STREAM_CONTRACT_VERSION,
    bind_causal_stream_contract_v2,
)
from .observation_contract import (
    OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION,
    ObservationBeliefExportV1,
    save_observation_belief_export,
)
from .observation_factors import (
    OBSERVATION_FACTOR_SCHEMA,
    PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
    ObservationFactorBundle,
)
from .observation_validation import load_observation_belief_export
from .provider_manifest import (
    PROB4D_PROVIDER_API_VERSION,
    PROB4D_PROVIDER_PACKAGE_VERSION,
    prob4d_provider_manifest as _historical_provider_manifest,
)

PROVIDER_API_VERSION = PROB4D_PROVIDER_API_VERSION
OBSERVATION_FACTOR_SCHEMA_VERSION = PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION
load_observation_factor_bundle = load_observation_factor_bundle_v3
write_observation_factor_bundle = write_observation_factor_bundle_v3


def prob4d_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Describe the artifact-only provider-v1 compatibility bridge."""

    inherited = dict(
        _historical_provider_manifest(provider_revision=provider_revision)
    )
    inherited.pop("manifest_id", None)
    metadata = dict(inherited["metadata"])
    metadata.update(
        {
            "artifact_compatibility_only": True,
            "execution_reproduction_release": "0.4.1",
            "python_import_boundary": "prob4d.provider_v1",
        }
    )
    limitations = dict(inherited["limitations"])
    limitations["provider_v1_execution_available"] = False
    descriptor = {
        **inherited,
        "metadata": metadata,
        "limitations": limitations,
    }
    manifest_id = hashlib.sha256(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {"manifest_id": manifest_id, **descriptor}


__all__ = [
    "GAUGE_COVARIANCE_CALIBRATION_SCHEMA",
    "GAUGE_COVARIANCE_CALIBRATION_VERSION",
    "METRIC_GAUGE_ANCHOR_SCHEMA",
    "METRIC_GAUGE_ANCHOR_VERSION",
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "OBSERVATION_FACTOR_SCHEMA",
    "OBSERVATION_FACTOR_SCHEMA_VERSION",
    "POINT_UNCERTAINTY_CALIBRATION_SCHEMA",
    "POINT_UNCERTAINTY_CALIBRATION_VERSION",
    "PROB4D_CAUSAL_STREAM_CONTRACT_VERSION",
    "PROB4D_PROVIDER_API_VERSION",
    "PROB4D_PROVIDER_PACKAGE_VERSION",
    "PROVIDER_API_VERSION",
    "GaugeCovarianceCalibrationV1",
    "MetricGaugeAnchor",
    "ObservationBeliefExportV1",
    "ObservationFactorBundle",
    "PointUncertaintyCalibrationV1",
    "bind_causal_stream_contract_v2",
    "load_gauge_covariance_calibration",
    "load_metric_gauge_anchor",
    "load_observation_belief_export",
    "load_observation_factor_bundle",
    "load_point_uncertainty_calibration",
    "prob4d_provider_manifest",
    "save_gauge_covariance_calibration",
    "save_metric_gauge_anchor",
    "save_observation_belief_export",
    "save_point_uncertainty_calibration",
    "write_observation_factor_bundle",
]
