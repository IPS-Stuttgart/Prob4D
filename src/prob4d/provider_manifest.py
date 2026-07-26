"""Versioned capability manifest for the Prob4D observation provider."""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, distribution, version
from typing import Any

PROB4D_PROVIDER_API_VERSION = 1
PROB4D_PROVIDER_PACKAGE_VERSION = "0.2.0"
PROB4D_PROVIDER_CAPABILITIES = (
    "append_invariant_causal_source_digest",
    "association_reliability_separation",
    "causal_prefix_selection",
    "conditional_point_covariance",
    "content_addressed_observation_belief",
    "fixed_metric_gauge_anchor",
    "immutable_prediction_window_inputs",
    "joint_cross_window_sim3_gauge_covariance",
    "strict_observation_belief_validation",
    "trace_audited_gauge_rank_reduction",
    "versioned_python_provider_api",
)
PROB4D_ARTIFACT_SCHEMA_VERSIONS = {
    "ObservationBeliefV1": 1,
    "ObservationFactorBundle": 3,
}
PROB4D_PROVIDER_LIMITATIONS = {
    "joint_cross_window_gauge_covariance_in_observation_belief_v1": True,
    "dense_alignment_edge_fusion_claim": False,
    "fixed_lag_boundary_covariance_exactness_claim": False,
    "prospective_covariance_calibration_claim": False,
    "physical_twin_improvement_claim": False,
}


def _installed_version() -> str:
    try:
        return version("prob4d")
    except PackageNotFoundError:
        return PROB4D_PROVIDER_PACKAGE_VERSION


def _installed_revision() -> str | None:
    try:
        direct_url = distribution("prob4d").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not direct_url:
        return None
    try:
        payload = json.loads(direct_url)
    except (TypeError, json.JSONDecodeError):
        return None
    commit_id = payload.get("vcs_info", {}).get("commit_id")
    return str(commit_id) if commit_id else None


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def prob4d_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return the auditable producer contract for Bayesian consumers."""

    revision = (
        provider_revision
        or os.environ.get("PROB4D_REVISION")
        or _installed_revision()
        or "unversioned-install"
    )
    descriptor: dict[str, object] = {
        "provider_name": "prob4d",
        "provider_version": _installed_version(),
        "provider_revision": revision,
        "provider_api_version": PROB4D_PROVIDER_API_VERSION,
        "capabilities": list(PROB4D_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(PROB4D_ARTIFACT_SCHEMA_VERSIONS),
        "limitations": dict(PROB4D_PROVIDER_LIMITATIONS),
        "metadata": {
            "source_repository": "FlorianPfaff/Prob4D",
            "python_import_boundary": "prob4d.provider_v1",
            "observation_stream": "prob4d:causal-overlap-window-points",
            "observation_belief_covariance_semantics": (
                "conditional local covariance plus one shared low-rank root of the "
                "joint Sim(3) covariance induced by the fixed metric anchor and "
                "selected causal gauge tree"
            ),
            "gauge_posterior_semantics": (
                "causal sequential spanning tree by default; fixed-lag block-diagonal "
                "covariance is an explicit approximate reconstruction control"
            ),
            "observation_factor_bundle_covariance_semantics": (
                "explicit gauge nuisance factors with schema-v3 reliability and "
                "correlation-group fields"
            ),
            "metric_boundary": (
                "ObservationBeliefV1 requires an independently checksummed fixed "
                "metric Sim(3) anchor"
            ),
        },
    }
    manifest_id = hashlib.sha256(_canonical_json(descriptor)).hexdigest()
    return {"manifest_id": manifest_id, **descriptor}


__all__ = [
    "PROB4D_ARTIFACT_SCHEMA_VERSIONS",
    "PROB4D_PROVIDER_API_VERSION",
    "PROB4D_PROVIDER_CAPABILITIES",
    "PROB4D_PROVIDER_LIMITATIONS",
    "PROB4D_PROVIDER_PACKAGE_VERSION",
    "prob4d_provider_manifest",
]
