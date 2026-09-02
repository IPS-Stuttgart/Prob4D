from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_tracking_cloth_rank_distortion_frontier_v1.py"
PROTOCOL = ROOT / "protocols/tracking-cloth-rank-distortion-frontier-v1.json"
WORKFLOW = ROOT / ".github/workflows/tracking-cloth-rank-distortion-frontier-v1.yml"
REQUEST = ROOT / "protocols/execution_requests/tracking_cloth_rank_distortion_frontier_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("tracking_cloth_rank_distortion", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_model(module):
    protocol = module.load_protocol(PROTOCOL)
    base, _ = module._load_base_module(protocol)
    rng = np.random.default_rng(20260902)
    dimension, rank, query_dimension = 12, 7, 3
    conditional = np.zeros((dimension, dimension))
    for index in range(dimension // 3):
        block = np.diag(rng.uniform(0.3, 1.2, 3))
        conditional[3 * index : 3 * index + 3, 3 * index : 3 * index + 3] = block
    shared = rng.normal(size=(dimension, rank)) / 3.0
    query_loading = rng.normal(size=(query_dimension, rank)) / 2.0
    prior = 0.4 * np.eye(query_dimension) + query_loading @ query_loading.T
    cross = query_loading @ shared.T
    return base, protocol, prior, cross, conditional, shared


def test_protocol_is_content_addressed_and_binds_frozen_model_construction() -> None:
    module = _module()
    protocol = module.load_protocol(PROTOCOL)
    assert protocol["expected_case_count"] == 55
    assert protocol["selection_uses_heldout_outcomes"] is False
    assert protocol["primary_trace_budget_per_query_dimension"] == 0.05
    assert protocol["base_script_git_blob_sha1"] == module.BASE_SCRIPT_GIT_BLOB_SHA1
    assert protocol["base_protocol_git_blob_sha1"] == module.BASE_PROTOCOL_GIT_BLOB_SHA1


def test_generalized_eigen_frontier_matches_dense_posteriors_and_beats_baselines() -> None:
    module = _module()
    base, protocol, prior, cross, conditional, shared = _synthetic_model(module)
    frontiers, projections, context = module.compute_projection_frontiers(
        base,
        prior=prior,
        cross=cross,
        conditional=conditional,
        shared=shared,
        numerical_relative_tolerance=protocol["numerical_relative_tolerance"],
        validity_margin=protocol["posterior_validity_margin"],
    )
    assert context["original_rank"] == 7
    assert context["numerical_exact_rank"] == 3
    assert context["maximum_identity_relative_error"] < 1e-10
    assert context["maximum_optimality_violation"] <= 1e-12
    for method in module.METHODS:
        assert len(frontiers[method]) == 8
        assert len(projections[method]) == 8
        assert frontiers[method][-1]["normalized_trace_loss"] < 1e-10
    strict_pca = 0
    for rank in range(8):
        optimum = frontiers[module.METHOD_OPTIMAL][rank]["normalized_trace_loss"]
        response = frontiers[module.METHOD_RESPONSE_SVD][rank]["normalized_trace_loss"]
        pca = frontiers[module.METHOD_FACTOR_PCA][rank]["normalized_trace_loss"]
        assert optimum <= response + 1e-10
        assert optimum <= pca + 1e-10
        strict_pca += int(pca > optimum + 1e-6)
    assert strict_pca >= 3


def test_registered_budget_selection_never_uses_more_rank_than_baselines() -> None:
    module = _module()
    base, protocol, prior, cross, conditional, shared = _synthetic_model(module)
    frontiers, _, _ = module.compute_projection_frontiers(
        base,
        prior=prior,
        cross=cross,
        conditional=conditional,
        shared=shared,
        numerical_relative_tolerance=protocol["numerical_relative_tolerance"],
        validity_margin=protocol["posterior_validity_margin"],
    )
    for budget in protocol["trace_budgets_per_query_dimension"]:
        ranks = {
            method: module.minimum_rank_for_budget(frontiers[method], budget)
            for method in module.METHODS
        }
        assert ranks[module.METHOD_OPTIMAL] <= ranks[module.METHOD_RESPONSE_SVD]
        assert ranks[module.METHOD_OPTIMAL] <= ranks[module.METHOD_FACTOR_PCA]


def test_self_hosted_workflow_is_push_only_and_fail_closed() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    request = REQUEST.read_text(encoding="utf-8")
    assert "pull_request_target:" not in workflow
    assert "pull_request:" not in workflow
    assert "runs-on: [self-hosted, Linux, X64, gpuserver4090]" in workflow
    assert "environment: trusted-self-hosted-validation" in workflow
    assert 'test "$RUNNER_NAME" = "workstation1"' in workflow
    assert "persist-credentials: false" in workflow
    assert "raw dataset payload appeared in evidence" in workflow.lower()
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1" in workflow
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0" in workflow
    assert '"branch": "science/tracking-cloth-rank-distortion-frontier-v1"' in request
    assert '"raw_data_publication_authorized": false' in request
