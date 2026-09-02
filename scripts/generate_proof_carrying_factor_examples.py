#!/usr/bin/env python3
"""Generate checked Proof4D certificates and caller-owned execution contexts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from prob4d.axial_query_certificate import HarmonicQuery
from prob4d.observable_gauge import (
    IID_OBSERVABLE_INFORMATION,
    CentroidGaugeChart,
    ObservableGaugeFactor,
)
from prob4d.proof_carrying_factor import (
    DEFAULT_ASSUMPTION_IDS,
    build_observable_gauge_query_certificate,
    render_proof_carrying_factor,
)
from prob4d.proof_carrying_orbit import (
    DEFAULT_ORBIT_ASSUMPTION_IDS,
    build_axial_orbit_action_certificate,
    render_proof_carrying_orbit,
)
from prob4d.sim3 import Sim3
from prob4d_independent_verifier.execution_gate import (
    build_execution_context,
    render_execution_context,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIRECTORY = ROOT / "examples" / "proof4d"
LOCAL_SUPPORTED_PATH = EXAMPLE_DIRECTORY / "linear-query-supported.json"
LOCAL_REJECTED_PATH = EXAMPLE_DIRECTORY / "linear-query-rejected.json"
ORBIT_SUPPORTED_PATH = EXAMPLE_DIRECTORY / "axial-action-supported.json"
ORBIT_REJECTED_PATH = EXAMPLE_DIRECTORY / "axial-action-rejected.json"
LOCAL_CONTEXT_PATH = EXAMPLE_DIRECTORY / "linear-query-supported.context.json"
ORBIT_CONTEXT_PATH = EXAMPLE_DIRECTORY / "axial-action-supported.context.json"


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _record(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("generated certificate section must be an object")
    return value


def _factor() -> ObservableGaugeFactor:
    identity = np.eye(7)
    return ObservableGaugeFactor(
        chart=CentroidGaugeChart(
            linearization=Sim3.identity(),
            source_centroid=np.zeros(3),
            cloud_scale=1.0,
        ),
        observable_basis=identity[:, [0, 2, 3, 4, 5, 6]],
        nullspace_basis=identity[:, 1:2],
        observable_information=10.0 * np.eye(6),
        normalized_geometry_spectrum=np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]),
        rank_threshold=0.1,
        residual_rms=0.01,
        residual_variance=1e-4,
        inlier_fraction=1.0,
        num_correspondences=8,
        covariance_method=IID_OBSERVABLE_INFORMATION,
    )


def _orbit_query(constant: float, cosine: float, sine: float) -> HarmonicQuery:
    return HarmonicQuery(
        constant=constant,
        cosine=cosine,
        sine=sine,
        orbit_key=(
            "example:shared-axial-gauge-v1",
            (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
    )


def _render_examples() -> dict[Path, str]:
    factor = _factor()
    fallback_id = "example:caller-owned-physical-fallback-v1"
    fallback_digest = _digest(b"proof4d controlled complete fallback belief v1")
    input_digest = _digest(b"proof4d controlled provider input v1")
    query_program_digest = _digest(b"proof4d controlled query program v1")
    producer = "prob4d-proof4d-example-generator"
    local_assumptions = (
        *DEFAULT_ASSUMPTION_IDS,
        "controlled-proof4d-example-v1",
    )
    local_supported = build_observable_gauge_query_certificate(
        factor,
        query_jacobian_local=np.eye(7)[[0, 4]],
        query_id="example:gauge-insensitive-local-query-v1",
        fallback_id=fallback_id,
        input_digest=input_digest,
        query_program_digest=query_program_digest,
        producer=producer,
        assumption_ids=local_assumptions,
    )
    local_rejected = build_observable_gauge_query_certificate(
        factor,
        query_jacobian_local=np.eye(7)[[1]],
        query_id="example:gauge-sensitive-local-query-v1",
        fallback_id=fallback_id,
        input_digest=input_digest,
        query_program_digest=query_program_digest,
        producer=producer,
        assumption_ids=local_assumptions,
    )

    orbit_assumptions = (
        *DEFAULT_ORBIT_ASSUMPTION_IDS,
        "controlled-proof4d-example-v1",
    )
    support_receipt_digest = _digest(b"proof4d controlled orbit support receipt v1")
    candidate_loss_program_digest = _digest(b"candidate loss program v1")
    fallback_loss_program_digest = _digest(b"fallback loss program v1")
    admission_policy_digest = _digest(b"proof4d robust action policy v1")
    orbit_supported = build_axial_orbit_action_certificate(
        fallback_loss=_orbit_query(4.0, 0.0, 0.0),
        candidate_loss=_orbit_query(1.0, 0.5, 0.0),
        scope_admitted=True,
        support_receipt_digest=support_receipt_digest,
        candidate_action_id="example:candidate-action-v1",
        fallback_action_id="example:fallback-action-v1",
        candidate_loss_program_digest=candidate_loss_program_digest,
        fallback_loss_program_digest=fallback_loss_program_digest,
        fallback_id=fallback_id,
        fallback_digest=fallback_digest,
        input_digest=input_digest,
        admission_policy_digest=admission_policy_digest,
        advantage_error_bound=0.1,
        required_margin=0.5,
        producer=producer,
        assumption_ids=orbit_assumptions,
    )
    orbit_rejected = build_axial_orbit_action_certificate(
        fallback_loss=_orbit_query(1.0, 0.0, 0.0),
        candidate_loss=_orbit_query(1.0, 1.0, 0.0),
        scope_admitted=True,
        support_receipt_digest=support_receipt_digest,
        candidate_action_id="example:candidate-action-v1",
        fallback_action_id="example:fallback-action-v1",
        candidate_loss_program_digest=candidate_loss_program_digest,
        fallback_loss_program_digest=fallback_loss_program_digest,
        fallback_id=fallback_id,
        fallback_digest=fallback_digest,
        input_digest=input_digest,
        admission_policy_digest=admission_policy_digest,
        advantage_error_bound=0.0,
        required_margin=0.0,
        producer=producer,
        assumption_ids=orbit_assumptions,
    )

    local_provenance = _record(local_supported["provenance"])
    local_query = _record(local_supported["query"])
    local_decision = _record(local_supported["decision"])
    local_context = build_execution_context(
        certificate_kind=str(local_supported["certificate_kind"]),
        bindings={
            "certificate_id": local_supported["certificate_id"],
            "input_digest": local_provenance["input_digest"],
            "source_factor_digest": local_provenance["source_factor_digest"],
            "query_id": local_query["query_id"],
            "query_program_digest": local_query["query_program_digest"],
            "fallback_id": local_decision["fallback_id"],
        },
    )
    orbit_provenance = _record(orbit_supported["provenance"])
    orbit_geometry = _record(orbit_supported["orbit"])
    orbit_actions = _record(orbit_supported["actions"])
    orbit_candidate = _record(orbit_actions["candidate"])
    orbit_fallback = _record(orbit_actions["fallback"])
    orbit_decision = _record(orbit_supported["decision"])
    orbit_context = build_execution_context(
        certificate_kind=str(orbit_supported["certificate_kind"]),
        bindings={
            "certificate_id": orbit_supported["certificate_id"],
            "input_digest": orbit_provenance["input_digest"],
            "shared_gauge_id": orbit_geometry["shared_gauge_id"],
            "support_receipt_digest": orbit_geometry["support_receipt_digest"],
            "candidate_action_id": orbit_candidate["action_id"],
            "fallback_action_id": orbit_fallback["action_id"],
            "candidate_loss_program_digest": orbit_candidate["loss_program_digest"],
            "fallback_loss_program_digest": orbit_fallback["loss_program_digest"],
            "fallback_id": orbit_decision["fallback_id"],
            "fallback_digest": orbit_decision["fallback_digest"],
            "admission_policy_digest": orbit_provenance["admission_policy_digest"],
        },
    )
    return {
        LOCAL_SUPPORTED_PATH: render_proof_carrying_factor(local_supported),
        LOCAL_REJECTED_PATH: render_proof_carrying_factor(local_rejected),
        ORBIT_SUPPORTED_PATH: render_proof_carrying_orbit(orbit_supported),
        ORBIT_REJECTED_PATH: render_proof_carrying_orbit(orbit_rejected),
        LOCAL_CONTEXT_PATH: render_execution_context(local_context),
        ORBIT_CONTEXT_PATH: render_execution_context(orbit_context),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = _render_examples()
    if arguments.check:
        mismatches = [
            path
            for path, expected in rendered.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != expected
        ]
        if mismatches:
            for path in mismatches:
                print(f"out of date: {path.relative_to(ROOT)}")
            return 1
        print("Proof4D examples and caller contexts match their implementations.")
        return 0
    EXAMPLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for path, content in rendered.items():
        path.write_text(content, encoding="utf-8")
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
