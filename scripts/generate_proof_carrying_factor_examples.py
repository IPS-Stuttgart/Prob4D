#!/usr/bin/env python3
"""Generate checked Proof4D v1 admit/reject certificate examples."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

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
from prob4d.sim3 import Sim3

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIRECTORY = ROOT / "examples" / "proof4d"
SUPPORTED_PATH = EXAMPLE_DIRECTORY / "linear-query-supported.json"
REJECTED_PATH = EXAMPLE_DIRECTORY / "linear-query-rejected.json"


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


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
        normalized_geometry_spectrum=np.array(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]
        ),
        rank_threshold=0.1,
        residual_rms=0.01,
        residual_variance=1e-4,
        inlier_fraction=1.0,
        num_correspondences=8,
        covariance_method=IID_OBSERVABLE_INFORMATION,
    )


def _render_examples() -> dict[Path, str]:
    factor = _factor()
    fallback_id = "example:caller-owned-physical-fallback-v1"
    input_digest = _digest(b"proof4d controlled provider input v1")
    query_program_digest = _digest(b"proof4d controlled query program v1")
    producer = "prob4d-proof4d-example-generator"
    assumption_ids = (
        *DEFAULT_ASSUMPTION_IDS,
        "controlled-proof4d-example-v1",
    )
    supported = build_observable_gauge_query_certificate(
        factor,
        query_jacobian_local=np.eye(7)[[0, 4]],
        query_id="example:gauge-insensitive-local-query-v1",
        fallback_id=fallback_id,
        input_digest=input_digest,
        query_program_digest=query_program_digest,
        producer=producer,
        assumption_ids=assumption_ids,
    )
    rejected = build_observable_gauge_query_certificate(
        factor,
        query_jacobian_local=np.eye(7)[[1]],
        query_id="example:gauge-sensitive-local-query-v1",
        fallback_id=fallback_id,
        input_digest=input_digest,
        query_program_digest=query_program_digest,
        producer=producer,
        assumption_ids=assumption_ids,
    )
    return {
        SUPPORTED_PATH: render_proof_carrying_factor(supported),
        REJECTED_PATH: render_proof_carrying_factor(rejected),
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
        print("Proof4D examples match the producer implementation.")
        return 0
    EXAMPLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for path, content in rendered.items():
        path.write_text(content, encoding="utf-8")
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
