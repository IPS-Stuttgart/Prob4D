from __future__ import annotations

import hashlib

import numpy as np

from prob4d.observable_gauge import (
    IID_OBSERVABLE_INFORMATION,
    CentroidGaugeChart,
    ObservableGaugeFactor,
)
from prob4d.proof_carrying_factor import build_observable_gauge_query_certificate
from prob4d.sim3 import Sim3
from prob4d_independent_verifier.proof_carrying import verify_proof_carrying_factor


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def test_full_rank_factor_has_a_valid_empty_nullspace_certificate() -> None:
    identity = np.eye(7)
    factor = ObservableGaugeFactor(
        chart=CentroidGaugeChart(
            linearization=Sim3.identity(),
            source_centroid=np.zeros(3),
            cloud_scale=1.0,
        ),
        observable_basis=identity,
        nullspace_basis=np.empty((7, 0)),
        observable_information=4.0 * identity,
        normalized_geometry_spectrum=np.ones(7),
        rank_threshold=0.1,
        residual_rms=0.01,
        residual_variance=1e-4,
        inlier_fraction=1.0,
        num_correspondences=8,
        covariance_method=IID_OBSERVABLE_INFORMATION,
    )
    certificate = build_observable_gauge_query_certificate(
        factor,
        query_jacobian_local=np.array(
            [
                [1.0, 0.2, 0.0, -0.1, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.1, 0.0, 1.0, 0.0],
            ]
        ),
        query_id="test:full-rank-query-v1",
        query_program_digest=_digest(b"full-rank query"),
        fallback_id="test:fallback-v1",
        input_digest=_digest(b"full-rank provider input"),
    )

    report = verify_proof_carrying_factor(certificate)

    assert report.valid
    assert report.admitted
    assert report.metrics["relative_query_nullspace_sensitivity"] == 0.0
