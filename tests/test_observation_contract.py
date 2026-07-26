from pathlib import Path

import numpy as np
import pytest

from prob4d.observation_contract import (
    ObservationBeliefExportV1,
    save_observation_belief_export,
)

GOLDEN_ARTIFACT_ID = (
    "9c02e638f60424cca7738d347d1258acd208eb562f422efacd077db4edb2fe80"
)


def _artifact() -> ObservationBeliefExportV1:
    local = np.repeat(np.eye(3)[None], 4, axis=0) * 1e-4
    factors = np.zeros((4, 3, 2))
    factors[:2, 0, 0] = 0.002
    factors[2:, 1, 1] = 0.003
    return ObservationBeliefExportV1(
        case_id="case-1",
        stream_id="prob4d:points",
        causal_frame_stop=12,
        view_names=("camera0",),
        window_names=("window0", "window1"),
        factor_names=("gauge_latent_0", "gauge_latent_1"),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=np.asarray([8, 9]),
        mean_xyz_m=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [1.1, 0.0, 1.0],
            ]
        ),
        frame_ids=np.asarray([8, 8, 9, 9]),
        entity_ids=np.asarray([0, 1, 0, 1]),
        view_indices=np.zeros(4, dtype=int),
        window_indices=np.asarray([0, 0, 1, 1]),
        correlation_group_ids=np.asarray([0, 0, 1, 1]),
        factor_group_ids=np.asarray([0, 0, 1, 1]),
        prior_reliability=np.asarray([0.9, 0.8, 0.7, 0.6]),
        association_probability=np.ones(4),
        local_covariance_m2=local,
        low_rank_factor_m=factors,
        group_ids=np.asarray([0, 1]),
        group_prior_nominal_probability=np.asarray([0.85, 0.65]),
        group_composite_weight=np.asarray([0.5, 0.5]),
        metadata={"causal_source": "prefix only"},
    )


def test_contract_matches_cross_repository_golden_digest(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    assert artifact.artifact_id == GOLDEN_ARTIFACT_ID
    path = tmp_path / "observation.npz"
    save_observation_belief_export(path, artifact)
    with np.load(path, allow_pickle=False) as archive:
        assert "descriptor_json" in archive.files
        assert archive["low_rank_factor_m"].shape == (4, 3, 2)


def test_contract_rejects_future_frame() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match="causal boundary"):
        ObservationBeliefExportV1(
            **{
                **artifact.__dict__,
                "declared_frame_ids": np.asarray([8, 12]),
                "frame_ids": np.asarray([8, 8, 12, 12]),
            }
        )


def test_contract_rejects_duplicate_observation_identity() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match="must be unique"):
        ObservationBeliefExportV1(
            **{
                **artifact.__dict__,
                "entity_ids": np.asarray([0, 0, 0, 1]),
            }
        )
