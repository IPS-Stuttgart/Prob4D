import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.observation_contract import (
    ObservationBeliefExportV1,
    save_observation_belief_export,
)
from prob4d.observation_validation import load_observation_belief_export
from prob4d.observation_validation import main as validate_main

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


def _descriptor(artifact: ObservationBeliefExportV1) -> dict[str, object]:
    return {"artifact_id": artifact.artifact_id, **artifact.descriptor()}


def _write_raw_descriptor(
    path: Path,
    artifact: ObservationBeliefExportV1,
    descriptor_text: str,
) -> None:
    np.savez_compressed(
        path,
        descriptor_json=np.asarray(descriptor_text),
        **artifact.arrays(),
    )


def _write_descriptor(
    path: Path,
    artifact: ObservationBeliefExportV1,
    descriptor: dict[str, object],
) -> None:
    _write_raw_descriptor(
        path,
        artifact,
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )


def test_contract_matches_cross_repository_golden_digest(tmp_path: Path) -> None:
    artifact = _artifact()
    assert artifact.artifact_id == GOLDEN_ARTIFACT_ID
    path = tmp_path / "observation.npz"
    save_observation_belief_export(path, artifact)
    with np.load(path, allow_pickle=False) as archive:
        assert "descriptor_json" in archive.files
        assert archive["low_rank_factor_m"].shape == (4, 3, 2)


def test_contract_round_trip_uses_strict_loader(tmp_path: Path) -> None:
    expected = _artifact()
    path = tmp_path / "observation.npz"
    save_observation_belief_export(path, expected)

    actual = load_observation_belief_export(path)

    assert actual.artifact_id == expected.artifact_id
    np.testing.assert_array_equal(actual.mean_xyz_m, expected.mean_xyz_m)
    np.testing.assert_array_equal(
        actual.low_rank_factor_m,
        expected.low_rank_factor_m,
    )


def test_contract_loader_rejects_tampered_payload(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"
    save_observation_belief_export(path, _artifact())
    with np.load(path, allow_pickle=False) as archive:
        descriptor = np.asarray(archive["descriptor_json"])
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "descriptor_json"
        }
    arrays["mean_xyz_m"] = arrays["mean_xyz_m"].copy()
    arrays["mean_xyz_m"][0, 0] += 1.0
    np.savez_compressed(path, descriptor_json=descriptor, **arrays)

    with pytest.raises(ValueError, match="digest"):
        load_observation_belief_export(path)


def test_contract_loader_rejects_extra_array(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"
    save_observation_belief_export(path, _artifact())
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    np.savez_compressed(path, **payload, unexpected=np.asarray([1]))

    with pytest.raises(ValueError, match="arrays changed"):
        load_observation_belief_export(path)


def test_contract_loader_rejects_duplicate_descriptor_keys(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    descriptor_text = json.dumps(
        _descriptor(artifact),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    descriptor_text = descriptor_text.replace(
        '"case_id":"case-1"',
        '"case_id":"ignored","case_id":"case-1"',
        1,
    )
    path = tmp_path / "duplicate-descriptor.npz"
    _write_raw_descriptor(path, artifact, descriptor_text)

    with pytest.raises(
        ValueError,
        match="duplicate JSON object key 'case_id'",
    ):
        load_observation_belief_export(path)


def test_contract_loader_rejects_boolean_schema_version_alias(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    descriptor = _descriptor(artifact)
    descriptor["schema_version"] = True
    path = tmp_path / "boolean-version.npz"
    _write_descriptor(path, artifact, descriptor)

    with pytest.raises(ValueError, match="schema_version must be an integer"):
        load_observation_belief_export(path)


def test_contract_loader_rejects_numeric_case_id_alias(tmp_path: Path) -> None:
    base = _artifact()
    artifact = ObservationBeliefExportV1(
        **{**base.__dict__, "case_id": "7"}
    )
    descriptor = _descriptor(artifact)
    descriptor["case_id"] = 7
    path = tmp_path / "numeric-case-id.npz"
    _write_descriptor(path, artifact, descriptor)

    with pytest.raises(ValueError, match="case_id must be a nonempty string"):
        load_observation_belief_export(path)


def test_contract_loader_rejects_numeric_name_alias(tmp_path: Path) -> None:
    base = _artifact()
    artifact = ObservationBeliefExportV1(
        **{**base.__dict__, "view_names": ("7",)}
    )
    descriptor = _descriptor(artifact)
    descriptor["view_names"] = [7]
    path = tmp_path / "numeric-view-name.npz"
    _write_descriptor(path, artifact, descriptor)

    with pytest.raises(
        ValueError,
        match="view_names must contain nonempty strings",
    ):
        load_observation_belief_export(path)


def test_validation_cli_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "observation.npz"
    save_observation_belief_export(path, _artifact())

    assert validate_main([str(path)]) == 0
    output = capsys.readouterr().out
    assert '"status": "valid"' in output
    assert GOLDEN_ARTIFACT_ID in output


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
