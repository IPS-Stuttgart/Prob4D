from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1
from prob4d.tree_sparse_observation_artifact import (
    TREE_SPARSE_OBSERVATION_ARTIFACT_SCHEMA,
    TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION,
    TreeSparseObservationArtifactV1,
    load_tree_sparse_observation_artifact,
    write_tree_sparse_observation_artifact,
)
from prob4d.tree_sparse_observation_factors import (
    TreeSparseStackedObservationFactors,
    build_tree_sparse_observation_factors,
)

SOURCE_REVISION = "a" * 40


def _prior(*, scale: float = 1.0) -> GaugeTreeSquareRootPriorV1:
    transitions = np.zeros((2, 7, 7), dtype=np.float64)
    transitions[1] = np.eye(7, dtype=np.float64) * 0.25
    innovations = np.stack(
        (
            np.eye(7, dtype=np.float64) * (2.0e-4 * scale),
            np.eye(7, dtype=np.float64) * (3.0e-4 * scale),
        )
    )
    return GaugeTreeSquareRootPriorV1.from_transition_covariances(
        gauge_ids=("window-0", "window-1"),
        parent_indices=np.asarray([-1, 0], dtype=np.int64),
        transition_matrices=transitions,
        innovation_covariances=innovations,
    )


def _factors(
    *,
    mean_offset: float = 0.0,
    prior: GaugeTreeSquareRootPriorV1 | None = None,
) -> TreeSparseStackedObservationFactors:
    selected_prior = _prior() if prior is None else prior
    local_jacobian = np.zeros((4, 3, 7), dtype=np.float64)
    local_jacobian[:, :, 4:7] = np.eye(3, dtype=np.float64)[None]
    return build_tree_sparse_observation_factors(
        selected_prior,
        world_mean_m=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.2, 0.0, 1.1],
                [0.1, 0.2, 1.2],
                [0.3, 0.1, 1.3],
            ],
            dtype=np.float64,
        )
        + mean_offset,
        conditional_world_covariance_m2=np.repeat(
            np.eye(3, dtype=np.float64)[None] * 1.0e-3,
            4,
            axis=0,
        ),
        local_gauge_jacobian=local_jacobian,
        gauge_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        association_probability=np.asarray([0.9, 0.8, 0.85, 0.75]),
        prior_reliability=np.asarray([0.95, 0.9, 0.88, 0.82]),
        prior_nominal_probability=np.asarray([0.94, 0.94, 0.91, 0.91]),
        composite_weight=np.asarray([0.5, 0.5, 0.4, 0.4]),
        point_ids=np.asarray([10, 11, 20, 21], dtype=np.int64),
        frame_indices=np.asarray([2, 2, 4, 4], dtype=np.int64),
        view_ids=("camera-1", "camera-1", "camera-0", "camera-0"),
        factor_ids=("factor-0", "factor-0", "factor-1", "factor-1"),
        correlation_group_ids=(
            "camera-1:frame-2",
            "camera-1:frame-2",
            "camera-0:frame-4",
            "camera-0:frame-4",
        ),
        causal_frame_stop=6,
    )


def _write(
    factors: TreeSparseStackedObservationFactors,
    path: Path,
):
    return write_tree_sparse_observation_artifact(
        factors,
        path,
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:tree-sparse:camera-panel",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision=SOURCE_REVISION,
        metadata={"split": "calibration", "seed": 17},
    )


def test_tree_sparse_observation_artifact_round_trip_without_dense_prior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factors = _factors()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("dense gauge covariance path was used")

    monkeypatch.setattr(
        GaugeTreeSquareRootPriorV1,
        "materialize_dense_covariance",
        forbidden,
    )
    monkeypatch.setattr(
        GaugeTreeSquareRootPriorV1,
        "verify_dense_covariance",
        forbidden,
    )
    monkeypatch.setattr(
        GaugeTreeSquareRootPriorV1,
        "selected_covariance",
        forbidden,
    )

    path = tmp_path / "observation.json"
    written = _write(factors, path)
    loaded = load_tree_sparse_observation_artifact(path)

    assert written.manifest.artifact_id == loaded.manifest.artifact_id
    assert loaded.manifest.sequence_id == "sequence-a"
    assert loaded.manifest.observation_count == 4
    assert loaded.manifest.view_id_table == ("camera-0", "camera-1")
    assert loaded.manifest.factor_id_table == ("factor-0", "factor-1")
    assert loaded.factors.gauge_tree_prior.prior_id == factors.gauge_tree_prior.prior_id
    assert loaded.factors.view_ids == factors.view_ids
    assert loaded.factors.factor_ids == factors.factor_ids
    assert loaded.factors.correlation_group_ids == factors.correlation_group_ids
    for name in (
        "world_mean_m",
        "conditional_world_covariance_m2",
        "marginal_world_covariance_m2",
        "local_gauge_jacobian",
        "gauge_indices",
        "association_probability",
        "prior_reliability",
        "prior_nominal_probability",
        "composite_weight",
        "point_ids",
        "frame_indices",
    ):
        np.testing.assert_allclose(
            getattr(loaded.factors, name),
            getattr(factors, name),
        )


def test_artifact_inventory_contains_no_dense_or_redundant_covariance(
    tmp_path: Path,
) -> None:
    loaded = _write(_factors(), tmp_path / "observation.json")
    names = set(loaded.manifest.array_members)

    assert "marginal_world_covariance_m2" not in names
    assert "joint_gauge_covariance" not in names
    assert "gauge_prior_covariance" not in names
    assert "conditional_world_covariance_m2" in names
    assert "local_gauge_jacobian" in names
    assert not hasattr(loaded.factors, "gauge_prior_covariance")


def test_tree_sparse_observation_artifact_is_transport_independent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    written = _write(_factors(), source / "observation.json")
    shutil.copytree(source, destination)

    transported = load_tree_sparse_observation_artifact(destination / "observation.json")
    assert transported.manifest.artifact_id == written.manifest.artifact_id
    np.testing.assert_array_equal(
        transported.factors.world_mean_m,
        written.factors.world_mean_m,
    )


def test_tree_sparse_observation_artifact_rejects_row_payload_tampering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observation.json"
    written = _write(_factors(), path)
    member = written.manifest.array_members["world_mean_m"]
    payload_path = tmp_path / member.path
    payload = bytearray(payload_path.read_bytes())
    payload[-1] ^= 1
    payload_path.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA-256 mismatch|content identity"):
        load_tree_sparse_observation_artifact(path)


def test_tree_sparse_observation_artifact_rejects_prior_tampering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observation.json"
    written = _write(_factors(), path)
    prior_path = tmp_path / written.manifest.gauge_tree_prior_manifest_filename
    record = json.loads(prior_path.read_text(encoding="utf-8"))
    record["prior_id"] = "0" * 64
    prior_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="identity|artifact ID"):
        load_tree_sparse_observation_artifact(path)


def test_tree_sparse_observation_artifact_rejects_manifest_path_escape(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observation.json"
    _write(_factors(), path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["array_members"]["point_ids"]["path"] = "../point-ids.npy"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="confined relative filename"):
        load_tree_sparse_observation_artifact(path)


def test_tree_sparse_observation_artifact_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observation.json"
    _write(_factors(), path)
    payload = path.read_text(encoding="utf-8")

    duplicate = payload.replace(
        '"schema":',
        '"schema":"duplicate","schema":',
        1,
    )
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_tree_sparse_observation_artifact(path)

    _write(_factors(), tmp_path / "fresh.json")
    fresh = tmp_path / "fresh.json"
    nonfinite = fresh.read_text(encoding="utf-8").replace(
        '"metadata":{"seed":17,"split":"calibration"}',
        '"metadata":{"value":NaN}',
    )
    fresh.write_text(nonfinite, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_tree_sparse_observation_artifact(fresh)


def test_tree_sparse_observation_artifact_publication_is_idempotent_no_clobber(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observation.json"
    first = _write(_factors(), path)
    retained = path.read_bytes()
    repeated = _write(_factors(), path)

    assert repeated.manifest.artifact_id == first.manifest.artifact_id
    assert path.read_bytes() == retained

    with pytest.raises((FileExistsError, ValueError)):
        _write(_factors(mean_offset=0.01), path)
    assert path.read_bytes() == retained
    assert load_tree_sparse_observation_artifact(path).manifest.artifact_id == (
        first.manifest.artifact_id
    )


def test_tree_sparse_observation_manifest_rejects_noncanonical_tables(
    tmp_path: Path,
) -> None:
    manifest = _write(_factors(), tmp_path / "observation.json").manifest

    with pytest.raises(ValueError, match="sorted and unique"):
        replace(
            manifest,
            view_id_table=("camera-1", "camera-0"),
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(
            manifest,
            factor_id_table=("factor-0", "factor-0"),
            artifact_id=None,
        )


def test_tree_sparse_observation_manifest_rejects_identity_and_shape_drift(
    tmp_path: Path,
) -> None:
    manifest = _write(_factors(), tmp_path / "observation.json").manifest
    member = manifest.array_members["point_ids"]

    with pytest.raises(ValueError, match="artifact ID mismatch"):
        replace(manifest, sequence_id="sequence-b")
    changed_members = dict(manifest.array_members)
    changed_members["point_ids"] = replace(member, shape=(3,))
    with pytest.raises(ValueError, match="shape must be"):
        replace(
            manifest,
            array_members=changed_members,
            artifact_id=None,
        )


def test_tree_sparse_observation_artifact_contract_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="TreeSparseStackedObservationFactors"):
        write_tree_sparse_observation_artifact(  # type: ignore[arg-type]
            object(),
            tmp_path / "invalid.json",
            sequence_id="sequence-a",
            case_id="case-a",
            stream_id="stream-a",
            source_repository="IPS-Stuttgart/Prob4D",
            source_revision=SOURCE_REVISION,
        )

    loaded = _write(_factors(), tmp_path / "observation.json")
    assert isinstance(loaded.manifest, TreeSparseObservationArtifactV1)
    assert loaded.manifest.to_record()["schema"] == (TREE_SPARSE_OBSERVATION_ARTIFACT_SCHEMA)
    assert loaded.manifest.to_record()["schema_version"] == (
        TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION
    )
    with pytest.raises(TypeError):
        loaded.manifest.metadata["x"] = 1  # type: ignore[index]
