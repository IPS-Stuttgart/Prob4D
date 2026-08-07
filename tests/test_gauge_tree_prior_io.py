from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.cli import main as grouped_main
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1, main
from prob4d.gauge_tree_prior_io import (
    GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA,
    gauge_tree_prior_artifact_id,
    load_gauge_tree_prior,
    write_gauge_tree_prior,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prior(*, gauge_count: int = 5) -> GaugeTreeSquareRootPriorV1:
    parents = np.asarray([-1] + [(index - 1) // 2 for index in range(1, gauge_count)])
    transitions = np.zeros((gauge_count, 7, 7), dtype=np.float64)
    scales = np.empty_like(transitions)
    scales[0] = np.diag(np.linspace(0.05, 0.11, 7))
    for index in range(1, gauge_count):
        transitions[index] = np.eye(7) * (0.7 + 0.02 * index)
        scales[index] = np.diag(
            np.linspace(0.02, 0.04, 7) * (1.0 + 0.05 * index)
        )
    return GaugeTreeSquareRootPriorV1(
        gauge_ids=tuple(f"window-{index}" for index in range(gauge_count)),
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
    )


def test_portable_sparse_prior_round_trip_omits_dense_covariance(tmp_path: Path) -> None:
    source = _prior()
    bound = GaugeTreeSquareRootPriorV1.from_dense_covariance(
        gauge_ids=source.gauge_ids,
        parent_indices=source.parent_indices,
        joint_covariance=source.materialize_dense_covariance(),
    )
    manifest, payload = write_gauge_tree_prior(bound, tmp_path / "prior.json")

    loaded = load_gauge_tree_prior(manifest)
    record = json.loads(manifest.read_text(encoding="utf-8"))

    assert loaded.prior_id == bound.prior_id
    assert loaded.source_joint_covariance_sha256 == bound.source_joint_covariance_sha256
    assert record["schema"] == GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA
    assert record["artifact_id"] == gauge_tree_prior_artifact_id(bound)
    with np.load(payload, allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "parent_indices",
            "transition_matrices",
            "innovation_scale_tril",
        }
    np.testing.assert_array_equal(loaded.parent_indices, bound.parent_indices)
    np.testing.assert_allclose(loaded.transition_matrices, bound.transition_matrices)
    np.testing.assert_allclose(loaded.innovation_scale_tril, bound.innovation_scale_tril)


def test_sparse_prior_writer_is_immutable(tmp_path: Path) -> None:
    prior = _prior()
    manifest = tmp_path / "prior.json"
    write_gauge_tree_prior(prior, manifest)

    with pytest.raises(FileExistsError, match="output already exists"):
        write_gauge_tree_prior(prior, manifest)


def test_sparse_prior_loader_rejects_payload_tampering(tmp_path: Path) -> None:
    manifest, payload = write_gauge_tree_prior(_prior(), tmp_path / "prior.json")
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="payload checksum mismatch"):
        load_gauge_tree_prior(manifest)


def test_sparse_prior_loader_rejects_manifest_tampering(tmp_path: Path) -> None:
    manifest, _ = write_gauge_tree_prior(_prior(), tmp_path / "prior.json")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["artifact_id"] = "0" * 64
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_id mismatch"):
        load_gauge_tree_prior(manifest)


def test_sparse_prior_loader_rejects_payload_path_escape(tmp_path: Path) -> None:
    manifest, _ = write_gauge_tree_prior(_prior(), tmp_path / "prior.json")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["payload"]["path"] = "../prior.npz"
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="relative path"):
        load_gauge_tree_prior(manifest)


def test_sparse_prior_cli_verifies_and_guards_dense_materialization(
    tmp_path: Path,
) -> None:
    manifest, _ = write_gauge_tree_prior(_prior(), tmp_path / "prior.json")

    assert main(["verify", str(manifest)]) == 0
    assert grouped_main(["gauge", "prior", "verify", str(manifest)]) == 0
    with pytest.raises(ValueError, match="limited to 4 gauges"):
        main(
            [
                "materialize",
                str(manifest),
                str(tmp_path / "dense.npy"),
                "--maximum-gauges",
                "4",
            ]
        )
    assert not (tmp_path / "dense.npy").exists()
    assert (
        main(
            [
                "materialize",
                str(manifest),
                str(tmp_path / "dense.npy"),
                "--maximum-gauges",
                "5",
            ]
        )
        == 0
    )
    assert np.load(tmp_path / "dense.npy", allow_pickle=False).shape == (35, 35)



def test_artifact_identity_is_path_independent_and_preserves_prior_identity(
    tmp_path: Path,
) -> None:
    prior = _prior()
    first, _ = write_gauge_tree_prior(prior, tmp_path / "first" / "prior.json")
    second, _ = write_gauge_tree_prior(prior, tmp_path / "second" / "renamed.json")
    first_record = json.loads(first.read_text(encoding="utf-8"))
    second_record = json.loads(second.read_text(encoding="utf-8"))

    assert first_record["artifact_id"] == second_record["artifact_id"]
    assert first_record["prior"]["prior_id"] == prior.prior_id
    assert first_record["prior"]["parent_indices"] == [
        int(value) for value in prior.parent_indices
    ]


def test_sparse_prior_loader_rejects_extra_payload_arrays(tmp_path: Path) -> None:
    manifest, payload = write_gauge_tree_prior(_prior(), tmp_path / "prior.json")
    with np.load(payload, allow_pickle=False) as arrays:
        values = {name: np.asarray(arrays[name]) for name in arrays.files}
    np.savez_compressed(payload, **values, unexpected=np.zeros(1))
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["payload"]["sha256"] = _sha256(payload)
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected array keys"):
        load_gauge_tree_prior(manifest)


def test_sparse_prior_loader_rejects_coercive_manifest_scalars(tmp_path: Path) -> None:
    manifest, _ = write_gauge_tree_prior(_prior(), tmp_path / "prior.json")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["schema_version"] = True
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be an integer"):
        load_gauge_tree_prior(manifest)


def test_sparse_prior_writer_rejects_payload_outside_manifest_tree(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="below the manifest directory"):
        write_gauge_tree_prior(
            _prior(),
            tmp_path / "manifests" / "prior.json",
            payload_path=tmp_path / "outside.npz",
        )
