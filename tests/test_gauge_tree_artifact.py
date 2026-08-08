from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.gauge_tree_artifact import (
    load_gauge_tree_prior_artifact,
    save_gauge_tree_prior_artifact,
)
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prior(
    *,
    gauge_count: int = 6,
    innovation_scale: float = 1.0,
) -> GaugeTreeSquareRootPriorV1:
    gauge_ids = tuple(f"window-{index}" for index in range(gauge_count))
    parents = np.asarray([-1] + [(index - 1) // 2 for index in range(1, gauge_count)])
    transitions = np.zeros((gauge_count, 7, 7), dtype=np.float64)
    scales = np.empty_like(transitions)
    scales[0] = np.diag(np.linspace(0.05, 0.11, 7) * innovation_scale)
    for index in range(1, gauge_count):
        transitions[index] = np.eye(7) * (0.7 + 0.02 * index)
        transitions[index, 4:, :3] = 0.01 * index
        scale = np.diag(np.linspace(0.02, 0.04, 7) * (1.0 + 0.05 * index) * innovation_scale)
        scale[3, 0] = 0.002 * innovation_scale
        scale[6, 2] = -0.001 * innovation_scale
        scales[index] = scale
    return GaugeTreeSquareRootPriorV1(
        gauge_ids=gauge_ids,
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
    )


def test_portable_sparse_artifact_round_trip_is_immutable(tmp_path: Path) -> None:
    prior = _prior(gauge_count=64)
    manifest, payload = save_gauge_tree_prior_artifact(tmp_path / "prior.json", prior)
    recovered = load_gauge_tree_prior_artifact(
        manifest,
        expected_prior_id=prior.prior_id,
    )

    assert recovered.prior_id == prior.prior_id
    assert recovered.to_dict() == prior.to_dict()
    assert not recovered.parent_indices.flags.writeable
    assert not recovered.transition_matrices.flags.writeable
    assert not recovered.innovation_scale_tril.flags.writeable
    assert payload.stat().st_size < prior.dense_covariance_nbytes


def test_artifact_identity_is_independent_of_transport_path(tmp_path: Path) -> None:
    prior = _prior()
    first_manifest, _ = save_gauge_tree_prior_artifact(tmp_path / "first" / "prior.json", prior)
    second_manifest, _ = save_gauge_tree_prior_artifact(
        tmp_path / "second" / "prior.json",
        prior,
    )

    first = json.loads(first_manifest.read_text(encoding="utf-8"))
    second = json.loads(second_manifest.read_text(encoding="utf-8"))
    assert first["artifact_id"] == second["artifact_id"]
    assert first["prior"]["prior_id"] == prior.prior_id
    assert second["prior"]["prior_id"] == prior.prior_id


def test_writer_is_idempotent_and_refuses_different_content(tmp_path: Path) -> None:
    manifest = tmp_path / "prior.json"
    first = _prior()
    first_paths = save_gauge_tree_prior_artifact(manifest, first)
    assert save_gauge_tree_prior_artifact(manifest, first) == first_paths

    with pytest.raises(FileExistsError, match="different sparse gauge-tree artifact"):
        save_gauge_tree_prior_artifact(
            manifest,
            _prior(innovation_scale=1.25),
        )


def test_writer_recovers_a_matching_orphan_payload(tmp_path: Path) -> None:
    manifest = tmp_path / "prior.json"
    prior = _prior()
    _, payload = save_gauge_tree_prior_artifact(manifest, prior)
    manifest.unlink()

    recovered_manifest, recovered_payload = save_gauge_tree_prior_artifact(manifest, prior)
    assert recovered_payload == payload
    assert load_gauge_tree_prior_artifact(recovered_manifest).prior_id == prior.prior_id


def test_payload_checksum_tampering_fails_closed(tmp_path: Path) -> None:
    manifest, payload = save_gauge_tree_prior_artifact(tmp_path / "prior.json", _prior())
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="payload checksum mismatch"):
        load_gauge_tree_prior_artifact(manifest)


def test_payload_with_unexpected_array_fails_closed(tmp_path: Path) -> None:
    manifest, payload = save_gauge_tree_prior_artifact(tmp_path / "prior.json", _prior())
    with np.load(payload, allow_pickle=False) as arrays:
        parents = np.asarray(arrays["parent_indices"])
        transitions = np.asarray(arrays["transition_matrices"])
        scales = np.asarray(arrays["innovation_scale_tril"])
    np.savez_compressed(
        payload,
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
        unexpected=np.asarray([1], dtype=np.int64),
    )
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["payload"]["sha256"] = _sha256(payload)
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected arrays"):
        load_gauge_tree_prior_artifact(manifest)


def test_non_npz_payload_fails_closed(tmp_path: Path) -> None:
    manifest, payload = save_gauge_tree_prior_artifact(tmp_path / "prior.json", _prior())
    with payload.open("wb") as stream:
        np.save(stream, np.asarray([1.0], dtype=np.float64))
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["payload"]["sha256"] = _sha256(payload)
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="must be an NPZ archive"):
        load_gauge_tree_prior_artifact(manifest)


def test_manifest_and_expected_identity_tampering_fail_closed(tmp_path: Path) -> None:
    prior = _prior()
    manifest, _ = save_gauge_tree_prior_artifact(tmp_path / "prior.json", prior)
    wrong_prior_id = "0" * 64 if prior.prior_id != "0" * 64 else "1" * 64
    with pytest.raises(ValueError, match="expected identity"):
        load_gauge_tree_prior_artifact(manifest, expected_prior_id=wrong_prior_id)

    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["prior"]["gauge_count"] = True
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the payload"):
        load_gauge_tree_prior_artifact(manifest)


def test_manifest_rejects_duplicate_keys_and_path_traversal(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid sparse gauge-tree artifact manifest"):
        load_gauge_tree_prior_artifact(duplicate)

    manifest, _ = save_gauge_tree_prior_artifact(tmp_path / "prior.json", _prior())
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["payload"]["path"] = "../outside.npz"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="below the manifest directory"):
        load_gauge_tree_prior_artifact(manifest)
