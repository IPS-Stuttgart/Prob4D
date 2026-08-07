from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.gauge_tree_artifact import (
    GAUGE_TREE_ARTIFACT_MANIFEST,
    GAUGE_TREE_ARTIFACT_SCHEMA,
    GAUGE_TREE_ARTIFACT_VERSION,
    load_gauge_tree_prior_artifact,
    verify_gauge_tree_prior_artifact,
    write_gauge_tree_prior_artifact,
)
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1


def _prior(*, gauge_count: int = 4, source_digest: str | None = None):
    parents = np.asarray([-1] + [index - 1 for index in range(1, gauge_count)])
    transitions = np.zeros((gauge_count, 7, 7), dtype=np.float64)
    scales = np.empty_like(transitions)
    scales[0] = np.diag(np.linspace(0.05, 0.11, 7))
    for index in range(1, gauge_count):
        transitions[index] = np.eye(7) * (0.75 + 0.02 * index)
        scales[index] = np.diag(np.linspace(0.02, 0.04, 7) * (1 + index / 10))
    return GaugeTreeSquareRootPriorV1(
        gauge_ids=tuple(f"window-{index}" for index in range(gauge_count)),
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
        source_joint_covariance_sha256=source_digest,
    )


def _manifest(root: Path) -> dict[str, object]:
    return json.loads((root / GAUGE_TREE_ARTIFACT_MANIFEST).read_text(encoding="utf-8"))


def test_round_trip_preserves_exact_prior_and_source_binding(tmp_path: Path) -> None:
    prior = _prior(source_digest="a" * 64)
    root = tmp_path / "prior"
    summary = write_gauge_tree_prior_artifact(prior, root)
    loaded = load_gauge_tree_prior_artifact(root)

    assert summary.artifact_id == verify_gauge_tree_prior_artifact(root).artifact_id
    assert summary.prior_id == prior.prior_id == loaded.prior_id
    assert summary.source_joint_covariance_sha256 == "a" * 64
    assert np.array_equal(loaded.parent_indices, prior.parent_indices)
    assert np.array_equal(loaded.transition_matrices, prior.transition_matrices)
    assert np.array_equal(loaded.innovation_scale_tril, prior.innovation_scale_tril)
    assert not loaded.parent_indices.flags.writeable
    assert not loaded.transition_matrices.flags.writeable
    assert not loaded.innovation_scale_tril.flags.writeable

    manifest = _manifest(root)
    assert manifest["schema"] == GAUGE_TREE_ARTIFACT_SCHEMA
    assert manifest["version"] == GAUGE_TREE_ARTIFACT_VERSION
    assert manifest["artifact_id"] == summary.artifact_id
    assert summary.serialized_nbytes == sum(path.stat().st_size for path in root.iterdir())


def test_artifact_identity_and_npy_members_are_deterministic(tmp_path: Path) -> None:
    prior = _prior()
    left = tmp_path / "left"
    right = tmp_path / "right"
    left_summary = write_gauge_tree_prior_artifact(prior, left)
    right_summary = write_gauge_tree_prior_artifact(prior, right)

    assert left_summary.artifact_id == right_summary.artifact_id
    assert _manifest(left) == _manifest(right)
    assert (left / "transition_matrices.npy").read_bytes() == (
        right / "transition_matrices.npy"
    ).read_bytes()
    assert (left / "innovation_scale_tril.npy").read_bytes() == (
        right / "innovation_scale_tril.npy"
    ).read_bytes()


def test_writer_is_no_clobber(tmp_path: Path) -> None:
    root = tmp_path / "prior"
    write_gauge_tree_prior_artifact(_prior(), root)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_gauge_tree_prior_artifact(_prior(), root)


def test_member_byte_tampering_fails_before_numpy_decode(tmp_path: Path) -> None:
    root = tmp_path / "prior"
    write_gauge_tree_prior_artifact(_prior(), root)
    member = root / "transition_matrices.npy"
    payload = bytearray(member.read_bytes())
    payload[-1] ^= 1
    member.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256"):
        load_gauge_tree_prior_artifact(root)


def test_manifest_duplicate_keys_and_unknown_files_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "prior"
    write_gauge_tree_prior_artifact(_prior(), root)
    manifest_path = root / GAUGE_TREE_ARTIFACT_MANIFEST
    original = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text('{"schema":"duplicate",' + original[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key 'schema'"):
        load_gauge_tree_prior_artifact(root)

    manifest_path.write_text(original, encoding="utf-8")
    (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="files changed"):
        load_gauge_tree_prior_artifact(root)


def test_manifest_cannot_redirect_members_or_alias_boolean_indices(tmp_path: Path) -> None:
    root = tmp_path / "prior"
    write_gauge_tree_prior_artifact(_prior(), root)
    manifest_path = root / GAUGE_TREE_ARTIFACT_MANIFEST
    manifest = _manifest(root)
    members = manifest["members"]
    assert isinstance(members, dict)
    transition = members["transition_matrices"]
    assert isinstance(transition, dict)
    transition["path"] = "../transition_matrices.npy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="must be exactly"):
        load_gauge_tree_prior_artifact(root)

    manifest = _manifest(root)
    transition = manifest["members"]["transition_matrices"]
    transition["path"] = "transition_matrices.npy"
    manifest["parent_indices"][1] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=r"parent_indices\[1\] must be an integer"):
        load_gauge_tree_prior_artifact(root)


def test_symlinked_root_and_member_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "prior"
    write_gauge_tree_prior_artifact(_prior(), root)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="root must not be a symlink"):
        load_gauge_tree_prior_artifact(alias)

    member = root / "transition_matrices.npy"
    external = tmp_path / "external.npy"
    member.replace(external)
    member.symlink_to(external)
    with pytest.raises(ValueError, match="members must not be symlinks"):
        load_gauge_tree_prior_artifact(root)


def test_member_size_and_npy_header_are_bounded_before_array_loading(tmp_path: Path) -> None:
    root = tmp_path / "prior"
    write_gauge_tree_prior_artifact(_prior(), root)
    manifest_path = root / GAUGE_TREE_ARTIFACT_MANIFEST
    manifest = _manifest(root)
    transition = manifest["members"]["transition_matrices"]
    transition["bytes"] += 5000
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="serialized byte count"):
        load_gauge_tree_prior_artifact(root)

    write_root = tmp_path / "header-prior"
    write_gauge_tree_prior_artifact(_prior(), write_root)
    manifest_path = write_root / GAUGE_TREE_ARTIFACT_MANIFEST
    manifest = _manifest(write_root)
    transition = manifest["members"]["transition_matrices"]
    wrong = np.zeros((4, 1, 49), dtype=np.float64)
    stream = io.BytesIO()
    np.save(stream, wrong, allow_pickle=False)
    payload = stream.getvalue()
    (write_root / "transition_matrices.npy").write_bytes(payload)
    transition["bytes"] = len(payload)
    transition["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="NPY header shape"):
        load_gauge_tree_prior_artifact(write_root)
