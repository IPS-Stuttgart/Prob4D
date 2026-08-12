from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from prob4d._gauge_tree_common import canonical_json_sha256
from prob4d.cli import main as grouped_main
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1
from prob4d.gauge_tree_prior_artifact import (
    GAUGE_TREE_PRIOR_STORAGE_SEMANTICS,
    artifact_summary,
    load_gauge_tree_prior_artifact,
    write_gauge_tree_prior_artifact,
)
from prob4d.gauge_tree_prior_artifact import (
    main as artifact_main,
)
from prob4d.gauge_tree_prior_io import (
    gauge_tree_prior_artifact_id,
    load_gauge_tree_prior,
    write_gauge_tree_prior,
)


def _prior(*, gauge_count: int = 4, offset: float = 0.0) -> GaugeTreeSquareRootPriorV1:
    parents = np.asarray([-1, *(index - 1 for index in range(1, gauge_count))])
    transitions = np.zeros((gauge_count, 7, 7), dtype=np.float64)
    for index in range(1, gauge_count):
        transitions[index] = np.eye(7) * (0.70 + 0.01 * index + offset)
    scales = np.repeat(np.eye(7, dtype=np.float64)[None], gauge_count, axis=0)
    scales *= 0.10 + offset
    return GaugeTreeSquareRootPriorV1(
        gauge_ids=tuple(f"gauge-{index}" for index in range(gauge_count)),
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
        source_joint_covariance_sha256="a" * 64,
    )


def _rewrite_manifest(path: Path, record: dict[str, object]) -> None:
    identity = {key: value for key, value in record.items() if key != "artifact_id"}
    record["artifact_id"] = canonical_json_sha256(identity)
    path.write_text(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_portable_prior_round_trip_without_dense_covariance(tmp_path: Path) -> None:
    prior = _prior(gauge_count=8)
    loaded = write_gauge_tree_prior_artifact(prior, tmp_path / "prior.json")

    assert loaded.prior.prior_id == prior.prior_id
    assert loaded.manifest.storage_semantics == GAUGE_TREE_PRIOR_STORAGE_SEMANTICS
    assert loaded.manifest.source_joint_covariance_sha256 == "a" * 64
    np.testing.assert_array_equal(loaded.prior.parent_indices, prior.parent_indices)
    np.testing.assert_array_equal(
        loaded.prior.transition_matrices,
        prior.transition_matrices,
    )
    np.testing.assert_array_equal(
        loaded.prior.innovation_scale_tril,
        prior.innovation_scale_tril,
    )
    assert len(tuple(tmp_path.glob("*.npy"))) == 3
    assert not tuple(tmp_path.glob("*.npz"))
    assert artifact_summary(loaded)["dense_covariance_nbytes"] == (8 * 7) ** 2 * 8
    with pytest.raises(ValueError):
        loaded.prior.parent_indices[0] = 0


def test_artifact_identity_and_manifest_bytes_are_location_independent(
    tmp_path: Path,
) -> None:
    prior = _prior()
    first_path = tmp_path / "first" / "prior.json"
    second_path = tmp_path / "second" / "different-name.json"

    first = write_gauge_tree_prior_artifact(prior, first_path)
    second = write_gauge_tree_prior_artifact(prior, second_path)

    assert first.manifest.artifact_id == second.manifest.artifact_id
    assert first_path.read_bytes() == second_path.read_bytes()
    assert sorted(path.name for path in first_path.parent.glob("*.npy")) == sorted(
        path.name for path in second_path.parent.glob("*.npy")
    )


def test_public_artifact_identity_matches_published_manifest(tmp_path: Path) -> None:
    prior = _prior()
    expected = gauge_tree_prior_artifact_id(prior)
    published = write_gauge_tree_prior(prior, tmp_path / "prior.json")
    reloaded = load_gauge_tree_prior(tmp_path / "prior.json")

    assert published.manifest.artifact_id == expected
    assert reloaded.prior_id == prior.prior_id
    np.testing.assert_array_equal(reloaded.parent_indices, prior.parent_indices)


def test_publication_is_idempotent_and_refuses_different_content(tmp_path: Path) -> None:
    path = tmp_path / "prior.json"
    first = write_gauge_tree_prior_artifact(_prior(), path)
    second = write_gauge_tree_prior_artifact(_prior(), path)

    assert first.manifest.artifact_id == second.manifest.artifact_id
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_gauge_tree_prior_artifact(_prior(offset=0.01), path)


def test_manifest_rejects_unbounded_npy_header_claim(tmp_path: Path) -> None:
    manifest_path = tmp_path / "prior.json"
    write_gauge_tree_prior_artifact(_prior(), manifest_path)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    member = record["parent_indices"]
    data_bytes = 4 * np.dtype("<i8").itemsize
    member["byte_count"] = data_bytes + 65_537
    _rewrite_manifest(manifest_path, record)

    with pytest.raises(ValueError, match="bounded NPY header"):
        load_gauge_tree_prior_artifact(manifest_path)


def test_loader_rejects_oversized_payload_before_decoding(tmp_path: Path) -> None:
    manifest_path = tmp_path / "prior.json"
    loaded = write_gauge_tree_prior_artifact(_prior(), manifest_path)
    member_path = tmp_path / loaded.manifest.transition_matrices.path
    member_path.write_bytes(member_path.read_bytes() + b"x" * 70_000)

    with pytest.raises(ValueError, match="maximum byte count"):
        load_gauge_tree_prior_artifact(manifest_path)


def test_loader_rejects_large_header_shape_before_array_allocation(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "prior.json"
    write_gauge_tree_prior_artifact(_prior(), manifest_path)

    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer,
        {
            "descr": "<i8",
            "fortran_order": False,
            "shape": (10**12,),
        },
    )
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    new_name = f"gauge-tree-prior-parent-indices-{digest}.npy"
    (tmp_path / new_name).write_bytes(payload)

    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    member = record["parent_indices"]
    member["path"] = new_name
    member["byte_count"] = len(payload)
    member["file_sha256"] = digest
    _rewrite_manifest(manifest_path, record)

    with pytest.raises(ValueError, match="NPY header shape mismatch"):
        load_gauge_tree_prior_artifact(manifest_path)


def test_loader_rejects_payload_byte_tampering(tmp_path: Path) -> None:
    loaded = write_gauge_tree_prior_artifact(_prior(), tmp_path / "prior.json")
    payload = tmp_path / loaded.manifest.transition_matrices.path
    corrupted = bytearray(payload.read_bytes())
    corrupted[-1] ^= 1
    payload.write_bytes(corrupted)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_gauge_tree_prior_artifact(tmp_path / "prior.json")


def test_loader_rejects_trailing_payload_bytes_even_when_rehashed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "prior.json"
    loaded = write_gauge_tree_prior_artifact(_prior(), manifest_path)
    member = loaded.manifest.parent_indices
    old_path = tmp_path / member.path
    payload = old_path.read_bytes() + b"trailing"
    digest = hashlib.sha256(payload).hexdigest()
    new_name = f"gauge-tree-prior-parent-indices-{digest}.npy"
    (tmp_path / new_name).write_bytes(payload)

    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["parent_indices"]["path"] = new_name
    record["parent_indices"]["byte_count"] = len(payload)
    record["parent_indices"]["file_sha256"] = digest
    _rewrite_manifest(manifest_path, record)

    with pytest.raises(ValueError, match="trailing bytes"):
        load_gauge_tree_prior_artifact(manifest_path)


def test_loader_rejects_manifest_identity_tampering(tmp_path: Path) -> None:
    manifest_path = tmp_path / "prior.json"
    write_gauge_tree_prior_artifact(_prior(), manifest_path)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["artifact_id"] = "b" * 64
    manifest_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact ID mismatch"):
        load_gauge_tree_prior_artifact(manifest_path)


def test_loader_rejects_boolean_schema_version(tmp_path: Path) -> None:
    manifest_path = tmp_path / "prior.json"
    write_gauge_tree_prior_artifact(_prior(), manifest_path)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["schema_version"] = True
    _rewrite_manifest(manifest_path, record)

    with pytest.raises(ValueError, match="schema_version must be a positive integer"):
        load_gauge_tree_prior_artifact(manifest_path)


def test_loader_rejects_unknown_fields_and_duplicate_keys(tmp_path: Path) -> None:
    manifest_path = tmp_path / "prior.json"
    write_gauge_tree_prior_artifact(_prior(), manifest_path)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["unexpected"] = True
    manifest_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="fields changed"):
        load_gauge_tree_prior_artifact(manifest_path)

    manifest_path.write_text('{"schema":1,"schema":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_gauge_tree_prior_artifact(manifest_path)


@pytest.mark.parametrize("member_path", ["../escape.npy", "nested/value.npy", "..\\escape.npy"])
def test_loader_rejects_unconfined_member_paths(
    tmp_path: Path,
    member_path: str,
) -> None:
    manifest_path = tmp_path / "prior.json"
    write_gauge_tree_prior_artifact(_prior(), manifest_path)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["parent_indices"]["path"] = member_path
    _rewrite_manifest(manifest_path, record)

    with pytest.raises(ValueError, match="confined relative filename"):
        load_gauge_tree_prior_artifact(manifest_path)


def test_loader_rejects_symbolic_link_payload(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symbolic links are unavailable")
    loaded = write_gauge_tree_prior_artifact(_prior(), tmp_path / "prior.json")
    member_path = tmp_path / loaded.manifest.innovation_scale_tril.path
    target = tmp_path / "real.npy"
    member_path.replace(target)
    try:
        member_path.symlink_to(target.name)
    except OSError:
        pytest.skip("symbolic links are not permitted in this environment")

    with pytest.raises(ValueError, match="symbolic link"):
        load_gauge_tree_prior_artifact(tmp_path / "prior.json")


def test_validation_cli_reports_compact_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "prior.json"
    write_gauge_tree_prior_artifact(_prior(gauge_count=5), manifest_path)

    assert artifact_main([str(manifest_path), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["valid"] is True
    assert summary["gauge_count"] == 5
    assert summary["factor_storage_nbytes"] < summary["dense_covariance_nbytes"]


def test_grouped_cli_verifies_and_materializes_dense_covariance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prior = _prior(gauge_count=5)
    manifest_path = tmp_path / "prior.json"
    dense_path = tmp_path / "dense.npy"
    write_gauge_tree_prior_artifact(prior, manifest_path)

    assert grouped_main(["gauge", "prior", "verify", str(manifest_path), "--json"]) == 0
    verify_summary = json.loads(capsys.readouterr().out)
    assert verify_summary["artifact_id"] == gauge_tree_prior_artifact_id(prior)

    assert (
        grouped_main(
            [
                "gauge",
                "prior",
                "materialize",
                str(manifest_path),
                str(dense_path),
                "--maximum-gauges",
                "5",
                "--json",
            ]
        )
        == 0
    )
    materialized = json.loads(capsys.readouterr().out)
    assert materialized["dense_output"] == str(dense_path)
    np.testing.assert_allclose(
        np.load(dense_path, allow_pickle=False),
        prior.materialize_dense_covariance(maximum_gauges=5),
        atol=0.0,
        rtol=0.0,
    )

    assert (
        grouped_main(
            [
                "gauge",
                "prior",
                "materialize",
                str(manifest_path),
                str(dense_path),
            ]
        )
        == 2
    )
    assert "refusing to replace dense output" in capsys.readouterr().err


def test_grouped_dense_materialization_obeys_gauge_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "prior.json"
    write_gauge_tree_prior_artifact(_prior(gauge_count=4), manifest_path)

    assert (
        grouped_main(
            [
                "gauge",
                "prior",
                "materialize",
                str(manifest_path),
                str(tmp_path / "dense.npy"),
                "--maximum-gauges",
                "3",
            ]
        )
        == 2
    )
    assert "limited to 3 gauges" in capsys.readouterr().err


def test_validation_cli_fails_closed_for_invalid_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "invalid.json"
    manifest_path.write_text("{}", encoding="utf-8")

    assert artifact_main([str(manifest_path)]) == 2
    assert "invalid gauge-tree prior artifact" in capsys.readouterr().err
