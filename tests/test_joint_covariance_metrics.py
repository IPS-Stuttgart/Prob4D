import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import prob4d.joint_covariance_metrics as joint_metrics
from prob4d.cli import main as grouped_main
from prob4d.joint_covariance_metrics import (
    JOINT_COVARIANCE_DIAGNOSTIC_SCHEMA,
    evaluate_joint_gaussian_group,
    evaluate_joint_gaussian_groups,
)


def test_woodbury_metrics_match_dense_gaussian() -> None:
    generator = np.random.default_rng(7)
    residual = generator.normal(size=(5, 3))
    basis = generator.normal(size=(5, 3, 3))
    local = basis @ np.swapaxes(basis, -1, -2) + 0.2 * np.eye(3)
    factor = generator.normal(scale=0.3, size=(5, 3, 4))

    result = evaluate_joint_gaussian_group(residual, local, factor)
    dense_local = np.zeros((15, 15))
    for index in range(5):
        dense_local[
            3 * index : 3 * index + 3,
            3 * index : 3 * index + 3,
        ] = local[index]
    dense_factor = factor.reshape(15, 4)
    covariance = dense_local + dense_factor @ dense_factor.T
    flat_residual = residual.reshape(-1)
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign == 1.0
    mahalanobis = float(flat_residual @ np.linalg.solve(covariance, flat_residual))
    nll = 0.5 * (15 * np.log(2.0 * np.pi) + logdet + mahalanobis)

    assert result["mahalanobis_squared"] == pytest.approx(mahalanobis, rel=1e-11)
    assert result["joint_log_determinant"] == pytest.approx(logdet, rel=1e-11)
    assert result["gaussian_nll"] == pytest.approx(nll, rel=1e-11)
    assert result["effective_shared_rank"] == 4


def test_subspace_energies_match_direct_svd_reference() -> None:
    generator = np.random.default_rng(11)
    residual = generator.normal(size=(8, 3))
    local_basis = generator.normal(size=(8, 3, 3))
    local = local_basis @ np.swapaxes(local_basis, -1, -2) + 0.4 * np.eye(3)
    factor = generator.normal(scale=0.2, size=(8, 3, 5))

    result = evaluate_joint_gaussian_group(residual, local, factor)

    cholesky = np.linalg.cholesky(local)
    whitened_residual = np.linalg.solve(
        cholesky,
        residual[..., None],
    )[..., 0].reshape(-1)
    whitened_factor = np.linalg.solve(cholesky, factor).reshape(-1, factor.shape[-1])
    left, singular_values, _ = np.linalg.svd(whitened_factor, full_matrices=False)
    rank = int(result["effective_shared_rank"])
    coefficients = left[:, :rank].T @ whitened_residual
    shared = float(
        np.sum(np.square(coefficients) / (1.0 + np.square(singular_values[:rank])))
    )
    local_energy = float(whitened_residual @ whitened_residual)
    conditional = max(local_energy - float(coefficients @ coefficients), 0.0)

    assert result["shared_subspace_normalized_energy"] == pytest.approx(shared / rank)
    assert result["conditional_subspace_normalized_energy"] == pytest.approx(
        conditional / (whitened_residual.size - rank)
    )


def test_evaluation_does_not_call_tall_svd(monkeypatch: pytest.MonkeyPatch) -> None:
    residual = np.ones((4, 3))
    local = np.repeat(np.eye(3)[None], 4, axis=0)
    factor = np.ones((4, 3, 2)) * 0.1

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("tall SVD should not be used")

    monkeypatch.setattr(joint_metrics.np.linalg, "svd", fail)
    result = evaluate_joint_gaussian_group(residual, local, factor)
    assert result["effective_shared_rank"] == 1


def test_zero_shared_factor_reduces_to_block_diagonal_model() -> None:
    residual = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 0.25]])
    local = np.repeat(np.eye(3)[None], 2, axis=0)
    factor = np.empty((2, 3, 0))

    result = evaluate_joint_gaussian_group(residual, local, factor)

    expected_energy = float(np.sum(np.square(residual)))
    assert result["mahalanobis_squared"] == pytest.approx(expected_energy)
    assert result["effective_shared_rank"] == 0
    assert result["shared_subspace_normalized_energy"] is None
    assert result["conditional_subspace_normalized_energy"] == pytest.approx(
        expected_energy / 6
    )


def test_groups_are_aggregated_with_equal_weight() -> None:
    residual = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    local = np.repeat(np.eye(3)[None], 3, axis=0)
    factor = np.empty((3, 3, 0))
    groups = np.array([10, 20, 20])

    result = evaluate_joint_gaussian_groups(
        residual,
        local,
        factor,
        factor_group_ids=groups,
    )

    by_group = {row["factor_group_id"]: row for row in result["groups"]}
    expected = 0.5 * (
        by_group[10]["normalized_nees"] + by_group[20]["normalized_nees"]
    )
    assert result["equal_group_mean"]["normalized_nees"] == pytest.approx(expected)
    assert result["group_count"] == 2


def test_rejects_non_positive_definite_local_covariance() -> None:
    residual = np.ones((1, 3))
    local = np.diag([1.0, 1.0, 0.0])[None]
    factor = np.empty((1, 3, 0))

    with pytest.raises(ValueError, match="positive definite"):
        evaluate_joint_gaussian_group(residual, local, factor)


def test_cli_writes_content_bound_report(tmp_path: Path) -> None:
    source = tmp_path / "matched.npz"
    output = tmp_path / "report.json"
    np.savez(
        source,
        residual_xyz_m=np.ones((2, 3)),
        local_covariance_m2=np.repeat(np.eye(3)[None], 2, axis=0),
        low_rank_factor_m=np.zeros((2, 3, 1)),
        factor_group_ids=np.array(["a", "b"]),
    )

    assert (
        grouped_main(
            ["diagnostic", "joint-covariance", str(source), "--output", str(output)]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["schema_name"] == JOINT_COVARIANCE_DIAGNOSTIC_SCHEMA
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["evaluation"]["group_count"] == 2
    with pytest.raises(FileExistsError):
        grouped_main(
            ["diagnostic", "joint-covariance", str(source), "--output", str(output)]
        )


def test_diagnostic_evaluates_the_exact_hashed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "matched.npz"
    replacement = tmp_path / "replacement.npz"
    output = tmp_path / "report.json"
    local = np.repeat(np.eye(3)[None], 2, axis=0)
    factor = np.zeros((2, 3, 1))
    np.savez(
        source,
        residual_xyz_m=np.zeros((2, 3)),
        local_covariance_m2=local,
        low_rank_factor_m=factor,
    )
    np.savez(
        replacement,
        residual_xyz_m=np.ones((2, 3)),
        local_covariance_m2=local,
        low_rank_factor_m=factor,
    )
    original_read_bytes = Path.read_bytes
    original_payload = original_read_bytes(source)
    replacement_payload = original_read_bytes(replacement)
    replaced = False

    def read_then_replace(path: Path) -> bytes:
        nonlocal replaced
        payload = original_read_bytes(path)
        if path == source and not replaced:
            source.write_bytes(replacement_payload)
            replaced = True
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    report = joint_metrics.run_joint_covariance_diagnostic(source, output)

    assert report["source_sha256"] == hashlib.sha256(original_payload).hexdigest()
    assert report["evaluation"]["equal_group_mean"]["normalized_nees"] == pytest.approx(0.0)


def test_report_publication_preserves_a_concurrent_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "matched.npz"
    output = tmp_path / "report.json"
    np.savez(
        source,
        residual_xyz_m=np.zeros((1, 3)),
        local_covariance_m2=np.eye(3)[None],
        low_rank_factor_m=np.zeros((1, 3, 0)),
    )
    original_link = joint_metrics.os.link

    def publish_competitor(source_path: Any, destination_path: Any) -> None:
        Path(destination_path).write_text("concurrent writer\n", encoding="utf-8")
        original_link(source_path, destination_path)

    monkeypatch.setattr(joint_metrics.os, "link", publish_competitor)

    with pytest.raises(FileExistsError):
        joint_metrics.run_joint_covariance_diagnostic(source, output)

    assert output.read_text(encoding="utf-8") == "concurrent writer\n"
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))


def test_diagnostic_rejects_unknown_npz_members(tmp_path: Path) -> None:
    source = tmp_path / "matched.npz"
    np.savez(
        source,
        residual_xyz_m=np.zeros((1, 3)),
        local_covariance_m2=np.eye(3)[None],
        low_rank_factor_m=np.zeros((1, 3, 0)),
        unexpected=np.asarray(1),
    )

    with pytest.raises(ValueError, match="extra=.*unexpected"):
        joint_metrics.run_joint_covariance_diagnostic(source, tmp_path / "report.json")
