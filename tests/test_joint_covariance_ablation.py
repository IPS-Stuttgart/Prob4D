from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import prob4d.joint_covariance_ablation as ablation
from prob4d.joint_covariance_ablation import (
    JOINT_COVARIANCE_ABLATION_SCHEMA,
    MAX_BOOTSTRAP_REPLICATES,
    compare_joint_covariance_ablations,
    run_joint_covariance_ablation,
)


def _correlated_groups(
    *,
    group_count: int = 48,
    rows_per_group: int = 6,
    rank: int = 2,
    seed: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    residuals: list[np.ndarray] = []
    local_covariances: list[np.ndarray] = []
    factors: list[np.ndarray] = []
    group_ids: list[int] = []
    for group_id in range(group_count):
        local = np.repeat((0.4 * np.eye(3))[None], rows_per_group, axis=0)
        factor = generator.normal(scale=0.45, size=(rows_per_group, 3, rank))
        dense_local = np.zeros((3 * rows_per_group, 3 * rows_per_group))
        for row in range(rows_per_group):
            dense_local[
                3 * row : 3 * row + 3,
                3 * row : 3 * row + 3,
            ] = local[row]
        dense_factor = factor.reshape(3 * rows_per_group, rank)
        covariance = dense_local + dense_factor @ dense_factor.T
        residual = generator.multivariate_normal(
            np.zeros(3 * rows_per_group),
            covariance,
        ).reshape(rows_per_group, 3)
        residuals.append(residual)
        local_covariances.append(local)
        factors.append(factor)
        group_ids.extend([group_id] * rows_per_group)
    return (
        np.concatenate(residuals),
        np.concatenate(local_covariances),
        np.concatenate(factors),
        np.asarray(group_ids, dtype=np.int64),
    )


def test_marginal_preserving_ablation_keeps_every_row_marginal() -> None:
    generator = np.random.default_rng(11)
    local_basis = generator.normal(size=(5, 3, 3))
    local = local_basis @ np.swapaxes(local_basis, -1, -2) + 0.2 * np.eye(3)
    factor = generator.normal(scale=0.3, size=(5, 3, 4))

    actual = ablation._marginal_preserving_local_covariance(local, factor)
    expected = local + np.einsum("nir,njr->nij", factor, factor)

    np.testing.assert_allclose(actual, expected)


def test_joint_proper_score_beats_dependence_ablations() -> None:
    residual, local, factor, groups = _correlated_groups()

    result = compare_joint_covariance_ablations(
        residual,
        local,
        factor,
        factor_group_ids=groups,
        bootstrap_replicates=500,
        bootstrap_seed=17,
    )

    marginal = result["comparisons"]["marginal_preserving_independence"]
    conditional = result["comparisons"]["conditional_only"]
    marginal_nll = marginal["equal_group_mean"][
        "gaussian_nll_per_dimension_advantage"
    ]
    marginal_interval = marginal["paired_group_bootstrap"]["metrics"][
        "gaussian_nll_per_dimension_advantage"
    ]

    assert marginal_nll > 0.0
    assert marginal_interval["lower"] > 0.0
    assert (
        marginal["joint_better_group_fraction"][
            "gaussian_nll_per_dimension_advantage"
        ]
        > 0.5
    )
    assert (
        conditional["equal_group_mean"]["gaussian_nll_per_dimension_advantage"]
        > 0.0
    )
    assert 0.75 < result["arms"]["joint"]["equal_group_mean"]["normalized_nees"] < 1.25


def test_zero_shared_factor_makes_all_arms_identical() -> None:
    generator = np.random.default_rng(19)
    residual = generator.normal(size=(8, 3))
    local = np.repeat(np.eye(3)[None], 8, axis=0)
    factor = np.empty((8, 3, 0))
    groups = np.repeat(np.arange(4), 2)

    result = compare_joint_covariance_ablations(
        residual,
        local,
        factor,
        factor_group_ids=groups,
        bootstrap_replicates=50,
    )

    for name in ("marginal_preserving_independence", "conditional_only"):
        comparison = result["comparisons"][name]
        assert comparison["equal_group_mean"][
            "gaussian_nll_per_dimension_advantage"
        ] == pytest.approx(0.0)
        assert comparison["equal_group_mean"][
            "normalized_nees_absolute_error_advantage"
        ] == pytest.approx(0.0)


def test_bootstrap_is_paired_equal_group_and_deterministic() -> None:
    residual, local, factor, groups = _correlated_groups(
        group_count=7,
        rows_per_group=3,
        seed=29,
    )

    first = compare_joint_covariance_ablations(
        residual,
        local,
        factor,
        factor_group_ids=groups,
        bootstrap_replicates=80,
        bootstrap_seed=23,
    )
    second = compare_joint_covariance_ablations(
        residual,
        local,
        factor,
        factor_group_ids=groups,
        bootstrap_replicates=80,
        bootstrap_seed=23,
    )

    comparison = first["comparisons"]["marginal_preserving_independence"]
    group_values = [
        row["gaussian_nll_per_dimension_advantage"]
        for row in comparison["groups"]
    ]
    assert comparison["equal_group_mean"][
        "gaussian_nll_per_dimension_advantage"
    ] == pytest.approx(float(np.mean(group_values)))
    assert first["bootstrap"]["unit"] == "factor_group"
    assert first == second


def test_unavailable_bootstrap_reasons_are_explicit() -> None:
    residual = np.zeros((2, 3))
    local = np.repeat(np.eye(3)[None], 2, axis=0)
    factor = np.zeros((2, 3, 1))

    single_group = compare_joint_covariance_ablations(
        residual,
        local,
        factor,
        bootstrap_replicates=20,
    )
    disabled = compare_joint_covariance_ablations(
        residual,
        local,
        factor,
        bootstrap_replicates=0,
    )

    single_summary = single_group["comparisons"]["conditional_only"][
        "paired_group_bootstrap"
    ]
    disabled_summary = disabled["comparisons"]["conditional_only"][
        "paired_group_bootstrap"
    ]
    assert single_summary["available"] is False
    assert single_summary["reason"] == "fewer-than-two-independent-groups"
    assert disabled_summary["available"] is False
    assert disabled_summary["reason"] == "bootstrap-disabled"


def test_report_is_bound_to_the_exact_bytes_read(
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
    report = run_joint_covariance_ablation(
        source,
        output,
        bootstrap_replicates=0,
    )

    assert report["source_sha256"] == hashlib.sha256(original_payload).hexdigest()
    assert report["evaluation"]["arms"]["joint"]["equal_group_mean"][
        "normalized_nees"
    ] == pytest.approx(0.0)


def test_cli_report_is_strict_and_no_clobber(tmp_path: Path) -> None:
    residual, local, factor, groups = _correlated_groups(
        group_count=3,
        rows_per_group=2,
    )
    source = tmp_path / "matched.npz"
    output = tmp_path / "report.json"
    np.savez(
        source,
        residual_xyz_m=residual,
        local_covariance_m2=local,
        low_rank_factor_m=factor,
        factor_group_ids=groups,
    )

    assert (
        ablation.main(
            [
                str(source),
                "--output",
                str(output),
                "--bootstrap-replicates",
                "20",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_name"] == JOINT_COVARIANCE_ABLATION_SCHEMA
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["evaluation"]["group_count"] == 3
    with pytest.raises(FileExistsError):
        run_joint_covariance_ablation(source, output)


def test_rejects_unknown_npz_members(tmp_path: Path) -> None:
    source = tmp_path / "matched.npz"
    np.savez(
        source,
        residual_xyz_m=np.zeros((1, 3)),
        local_covariance_m2=np.eye(3)[None],
        low_rank_factor_m=np.zeros((1, 3, 0)),
        unexpected=np.asarray(1),
    )

    with pytest.raises(ValueError, match="extra=.*unexpected"):
        run_joint_covariance_ablation(source, tmp_path / "report.json")


def test_rejects_invalid_bootstrap_controls() -> None:
    residual = np.zeros((1, 3))
    local = np.eye(3)[None]
    factor = np.zeros((1, 3, 0))

    with pytest.raises(ValueError, match="bootstrap_replicates"):
        compare_joint_covariance_ablations(
            residual,
            local,
            factor,
            bootstrap_replicates=-1,
        )
    with pytest.raises(ValueError, match="bootstrap_replicates"):
        compare_joint_covariance_ablations(
            residual,
            local,
            factor,
            bootstrap_replicates=MAX_BOOTSTRAP_REPLICATES + 1,
        )
    with pytest.raises(ValueError, match="bootstrap_seed"):
        compare_joint_covariance_ablations(
            residual,
            local,
            factor,
            bootstrap_seed=-1,
        )
    with pytest.raises(ValueError, match="confidence_level"):
        compare_joint_covariance_ablations(
            residual,
            local,
            factor,
            confidence_level=1.0,
        )
