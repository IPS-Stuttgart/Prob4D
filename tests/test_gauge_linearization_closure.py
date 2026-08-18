from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from prob4d.gauge_linearization_closure import (
    GaugeLinearizationCaseV1,
    GaugeLinearizationClosureV1,
    GaugeLinearizationPolicyV1,
    build_gauge_linearization_closure,
    evaluate_gauge_linearization_case,
    linearize_sim3_chain,
    load_gauge_linearization_closure,
    main,
    write_gauge_linearization_closure,
)
from prob4d.sim3 import Sim3


def policy(
    *,
    minimum_group_count: int = 1,
    minimum_group_pass_fraction: float = 1.0,
) -> GaugeLinearizationPolicyV1:
    return GaugeLinearizationPolicyV1(
        minimum_group_count=minimum_group_count,
        minimum_group_pass_fraction=minimum_group_pass_fraction,
        minimum_branch_cut_clearance_radians=0.05,
        maximum_normalized_mean_shift=0.02,
        maximum_relative_covariance_frobenius_error=0.05,
        maximum_directional_variance_ratio_deviation=0.1,
        maximum_variance_outside_linear_support_fraction=0.01,
    )


def small_case(case_id: str = "small", group_id: str = "object-a") -> GaugeLinearizationCaseV1:
    vectors = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.02, -0.01, 0.03, 0.1, 0.0, 0.0],
        ]
    )
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, 0.2, 0.3]])
    query = np.zeros((2, len(points), 3))
    query[0, 0, 0] = 1.0
    query[0, 2, 0] = -1.0
    query[1, 1, 1] = 1.0
    return GaugeLinearizationCaseV1(
        case_id=case_id,
        group_id=group_id,
        transform_vectors=vectors,
        joint_covariance=np.eye(14) * 1e-5,
        points_local_m=points,
        query_jacobian=query,
        metadata={"evidence_partition": "source-diagnostic"},
    )


def nonlinear_case(
    case_id: str = "nonlinear",
    group_id: str = "object-b",
) -> GaugeLinearizationCaseV1:
    return GaugeLinearizationCaseV1(
        case_id=case_id,
        group_id=group_id,
        transform_vectors=np.array(
            [
                [0.0, 0.8, -0.4, 0.5, 0.1, 0.0, 0.0],
                [0.1, 0.5, 0.2, -0.4, 0.0, 0.2, 0.0],
            ]
        ),
        joint_covariance=np.eye(14) * 0.04,
        points_local_m=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, 0.2, 0.3]]
        ),
    )


def _compose_vectors(flat: np.ndarray, transform_count: int) -> np.ndarray:
    vectors = flat.reshape(transform_count, 7)
    current = Sim3.from_vector(vectors[0])
    for vector in vectors[1:]:
        current = current.compose(Sim3.from_vector(vector))
    return current.as_vector()


def test_chain_jacobian_matches_central_difference() -> None:
    vectors = small_case().transform_vectors.copy()
    _, jacobian = linearize_sim3_chain(vectors)
    flat = vectors.reshape(-1)
    numerical = np.empty_like(jacobian)
    for index in range(flat.size):
        step = 1e-6 * max(1.0, abs(float(flat[index])))
        plus = flat.copy()
        minus = flat.copy()
        plus[index] += step
        minus[index] -= step
        numerical[:, index] = (
            _compose_vectors(plus, len(vectors))
            - _compose_vectors(minus, len(vectors))
        ) / (2.0 * step)
    np.testing.assert_allclose(jacobian, numerical, atol=2e-8, rtol=2e-8)


def test_small_joint_gauge_case_passes_point_and_query_closure() -> None:
    report = evaluate_gauge_linearization_case(small_case(), policy())

    assert report["closure_passed"] is True
    assert report["failure_reasons"] == []
    assert report["gauge_rank"] == 14
    assert report["maximum_point_normalized_mean_shift"] is not None
    assert report["maximum_point_normalized_mean_shift"] < 0.002
    assert report["query_relative_covariance_frobenius_error"] is not None
    assert report["query_relative_covariance_frobenius_error"] < 0.001


def test_material_nonlinearity_fails_before_point_covariance_development() -> None:
    report = evaluate_gauge_linearization_case(nonlinear_case(), policy())

    assert report["closure_passed"] is False
    assert "point-mean-shift" in report["failure_reasons"]
    assert "point-covariance-frobenius" in report["failure_reasons"]
    assert "point-directional-variance" in report["failure_reasons"]

    artifact = build_gauge_linearization_closure(
        representation_name="joint-sim3-chain-v1",
        policy=policy(),
        cases=(nonlinear_case(),),
    )
    assert artifact.decision == "linearization-closure-negative"
    assert artifact.to_dict()["point_covariance_development_authorized"] is False


def test_exact_pi_mean_chain_is_a_valid_branch_cut_negative() -> None:
    case = GaugeLinearizationCaseV1(
        case_id="pi",
        group_id="object-pi",
        transform_vectors=np.array([[0.0, np.pi, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        joint_covariance=np.eye(7) * 1e-5,
        points_local_m=np.array([[1.0, 0.0, 0.0]]),
    )

    report = evaluate_gauge_linearization_case(case, policy())

    assert report["closure_passed"] is False
    assert report["branch_cut_safe"] is False
    assert report["failure_reasons"] == ["mean-chain-branch-cut"]
    assert report["maximum_point_normalized_mean_shift"] is None


def test_groups_receive_equal_mass_and_require_all_registered_cases() -> None:
    artifact = build_gauge_linearization_closure(
        representation_name="joint-sim3-chain-v1",
        policy=policy(minimum_group_count=2, minimum_group_pass_fraction=0.5),
        cases=(
            small_case("a-pass", "object-a"),
            small_case("a-pass-2", "object-a"),
            nonlinear_case("b-fail", "object-b"),
        ),
    )

    assert artifact.group_count == 2
    assert artifact.passing_group_count == 1
    assert artifact.group_pass_fraction == pytest.approx(0.5)
    assert artifact.decision == "linearization-closure-adequate"
    assert artifact.group_reports[1]["closure_passed"] is False


def test_artifact_is_permutation_invariant_and_tamper_evident(tmp_path) -> None:
    first = build_gauge_linearization_closure(
        representation_name="joint-sim3-chain-v1",
        policy=policy(minimum_group_count=2, minimum_group_pass_fraction=0.5),
        cases=(small_case(), nonlinear_case()),
        metadata={"target_outcomes_used": False},
    )
    second = build_gauge_linearization_closure(
        representation_name="joint-sim3-chain-v1",
        policy=policy(minimum_group_count=2, minimum_group_pass_fraction=0.5),
        cases=(nonlinear_case(), small_case()),
        metadata={"target_outcomes_used": False},
    )
    assert first.to_dict() == second.to_dict()

    path = tmp_path / "closure.json"
    write_gauge_linearization_closure(first, path)
    assert load_gauge_linearization_closure(path).to_dict() == first.to_dict()
    write_gauge_linearization_closure(first, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reports"][0]["maximum_point_normalized_mean_shift"] += 0.1
    with pytest.raises(ValueError, match="deterministic replay"):
        GaugeLinearizationClosureV1.from_dict(payload)

    different = replace(first, metadata={"variant": "different"})
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_gauge_linearization_closure(different, path)


def test_case_loader_rejects_coercive_numbers() -> None:
    payload = small_case().to_dict()
    payload["transform_vectors"][0][0] = "0.0"
    with pytest.raises(ValueError, match="real number"):
        GaugeLinearizationCaseV1.from_dict(payload)


def test_cli_build_verify_and_require_pass(tmp_path, capsys) -> None:
    passing_case = small_case()
    raw = {
        "representation_name": "joint-sim3-chain-v1",
        "policy": policy().to_dict(),
        "cases": [passing_case.to_dict()],
        "evidence_partition": "source-diagnostic",
        "target_outcomes_used": False,
        "metadata": {},
    }
    input_path = tmp_path / "raw.json"
    output_path = tmp_path / "closure.json"
    input_path.write_text(json.dumps(raw), encoding="utf-8")

    assert main(["build", str(input_path), "--output", str(output_path), "--require-pass"]) == 0
    artifact_id = capsys.readouterr().out.strip()
    assert len(artifact_id) == 64
    assert main(["verify", str(output_path), "--require-pass"]) == 0
    assert capsys.readouterr().out.strip() == artifact_id

    raw["cases"] = [nonlinear_case().to_dict()]
    negative_input = tmp_path / "negative.json"
    negative_output = tmp_path / "negative-artifact.json"
    negative_input.write_text(json.dumps(raw), encoding="utf-8")
    assert (
        main(
            [
                "build",
                str(negative_input),
                "--output",
                str(negative_output),
                "--require-pass",
            ]
        )
        == 3
    )


def test_zero_rank_covariance_replays_exact_mean() -> None:
    case = GaugeLinearizationCaseV1(
        case_id="deterministic",
        group_id="object-deterministic",
        transform_vectors=np.zeros((1, 7)),
        joint_covariance=np.zeros((7, 7)),
        points_local_m=np.array([[1.0, 2.0, 3.0]]),
    )

    report = evaluate_gauge_linearization_case(case, policy())

    assert report["gauge_rank"] == 0
    assert report["closure_passed"] is True
    assert report["maximum_point_normalized_mean_shift"] == pytest.approx(0.0)
    assert report["maximum_point_relative_covariance_frobenius_error"] == pytest.approx(
        0.0
    )


def test_insufficient_group_count_is_not_a_positive_decision() -> None:
    artifact = build_gauge_linearization_closure(
        representation_name="joint-sim3-chain-v1",
        policy=policy(minimum_group_count=2),
        cases=(small_case(),),
    )

    assert artifact.group_count == 1
    assert artifact.decision == "insufficient-independent-groups"
    assert artifact.passed is False


def test_artifact_loader_rejects_coercive_summary_scalars(tmp_path) -> None:
    artifact = build_gauge_linearization_closure(
        representation_name="joint-sim3-chain-v1",
        policy=policy(),
        cases=(small_case(),),
    )
    payload = artifact.to_dict()
    payload["group_count"] = 1.0
    with pytest.raises(ValueError, match="group_count"):
        GaugeLinearizationClosureV1.from_dict(payload)

    payload = artifact.to_dict()
    payload["passed"] = 1
    with pytest.raises(ValueError, match="passed"):
        GaugeLinearizationClosureV1.from_dict(payload)


def test_loader_rejects_duplicate_keys_and_nonfinite_values(tmp_path) -> None:
    artifact = build_gauge_linearization_closure(
        representation_name="joint-sim3-chain-v1",
        policy=policy(),
        cases=(small_case(),),
    )
    path = tmp_path / "closure.json"
    write_gauge_linearization_closure(artifact, path)
    original = path.read_text(encoding="utf-8")

    schema_line = f'  "schema_name": "{artifact.to_dict()["schema_name"]}",'
    path.write_text(
        original.replace(schema_line, f"{schema_line}\n{schema_line}", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_gauge_linearization_closure(path)

    path.write_text(
        original.replace(
            '  "representation_name":',
            '  "poison": NaN,\n  "representation_name":',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_gauge_linearization_closure(path)
