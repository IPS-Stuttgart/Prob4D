from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from prob4d.moment_collapse_diagnostic import (
    MomentCollapseDiagnosticV1,
    MomentCollapseThresholdsV1,
    SymmetricGaussianMixtureCaseV1,
    build_moment_collapse_diagnostic,
    evaluate_moment_collapse_case,
    load_moment_collapse_diagnostic,
    main,
    moment_matched_gaussian,
    write_moment_collapse_diagnostic,
)


def thresholds() -> MomentCollapseThresholdsV1:
    return MomentCollapseThresholdsV1(
        midpoint_half_width_component_sigma=1.0,
        minimum_component_mean_separation_sigma=4.0,
        maximum_mixture_midpoint_to_component_mean_density_ratio=0.1,
        minimum_moment_gaussian_central_mass_inflation=0.1,
    )


def separated_case(case_id: str = "separated") -> SymmetricGaussianMixtureCaseV1:
    return SymmetricGaussianMixtureCaseV1(
        case_id=case_id,
        offset_xyz=np.array([3.0, 0.0, 0.0]),
        component_covariance=np.eye(3),
        metadata={"mechanism": "symmetric-two-mode"},
    )


def weak_case(case_id: str = "weak") -> SymmetricGaussianMixtureCaseV1:
    return SymmetricGaussianMixtureCaseV1(
        case_id=case_id,
        offset_xyz=np.array([0.2, 0.0, 0.0]),
        component_covariance=np.eye(3),
    )


def test_separated_mixture_exposes_material_moment_collapse() -> None:
    report = evaluate_moment_collapse_case(separated_case(), thresholds())

    assert report.mahalanobis_offset_squared == pytest.approx(9.0)
    assert report.component_mean_separation_sigma == pytest.approx(6.0)
    assert report.mixture_midpoint_to_component_mean_density_ratio < 0.03
    assert report.mixture_central_mass < 0.03
    assert report.moment_gaussian_central_mass > 0.24
    assert report.moment_gaussian_central_mass_inflation > 0.20
    assert report.moment_matched_excess_kurtosis == pytest.approx(-1.62)
    assert report.material_moment_collapse is True


def test_weak_mixture_does_not_pass_materiality_gates() -> None:
    report = evaluate_moment_collapse_case(weak_case(), thresholds())

    assert report.component_mean_separation_sigma == pytest.approx(0.4)
    assert report.material_moment_collapse is False


def test_moment_matched_gaussian_preserves_only_first_two_moments() -> None:
    case = separated_case()
    mean, covariance = moment_matched_gaussian(case)

    np.testing.assert_array_equal(mean, np.zeros(3))
    np.testing.assert_array_equal(covariance, np.diag([10.0, 1.0, 1.0]))
    assert mean.flags.writeable is False
    assert covariance.flags.writeable is False
    with pytest.raises(ValueError):
        covariance.setflags(write=True)


def test_case_rejects_nonpositive_definite_covariance() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        SymmetricGaussianMixtureCaseV1(
            case_id="invalid",
            offset_xyz=np.ones(3),
            component_covariance=np.diag([1.0, 0.0, 1.0]),
        )



def test_case_loader_rejects_coercive_numeric_strings() -> None:
    payload = separated_case().to_dict()
    payload["offset_xyz"][0] = "3.0"
    with pytest.raises(ValueError, match="real number"):
        SymmetricGaussianMixtureCaseV1.from_dict(payload)


def test_diagnostic_is_permutation_invariant() -> None:
    first = build_moment_collapse_diagnostic(
        representation_name="mean-and-covariance-only",
        cases=(separated_case(), weak_case()),
        thresholds=thresholds(),
        metadata={"uses_real_provider_outcomes": False},
    )
    second = build_moment_collapse_diagnostic(
        representation_name="mean-and-covariance-only",
        cases=(weak_case(), separated_case()),
        thresholds=thresholds(),
        metadata={"uses_real_provider_outcomes": False},
    )

    assert first.artifact_id == second.artifact_id
    assert first.material_case_count == 1
    assert first.any_material_moment_collapse is True
    assert first.to_dict() == second.to_dict()


def test_round_trip_tamper_rejection_and_no_clobber(tmp_path) -> None:
    artifact = build_moment_collapse_diagnostic(
        representation_name="mean-and-covariance-only",
        cases=(separated_case(), weak_case()),
        thresholds=thresholds(),
    )
    path = tmp_path / "moment-collapse.json"
    write_moment_collapse_diagnostic(artifact, path)

    loaded = load_moment_collapse_diagnostic(path)
    assert loaded.to_dict() == artifact.to_dict()
    write_moment_collapse_diagnostic(artifact, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reports"][0]["moment_gaussian_central_mass"] -= 0.1
    with pytest.raises(ValueError, match="deterministic replay"):
        MomentCollapseDiagnosticV1.from_dict(payload)

    different = replace(artifact, metadata={"variant": "different"})
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_moment_collapse_diagnostic(different, path)


def test_loader_rejects_duplicate_keys_and_nonfinite_values(tmp_path) -> None:
    artifact = build_moment_collapse_diagnostic(
        representation_name="mean-and-covariance-only",
        cases=(separated_case(),),
        thresholds=thresholds(),
    )
    path = tmp_path / "moment-collapse.json"
    write_moment_collapse_diagnostic(artifact, path)
    original = path.read_text(encoding="utf-8")

    schema_line = f'  "schema_name": "{artifact.to_dict()["schema_name"]}",'
    path.write_text(
        original.replace(schema_line, f"{schema_line}\n{schema_line}", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_moment_collapse_diagnostic(path)

    path.write_text(
        original.replace(
            '  "representation_name":',
            '  "poison": NaN,\n  "representation_name":',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_moment_collapse_diagnostic(path)


def test_cli_build_and_verify(tmp_path, capsys) -> None:
    artifact = build_moment_collapse_diagnostic(
        representation_name="mean-and-covariance-only",
        cases=(separated_case(), weak_case()),
        thresholds=thresholds(),
    )
    raw = artifact.to_dict()
    for key in (
        "schema_name",
        "schema_version",
        "reports",
        "material_case_count",
        "any_material_moment_collapse",
        "claim_boundary",
        "artifact_id",
    ):
        raw.pop(key)
    input_path = tmp_path / "raw.json"
    output_path = tmp_path / "artifact.json"
    input_path.write_text(json.dumps(raw), encoding="utf-8")

    assert main(["build", str(input_path), "--output", str(output_path)]) == 0
    assert capsys.readouterr().out.strip() == artifact.artifact_id
    assert main(["verify", str(output_path)]) == 0
    assert capsys.readouterr().out.strip() == artifact.artifact_id
