#!/usr/bin/env python3
"""Audit and exercise a built Prob4D source distribution."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

REQUIRED_PATHS = frozenset(
    {
        ".github/CODEOWNERS",
        ".github/dependabot.yml",
        ".github/workflows/ecosystem-release-capsule.yml",
        ".github/workflows/finite-sample-capability.yml",
        ".github/workflows/gauge-tree-prior-artifact.yml",
        ".github/workflows/generic-provider-import.yml",
        ".github/workflows/heldout-cohort-binding.yml",
        ".github/workflows/heldout-provider-promotion.yml",
        ".github/workflows/observation-bias-binding.yml",
        ".github/workflows/provider-evaluation-decision.yml",
        ".github/workflows/provider-neutral-common-mode.yml",
        ".github/workflows/provider-runtime.yml",
        ".github/workflows/recursive-visual-bias.yml",
        ".github/workflows/security-scanning.yml",
        ".github/workflows/target-provider-admission.yml",
        ".github/workflows/tests.yml",
        ".github/workflows/trusted-self-hosted-validation.yml",
        "CHANGELOG.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "SECURITY.md",
        "docs/ecosystem-release-capsule.md",
        "docs/examples/deform360-heldout-provider-promotion-config.json",
        "docs/examples/deform360-target-provider-admission-config.json",
        "docs/examples/heldout-provider-promotion-config.json",
        "docs/examples/material-identity-mixture-config.json",
        "docs/examples/provider-neutral-import-spec.json",
        "docs/finite-sample-capability.md",
        "docs/gauge-tree-prior-artifact.md",
        "docs/generic-provider-import.md",
        "docs/heldout-provider-promotion.md",
        "docs/joint-covariance-diagnostics.md",
        "docs/material-identity-cli.md",
        "docs/observation-bias-binding.md",
        "docs/prediction-window-storage.md",
        "docs/provider-evaluation-decisions.md",
        "docs/provider-neutral-predictions.md",
        "docs/provider-neutral-runtime.md",
        "docs/provider-v2.md",
        "docs/recursive-visual-bias.md",
        "docs/repository-identity.md",
        "docs/target-provider-admission.md",
        "docs/trusted-self-hosted-validation.md",
        "protocols/cycle-guard-conformal-v1.json",
        "protocols/cycle-guard-normalization-v1.json",
        "requirements/ci/minimum.txt",
        "scripts/ci/build_ecosystem_release_capsule.py",
        "scripts/ci/check_sdist.py",
        "src/prob4d/_deform360_cohort_binding.py",
        "src/prob4d/_deform360_cohort_io.py",
        "src/prob4d/_deform360_cohort_schema.py",
        "src/prob4d/_finite_sample_capability_build.py",
        "src/prob4d/_finite_sample_capability_common.py",
        "src/prob4d/_finite_sample_capability_derive.py",
        "src/prob4d/_finite_sample_capability_io.py",
        "src/prob4d/_finite_sample_capability_model.py",
        "src/prob4d/_finite_sample_capability_output.py",
        "src/prob4d/_finite_sample_capability_records.py",
        "src/prob4d/_finite_sample_capability_state.py",
        "src/prob4d/_gauge_tree_artifact_common.py",
        "src/prob4d/_gauge_tree_artifact_io.py",
        "src/prob4d/_heldout_promotion_common.py",
        "src/prob4d/_heldout_promotion_diagnosis.py",
        "src/prob4d/_heldout_promotion_evaluation.py",
        "src/prob4d/_heldout_promotion_lock.py",
        "src/prob4d/_heldout_promotion_query.py",
        "src/prob4d/_heldout_promotion_report.py",
        "src/prob4d/deform360_cohort_binding.py",
        "src/prob4d/finite_sample_capability.py",
        "src/prob4d/gauge_tree_prior_artifact.py",
        "src/prob4d/heldout_promotion.py",
        "src/prob4d/joint_covariance_metrics.py",
        "src/prob4d/material_identity_cli.py",
        "src/prob4d/observation_bias_binding.py",
        "src/prob4d/prediction_cli.py",
        "src/prob4d/prediction_provider_import.py",
        "src/prob4d/prediction_provider_manifest.py",
        "src/prob4d/prediction_provider_scaffold.py",
        "src/prob4d/promotion_evidence.py",
        "src/prob4d/provider_evaluation.py",
        "src/prob4d/provider_evaluation_target_authorization.py",
        "src/prob4d/provider_runtime.py",
        "src/prob4d/target_admission_enforcement.py",
        "src/prob4d/target_provider_admission.py",
        "src/prob4d/target_provider_admission_cli.py",
        "src/prob4d/visual_bias_stream.py",
        "tests/fixtures/prob4d_joint_observation_v1.json",
        "tests/test_cli.py",
        "tests/test_data_storage.py",
        "tests/test_deform360_cohort_binding.py",
        "tests/test_ecosystem_release_capsule.py",
        "tests/test_finite_sample_capability.py",
        "tests/test_finite_sample_capability_io.py",
        "tests/test_gauge_tree_prior_artifact.py",
        "tests/test_github_action_pins.py",
        "tests/test_heldout_promotion.py",
        "tests/test_heldout_promotion_diagnosis.py",
        "tests/test_joint_covariance_metrics.py",
        "tests/test_joint_observation_contract_fixture.py",
        "tests/test_material_identity_cli.py",
        "tests/test_observation_bias_binding.py",
        "tests/test_prediction_provider_import.py",
        "tests/test_prediction_provider_manifest.py",
        "tests/test_prediction_provider_scaffold.py",
        "tests/test_project_identity.py",
        "tests/test_promotion_evidence.py",
        "tests/test_provider_evaluation.py",
        "tests/test_provider_evaluation_decision.py",
        "tests/test_provider_evaluation_target_authorization.py",
        "tests/test_provider_runtime.py",
        "tests/test_release_metadata.py",
        "tests/test_security_scanning_workflow_policy.py",
        "tests/test_target_admission_enforcement.py",
        "tests/test_target_provider_admission.py",
        "tests/test_trusted_self_hosted_validation_policy.py",
        "tests/test_visual_bias_stream.py",
    }
)
REPRESENTATIVE_TESTS = (
    "tests/test_sim3.py",
    "tests/test_data_storage.py",
    "tests/test_deform360_cohort_binding.py",
    "tests/test_ecosystem_release_capsule.py",
    "tests/test_finite_sample_capability.py",
    "tests/test_finite_sample_capability_io.py",
    "tests/test_gauge_tree_prior_artifact.py",
    "tests/test_heldout_promotion.py",
    "tests/test_heldout_promotion_diagnosis.py",
    "tests/test_joint_covariance_metrics.py",
    "tests/test_material_identity_cli.py",
    "tests/test_observation_bias_binding.py",
    "tests/test_prediction_provider_import.py",
    "tests/test_prediction_provider_scaffold.py",
    "tests/test_promotion_evidence.py",
    "tests/test_provider_evaluation.py",
    "tests/test_provider_evaluation_decision.py",
    "tests/test_provider_evaluation_target_authorization.py",
    "tests/test_provider_manifest.py",
    "tests/test_joint_observation_contract_fixture.py",
    "tests/test_project_identity.py",
    "tests/test_release_metadata.py",
    "tests/test_security_scanning_workflow_policy.py",
    "tests/test_target_admission_enforcement.py",
    "tests/test_target_provider_admission.py",
    "tests/test_trusted_self_hosted_validation_policy.py",
    "tests/test_visual_bias_stream.py",
    "tests/test_github_action_pins.py",
)


def _validated_members(archive: Path) -> tuple[str, tuple[tarfile.TarInfo, ...]]:
    with tarfile.open(archive, "r:gz") as handle:
        members = tuple(handle.getmembers())
    if not members:
        raise RuntimeError("source distribution is empty")

    roots: set[str] = set()
    relative_paths: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise RuntimeError(f"unsafe source-distribution path: {member.name}")
        roots.add(path.parts[0])
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise RuntimeError(f"source distribution contains a non-regular member: {member.name}")
        if len(path.parts) > 1:
            relative_paths.add(PurePosixPath(*path.parts[1:]).as_posix())

    if len(roots) != 1:
        raise RuntimeError("source distribution must have exactly one root directory")
    missing = sorted(REQUIRED_PATHS - relative_paths)
    if missing:
        raise RuntimeError(f"source distribution omitted required assets: {missing}")
    return roots.pop(), members


def _extract_regular_files(archive: Path, destination: Path) -> str:
    root_name, members = _validated_members(archive)
    with tarfile.open(archive, "r:gz") as handle:
        for member in members:
            relative = PurePosixPath(member.name)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError(f"unable to read archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
    return root_name


def audit_sdist(archive: Path) -> None:
    archive = archive.resolve()
    if not archive.is_file():
        raise RuntimeError(f"source distribution does not exist: {archive}")
    with tempfile.TemporaryDirectory(prefix="prob4d-sdist-") as temporary:
        destination = Path(temporary)
        root_name = _extract_regular_files(archive, destination)
        source_root = destination / root_name
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_root / "src")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *REPRESENTATIVE_TESTS],
            cwd=source_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout + result.stderr)
        print(result.stdout, end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    arguments = parser.parse_args(argv)
    audit_sdist(arguments.archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
