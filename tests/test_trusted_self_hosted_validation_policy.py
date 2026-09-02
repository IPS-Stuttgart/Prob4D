"""Load the unchanged self-hosted policy suite and extend its reviewed allowlist.

The historical policy implementation is retained byte-for-byte in the adjacent
non-test module. This wrapper adds the separately reviewed DOT R11--R30,
DOT R04--R10 recovery/relay/camera-audit, Tracking-Cloth query-portfolio,
continuous-SO(2) calibration, header-only augmented-layout audit, and labelled
rod-pair source audit, DOT R11--R20 source-support execution, and its fail-closed
gpuserver6000 recovery workflow before exposing the original test functions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).with_name("_trusted_self_hosted_validation_policy_base.py")
MODULE_NAME = "trusted_self_hosted_validation_policy_base"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, BASE)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = POLICY
SPEC.loader.exec_module(POLICY)

DOT_ROPE_QUERY_SELECTIVE_HELDOUT_WORKFLOW = (
    POLICY.WORKFLOW_ROOT / "dot-rope-query-selective-heldout-v1.yml"
)
DOT_ROPE_R04_R10_ARCHIVE_RECOVERY_V2_WORKFLOW = (
    POLICY.WORKFLOW_ROOT / "recover-dot-r04-r10-archive-cache-v2.yml"
)
DOT_ROPE_R04_R10_HOSTED_ARCHIVE_RELAY_WORKFLOW = (
    POLICY.WORKFLOW_ROOT / "relay-dot-r04-r10-archive-v1.yml"
)
DOT_R04_R10_CAMERA_SUPPORT_AUDIT_WORKFLOW = (
    POLICY.WORKFLOW_ROOT / "dot-r04-r10-camera-support-audit-v1.yml"
)
TRACKING_CLOTH_QUERY_PORTFOLIO_WORKFLOW = (
    POLICY.WORKFLOW_ROOT / "tracking-cloth-query-portfolio-v1.yml"
)
TRACKING_CLOTH_CONTINUOUS_CALIBRATED_SO2_WORKFLOW = (
    POLICY.WORKFLOW_ROOT / "tracking-cloth-continuous-calibrated-so2-v1.yml"
)
TRACKING_CLOTH_AUGMENTED_HEADER_AUDIT_WORKFLOW = (
    POLICY.WORKFLOW_ROOT / "tracking-cloth-augmented-header-audit-v1.yml"
)
TRACKING_CLOTH_AUGMENTED_ROD_SOURCE_AUDIT_WORKFLOW = (
    POLICY.WORKFLOW_ROOT / "tracking-cloth-augmented-rod-source-audit-v1.yml"
)
DOT_ROPE_QUERY_SELECTIVE_SOURCE_SUPPORT_V2_EXECUTION_WORKFLOW = (
    POLICY.WORKFLOW_ROOT / "dot-rope-query-selective-source-support-v2-execute.yml"
)
DOT_ROPE_QUERY_SELECTIVE_SOURCE_SUPPORT_V2_RECOVERY_WORKFLOW = (
    POLICY.WORKFLOW_ROOT / "dot-rope-query-selective-source-support-v2-gpuserver6000-recovery.yml"
)
POLICY.TRUSTED_SELF_HOSTED_WORKFLOWS = (
    *POLICY.TRUSTED_SELF_HOSTED_WORKFLOWS,
    DOT_ROPE_QUERY_SELECTIVE_HELDOUT_WORKFLOW,
    DOT_ROPE_R04_R10_ARCHIVE_RECOVERY_V2_WORKFLOW,
    DOT_ROPE_R04_R10_HOSTED_ARCHIVE_RELAY_WORKFLOW,
    DOT_R04_R10_CAMERA_SUPPORT_AUDIT_WORKFLOW,
    TRACKING_CLOTH_QUERY_PORTFOLIO_WORKFLOW,
    TRACKING_CLOTH_CONTINUOUS_CALIBRATED_SO2_WORKFLOW,
    TRACKING_CLOTH_AUGMENTED_HEADER_AUDIT_WORKFLOW,
    TRACKING_CLOTH_AUGMENTED_ROD_SOURCE_AUDIT_WORKFLOW,
    DOT_ROPE_QUERY_SELECTIVE_SOURCE_SUPPORT_V2_EXECUTION_WORKFLOW,
    DOT_ROPE_QUERY_SELECTIVE_SOURCE_SUPPORT_V2_RECOVERY_WORKFLOW,
)


def test_continuous_so2_workflow_is_branch_bound_read_only_and_protected() -> None:
    text = TRACKING_CLOTH_CONTINUOUS_CALIBRATED_SO2_WORKFLOW.read_text(
        encoding="utf-8"
    )
    assert "science/tracking-cloth-continuous-calibrated-so2-v1" in text
    assert "pull_request_target:" not in text
    assert "environment: trusted-self-hosted-validation" in text
    assert "runs-on: [self-hosted, Linux, X64, gpuserver4090]" in text
    assert 'test "$RUNNER_NAME" = "workstation1"' in text
    assert 'test "$RUNNER_OS" = "Linux"' in text
    assert 'test "$RUNNER_ARCH" = "X64"' in text
    assert "persist-credentials: false" in text
    assert "target_side_threshold_tuning_allowed" in text
    assert "raw_data_publication_authorized" in text
    assert "learned_provider_promotion_authorized" in text
    assert 'evidence="${{ steps.runtime.outputs.root }}/evidence"' in text
    assert 'find "$evidence" -type f' in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "secrets." not in text
    assert "git push" not in text


def test_augmented_header_audit_is_header_only_read_only_and_protected() -> None:
    text = TRACKING_CLOTH_AUGMENTED_HEADER_AUDIT_WORKFLOW.read_text(
        encoding="utf-8"
    )
    assert "science/tracking-cloth-continuous-calibrated-so2-v1" in text
    assert "pull_request_target:" not in text
    assert "environment: trusted-self-hosted-validation" in text
    assert "runs-on: [self-hosted, Linux, X64, gpuserver4090]" in text
    assert 'test "$RUNNER_NAME" = "workstation1"' in text
    assert "marker_trajectory_value_parsing_allowed" in text
    assert "trajectory_hashing_allowed" in text
    assert "target_side_threshold_tuning_allowed" in text
    assert "raw_data_publication_authorized" in text
    assert 'evidence="${{ steps.runtime.outputs.root }}/evidence"' in text
    assert 'find "$evidence" -type f' in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "secrets." not in text
    assert "git push" not in text


def test_augmented_rod_source_audit_is_source_only_read_only_and_protected() -> None:
    text = TRACKING_CLOTH_AUGMENTED_ROD_SOURCE_AUDIT_WORKFLOW.read_text(
        encoding="utf-8"
    )
    assert "science/tracking-cloth-continuous-calibrated-so2-v1" in text
    assert "pull_request_target:" not in text
    assert "environment: trusted-self-hosted-validation" in text
    assert "runs-on: [self-hosted, Linux, X64, gpuserver4090]" in text
    assert 'test "$RUNNER_NAME" = "workstation1"' in text
    assert "self_collision_trajectory_access_allowed" in text
    assert "target_side_tuning_allowed" in text
    assert "raw_data_publication_authorized" in text
    assert 'evidence="${{ steps.runtime.outputs.root }}/evidence"' in text
    assert 'find "$evidence" -type f' in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "secrets." not in text
    assert "git push" not in text


for _name, _value in vars(POLICY).items():
    if _name.startswith("test_"):
        globals()[_name] = _value
