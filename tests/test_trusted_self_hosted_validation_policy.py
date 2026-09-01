"""Load the unchanged self-hosted policy suite and extend its reviewed allowlist.

The historical policy implementation is retained byte-for-byte in the adjacent
non-test module. This wrapper adds separately reviewed DOT continuation and
Tracking-Cloth query-portfolio workflows before exposing the original test
functions to pytest.
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
DOT_ROPE_QUERY_SELECTIVE_SOURCE_SUPPORT_V2_EXECUTION_WORKFLOW = (
    POLICY.WORKFLOW_ROOT / "dot-rope-query-selective-source-support-v2-execute.yml"
)
POLICY.TRUSTED_SELF_HOSTED_WORKFLOWS = (
    *POLICY.TRUSTED_SELF_HOSTED_WORKFLOWS,
    DOT_ROPE_QUERY_SELECTIVE_HELDOUT_WORKFLOW,
    DOT_ROPE_R04_R10_ARCHIVE_RECOVERY_V2_WORKFLOW,
    DOT_ROPE_R04_R10_HOSTED_ARCHIVE_RELAY_WORKFLOW,
    DOT_R04_R10_CAMERA_SUPPORT_AUDIT_WORKFLOW,
    TRACKING_CLOTH_QUERY_PORTFOLIO_WORKFLOW,
    DOT_ROPE_QUERY_SELECTIVE_SOURCE_SUPPORT_V2_EXECUTION_WORKFLOW,
)

for _name, _value in vars(POLICY).items():
    if _name.startswith("test_"):
        globals()[_name] = _value
