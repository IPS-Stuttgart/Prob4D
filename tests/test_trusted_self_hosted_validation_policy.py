"""Load the unchanged self-hosted policy suite and extend its reviewed allowlist.

The historical policy implementation is retained byte-for-byte in the adjacent
non-test module. This wrapper adds the separately reviewed DOT R11--R30 and
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
TRACKING_CLOTH_QUERY_PORTFOLIO_WORKFLOW = (
    POLICY.WORKFLOW_ROOT / "tracking-cloth-query-portfolio-v1.yml"
)
POLICY.TRUSTED_SELF_HOSTED_WORKFLOWS = (
    *POLICY.TRUSTED_SELF_HOSTED_WORKFLOWS,
    DOT_ROPE_QUERY_SELECTIVE_HELDOUT_WORKFLOW,
    TRACKING_CLOTH_QUERY_PORTFOLIO_WORKFLOW,
)

for _name, _value in vars(POLICY).items():
    if _name.startswith("test_"):
        globals()[_name] = _value
